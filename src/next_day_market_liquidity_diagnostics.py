"""Diagnostic slices for preselected 1-day models; never used for deployment.

Market regime is based on trailing 20-day median cross-sectional stock return,
not an official market index. Liquidity is close*volume, a daily traded-value proxy.
2025+ is a previously inspected retrospective period, NOT an untouched holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from . import next_day_validation_feature_models as exp

TRAIN_END = pd.Timestamp("2024-12-30")
TEST_START = pd.Timestamp("2025-01-02")
THRESHOLD = 0.52
VALUE_FLOORS_KRW = (0, 100_000_000, 1_000_000_000, 10_000_000_000)


def summarize(model_name, regime, floor, frame, selected):
    subset = frame.loc[selected]
    return {
        "model": model_name,
        "regime": regime,
        "min_daily_value_proxy_krw": floor,
        "eligible_rows": len(frame),
        "signals": len(subset),
        "signal_days": int(subset["date"].nunique()),
        "upward_hit_pct": float(100 * subset["target"].mean()) if len(subset) else np.nan,
        "avg_next_return_pct": float(100 * subset["next_return"].mean()) if len(subset) else np.nan,
        "median_next_return_pct": float(100 * subset["next_return"].median()) if len(subset) else np.nan,
    }


def run():
    previous = exp.LABEL_DATA_END
    try:
        exp.LABEL_DATA_END = pd.Timestamp("2100-01-01")
        data = exp.build_dataset(exp.DATA_DIR)
    finally:
        exp.LABEL_DATA_END = previous
    train = data.loc[data["date"] <= TRAIN_END]
    test = data.loc[data["date"] >= TEST_START].copy()
    if train.empty or test.empty:
        raise RuntimeError("Empty train or test")
    if train["date"].max() >= test["date"].min():
        raise RuntimeError("Train/test overlap")
    if not {"market_return_20d", "close", "volume"}.issubset(test.columns):
        raise RuntimeError("Missing required market/liquidity columns")

    # The market feature is known at the signal date; no future returns in regimes.
    market = test["market_return_20d"]
    test["regime"] = np.select(
        [market > 0.03, market < -0.03, market.notna()],
        ["up", "down", "sideways"],
        default="unknown",
    )
    # Approximation, not official traded value: assumes close*share volume in KRW.
    test["value_proxy"] = test["close"] * test["volume"]
    variants = {
        "baseline": exp.BASE_FEATURES,
        "volume_value_obv": exp.BASE_FEATURES + exp.FLOW_FEATURES,
    }
    results = []
    for model_name, features in variants.items():
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(**exp.HGB_BASELINE_PARAMS),
        )
        model.fit(train[features], train["target"].astype(int))
        test["signal"] = model.predict_proba(test[features])[:, 1] >= THRESHOLD
        for regime in ("all", "up", "sideways", "down"):
            group = test if regime == "all" else test.loc[test["regime"] == regime]
            for floor in VALUE_FLOORS_KRW:
                eligible = group.loc[
                    group["value_proxy"].notna() & (group["value_proxy"] >= floor)
                ]
                results.append(summarize(model_name, regime, floor, eligible, eligible["signal"]))

    output = pd.DataFrame(results)
    exp.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = exp.OUTPUT_DIR / "next_day_market_liquidity_diagnostics.csv"
    output.to_csv(path, index=False, encoding="utf-8-sig")
    print("DIAGNOSTIC ONLY; retrospective 2025+ data were previously inspected.")
    print("Regimes: trailing 20d median-stock proxy >3%, <-3%, otherwise sideways.")
    print("Value proxy = close * volume; no fees, slippage, or independent holdout.")
    print(output.to_string(index=False))
    return output


if __name__ == "__main__":
    run()
