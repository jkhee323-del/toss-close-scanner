"""Predeclared regime/liquidity strategy audit (research only).

Rule is specified using 2023-24 dates only; 2025+ is a previously inspected
retrospective check, NOT an independent final holdout. No deployment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from . import next_day_validation_feature_models as exp

DEV_END = pd.Timestamp("2024-12-30")
TEST_START = pd.Timestamp("2025-01-02")
THRESHOLD = 0.52
VALUE_FLOOR_KRW = 10_000_000_000
REGIME_DOWN_CUTOFF = -0.03
COSTS_ROUND_TRIP_PCT = (0.0, 0.2, 0.5)


def run():
    previous = exp.LABEL_DATA_END
    try:
        exp.LABEL_DATA_END = pd.Timestamp("2100-01-01")
        data = exp.build_dataset(exp.DATA_DIR)
    finally:
        exp.LABEL_DATA_END = previous

    # Exclude labels crossing the training cutoff.
    train = data.loc[(data["date"] <= DEV_END) & (data["next_date"] <= DEV_END)].copy()
    test = data.loc[data["date"] >= TEST_START].copy()
    if train.empty or test.empty:
        raise RuntimeError("Training or evaluation dataset empty")
    if train["next_date"].max() > DEV_END or train["date"].max() >= test["date"].min():
        raise RuntimeError("Temporal leakage detected")

    variants = {
        "baseline_unfiltered": (exp.BASE_FEATURES, False),
        "flow_unfiltered": (exp.BASE_FEATURES + exp.FLOW_FEATURES, False),
        "flow_fixed_filter": (exp.BASE_FEATURES + exp.FLOW_FEATURES, True),
    }
    results = []
    daily_rows = []
    for name, (features, filtered) in variants.items():
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(**exp.HGB_BASELINE_PARAMS),
        )
        model.fit(train[features], train["target"].astype(int))
        predicted = test.copy()
        predicted["probability"] = model.predict_proba(test[features])[:, 1]
        signal = predicted["probability"] >= THRESHOLD
        if filtered:
            signal &= (
                (predicted["market_return_20d"] < REGIME_DOWN_CUTOFF)
                & ((predicted["close"] * predicted["volume"]) >= VALUE_FLOOR_KRW)
            )
        chosen = predicted.loc[signal].copy()
        # Equal-weight same-day signals; flat on no-signal days.
        per_day = chosen.groupby("date")["next_return"].mean()
        calendar = pd.Index(sorted(test["date"].unique()), name="date")
        daily = per_day.reindex(calendar).fillna(0.0)
        counts = chosen.groupby("date").size().reindex(calendar).fillna(0).astype(int)
        # Conservative daily round-trip cost per active day; zero on idle days.
        for cost in COSTS_ROUND_TRIP_PCT:
            net = daily - (counts.gt(0).astype(float) * cost / 100.0)
            equity = (1.0 + net).cumprod()
            max_drawdown = (equity / equity.cummax() - 1).min() * 100
            for period, mask in {
                "all": np.ones(len(chosen), dtype=bool),
                "2025_H1": (chosen["date"] >= "2025-01-01") & (chosen["date"] < "2025-07-01"),
                "2025_H2": (chosen["date"] >= "2025-07-01") & (chosen["date"] < "2026-01-01"),
                "2026_H1": (chosen["date"] >= "2026-01-01") & (chosen["date"] < "2026-07-01"),
                "2026_H2": (chosen["date"] >= "2026-07-01") & (chosen["date"] < "2027-01-01"),
            }.items():
                sub = chosen.loc[mask]
                if period != "all" and sub.empty:
                    continue
                results.append({
                    "model": name, "period": period, "cost_round_trip_pct": cost,
                    "signals": len(sub), "signal_days": int(sub["date"].nunique()),
                    "hit_pct": float(sub["target"].mean() * 100) if len(sub) else np.nan,
                    "avg_gross_return_pct": float(sub["next_return"].mean() * 100) if len(sub) else np.nan,
                    "avg_net_return_pct": float(sub["next_return"].mean() * 100 - cost) if len(sub) else np.nan,
                    "idle_days_all_period": int(counts.eq(0).sum()),
                    "max_drawdown_all_period_pct": float(max_drawdown),
                    "compounded_net_all_period_pct": float((equity.iloc[-1] - 1) * 100),
                })
        for date, gross in daily.items():
            daily_rows.append({"model": name, "date": date, "signals": int(counts.loc[date]), "equal_weight_gross_return": float(gross)})

    exp.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(results)
    summary.to_csv(exp.OUTPUT_DIR / "next_day_fixed_strategy_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(daily_rows).to_csv(exp.OUTPUT_DIR / "next_day_fixed_strategy_daily.csv", index=False)
    print("RETROSPECTIVE, NOT INDEPENDENT. Predeclared fixed regime<-3%, close*volume>=10bn KRW, probability>=0.52.")
    print("Equal-weight daily hypothetical returns; no execution model; costs are scenarios.")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
