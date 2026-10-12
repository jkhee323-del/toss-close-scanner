"""Retrospective 2025+ check of preselected 1-day feature sets.

This period has been inspected in previous experiments and is NOT a pristine holdout.
No threshold or hyperparameter selection is performed here.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from . import next_day_validation_feature_models as exp

TRAIN_END = pd.Timestamp("2024-12-30")
TEST_START = pd.Timestamp("2025-01-02")
THRESHOLD = 0.52


def evaluate(name, period, frame, probabilities):
    selected = probabilities >= THRESHOLD
    signals = frame.loc[selected]
    return {
        "model": name,
        "period": period,
        "rows": len(frame),
        "trading_days": frame["date"].nunique(),
        "signals": len(signals),
        "signal_days": signals["date"].nunique(),
        "signal_coverage_pct": 100 * len(signals) / len(frame),
        "upward_hit_pct": 100 * signals["target"].mean() if len(signals) else np.nan,
        "avg_next_return_pct": 100 * signals["next_return"].mean() if len(signals) else np.nan,
        "median_next_return_pct": 100 * signals["next_return"].median() if len(signals) else np.nan,
    }


def run():
    previous = exp.LABEL_DATA_END
    try:
        exp.LABEL_DATA_END = pd.Timestamp("2100-01-01")
        data = exp.build_dataset(exp.DATA_DIR)
    finally:
        exp.LABEL_DATA_END = previous

    train = data.loc[data["date"] <= TRAIN_END].copy()
    test = data.loc[data["date"] >= TEST_START].copy()
    if train.empty or test.empty:
        raise RuntimeError("Train or retrospective test is empty")
    if train["date"].max() >= test["date"].min():
        raise RuntimeError("Train/test chronology overlaps")

    variants = {
        "baseline": exp.BASE_FEATURES,
        "baseline_plus_obv": exp.BASE_FEATURES + exp.OBV_FEATURES,
        "volume_value_obv": exp.BASE_FEATURES + exp.FLOW_FEATURES,
    }
    results = []
    for name, columns in variants.items():
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(**exp.HGB_BASELINE_PARAMS),
        )
        model.fit(train[columns], train["target"].astype(int))
        probabilities = model.predict_proba(test[columns])[:, 1]
        periods = {
            "all_2025_plus": np.ones(len(test), dtype=bool),
            "2025_H1": (test["date"] >= "2025-01-01") & (test["date"] < "2025-07-01"),
            "2025_H2": (test["date"] >= "2025-07-01") & (test["date"] < "2026-01-01"),
            "2026_H1": (test["date"] >= "2026-01-01") & (test["date"] < "2026-07-01"),
            "2026_H2": (test["date"] >= "2026-07-01") & (test["date"] < "2027-01-01"),
        }
        for period, mask in periods.items():
            if np.any(mask):
                results.append(evaluate(name, period, test.loc[mask], probabilities[mask]))

    output = pd.DataFrame(results)
    exp.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = exp.OUTPUT_DIR / "next_day_flow_2025_2026_retrospective.csv"
    output.to_csv(destination, index=False, encoding="utf-8-sig")
    print("RETROSPECTIVE ONLY: 2025+ was previously inspected; not a pristine holdout.")
    print("Fixed HGB params and threshold=0.52; no model selection on this test.")
    print(f"Train through {train['date'].max().date()}, test {test['date'].min().date()} to {test['date'].max().date()}")
    print(output.to_string(index=False))
    return output


if __name__ == "__main__":
    run()
