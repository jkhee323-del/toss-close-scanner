from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .next_day_validation_feature_models import (
    build_dataset, DATA_DIR, EXPANDED_FEATURES, BASE_FEATURES,
    HGB_BASELINE_PARAMS, RANDOM_STATE, PREDICTION_THRESHOLD,
)

TRAIN_END = pd.Timestamp("2024-12-30")
HOLDOUT_START = pd.Timestamp("2025-01-02")


def metrics(name, prob, frame):
    signal = prob >= PREDICTION_THRESHOLD
    s = frame.loc[signal]
    return {
        "model": name,
        "accuracy_pct": float(np.mean(signal == frame["target"].to_numpy()) * 100),
        "upward_hit_rate_pct": float(s["target"].mean() * 100) if len(s) else None,
        "signals": int(len(s)),
        "sample_coverage_pct": float(len(s) / len(frame) * 100),
        "signal_days": int(s["date"].nunique()),
        "trading_days": int(frame["date"].nunique()),
        "avg_signals_per_day": float(len(s) / frame["date"].nunique()),
        "avg_next_return_pct": float(s["next_return"].mean() * 100) if len(s) else None,
        "median_next_return_pct": float(s["next_return"].median() * 100) if len(s) else None,
    }


def run():
    # Candidate and all hyperparameters were fixed from 2024 validation before this holdout run.
    data = build_dataset(DATA_DIR)
    train = data.loc[data["date"] <= TRAIN_END].copy()
    holdout = data.loc[data["date"] >= HOLDOUT_START].copy()
    if train.empty or holdout.empty:
        raise RuntimeError("empty train or holdout")
    if train["date"].max() >= holdout["date"].min():
        raise RuntimeError("time split overlap")

    baseline = HistGradientBoostingClassifier(**HGB_BASELINE_PARAMS)
    candidate = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=50,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    )
    baseline.fit(train[BASE_FEATURES], train["target"].astype(int))
    candidate.fit(train[EXPANDED_FEATURES], train["target"].astype(int))
    rows = [
        metrics("HGB baseline", baseline.predict_proba(holdout[BASE_FEATURES])[:, 1], holdout),
        metrics("RandomForest expanded FIXED", candidate.predict_proba(holdout[EXPANDED_FEATURES])[:, 1], holdout),
    ]
    out = pd.DataFrame(rows)
    print(f"train={train['date'].min().date()}..{train['date'].max().date()} rows={len(train)}")
    print(f"holdout={holdout['date'].min().date()}..{holdout['date'].max().date()} rows={len(holdout)}")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return out


if __name__ == "__main__":
    run()
