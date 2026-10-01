from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler

from . import next_day_validation_feature_models as exp

DATA_DIR = exp.DATA_DIR
EXPANDED_FEATURES = exp.EXPANDED_FEATURES
BASE_FEATURES = exp.BASE_FEATURES
HGB_BASELINE_PARAMS = exp.HGB_BASELINE_PARAMS
RANDOM_STATE = exp.RANDOM_STATE
PREDICTION_THRESHOLD = exp.PREDICTION_THRESHOLD

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
    # Locked candidate: features/model/threshold were selected on 2024 validation.
    # Only widen raw label availability so later dates can be evaluated.
    exp.LABEL_DATA_END = pd.Timestamp("2100-01-01")
    data = exp.build_dataset(DATA_DIR)

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
    logistic = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    )

    baseline.fit(train[BASE_FEATURES], train["target"].astype(int))
    candidate.fit(train[EXPANDED_FEATURES], train["target"].astype(int))\n    logistic.fit(train[EXPANDED_FEATURES], train["target"].astype(int))

    out = pd.DataFrame([
        metrics("HGB baseline", baseline.predict_proba(holdout[BASE_FEATURES])[:, 1], holdout),
        metrics("RandomForest expanded FIXED", candidate.predict_proba(holdout[EXPANDED_FEATURES])[:, 1], holdout),\n        metrics("LogisticRegression expanded FIXED", logistic.predict_proba(holdout[EXPANDED_FEATURES])[:, 1], holdout),
    ])
    print(f"train={train['date'].min().date()}..{train['date'].max().date()} rows={len(train)}")
    print(f"holdout={holdout['date'].min().date()}..{holdout['date'].max().date()} rows={len(holdout)}")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return out


if __name__ == "__main__":
    run()
