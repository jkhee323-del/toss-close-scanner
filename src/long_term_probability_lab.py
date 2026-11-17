"""Experimental long-horizon classifier with a strictly chronological holdout.

This module NEVER overwrites production models or claims calibrated probabilities.
Horizon targets are known only if the future exit date is observed.
Signals are end-of-day, so returns use next-session open to horizon close.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from .indicators import add_indicators
from .lab_sampling import sample_history_paths

HORIZONS = (20, 60, 120, 240)
FEATURES = [
    "ret_1d", "ret_5d", "ret_20d", "ma20_distance",
    "volume_ratio", "near_20d_high", "rsi14", "volatility_20d",
    "obv_change_5d", "obv_change_20d",
]
DEVELOPMENT_END = pd.Timestamp("2024-12-30")
HOLDOUT_START = pd.Timestamp("2025-01-01")
COST = 0.002


def prepare_symbol(path: Path, horizon: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"date", "open", "high", "low", "close", "volume"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in needed - {"date"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(needed)).sort_values("date").drop_duplicates("date")
    if len(df) < 260 or (df[["open", "high", "low", "close"]] <= 0).any().any():
        return pd.DataFrame()
    if (df["volume"] < 0).any():
        return pd.DataFrame()
    df = add_indicators(df.reset_index(drop=True))
    df["future_date"] = df["date"].shift(-horizon)
    future_close = df["close"].shift(-horizon)
    # Signals are formed after the close: earliest executable entry is next open.
    entry_open = df["open"].shift(-1)
    df["future_net_return"] = future_close / entry_open - 1 - COST
    valid_future = future_close.notna() & entry_open.notna() & (entry_open > 0)
    df["target"] = np.where(valid_future, (df["future_net_return"] > 0).astype(float), np.nan)
    df["symbol"] = path.stem
    return df[["symbol", "date", "future_date", "future_net_return", "target"] + FEATURES].replace([np.inf, -np.inf], np.nan).dropna()


def split_no_leakage(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training outcomes must fully mature before the development cutoff."""
    development = dataset.loc[
        (dataset["date"] <= DEVELOPMENT_END) &
        (dataset["future_date"] <= DEVELOPMENT_END)
    ].copy()
    holdout = dataset.loc[dataset["date"] >= HOLDOUT_START].copy()
    return development, holdout


def evaluate_horizon(dataset: pd.DataFrame, horizon: int) -> dict:
    train, holdout = split_no_leakage(dataset)
    if len(train) < 1000 or len(holdout) < 100 or train["target"].nunique() < 2 or holdout["target"].nunique() < 2:
        return {"horizon": horizon, "status": "insufficient data", "train_rows": len(train), "holdout_rows": len(holdout)}
    model = HistGradientBoostingClassifier(
        max_iter=120, max_leaf_nodes=15, min_samples_leaf=100,
        learning_rate=0.05, l2_regularization=1.0, random_state=42,
    )
    model.fit(train[FEATURES], train["target"].astype(int))
    score = model.predict_proba(holdout[FEATURES])[:, 1]
    truth = holdout["target"].to_numpy(dtype=int)
    cutoff = np.quantile(score, 0.90)
    selected = truth[score >= cutoff]
    baseline = float(train["target"].mean())
    return {
        "horizon": horizon,
        "status": "evaluated",
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "holdout_positive_rate": float(truth.mean()),
        "model_auc": float(roc_auc_score(truth, score)),
        "model_brier": float(brier_score_loss(truth, score)),
        "constant_brier": float(brier_score_loss(truth, np.full(len(truth), baseline))),
        "top_decile_positive_rate": float(selected.mean()),
        "top_decile_count": len(selected),
        "model_beats_constant_brier": bool(brier_score_loss(truth, score) < brier_score_loss(truth, np.full(len(truth), baseline))),
    }


def run_lab(history_dir: Path, max_symbols: int | None = None) -> pd.DataFrame:
    paths = sample_history_paths(list(history_dir.glob("*.csv")), max_symbols)
    outputs = []
    for horizon in HORIZONS:
        frames = [prepare_symbol(path, horizon) for path in paths]
        valid = [frame for frame in frames if not frame.empty]
        if not valid:
            outputs.append({"horizon": horizon, "status": "no observations"})
            continue
        dataset = pd.concat(valid, ignore_index=True)
        outputs.append(evaluate_horizon(dataset, horizon))
    return pd.DataFrame(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", type=Path, default=Path("data/history"))
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/long_term_probability_lab.csv"))
    args = parser.parse_args()
    result = run_lab(args.history_dir, args.max_symbols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print("EXPERIMENT ONLY: model scores are not calibrated probabilities; production models unchanged.")


if __name__ == "__main__":
    main()
