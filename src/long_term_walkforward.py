"""Quarterly point-in-time comparison of fixed long-horizon model candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_integrity_audit import valid_ohlcv_rows
from .indicators import add_indicators
from .lab_sampling import sample_history_paths
from .long_term_probability_lab import COST, FEATURES, HOLDOUT_START, HORIZONS


def prepare_history(path: Path) -> pd.DataFrame:
    bars, valid = valid_ohlcv_rows(pd.read_csv(path))
    outcome_columns = [f"{kind}_{horizon}" for horizon in HORIZONS for kind in ("exit", "net")]
    pieces = []
    for _, segment in bars.loc[valid].groupby((~valid).cumsum()):
        if len(segment) <= 20:
            continue
        frame = add_indicators(segment)
        frame["symbol"] = path.stem
        for horizon in HORIZONS:
            frame[f"exit_{horizon}"] = frame["date"].shift(-horizon)
            frame[f"net_{horizon}"] = frame["close"].shift(-horizon) / frame["open"].shift(-1) - 1 - COST
        pieces.append(frame[["symbol", "date", *FEATURES, *outcome_columns]])
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces).replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)


def evaluate(dataset: pd.DataFrame, horizon: int, *, relative: bool = False) -> list[dict]:
    if relative:
        dataset = dataset.copy()
        # Same-close observations only. Center each stock against its contemporaneous peers.
        dataset[FEATURES] = dataset.groupby("date")[FEATURES].rank(pct=True)
    exit_col, net_col = f"exit_{horizon}", f"net_{horizon}"
    results = []
    for start, quarter in dataset.loc[dataset["date"] >= HOLDOUT_START].groupby(dataset["date"].dt.to_period("Q")):
        boundary = start.start_time
        train = dataset.loc[(dataset[exit_col] < boundary) & dataset[net_col].notna()]
        truth_train = train[net_col] > 0
        matured = quarter[net_col].notna()
        if len(train) < 1000 or truth_train.nunique() < 2 or not matured.any():
            results.append({"horizon": horizon, "quarter": str(start), "status": "insufficient",
                            "train_rows": len(train), "matured_rows": int(matured.sum())})
            continue
        # Fixed regularization and shrinkage. No holdout-driven parameter selection.
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=300))
        model.fit(train[FEATURES], truth_train)
        baseline = float(truth_train.mean())
        prediction = model.predict_proba(quarter[FEATURES])[:, 1]
        if relative:
            daily_mean = pd.Series(prediction, index=quarter.index).groupby(quarter["date"]).transform("mean")
            score = np.clip(baseline + 0.5 * (prediction - daily_mean.to_numpy()), 0, 1)
        else:
            score = 0.5 * prediction + 0.5 * baseline
        truth = quarter.loc[matured, net_col] > 0
        selected = quarter.assign(score=score).sort_values(
            ["date", "score", "symbol"], ascending=[True, False, True]
        ).groupby("date").head(10)
        # Rank before checking whether the holding outcome is observable.
        selected_net = selected[net_col].dropna()
        comparable = quarter.loc[quarter["date"].isin(selected.loc[selected[net_col].notna(), "date"]), net_col].dropna()
        results.append({
            "horizon": horizon, "quarter": str(start), "status": "evaluated",
            "train_rows": len(train), "last_train_exit": train[exit_col].max().date(),
            "matured_rows": len(truth), "scored_rows": len(quarter),
            "model_auc": roc_auc_score(truth, score[matured]) if truth.nunique() == 2 else np.nan,
            "model_brier": brier_score_loss(truth, score[matured]),
            "constant_brier": brier_score_loss(truth, np.full(len(truth), baseline)),
            "model_accuracy": ((score[matured] >= 0.5) == truth).mean(),
            "constant_accuracy": (truth == (baseline >= 0.5)).mean(),
            "top10_selected": len(selected), "top10_matured": len(selected_net),
            "top10_hit_rate": (selected_net > 0).mean(), "top10_mean_net": selected_net.mean(),
            "universe_mean_net": comparable.mean(),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", type=Path, default=Path("data/history"))
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--relative", action="store_true", help="Evaluate the same-close peer rank model")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/long_term_walkforward.csv"))
    args = parser.parse_args()
    frames = [prepare_history(path) for path in sample_history_paths(
        list(args.history_dir.glob("*.csv")), args.max_symbols
    )]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise SystemExit("No eligible OHLCV histories")
    dataset = pd.concat(frames, ignore_index=True)
    result = pd.DataFrame([row for horizon in HORIZONS for row in evaluate(dataset, horizon, relative=args.relative)])
    if result.empty or not result["status"].eq("evaluated").any():
        raise SystemExit("No mature walk-forward folds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
