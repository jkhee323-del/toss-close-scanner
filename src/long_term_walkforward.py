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


LONG_FEATURES = FEATURES + [
    "ret_60d", "ret_120d", "ret_240d", "ma120_distance",
    "ma240_distance", "volatility_120d", "drawdown_240d",
]


def prepare_history(path: Path, *, long_features: bool = False) -> pd.DataFrame:
    columns = LONG_FEATURES if long_features else FEATURES
    bars, valid = valid_ohlcv_rows(pd.read_csv(path))
    outcome_columns = [f"{kind}_{horizon}" for horizon in HORIZONS for kind in ("exit", "net")]
    pieces = []
    for _, segment in bars.loc[valid].groupby((~valid).cumsum()):
        if len(segment) <= 20:
            continue
        frame = add_indicators(segment)
        frame["symbol"] = path.stem
        if long_features:
            for days in (60, 120, 240):
                frame[f"ret_{days}d"] = frame["close"].pct_change(days, fill_method=None)
            for days in (120, 240):
                frame[f"ma{days}_distance"] = frame["close"] / frame["close"].rolling(days).mean() - 1
            frame["volatility_120d"] = frame["ret_1d"].rolling(120).std()
            frame["drawdown_240d"] = frame["close"] / frame["close"].rolling(240).max() - 1
        for horizon in HORIZONS:
            frame[f"exit_{horizon}"] = frame["date"].shift(-horizon)
            frame[f"net_{horizon}"] = frame["close"].shift(-horizon) / frame["open"].shift(-1) - 1 - COST
        pieces.append(frame[["symbol", "date", *columns, *outcome_columns]])
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces).replace([np.inf, -np.inf], np.nan).dropna(subset=columns)


def validation_blend(train: pd.DataFrame, columns: list[str], horizon: int, *, relative: bool) -> float:
    """Estimate a bounded blend using an inner, purged chronological window."""
    dates = train["date"].sort_values().unique()
    if len(dates) < 252:
        return 0.0
    validation_start = dates[-126]
    inner = train.loc[train[f"exit_{horizon}"] < validation_start]
    validation = train.loc[train["date"] >= validation_start]
    target = inner[f"net_{horizon}"] > 0
    if len(inner) < 1000 or target.nunique() < 2 or validation.empty:
        return 0.0
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=300))
    model.fit(inner[columns], target)
    predicted = model.predict_proba(validation[columns])[:, 1]
    baseline = float(target.mean())
    if relative:
        center = (
            pd.Series(predicted, index=validation.index)
            .groupby(validation["date"]).transform("mean").to_numpy()
        )
    else:
        center = baseline
    residual = predicted - center
    outcome = (validation[f"net_{horizon}"] > 0).to_numpy(dtype=float)
    # Equal signal-day weights reduce dominance from unequal cross-section sizes.
    weights = 1 / validation.groupby("date")["symbol"].transform("size").to_numpy()
    numerator = np.sum(weights * residual * (outcome - baseline))
    denominator = np.sum(weights * residual ** 2) + 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def evaluate(
    dataset: pd.DataFrame,
    horizon: int,
    *,
    relative: bool = False,
    held_out: set[str] | None = None,
    long_features: bool = False,
    validate_blend: bool = False,
) -> list[dict]:
    columns = LONG_FEATURES if long_features else FEATURES
    test_universe = dataset if held_out is None else dataset.loc[dataset["symbol"].isin(held_out)]
    train_universe = dataset if held_out is None else dataset.loc[~dataset["symbol"].isin(held_out)]
    if relative:
        # Same-close peer ranks. Held-out stocks never enter training ranks.
        train_universe = train_universe.assign(**train_universe.groupby("date")[columns].rank(pct=True))
        test_universe = test_universe.assign(**test_universe.groupby("date")[columns].rank(pct=True))
    exit_col, net_col = f"exit_{horizon}", f"net_{horizon}"
    results = []
    for start, quarter in test_universe.loc[test_universe["date"] >= HOLDOUT_START].groupby(
        test_universe["date"].dt.to_period("Q")
    ):
        boundary = start.start_time
        train = train_universe.loc[(train_universe[exit_col] < boundary) & train_universe[net_col].notna()]
        truth_train = train[net_col] > 0
        matured = quarter[net_col].notna()
        if len(train) < 1000 or truth_train.nunique() < 2 or not matured.any():
            results.append({"horizon": horizon, "quarter": str(start), "status": "insufficient",
                            "train_rows": len(train), "matured_rows": int(matured.sum())})
            continue
        # Fixed regularization and shrinkage. No holdout-driven parameter selection.
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=300))
        model.fit(train[columns], truth_train)
        baseline = float(truth_train.mean())
        prediction = model.predict_proba(quarter[columns])[:, 1]
        blend = validation_blend(train, columns, horizon, relative=relative) if validate_blend else 0.5
        if relative:
            daily_mean = pd.Series(prediction, index=quarter.index).groupby(quarter["date"]).transform("mean")
            score = np.clip(baseline + blend * (prediction - daily_mean.to_numpy()), 0, 1)
        else:
            score = blend * prediction + (1 - blend) * baseline
        truth = quarter.loc[matured, net_col] > 0
        selected = quarter.assign(score=score).sort_values(
            ["date", "score", "symbol"], ascending=[True, False, True]
        ).groupby("date").head(10)
        # Rank before checking whether the holding outcome is observable.
        selected_net = selected[net_col].dropna()
        comparable = quarter.loc[quarter["date"].isin(selected.loc[selected[net_col].notna(), "date"]), net_col].dropna()
        results.append({
            "horizon": horizon, "quarter": str(start), "status": "evaluated", "blend": blend,
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
    parser.add_argument("--validate-blend", action="store_true", help="Choose blending weight on an inner purged historical window")
    parser.add_argument("--long-features", action="store_true", help="Compare long-range features with the original model on identical observations")
    parser.add_argument("--symbol-folds", action="store_true", help="Evaluate three disjoint held-out symbol folds")
    parser.add_argument("--relative", action="store_true", help="Evaluate the same-close peer rank model")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/long_term_walkforward.csv"))
    args = parser.parse_args()
    frames = [prepare_history(path, long_features=args.long_features) for path in sample_history_paths(
        list(args.history_dir.glob("*.csv")), args.max_symbols
    )]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise SystemExit("No eligible OHLCV histories")
    dataset = pd.concat(frames, ignore_index=True)
    if args.symbol_folds:
        ranked = sample_history_paths([Path(symbol) for symbol in dataset["symbol"].unique()],
                                      dataset["symbol"].nunique(), seed="held-out-symbols-v1")
        groups = [set(path.stem for path in ranked[i::3]) for i in range(3)]
    else:
        groups = [None]
    result = pd.DataFrame([
        {"symbol_fold": fold, "feature_set": "long" if extended else "short", "blend_method": "validated" if args.validate_blend else "fixed", **row}
        for fold, group in enumerate(groups, 1)
        for horizon in HORIZONS
        for extended in ([False, True] if args.long_features else [False])
        for row in evaluate(dataset, horizon, relative=args.relative, held_out=group, long_features=extended, validate_blend=args.validate_blend)
    ])
    if result.empty or not result["status"].eq("evaluated").any():
        raise SystemExit("No mature walk-forward folds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
