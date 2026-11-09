"""Research-only 1/3/6/12-month US equity direction experiments.

No live predictions are published. Train/holdout labels are separated using
ACTUAL exit dates rather than a fixed calendar gap. Data after the requested
as-of date is excluded. Saved artifacts, if explicitly requested, are stamped
EXPERIMENTAL and must not be interpreted as calibrated trading probabilities.

This cannot remove survivorship bias, vendor revisions, corporate actions,
or overlapping-label dependence in the evaluation set.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .global_market_safety import PriceDataError, validate_ohlcv
from .indicators import add_indicators
from .search import US_NAME_MAP

HORIZONS = {"1m": 20, "3m": 60, "6m": 120, "12m": 240}
FEATURES = [
    "ret_5d", "ret_20d", "ma20_distance", "volatility_20d",
    "volume_ratio", "rsi14", "near_20d_high",
    "obv_change_5d", "obv_change_20d", "range_pct",
]
MIN_TRAIN = 500
MIN_TEST = 250


def make_labeled_rows(
    history: pd.DataFrame, ticker: str, horizon: int, *, as_of: str | date | None = None
) -> pd.DataFrame:
    """Make past-only features and labels whose outcomes are already known.

    A row at signal date t uses historical features at t; its label requires
    the observed close at t + horizon stock-specific trading sessions.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    # Minimum length also prevents a one-year model from training on 1y of bars.
    minimum = max(220, 2 * horizon + 120)
    x = validate_ohlcv(history, min_rows=minimum)
    if as_of is not None:
        cutoff = pd.Timestamp(as_of).normalize()
        x = x.loc[x["date"] <= cutoff].copy()
    if len(x) < minimum:
        raise PriceDataError(f"not enough completed sessions: {len(x)} < {minimum}")
    x = add_indicators(x)
    x["exit_date"] = x["date"].shift(-horizon)
    x["future_return"] = x["close"].shift(-horizon) / x["close"] - 1.0
    x["target"] = (x["future_return"] > 0).astype(int)
    x["ticker"] = ticker
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["date", "exit_date", "future_return", *FEATURES]).copy()
    if as_of is not None:
        x = x.loc[x["exit_date"] <= pd.Timestamp(as_of)].copy()
    return x[["date", "exit_date", "ticker", "target", "future_return", *FEATURES]]


def purged_split(rows: pd.DataFrame, *, cutoff: pd.Timestamp):
    """No training label may be determined on or after the test start."""
    start = pd.Timestamp(cutoff).normalize()
    train = rows.loc[rows["exit_date"] < start].copy()
    test = rows.loc[rows["date"] >= start].copy()
    if not train.empty and not test.empty:
        assert train["exit_date"].max() < test["date"].min(), "label leakage"
    return train, test


def evaluate_horizon(
    rows: pd.DataFrame, horizon: int, *, min_train: int = MIN_TRAIN,
    min_test: int = MIN_TEST, cutoff: pd.Timestamp | None = None,
):
    """One chronological, outcome-purged holdout (not a live trading audit)."""
    if rows.empty:
        return {"status": "INSUFFICIENT_LABELLED_DATA", "horizon_days": horizon}, None
    dates = np.sort(pd.to_datetime(rows["date"]).unique())
    if cutoff is None:
        if len(dates) < 100:
            return {"status": "INSUFFICIENT_DATES", "horizon_days": horizon}, None
        cutoff = pd.Timestamp(dates[int(len(dates) * 0.75)])
    train, test = purged_split(rows, cutoff=cutoff)
    base = {
        "horizon_days": horizon, "cutoff": str(pd.Timestamp(cutoff).date()),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "stocks": int(rows["ticker"].nunique()),
        "train_last_signal_date": str(train["date"].max().date()) if len(train) else None,
        "train_last_exit_date": str(train["exit_date"].max().date()) if len(train) else None,
    }
    if (len(train) < min_train or len(test) < min_test
            or train["target"].nunique() < 2 or test["target"].nunique() < 2):
        return {"status": "INSUFFICIENT_HOLDOUT", **base}, None

    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=0.05, max_iter=500, random_state=42),
    )
    model.fit(train[FEATURES], train["target"].astype(int))
    prob = model.predict_proba(test[FEATURES])[:, 1]
    y = test["target"].to_numpy(dtype=int)
    train_rate = float(train["target"].mean())
    model_accuracy = float(accuracy_score(y, prob >= 0.5))
    # Always-up is especially important in upward-trending equity markets.
    always_up_accuracy = float(y.mean())
    train_majority_accuracy = float(accuracy_score(
        y, np.full(len(y), train_rate >= 0.5)
    ))
    brier_model = float(brier_score_loss(y, prob))
    brier_prior = float(brier_score_loss(y, np.full(len(y), train_rate)))
    auc = float(roc_auc_score(y, prob))
    yearly = {}
    for year, group in test.assign(pred=prob).groupby(test["date"].dt.year):
        # Report annual sample size and direction hit rate, not independent trials.
        yearly[str(int(year))] = {
            "rows": int(len(group)),
            "positive_rate": round(float(group["target"].mean()), 5),
            "model_accuracy": round(
                float(accuracy_score(group["target"], group["pred"] >= 0.5)), 5
            ),
        }
    # This gate is intentionally stringent and only experimental.
    passed = (
        len(test) >= min_test
        and brier_model < brier_prior
        and auc >= 0.53
        and model_accuracy > max(always_up_accuracy, train_majority_accuracy) + 0.005
    )
    result = {
        "status": "OOS_GATE_PASSED_EXPERIMENTAL" if passed else "OOS_GATE_FAILED",
        **base,
        "test_start": str(test["date"].min().date()),
        "test_last_signal": str(test["date"].max().date()),
        "test_last_exit": str(test["exit_date"].max().date()),
        "train_positive_rate": round(train_rate, 5),
        "test_positive_rate": round(float(y.mean()), 5),
        "always_up_accuracy": round(always_up_accuracy, 5),
        "train_majority_accuracy": round(train_majority_accuracy, 5),
        "model_accuracy": round(model_accuracy, 5),
        "model_brier": round(brier_model, 6),
        "prior_brier": round(brier_prior, 6),
        "model_auc": round(auc, 5),
        "test_years": yearly,
        "verified_for_live": False,
        "calibrated_probability": False,
    }
    return result, model if passed else None


def run_all(
    history_dir: Path, output_dir: Path, *,
    as_of: str | date | None = None, save_experimental: bool = False,
):
    """Read only ticker catalog entries explicitly listed as US equities."""
    as_of = as_of or date.today()
    data = {}
    excluded = {}
    for path in sorted(Path(history_dir).glob("*.csv")):
        ticker = path.stem
        if US_NAME_MAP.get(ticker, (None, None))[1] != "미국":
            continue
        try:
            data[ticker] = validate_ohlcv(pd.read_csv(path), min_rows=70)
        except (PriceDataError, ValueError, OSError) as exc:
            excluded[ticker] = str(exc)
    report = {
        "created_on": date.today().isoformat(),
        "as_of": str(as_of),
        "status": "RESEARCH_ONLY",
        "disclaimer": "Not verified for live signals; scores are not calibrated probabilities",
        "eligible_price_histories": len(data),
        "excluded_price_histories": excluded,
        "results": {},
    }
    for label, horizon in HORIZONS.items():
        parts = []
        for ticker, history in data.items():
            try:
                rows = make_labeled_rows(history, ticker, horizon, as_of=as_of)
                if not rows.empty:
                    parts.append(rows)
            except PriceDataError:
                continue
        combined = pd.concat(parts, ignore_index=True).sort_values("date") if parts else pd.DataFrame()
        stats, model = evaluate_horizon(combined, horizon)
        report["results"][label] = stats
        if save_experimental and model is not None:
            import joblib

            output_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"model": model, "features": FEATURES, "report": stats,
                 "verified_for_live": False, "calibrated_probability": False},
                output_dir / f"us_{label}_EXPERIMENTAL.joblib",
            )
        print(f"{label} ({horizon} sessions): {stats['status']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "us_horizon_research.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Research report: {target}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=Path("data/global_history"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/us_horizon_research"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--save-experimental", action="store_true")
    args = parser.parse_args()
    run_all(args.history, args.output_dir, as_of=args.as_of, save_experimental=args.save_experimental)


if __name__ == "__main__":
    main()
