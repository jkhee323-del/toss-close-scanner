from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .indicators import add_indicators


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "history"
OUTPUT_DIR = ROOT / "data" / "experiments"
SUMMARY_PATH = OUTPUT_DIR / "validation_feature_model_comparison.csv"
DETAILS_PATH = OUTPUT_DIR / "validation_feature_model_comparison.json"

BASE_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "volume_ratio",
    "close_position",
    "near_20d_high",
    "rsi14",
]

ENGINEERED_FEATURES = [
    "ma5_distance",
    "ma20_distance",
    "ma60_distance",
    "ma5_slope_5d",
    "ma20_slope_5d",
    "ma60_slope_5d",
    "ma_bull_alignment",
    "ma_bear_alignment",
    "ret_3d",
    "ret_10d",
    "ret1d_x_ret5d",
    "ret1d_x_ret20d",
    "ret5d_minus_ret20d",
    "volume_change_1d",
    "volume_ratio_5d",
    "volume_ratio_change_5d",
    "log_trading_value",
    "trading_value_ratio_20d",
    "trading_value_change_1d",
    "trading_value_change_5d",
    "atr14_pct",
    "volatility_10d",
    "range_pct",
    "candle_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "upper_wick_range_ratio",
    "lower_wick_range_ratio",
    "distance_from_20d_high",
    "distance_from_20d_low",
    "market_return_1d",
    "market_return_5d",
    "market_return_20d",
    "relative_strength_5d",
    "relative_strength_20d",
]

EXPANDED_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

TRAIN_END = pd.Timestamp("2023-12-28")
VAL_START = pd.Timestamp("2024-01-02")
VAL_END = pd.Timestamp("2024-12-30")
LABEL_DATA_END = pd.Timestamp("2024-12-31")
PREDICTION_THRESHOLD = 0.52
RANDOM_STATE = 42
MIN_SIGNALS_FOR_CANDIDATE = 2000

HGB_BASELINE_PARAMS: dict[str, object] = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "max_iter": 300,
    "random_state": RANDOM_STATE,
}


def load_raw_history(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("*.csv")):
        frame = pd.read_csv(path, parse_dates=["date"])
        if frame.empty:
            continue
        frame = frame.loc[frame["date"] <= LABEL_DATA_END].copy()
        if frame.empty:
            continue
        frame = frame.sort_values("date").reset_index(drop=True)
        frame["symbol"] = path.stem
        frames.append(frame)

    if not frames:
        raise ValueError(f"No history rows found on or before {LABEL_DATA_END.date()} in {data_dir}")

    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_indicators(frame)
    previous_close = result["close"].shift(1)
    daily_range = (result["high"] - result["low"]).replace(0, np.nan)
    trading_value = result["close"] * result["volume"]

    result["ma60_distance"] = result["close"] / result["ma60"] - 1
    for window in (5, 20, 60):
        result[f"ma{window}_slope_5d"] = result[f"ma{window}"].pct_change(5, fill_method=None)

    result["ma_bull_alignment"] = (
        (result["ma5"] > result["ma20"]) & (result["ma20"] > result["ma60"])
    ).astype("int8")
    result["ma_bear_alignment"] = (
        (result["ma5"] < result["ma20"]) & (result["ma20"] < result["ma60"])
    ).astype("int8")
    result["ret_3d"] = result["close"].pct_change(3)
    result["ret_10d"] = result["close"].pct_change(10)
    result["ret1d_x_ret5d"] = result["ret_1d"] * result["ret_5d"]
    result["ret1d_x_ret20d"] = result["ret_1d"] * result["ret_20d"]
    result["ret5d_minus_ret20d"] = result["ret_5d"] - result["ret_20d"]
    result["volume_change_1d"] = result["volume"].pct_change(fill_method=None)
    result["volume_ratio_5d"] = result["volume"] / result["volume"].rolling(5).mean()
    result["volume_ratio_change_5d"] = result["volume_ratio"].pct_change(5, fill_method=None)
    result["log_trading_value"] = np.log1p(trading_value.clip(lower=0))
    result["trading_value_ratio_20d"] = trading_value / trading_value.rolling(20).mean()
    result["trading_value_change_1d"] = trading_value.pct_change(fill_method=None)
    result["trading_value_change_5d"] = trading_value.pct_change(5, fill_method=None)

    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr14_pct"] = true_range.rolling(14).mean() / result["close"]
    result["volatility_10d"] = result["ret_1d"].rolling(10).std()
    result["candle_body_pct"] = (result["close"] - result["open"]) / result["open"]

    upper_wick = result["high"] - result[["open", "close"]].max(axis=1)
    lower_wick = result[["open", "close"]].min(axis=1) - result["low"]
    result["upper_wick_pct"] = upper_wick / result["close"]
    result["lower_wick_pct"] = lower_wick / result["close"]
    result["upper_wick_range_ratio"] = upper_wick / daily_range
    result["lower_wick_range_ratio"] = lower_wick / daily_range
    result["distance_from_20d_high"] = result["close"] / result["high20"] - 1
    low20 = result["low"].rolling(20).min()
    result["distance_from_20d_low"] = result["close"] / low20 - 1
    return result


def build_dataset(data_dir: Path) -> pd.DataFrame:
    raw = load_raw_history(data_dir)
    frames = []
    for _, symbol_frame in raw.groupby("symbol", sort=False):
        symbol_frame = symbol_frame.sort_values("date").reset_index(drop=True)
        features = add_engineered_features(symbol_frame)
        features["next_date"] = features["date"].shift(-1)
        features["next_return"] = features["close"].shift(-1) / features["close"] - 1
        features["target"] = (features["next_return"] > 0).astype("int8")
        frames.append(features)

    data = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    market_daily_return = data.groupby("date", sort=True)["ret_1d"].median()
    market_returns = pd.DataFrame({"market_return_1d": market_daily_return})
    for window in (5, 20):
        market_returns[f"market_return_{window}d"] = (
            (1 + market_returns["market_return_1d"]).rolling(window).apply(np.prod, raw=True) - 1
        )
    data = data.join(market_returns, on="date")
    data["relative_strength_5d"] = data["ret_5d"] - data["market_return_5d"]
    data["relative_strength_20d"] = data["ret_20d"] - data["market_return_20d"]

    data = data.loc[data["next_date"] <= LABEL_DATA_END].copy()
    numeric_columns = list(dict.fromkeys(EXPANDED_FEATURES + ["next_return", "target"]))
    data[numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.nan)
    required_columns = BASE_FEATURES + ["next_return", "target"]
    return data.dropna(subset=required_columns).reset_index(drop=True)


def get_time_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.loc[data["date"] <= TRAIN_END].copy()
    validation = data.loc[data["date"].between(VAL_START, VAL_END)].copy()
    if train.empty or validation.empty:
        raise ValueError("Fixed chronological Train/Validation split produced an empty partition")
    if train["date"].max() >= validation["date"].min():
        raise ValueError("Train and Validation date ranges overlap")
    if data["date"].max() > VAL_END or data["next_date"].max() > LABEL_DATA_END:
        raise ValueError("Rows after the fixed Validation label boundary entered the experiment")
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def make_models() -> dict[str, tuple[object, list[str]]]:
    return {
        "HGB baseline (7 features, p>=0.52)": (
            HistGradientBoostingClassifier(**HGB_BASELINE_PARAMS),
            BASE_FEATURES,
        ),
        "HGB expanded": (
            make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingClassifier(**HGB_BASELINE_PARAMS),
            ),
            EXPANDED_FEATURES,
        ),
        "LogisticRegression expanded": (
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE),
            ),
            EXPANDED_FEATURES,
        ),
        "RandomForest expanded": (
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=50,
                max_features="sqrt",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            EXPANDED_FEATURES,
        ),
        "ExtraTrees expanded": (
            make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_leaf=50,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
            EXPANDED_FEATURES,
        ),
    }


def summarize_predictions(
    name: str,
    probabilities: np.ndarray,
    validation: pd.DataFrame,
) -> dict[str, object]:
    predicted_up = probabilities >= PREDICTION_THRESHOLD
    signals = validation.loc[predicted_up]
    sample_count = len(validation)
    trading_days = int(validation["date"].nunique())
    signal_count = len(signals)
    signal_days = int(signals["date"].nunique())

    return {
        "model": name,
        "threshold_fixed": PREDICTION_THRESHOLD,
        "validation_samples": sample_count,
        "validation_trading_days": trading_days,
        "accuracy_pct": float(np.mean(predicted_up == validation["target"].to_numpy()) * 100),
        "upward_hit_rate_pct": float(signals["target"].mean() * 100) if signal_count else None,
        "upward_signals": signal_count,
        "coverage_pct": signal_count / sample_count * 100,
        "signal_days": signal_days,
        "trading_day_coverage_pct": signal_days / trading_days * 100,
        "average_recommendations_per_day": signal_count / trading_days,
        "average_next_return_pct": float(signals["next_return"].mean() * 100) if signal_count else None,
        "median_next_return_pct": float(signals["next_return"].median() * 100) if signal_count else None,
        "eligible_signal_count": signal_count >= MIN_SIGNALS_FOR_CANDIDATE,
    }


def run_validation_experiment(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    if sklearn.__version__ != "1.8.0":
        raise RuntimeError(f"Expected scikit-learn 1.8.0, found {sklearn.__version__}")

    data = build_dataset(data_dir)
    train, validation = get_time_split(data)
    results = []

    for name, (model, feature_names) in make_models().items():
        model.fit(train[feature_names], train["target"].astype(int))
        probabilities = model.predict_proba(validation[feature_names])[:, 1]
        results.append(summarize_predictions(name, probabilities, validation))

    summary = pd.DataFrame(results)
    summary["candidate_rank"] = pd.NA
    candidates = summary.loc[
        (summary["model"] != "HGB baseline (7 features, p>=0.52)")
        & summary["eligible_signal_count"]
    ].sort_values(
        ["upward_hit_rate_pct", "average_next_return_pct", "coverage_pct"],
        ascending=[False, False, False],
    )
    for rank, index in enumerate(candidates.index[:3], start=1):
        summary.loc[index, "candidate_rank"] = rank

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.6f", encoding="utf-8-sig")
    metadata = {
        "train_end_inclusive": TRAIN_END.date().isoformat(),
        "validation_start_inclusive": VAL_START.date().isoformat(),
        "validation_end_inclusive": VAL_END.date().isoformat(),
        "label_data_end_inclusive": LABEL_DATA_END.date().isoformat(),
        "threshold_fixed_for_all_models": PREDICTION_THRESHOLD,
        "random_state": RANDOM_STATE,
        "hgb_baseline_params": HGB_BASELINE_PARAMS,
        "base_features": BASE_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "minimum_signals_for_candidate": MIN_SIGNALS_FOR_CANDIDATE,
        "train_rows": len(train),
        "train_actual_range": [train["date"].min().date().isoformat(), train["date"].max().date().isoformat()],
        "validation_rows": len(validation),
        "validation_actual_range": [validation["date"].min().date().isoformat(), validation["date"].max().date().isoformat()],
        "validation_trading_days": int(validation["date"].nunique()),
        "market_proxy": "per-date cross-sectional median of ret_1d in the training/validation universe",
        "scikit_learn_version": sklearn.__version__,
        "holdout_accessed": False,
        "model_or_threshold_search": False,
    }
    DETAILS_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"train_rows={len(train)} train_range={metadata['train_actual_range']}")
    print(f"validation_rows={len(validation)} validation_range={metadata['validation_actual_range']} days={metadata['validation_trading_days']}")
    print(f"summary_csv={SUMMARY_PATH.relative_to(ROOT)}")
    print(f"details_json={DETAILS_PATH.relative_to(ROOT)}")
    return summary


if __name__ == "__main__":
    run_validation_experiment()