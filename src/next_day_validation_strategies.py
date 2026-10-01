from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score

from .indicators import add_indicators

FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "volume_ratio",
    "close_position",
    "near_20d_high",
    "rsi14",
]

TRAIN_END = pd.Timestamp("2023-12-28")
VAL_START = pd.Timestamp("2024-01-02")
VAL_END = pd.Timestamp("2024-12-30")


def load_next_day_dataset(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    frames = []
    for path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        df = add_indicators(df)
        df["next_return"] = df["close"].shift(-1) / df["close"] - 1
        df["target"] = (df["next_return"] > 0).astype(int)
        df["symbol"] = path.stem
        frames.append(df)

    if not frames:
        raise ValueError(f"No data found in {data_dir}")

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["date", "symbol"]).reset_index(drop=True)
    return data.dropna(subset=FEATURES + ["next_return", "target"]).reset_index(drop=True)


def get_split(data: pd.DataFrame):
    train = data[(data["date"] <= TRAIN_END)].copy()
    val = data[(data["date"] >= VAL_START) & (data["date"] <= VAL_END)].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


def score_rule(row: pd.Series) -> float:
    def clip(v, lo=0.0, hi=1.0):
        v = float(v)
        return max(lo, min(hi, v))

    ret_5d = float(row["ret_5d"])
    ma5 = float(row["ma5"])
    ma20 = float(row["ma20"])
    ma60 = float(row["ma60"])
    vol_ratio = float(row["volume_ratio"])
    close_position = float(row["close_position"])
    near_20d_high = float(row["near_20d_high"])
    rsi = float(row["rsi14"])

    momentum = 18 * clip((ret_5d + 0.02) / 0.10)
    trend = 18 * (
        0.5 * clip((ma5 / ma20 - 0.98) / 0.06)
        + 0.5 * clip((ma20 / ma60 - 0.98) / 0.08)
    )
    volume = 18 * clip((vol_ratio - 0.8) / 2.2)
    close_strength = 18 * clip(close_position)
    breakout = 18 * clip((near_20d_high - 0.94) / 0.06)
    rsi_part = 10 * clip((rsi - 45) / 30)
    score = momentum + trend + volume + close_strength + breakout + rsi_part
    return max(0.0, min(100.0, score))


def summarize_strategy(frame: pd.DataFrame, strategy_name: str, min_signals: int = 50):
    n = len(frame)
    if n == 0:
        return {
            "strategy": strategy_name,
            "signals": 0,
            "hit_rate": None,
            "coverage": None,
            "days_with_signals": None,
            "avg_signals_per_day": None,
            "avg_next_return_pct": None,
            "median_next_return_pct": None,
            "min_signals": min_signals,
            "flag_low_signal": True,
        }

    hit_rate = float(frame["target"].mean()) * 100.0
    coverage = len(frame) / len(frame["date"].unique()) if frame["date"].nunique() > 0 else 0.0
    avg_signals_per_day = len(frame) / frame["date"].nunique() if frame["date"].nunique() > 0 else 0.0
    avg_ret = frame["next_return"].mean() * 100.0
    med_ret = frame["next_return"].median() * 100.0

    return {
        "strategy": strategy_name,
        "signals": int(n),
        "hit_rate": round(hit_rate, 2),
        "days_with_signals": int(frame["date"].nunique()),
        "coverage": round((n / len(frame["date"].unique())) * 100.0 if frame["date"].nunique() > 0 else 0.0, 2),
        "avg_signals_per_day": round(avg_signals_per_day, 2),
        "avg_next_return_pct": round(avg_ret, 2),
        "median_next_return_pct": round(med_ret, 2),
        "min_signals": min_signals,
        "flag_low_signal": n < min_signals,
    }


def build_validation_strategies(data_dir: str | Path):
    data = load_next_day_dataset(data_dir)
    train, val = get_split(data)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=42,
    )
    model.fit(train[FEATURES], train["target"].astype(int))

    val = val.copy()
    val["probability"] = model.predict_proba(val[FEATURES])[:, 1]

    # baseline rule-based score
    val["rule_score"] = val.apply(score_rule, axis=1)
    rule_baseline = val[val["rule_score"] >= 70].copy()
    baseline_summary = summarize_strategy(rule_baseline, "old_rule_70", min_signals=50)

    # HGB basic default threshold=0.50
    hgb_basic = val[val["probability"] >= 0.50].copy()
    hgb_basic_summary = summarize_strategy(hgb_basic, "hgb_default_0.50", min_signals=50)

    thresholds = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70]
    threshold_results = []
    for t in thresholds:
        subset = val[val["probability"] >= t].copy()
        threshold_results.append(summarize_strategy(subset, f"hgb_thresh_{t}", min_signals=50))

    top_results = []
    for top_n in [3, 5, 10]:
        top = (
            val.sort_values(["date", "probability"], ascending=[True, False])
            .groupby("date", group_keys=False)
            .head(top_n)
            .copy()
        )
        top_results.append(summarize_strategy(top, f"top_{top_n}", min_signals=50))

    combo_results = []
    for t in thresholds:
        for top_n in [3, 5, 10]:
            tbl = val[val["probability"] >= t].copy()
            if tbl.empty:
                continue
            top = (
                tbl.sort_values(["date", "probability"], ascending=[True, False])
                .groupby("date", group_keys=False)
                .head(top_n)
                .copy()
            )
            combo_results.append(summarize_strategy(top, f"thresh_{t}_top_{top_n}", min_signals=50))

    # display each strategy with signal count and hit rate; also flag very-low-signal strategies
    all_results = [baseline_summary, hgb_basic_summary] + threshold_results + top_results + combo_results
    return {
        "train_rows": len(train),
        "val_rows": len(val),
        "train_range": (train["date"].min(), train["date"].max()),
        "val_range": (val["date"].min(), val["date"].max()),
        "results": all_results,
        "val_raw": val,
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1] / "data" / "history"
    out = build_validation_strategies(base_dir)
    print(f"train_rows={out['train_rows']} val_rows={out['val_rows']}")
    print(f"train_range={out['train_range']}")
    print(f"val_range={out['val_range']}")
    for item in out["results"]:
        print(item)
