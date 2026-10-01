from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

TRAIN_END = pd.Timestamp("2023-12-29")
VAL_START = pd.Timestamp("2024-01-02")
VAL_END = pd.Timestamp("2024-12-31")


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
    data = data.dropna(subset=FEATURES + ["next_return", "target"]).reset_index(drop=True)
    return data


def make_time_split(data: pd.DataFrame):
    train = data[(data["date"] <= TRAIN_END)].copy()
    val = data[(data["date"] >= VAL_START) & (data["date"] <= VAL_END)].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


def evaluate_model(name: str, model, X_train, y_train, X_val, y_val, val_df: pd.DataFrame):
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(y_val, pred)
    precision = precision_score(y_val, pred, zero_division=0)
    recall = recall_score(y_val, pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, prob)

    predicted_up = pred == 1
    signal_count = int(predicted_up.sum())
    coverage = signal_count / len(X_val)

    if signal_count > 0:
        upward_hits = y_val[predicted_up].mean()
        avg_return = val_df.loc[predicted_up, "next_return"].mean() * 100.0
        median_return = val_df.loc[predicted_up, "next_return"].median() * 100.0
    else:
        upward_hits = float("nan")
        avg_return = float("nan")
        median_return = float("nan")

    return {
        "model": name,
        "accuracy": round(acc * 100, 2),
        "upward_hit_rate": round(upward_hits * 100, 2) if pd.notna(upward_hits) else None,
        "signals": signal_count,
        "coverage": round(coverage * 100, 2),
        "avg_next_return_pct": round(avg_return, 2) if pd.notna(avg_return) else None,
        "median_next_return_pct": round(median_return, 2) if pd.notna(median_return) else None,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "roc_auc": round(roc_auc, 3),
        "raw": {
            "pred_prob": prob,
            "pred": pred,
            "y_true": y_val.to_numpy(),
            "next_return": val_df["next_return"].to_numpy(),
        },
    }


def run_validation_experiment(data_dir: str | Path):
    data = load_next_day_dataset(data_dir)
    train, val = make_time_split(data)

    X_train = train[FEATURES]
    y_train = train["target"].astype(int)
    X_val = val[FEATURES]
    y_val = val["target"].astype(int)

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=42,
        ),
    }

    results = []
    for name, model in models.items():
        results.append(evaluate_model(name, model, X_train, y_train, X_val, y_val, val))

    return train, val, results


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1] / "data" / "history"
    train, val, results = run_validation_experiment(base_dir)

    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(val)}")
    print(f"Train range: {train['date'].min().date()} ~ {train['date'].max().date()}")
    print(f"Validation range: {val['date'].min().date()} ~ {val['date'].max().date()}")
    print()
    print("Model comparison on validation:")
    for row in results:
        print(row)
