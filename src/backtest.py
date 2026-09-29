from __future__ import annotations
from pathlib import Path
import pandas as pd
from .indicators import add_indicators
from .scoring import score_row


def backtest_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    x = add_indicators(df)
    rows = []

    needed = [
        "ret_5d", "ma5", "ma20", "ma60",
        "volume_ratio", "close_position",
        "near_20d_high", "rsi14"
    ]

    for i in range(len(x) - 1):
        row = x.iloc[i]
        if row[needed].isna().any():
            continue

        score, _ = score_row(row)
        next_return = x.iloc[i + 1]["close"] / row["close"] - 1

        rows.append({
            "date": row["date"].date().isoformat(),
            "score": score,
            "next_return": next_return,
            "next_up": int(next_return > 0),
        })

    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame, threshold: float = 70) -> dict:
    selected = bt[bt["score"] >= threshold].copy()

    if selected.empty:
        return {
            "signals": 0,
            "win_rate": None,
            "avg_next_return": None,
            "median_next_return": None,
        }

    return {
        "signals": len(selected),
        "win_rate": round(float(selected["next_up"].mean()) * 100, 2),
        "avg_next_return": round(float(selected["next_return"].mean()) * 100, 3),
        "median_next_return": round(float(selected["next_return"].median()) * 100, 3),
    }


def run_backtest(data_dir: str | Path, threshold: float = 70):
    data_dir = Path(data_dir)
    summaries = []

    for path in sorted(data_dir.glob("*.csv")):
        bt = backtest_file(path)
        if bt.empty:
            continue
        summary = summarize(bt, threshold)
        summary["symbol"] = path.stem
        summaries.append(summary)

    return pd.DataFrame(summaries)
