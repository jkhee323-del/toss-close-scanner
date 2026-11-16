"""Historical, close-of-day long-term scanner audit.

For each month-end session, rank stocks using only data available at that close.
Enter at the next available session's open, exit at the H-th later session close.
Require the exit observation to exist; do not grade unfinished positions.
The stored universe can have survivorship bias, so results are exploratory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .long_term_scanner import HORIZONS, analyze_stock


def _load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "open", "close"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for key in ("open", "close"):
        df[key] = pd.to_numeric(df[key], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"]).sort_values("date").drop_duplicates("date")
    return df.reset_index(drop=True)


def realized_return(df: pd.DataFrame, signal_date: pd.Timestamp, horizon: int, cost_rate: float = 0.002) -> float | None:
    """Return from next-session open to signal+horizon close, net round-trip cost."""
    matching = df.index[df["date"] == signal_date]
    if len(matching) != 1:
        return None
    signal_index = int(matching[0])
    exit_index = signal_index + horizon
    if exit_index >= len(df):
        return None
    entry = float(df.iloc[signal_index + 1]["open"])
    exit_price = float(df.iloc[exit_index]["close"])
    if not (np.isfinite(entry) and np.isfinite(exit_price)) or entry <= 0 or exit_price <= 0:
        return None
    return exit_price / entry - 1.0 - cost_rate


def audit(history_dir: Path, start: str = "2025-01-01", top_n: int = 10, cost_rate: float = 0.002) -> pd.DataFrame:
    if top_n <= 0 or cost_rate < 0:
        raise ValueError("top_n must be positive and cost_rate nonnegative")
    history = {}
    for path in sorted(history_dir.glob("*.csv")):
        frame = _load_history(path)
        if not frame.empty:
            history[path.stem] = (path, frame)
    if not history:
        return pd.DataFrame()
    sessions = sorted(set().union(*(set(frame["date"]) for _, frame in history.values())))
    sessions = pd.DatetimeIndex(sessions)
    sessions = sessions[sessions >= pd.Timestamp(start)]
    # One signal per month: last observed session on the shared market calendar.
    month_ends = pd.Series(sessions, index=sessions).groupby(sessions.to_period("M")).max().tolist()
    output = []
    for signal_day in month_ends:
        for horizon_name, (_, days) in HORIZONS.items():
            candidates = []
            for symbol, (path, frame) in history.items():
                # Require a tradable and fully observed future holding window.
                ret = realized_return(frame, signal_day, days, cost_rate)
                if ret is None:
                    continue
                scan = analyze_stock(path, days, as_of_date=str(signal_day.date()))
                if scan is None or scan["기준일"] != str(signal_day.date()):
                    continue
                candidates.append((symbol, scan["점수"], ret))
            if len(candidates) < top_n:
                continue
            candidates.sort(key=lambda x: (-x[1], x[0]))
            winners = candidates[:top_n]
            selected = np.array([x[2] for x in winners], dtype=float)
            full = np.array([x[2] for x in candidates], dtype=float)
            output.append({
                "signal_date": str(signal_day.date()),
                "horizon_days": days,
                "top_n": top_n,
                "universe_count": len(candidates),
                "top_mean_net_return": float(selected.mean()),
                "top_median_net_return": float(np.median(selected)),
                "top_win_rate": float((selected > 0).mean()),
                "universe_mean_net_return": float(full.mean()),
                "excess_vs_universe": float(selected.mean() - full.mean()),
            })
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", type=Path, default=Path("data/history"))
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--cost", type=float, default=0.002, help="round-trip cost fraction")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/long_term_holdout.csv"))
    args = parser.parse_args()
    report = audit(args.history_dir, start=args.start, top_n=args.top, cost_rate=args.cost)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    if report.empty:
        print("No fully observed holding periods. No accuracy claim can be made.")
    else:
        print(report.groupby("horizon_days")[["top_mean_net_return", "top_win_rate", "excess_vs_universe"]].mean().to_string())
    print(f"Saved {len(report)} evaluated month/horizon pairs to {args.output}")


if __name__ == "__main__":
    main()
