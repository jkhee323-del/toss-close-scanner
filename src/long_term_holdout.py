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


def rank_and_grade(candidates: list[tuple[str, float, float | None]], top_n: int) -> dict | None:
    """Rank *before* inspecting later returns. Missing exits cannot promote other names."""
    if len(candidates) < top_n:
        return None
    ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
    top = ranked[:top_n]
    observed = np.asarray([ret for _, _, ret in top if ret is not None], dtype=float)
    benchmark = np.asarray([ret for _, _, ret in ranked if ret is not None], dtype=float)
    # Do not silently report profits for unfinished holdings.
    return {
        "top_n": top_n,
        "universe_count": len(ranked),
        "top_evaluated_count": len(observed),
        "top_unfinished_count": top_n - len(observed),
        "universe_evaluated_count": len(benchmark),
        "top_coverage": len(observed) / top_n,
        "top_mean_net_return": float(observed.mean()) if len(observed) else float("nan"),
        "top_median_net_return": float(np.median(observed)) if len(observed) else float("nan"),
        "top_win_rate": float((observed > 0).mean()) if len(observed) else float("nan"),
        "universe_mean_net_return": float(benchmark.mean()) if len(benchmark) else float("nan"),
        "excess_vs_universe": float(observed.mean() - benchmark.mean()) if len(observed) and len(benchmark) else float("nan"),
        "top_symbols": ",".join(symbol for symbol, _, _ in top),
    }


def audit(history_dir: Path, start: str = "2025-01-01", top_n: int = 10, cost_rate: float = 0.002, max_symbols: int | None = None) -> pd.DataFrame:
    if top_n <= 0 or cost_rate < 0:
        raise ValueError("top_n must be positive and cost_rate nonnegative")
    history = {}
    paths = sorted(history_dir.glob("*.csv"))
    if max_symbols is not None:
        paths = paths[:max_symbols]
    for path in paths:
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
                # Rank on the signal date WITHOUT knowing which positions later mature.
                scan = analyze_stock(path, days, as_of_date=str(signal_day.date()), history_frame=frame)
                if scan is None or scan["기준일"] != str(signal_day.date()):
                    continue
                ret = realized_return(frame, signal_day, days, cost_rate)
                candidates.append((symbol, scan["점수"], ret))
            graded = rank_and_grade(candidates, top_n)
            if graded is None:
                continue
            output.append({"signal_date": str(signal_day.date()), "horizon_days": days, **graded})
    return pd.DataFrame(output)


def summarize_completed_periods(report: pd.DataFrame) -> pd.DataFrame:
    """Only fully matured portfolios qualify for aggregate performance claims."""
    if report.empty:
        return pd.DataFrame()
    valid = report.loc[report["top_evaluated_count"] == report["top_n"]].copy()
    summary = []
    for horizon in sorted(report["horizon_days"].unique()):
        all_periods = report.loc[report["horizon_days"] == horizon]
        completed = valid.loc[valid["horizon_days"] == horizon]
        summary.append({
            "horizon_days": int(horizon),
            "signal_periods": len(all_periods),
            "fully_matured_periods": len(completed),
            "excluded_incomplete_periods": len(all_periods) - len(completed),
            "mean_return_completed_only": float(completed["top_mean_net_return"].mean()) if len(completed) else float("nan"),
            "win_rate_completed_only": float(completed["top_win_rate"].mean()) if len(completed) else float("nan"),
            "excess_return_completed_only": float(completed["excess_vs_universe"].mean()) if len(completed) else float("nan"),
        })
    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", type=Path, default=Path("data/history"))
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--cost", type=float, default=0.002, help="round-trip cost fraction")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/long_term_holdout.csv"))
    args = parser.parse_args()
    report = audit(args.history_dir, start=args.start, top_n=args.top, cost_rate=args.cost, max_symbols=args.max_symbols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    summary = summarize_completed_periods(report)
    summary_path = args.output.with_name(args.output.stem + "_matured_summary.csv")
    summary.to_csv(summary_path, index=False)
    if report.empty:
        print("No evaluated signals. No accuracy claim can be made.")
    else:
        print("COMPLETED portfolios only (every selected stock has a known exit):")
        print(summary.to_string(index=False))
        print("WARNING: survivorship bias, limited stock universe and correlated positions remain.")
    print(f"Saved {len(report)} signal/horizon pairs to {args.output}")
    print(f"Saved fully matured summary to {summary_path}")


if __name__ == "__main__":
    main()
