"""Isolated five-year US research downloader — never edits deployed stock cache.

Uses existing exchange-close and OHLCV guards. Research universe is a small
survivorship-biased convenience sample, not representative of all US equities.
Requires live network; provider errors are recorded, not filled with guesses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .auto_update import _fetch_global
from .global_market_safety import atomic_save_history
from .us_horizon_research import run_all

DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
                   "AMD", "JPM", "COST", "WMT", "XOM", "KO")


def download_research_histories(
    output: Path, *, tickers=DEFAULT_TICKERS, pause_seconds: float = 0.5,
    fetcher=None,
) -> dict:
    """Collect final-session bars to a detached directory and track failures."""
    from .search import US_NAME_MAP

    if pause_seconds < 0:
        raise ValueError("pause_seconds must be nonnegative")
    if fetcher is None:
        fetcher = _fetch_global
    unique = list(dict.fromkeys(str(symbol).upper() for symbol in tickers))
    if not unique:
        raise ValueError("research ticker selection is empty")
    status = {"requested": unique, "written": [], "failed": {}}
    for symbol in unique:
        if US_NAME_MAP.get(symbol, (None, None))[1] != "미국":
            status["failed"][symbol] = "not an explicitly US-listed catalogue ticker"
            continue
        try:
            df = fetcher(symbol, country="미국", period="5y")
            if len(df) < 600:
                raise ValueError(f"too short for long-horizon study: {len(df)} rows")
            destination = Path(output) / f"{symbol}.csv"
            atomic_save_history(destination, df)
            status["written"].append(symbol)
            print(f"{symbol}: {len(df)} validated historical trading sessions")
        except (ValueError, OSError) as exc:
            status["failed"][symbol] = str(exc)
            print(f"[EXCLUDED] {symbol}: {exc}")
        if pause_seconds:
            time.sleep(pause_seconds)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-output", type=Path, default=Path("reports/us_research_5y/history"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/us_research_5y/analysis"))
    parser.add_argument("--max-symbols", type=int, default=12)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if not 1 <= args.max_symbols <= len(DEFAULT_TICKERS):
        parser.error(f"--max-symbols must be 1..{len(DEFAULT_TICKERS)}")
    report = download_research_histories(
        args.history_output, tickers=DEFAULT_TICKERS[:args.max_symbols],
        pause_seconds=args.pause_seconds,
    )
    args.report_output.mkdir(parents=True, exist_ok=True)
    (args.report_output / "data_acquisition.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if len(report["written"]) < 5:
        raise RuntimeError("fewer than five usable US research series; no AI study performed")
    run_all(args.history_output, args.report_output, save_experimental=False)
    print("Research only; no live model artifacts or production cache changes.")


if __name__ == "__main__":
    main()
