"""Build a verified KRX snapshot for a configured lab watchlist."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from .investor_flow import fetch_investor_flows, latest_published_flow_date


def refresh(tickers: list[str], sessions: pd.DatetimeIndex, output: Path) -> dict:
    if not tickers or len(set(tickers)) != len(tickers) or any(
        not re.fullmatch(r"[0-9]{6}", ticker) for ticker in tickers
    ):
        raise ValueError("Provide distinct six-digit KRX tickers")
    cutoff = latest_published_flow_date()
    expected = sessions[sessions <= cutoff].sort_values()
    if len(expected) < 20 or expected.has_duplicates or expected.hasnans:
        raise ValueError("Invalid KRX reference calendar")
    as_of = expected[-1].date().isoformat()
    symbols = {}
    for ticker in tickers:
        result = fetch_investor_flows(ticker, as_of, expected_sessions=expected)
        if result is None or result["as_of"] != as_of or result["sessions"] != 20:
            raise RuntimeError(f"{ticker}: incomplete KRX investor-flow window for {as_of}")
        symbols[ticker] = result
    snapshot = {"source": "KRX", "as_of": as_of, "symbols": symbols}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False)
    pending = Path(temp.name)
    try:
        with temp:
            json.dump(snapshot, temp, ensure_ascii=False)
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    return snapshot


def load_snapshot(path: Path, ticker: str, as_of: str) -> dict | None:
    """Read a trusted collector output only for the exact requested session."""
    with path.open(encoding="utf-8") as handle:
        snapshot = json.load(handle)
    if not isinstance(snapshot, dict) or snapshot.get("source") != "KRX" or snapshot.get("as_of") != as_of:
        return None
    symbols = snapshot.get("symbols")
    result = symbols.get(ticker) if isinstance(symbols, dict) else None
    if not isinstance(result, dict) or result.get("as_of") != as_of or result.get("sessions") != 20:
        return None
    if any(type(result.get(column)) is not int for column in ("foreign_net_krw", "institution_net_krw")):
        return None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", nargs="+")
    parser.add_argument("--calendar", type=Path, default=Path("data/history/005930.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sessions = pd.DatetimeIndex(pd.read_csv(args.calendar, usecols=["date"])["date"])
    result = refresh(args.ticker, sessions, args.output)
    print(f"Verified KRX data for {result['as_of']}: {len(result['symbols'])} symbols")


if __name__ == "__main__":
    main()
