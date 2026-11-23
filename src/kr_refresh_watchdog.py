"""Read-only Korean daily cache watchdog using an independently sourced session.

The watchdog distinguishes 'no completed new market day' from 'new independently
observed market session not yet cached'. It never uses a guessed holiday list,
and refuses to mutate prices when the independent price basis differs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .kr_naver_provider import fetch_naver_ohlcv, validate_naver_overlap
from .auto_update import _latest_complete_kr_date

REFERENCE_SYMBOLS = ("005930", "000660")
MIN_MATCHES = 3


def probe(reference_dir: Path, cutoff: pd.Timestamp | None = None) -> dict:
    as_of = _latest_complete_kr_date() if cutoff is None else pd.Timestamp(cutoff).normalize()
    results = []
    for symbol in REFERENCE_SYMBOLS:
        file = reference_dir / f"{symbol}.csv"
        cached = pd.read_csv(file)
        cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.normalize()
        if cached.empty or cached["date"].isna().any() or cached["date"].duplicated().any():
            raise ValueError(f"{symbol}: invalid cached session dates")
        last = cached["date"].max()
        start = (last - pd.Timedelta(days=28)).date().isoformat()
        end = (as_of + pd.Timedelta(days=1)).date().isoformat()
        live = fetch_naver_ohlcv(symbol, start, end)
        live = live.loc[live["date"] <= as_of].copy()
        if live.empty:
            raise ValueError(f"{symbol}: no completed independent market sessions")
        matches = validate_naver_overlap(cached.tail(25), live, minimum_matches=MIN_MATCHES)
        observed = live["date"].max()
        results.append({
            "symbol": symbol,
            "stored": last.date().isoformat(),
            "provider": observed.date().isoformat(),
            "overlap": matches,
            "missing": bool(observed > last),
        })
    # Require corroboration across two independent symbol histories
    # sourced from the same feed. One symbol could be suspended or halted.
    missing = [item for item in results if item["missing"]]
    return {
        "as_of_cutoff": as_of.date().isoformat(),
        "needs_refresh": bool(missing),
        "reference_symbols": results,
        "provider_latest": max(row["provider"] for row in results),
        "cached_latest": max(row["stored"] for row in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/history"))
    parser.add_argument("--refresh-if-needed", action="store_true")
    args = parser.parse_args()
    before = probe(args.root)
    print("Korea watchdog:", json.dumps(before, ensure_ascii=False), flush=True)
    if not args.refresh_if_needed:
        return
    if not before["needs_refresh"]:
        print("Korea watchdog: cached completed session matches available independent feed.")
        return
    from .auto_update import update_korea
    update_korea()
    after = probe(args.root)
    print("Korea watchdog after refresh:", json.dumps(after, ensure_ascii=False), flush=True)
    if after["needs_refresh"]:
        raise RuntimeError("Korea cache still lacks an independently confirmed completed session")


if __name__ == "__main__":
    main()
