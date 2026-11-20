"""Read-only production-path smoke checks for lab deployment readiness.

Runs the actual cached models and top-level scan functions without starting a
Streamlit server, downloading new prices or writing to production CSV files.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .long_term_scanner import get_recommendations
from .signals import compute_scored_window, pick_signals
from .global_cache_validation import validate_global_candidates
from .data_integrity_audit import inspect_market


def run_smoke(data: str | Path = "data") -> dict:
    data = Path(data)
    window, meta = compute_scored_window(data, n_days=4)
    if window.empty:
        raise RuntimeError("No executable Korean close-scan signals")
    if set(window["day_offset"].astype(int).unique()) != {0, 1, 2, 3}:
        raise RuntimeError("Korean scanner missing one or more day tabs")
    if window["scanner_score"].isna().any() or not window["scanner_score"].between(0, 100).all():
        raise RuntimeError("Non-finite Korean model score")
    if meta.get("training_symbols", 0) < 100:
        raise RuntimeError("Insufficient Korean symbol coverage")
    if pd.Timestamp(meta["latest_date"]) < pd.Timestamp("2026-11-13"):
        raise RuntimeError("Korean cache is older than latest audit baseline")
    for offset in range(4):
        top, count = pick_signals(window, offset, top_n=10, min_trading_value=0)
        if top.empty or count <= 0:
            raise RuntimeError(f"Korean top signal tab {offset} is empty")
    long_kr = get_recommendations("1", "KR")
    long_us = get_recommendations("1", "US")
    if long_kr.empty or long_us.empty:
        raise RuntimeError("KR or US medium-horizon scan is empty")
    if "기준일" not in long_kr or "기준일" not in long_us:
        raise RuntimeError("Missing reference dates")
    global_health = inspect_market(data / "global_history")
    if global_health["status"] != "pass":
        raise RuntimeError(f"Overseas cache corrupted: {global_health.get('total_errors')} invalid histories")
    global_rows = validate_global_candidates(data)
    result = {
        "global_history_verified": global_health["valid"],
        "global_candidates_verified": global_rows["candidate_rows_verified"],
        "kr_date": str(pd.Timestamp(meta["latest_date"]).date()),
        "kr_symbols": int(meta["training_symbols"]),
        "kr_corrupted_excluded": int(meta.get("excluded_invalid_histories", 0)),
        "kr_old_history_recovered": int(meta.get("recovered_recent_histories", 0)),
        "long_kr_symbols": len(long_kr),
        "long_us_symbols": len(long_us),
        "long_kr_date": str(long_kr.iloc[0]["기준일"]),
        "long_us_date": str(long_us.iloc[0]["기준일"]),
        "tabs_verified": 4,
    }
    print(result)
    return result


if __name__ == "__main__":
    run_smoke()
