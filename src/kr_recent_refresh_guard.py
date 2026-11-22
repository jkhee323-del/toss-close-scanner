"""Safety gates for alternative completed-session Korean daily-price refresh.

This routine never overwrites any stored historical session. New NAVER or
KRX bars are append-only, with at least three matched existing close prices.
Data that disagree with the stored share-adjustment basis are rejected.
"""
from __future__ import annotations

import pandas as pd

from .kr_naver_provider import validate_naver_overlap
from .data_integrity_audit import check_ohlcv_quality

NEEDED = ["date", "open", "high", "low", "close", "volume"]


def verified_new_candles(
    cached: pd.DataFrame,
    provider: pd.DataFrame,
    latest_complete: pd.Timestamp,
    *,
    minimum_overlap: int = 3,
) -> pd.DataFrame:
    """Return append-only confirmed candles, or raise and preserve the cache."""
    if not set(NEEDED).issubset(cached.columns):
        raise ValueError("Stored history missing OHLCV")
    if provider is None or provider.empty or not set(NEEDED).issubset(provider.columns):
        raise ValueError("Provider did not return candidate OHLCV")
    current = cached[NEEDED].copy()
    incoming = provider[NEEDED].copy()
    current["date"] = pd.to_datetime(current["date"], errors="coerce").dt.normalize()
    incoming["date"] = pd.to_datetime(incoming["date"], errors="coerce").dt.normalize()
    if current["date"].isna().any() or current["date"].duplicated().any():
        raise ValueError("Ambiguous stored dates")
    if incoming["date"].isna().any() or incoming["date"].duplicated().any():
        raise ValueError("Ambiguous provider dates")
    incoming = incoming.loc[incoming["date"] <= pd.Timestamp(latest_complete).normalize()]
    incoming = incoming.sort_values("date").reset_index(drop=True)
    if incoming.empty:
        raise ValueError("No completed trading sessions in provider response")
    # A valid incoming price history is required even if overlapping days
    # have price discrepancies. No invalid zeros accepted as a trading price.
    problem, _ = check_ohlcv_quality(incoming)
    if problem:
        raise ValueError(f"Invalid provider OHLCV: {problem}")
    validate_naver_overlap(current.tail(20), incoming, minimum_matches=minimum_overlap)
    max_stored = current["date"].max()
    new_only = incoming.loc[incoming["date"] > max_stored].copy()
    return new_only.reset_index(drop=True)
