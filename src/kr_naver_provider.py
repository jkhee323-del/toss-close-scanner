"""Optional independent NAVER daily OHLCV provider via FinanceDataReader.

This module never writes historical cache files. NAVER is a separately
sourced quote feed, not authoritative KRX adjusted-price provenance.
"""
from __future__ import annotations

import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def fetch_naver_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch a date-bounded historical alternative using NAVER:<six-digit code>."""
    if len(symbol) != 6 or not symbol.isdigit():
        raise ValueError("Korean stock symbol must have exactly six digits")
    import FinanceDataReader as fdr

    raw = fdr.DataReader(f"NAVER:{symbol}", start, end)
    if raw is None or raw.empty:
        raise ValueError(f"{symbol}: NAVER returned no historical price records")
    if raw.columns.duplicated().any():
        raise ValueError(f"{symbol}: duplicate price columns from NAVER")
    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(raw.columns):
        raise ValueError(f"{symbol}: missing NAVER OHLCV columns")
    frame = raw.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    frame = frame.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    frame = frame.reset_index()
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    for col in COLUMNS[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[COLUMNS].isna().any().any():
        raise ValueError(f"{symbol}: NAVER returned missing OHLCV values")
    if frame["date"].duplicated().any() or frame["date"].isna().any():
        raise ValueError(f"{symbol}: duplicate or missing NAVER sessions")
    # Preserve zero/invalid provider rows for explicit downstream validation.
    return frame[COLUMNS].sort_values("date").reset_index(drop=True)


def fetch_naver_review_window(
    symbol: str, first_bad: pd.Timestamp, last_bad: pd.Timestamp,
) -> pd.DataFrame:
    """Read-only source for historical repair analysis, plus adjacent sessions."""
    start = (pd.Timestamp(first_bad) - pd.Timedelta(days=20)).date().isoformat()
    end = (pd.Timestamp(last_bad) + pd.Timedelta(days=21)).date().isoformat()
    return fetch_naver_ohlcv(symbol, start, end)


def validate_naver_overlap(
    original: pd.DataFrame,
    replacement: pd.DataFrame,
    *,
    minimum_matches: int = 3,
    max_close_disagreement: float = 0.01,
) -> int:
    """Require matching independent close prices before accepting fresh bars.

    Returns number of matching *actual sessions*. Any overlapping close
    disagreement above tolerance rejects the entire candidate feed.
    """
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(original.columns) or not required.issubset(replacement.columns):
        raise ValueError("Missing OHLCV data for feed comparison")
    from .data_integrity_audit import check_ohlcv_quality

    issue, _ = check_ohlcv_quality(replacement)
    if issue:
        raise ValueError(f"Invalid alternative provider feed: {issue}")
    old = original.copy()
    new = replacement.copy()
    old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.normalize()
    new["date"] = pd.to_datetime(new["date"], errors="coerce").dt.normalize()
    old["close"] = pd.to_numeric(old["close"], errors="coerce")
    new["close"] = pd.to_numeric(new["close"], errors="coerce")
    comparable = old[["date", "close"]].merge(new[["date", "close"]], on="date", suffixes=("_old", "_new"))
    # Invalidate known-bad original rows, but require several good overlaps.
    comparable = comparable.loc[comparable["close_old"].gt(0) & comparable["close_new"].gt(0)]
    if len(comparable) < minimum_matches:
        raise ValueError("Insufficient independent overlap to verify NAVER price adjustment basis")
    ratios = (comparable["close_old"] / comparable["close_new"] - 1).abs()
    if ratios.gt(max_close_disagreement).any():
        raise ValueError("NAVER and stored price adjustment bases disagree")
    return len(comparable)
