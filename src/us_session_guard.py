"""Conservative end-of-session guard for daily Yahoo OHLCV ingestion.

Yahoo's timezone-naive Date column is a local exchange session label.
Timezone-aware timestamps are first converted to the exchange's zone.
The release buffer does not prove vendor finality or validate exchange holidays.
"""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

# A deliberately conservative one-hour (or greater) margin after the
# principal cash session closes. Country labels come from GLOBAL_SCAN_UNIVERSE.
MARKETS = {
    "미국": ("America/New_York", time(17, 0)),     # NYSE 16:00
    "일본": ("Asia/Tokyo", time(16, 30)),         # TSE 15:30
    "독일": ("Europe/Berlin", time(18, 30)),      # Xetra 17:30
    "네덜란드": ("Europe/Amsterdam", time(18, 30)),
    "프랑스": ("Europe/Paris", time(18, 30)),
    "스위스": ("Europe/Zurich", time(18, 30)),
}
NEW_YORK = ZoneInfo("America/New_York")
SAFE_RELEASE_TIME = time(17, 0)


def market_cutoff(country: str, *, now: datetime | None = None):
    """Return (local calendar date, today's bar can be released)."""
    if country not in MARKETS:
        raise ValueError(f"Unsupported exchange country: {country}")
    zone_name, release = MARKETS[country]
    zone = ZoneInfo(zone_name)
    if now is None:
        local = datetime.now(zone)
    else:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        local = now.astimezone(zone)
    return local.date(), local.time() >= release


def cutoff_date(*, now: datetime | None = None):
    """Backwards-compatible US cutoff accessor."""
    return market_cutoff("미국", now=now)


def keep_confirmed_market_daily(
    frame: pd.DataFrame, country: str, *, now: datetime | None = None
) -> pd.DataFrame:
    """Remove future, weekend, and still-open exchange-session rows.

    Naive dates are treated as *session labels* (no UTC conversion).
    Aware timestamps are converted to the country's exchange time zone.
    Always returns naive, normalized local session dates on a copy.

    This is an ingestion-time protection only; a historical row collected
    intraday cannot be certified as final using its date alone.
    """
    if not isinstance(frame, pd.DataFrame) or "date" not in frame:
        raise ValueError("OHLCV frame with date column required")
    # Validate the market even for an empty frame.
    today, complete = market_cutoff(country, now=now)
    if frame.empty:
        return frame.copy()
    parsed = pd.to_datetime(frame["date"], errors="coerce")
    if parsed.isna().any():
        raise ValueError("invalid daily-bar date")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert(MARKETS[country][0]).dt.tz_localize(None)
    elif not pd.api.types.is_datetime64_any_dtype(parsed):
        raise ValueError("mixed or unsupported time zones in date column")
    sessions = parsed.dt.normalize()
    mask = (sessions.dt.date < today) | ((sessions.dt.date == today) & complete)
    mask &= sessions.dt.dayofweek < 5
    out = frame.loc[mask].copy()
    out["date"] = sessions.loc[mask].to_numpy()
    return out.reset_index(drop=True)


def keep_confirmed_us_daily(frame: pd.DataFrame, *, now: datetime | None = None) -> pd.DataFrame:
    """Backwards-compatible US daily guard."""
    return keep_confirmed_market_daily(frame, "미국", now=now)
