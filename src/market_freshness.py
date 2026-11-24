"""Conservative display-only completed-session freshness checks.

Never claim that inferred weekdays are confirmed exchange sessions:
exchange holidays and late vendor publication can change the expected date.
All comparisons are advisory; missing candidate prices must not be filled.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd

# The grace time is deliberately later than each venue's regular close.
# Holiday calendars are not available, so these checks issue warnings
# rather than declaring a trading day definitively missing.
VENUES = {
    "KR": ("Asia/Seoul", time(17, 30)),
    "US": ("America/New_York", time(18, 30)),
    "JP": ("Asia/Tokyo", time(17, 30)),
    "EU": ("Europe/Berlin", time(20, 0)),
    "UK": ("Europe/London", time(18, 30)),
}

COUNTRY_TO_VENUE = {
    "한국": "KR", "미국": "US", "일본": "JP", "영국": "UK",
    "독일": "EU", "네덜란드": "EU", "프랑스": "EU",
    "스위스": "EU", "스페인": "EU", "이탈리아": "EU",
}


def venue_for_country(country: str | None) -> str | None:
    """Unknown exchanges do not receive a made-up freshness verdict."""
    return COUNTRY_TO_VENUE.get(str(country or "").strip())


def estimated_last_completed_session(
    venue: str, *, now: datetime | pd.Timestamp | None = None
) -> pd.Timestamp:
    """Latest plausible *weekday*, not a guaranteed exchange trading day."""
    if venue not in VENUES:
        raise ValueError(f"Unsupported exchange venue: {venue}")
    tz, cutoff = VENUES[venue]
    instant = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if instant.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = instant.tz_convert(ZoneInfo(tz))
    day = pd.Timestamp(local.date())
    if day.weekday() >= 5 or local.time() < cutoff:
        day = day - pd.offsets.BDay(1)
    return day.normalize()


def completed_weekday_lag(
    observed: str | datetime | pd.Timestamp | None,
    venue: str,
    *,
    now: datetime | pd.Timestamp | None = None,
) -> int | None:
    """Number of potential completed weekdays since the observed session.

    A negative date/future or invalid date yields None: callers should
    display uncertainty rather than silently call it fresh.
    """
    day = pd.to_datetime(observed, errors="coerce")
    if pd.isna(day):
        return None
    day = pd.Timestamp(day)
    if day.tzinfo is not None:
        day = day.tz_convert("UTC").tz_localize(None)
    day = day.normalize()
    expected = estimated_last_completed_session(venue, now=now)
    if day > expected:
        return None
    return len(pd.bdate_range(start=day + pd.offsets.BDay(1), end=expected))


def session_notice(
    observed: str | datetime | pd.Timestamp | None,
    venue: str | None,
    *,
    now: datetime | pd.Timestamp | None = None,
) -> dict:
    """Warning-only metadata for user-facing stale-price indicators."""
    date = pd.to_datetime(observed, errors="coerce")
    date_label = "확인 불가" if pd.isna(date) else pd.Timestamp(date).strftime("%Y-%m-%d")
    if venue not in VENUES:
        return {"date": date_label, "lag": None, "warning": True, "status": "unknown_exchange"}
    lag = completed_weekday_lag(observed, venue, now=now)
    return {
        "date": date_label,
        "lag": lag,
        "warning": lag is None or lag >= 1,
        "status": "unknown_date" if lag is None else "possibly_stale" if lag >= 1 else "recent",
    }
