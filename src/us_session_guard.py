"""Conservative US daily-bar close guard (research/data-ingestion use).

This rejects the current NYSE date until 17:00 America/New_York. It accounts
for US daylight saving automatically, but does NOT certify that a vendor bar
is final, corrected, or exchange-calendar-valid. Re-download suspect history.
"""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

NEW_YORK = ZoneInfo("America/New_York")
SAFE_RELEASE_TIME = time(17, 0)


def cutoff_date(*, now: datetime | None = None):
    """Latest *calendar date* permitted by a conservative US closing clock."""
    if now is None:
        clock = datetime.now(NEW_YORK)
    else:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        clock = now.astimezone(NEW_YORK)
    # Compare local date and local clock; prior days are allowed during trading.
    return clock.date(), clock.time() >= SAFE_RELEASE_TIME


def keep_confirmed_us_daily(frame: pd.DataFrame, *, now: datetime | None = None) -> pd.DataFrame:
    """Drop uncompleted, future-dated, and weekend US daily bars.

    Date-only or tz-naive inputs are interpreted as vendor session-date labels.
    Tz-aware inputs are first converted to New York, preventing UTC date drift.
    Output date is tz-naive midnight local session date. Does not modify input.

    The 17:00 clock is a conservative buffer, not proof of provider finality.
    """
    if not isinstance(frame, pd.DataFrame) or "date" not in frame:
        raise ValueError("OHLCV frame with date column required")
    if frame.empty:
        return frame.copy()

    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("invalid daily-bar date")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_convert(NEW_YORK).dt.tz_localize(None)
    elif not isinstance(dates.dtype, pd.DatetimeTZDtype) and not pd.api.types.is_datetime64_any_dtype(dates):
        raise ValueError("mixed or unsupported time zones in date column")

    dates = dates.dt.normalize()
    today, complete = cutoff_date(now=now)
    # .dt.weekday avoids accepting an accidental Saturday/Sunday daily bar.
    mask = (dates.dt.date < today) | ((dates.dt.date == today) & complete)
    mask &= dates.dt.dayofweek < 5
    out = frame.loc[mask].copy()
    out["date"] = dates.loc[mask].to_numpy()
    return out.reset_index(drop=True)
