"""US daily session guard tests. All times explicitly specify UTC."""
from datetime import datetime

import pandas as pd
import pytest

from src.us_session_guard import cutoff_date, keep_confirmed_us_daily


@pytest.mark.parametrize(
    "instant, expected_count",
    [
        ("2026-11-02T21:59:00+00:00", 1),  # EST 16:59
        ("2026-11-02T22:00:00+00:00", 2),  # EST 17:00
        ("2026-07-01T20:59:00+00:00", 1),  # EDT 16:59
        ("2026-07-01T21:00:00+00:00", 2),  # EDT 17:00
    ],
)
def test_dst_cutoff(instant, expected_count):
    date = "2026-07-01" if instant.startswith("2026-07") else "2026-11-02"
    frame = pd.DataFrame({"date": [pd.Timestamp(date) - pd.Timedelta(days=3), pd.Timestamp(date)]})
    result = keep_confirmed_us_daily(frame, now=datetime.fromisoformat(instant))
    assert len(result) == expected_count


def test_future_and_weekend_rows_are_removed():
    data = pd.DataFrame({"date": ["2026-10-30", "2026-10-31", "2026-11-02", "2026-11-03"]})
    result = keep_confirmed_us_daily(data, now=datetime.fromisoformat("2026-11-02T23:00:00+00:00"))
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-30", "2026-11-02"]
    assert data["date"].tolist() == ["2026-10-30", "2026-10-31", "2026-11-02", "2026-11-03"]


def test_utc_timestamp_is_converted_to_new_york_trading_date():
    data = pd.DataFrame({"date": pd.to_datetime([
        "2026-11-03T01:00:00+00:00",  # New York Monday 20:00
        "2026-11-03T05:00:00+00:00",  # New York Tuesday midnight
    ])})
    result = keep_confirmed_us_daily(data, now=datetime.fromisoformat("2026-11-03T01:30:00+00:00"))
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-11-02"]


def test_naive_session_dates_do_not_shift_to_previous_day():
    data = pd.DataFrame({"date": pd.to_datetime(["2026-11-02"])})
    result = keep_confirmed_us_daily(data, now=datetime.fromisoformat("2026-11-02T22:01:00+00:00"))
    assert result["date"].iloc[0] == pd.Timestamp("2026-11-02")


def test_bad_dates_are_rejected():
    with pytest.raises(ValueError, match="invalid"):
        keep_confirmed_us_daily(pd.DataFrame({"date": ["not a day"]}))


def test_naive_clock_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        cutoff_date(now=datetime(2026, 11, 2, 17, 0))


def test_empty_frame_is_safe():
    x = pd.DataFrame(columns=["date", "close"])
    assert keep_confirmed_us_daily(x).empty
