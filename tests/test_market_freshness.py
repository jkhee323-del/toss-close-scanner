"""Conservative, timezone-aware labels for saved closing-price data."""
import pandas as pd
import pytest

from src.market_freshness import (
    completed_weekday_lag,
    estimated_last_completed_session,
    session_notice,
    venue_for_country,
)


def test_kr_tuesday_after_close_flags_yesterday_as_possible_stale():
    now = pd.Timestamp("2026-11-24T22:58:00+09:00")
    assert estimated_last_completed_session("KR", now=now) == pd.Timestamp("2026-11-24")
    assert completed_weekday_lag("2026-11-23", "KR", now=now) == 1
    assert session_notice("2026-11-23", "KR", now=now)["warning"]
    assert not session_notice("2026-11-24", "KR", now=now)["warning"]


def test_us_market_is_not_expected_to_have_closed_on_korean_tuesday_evening():
    now = pd.Timestamp("2026-11-24T22:58:00+09:00")
    assert estimated_last_completed_session("US", now=now) == pd.Timestamp("2026-11-23")
    assert completed_weekday_lag("2026-11-23", "US", now=now) == 0
    assert completed_weekday_lag("2026-11-20", "US", now=now) == 1


def test_monday_before_korea_market_close_expects_previous_friday():
    now = pd.Timestamp("2026-11-23T09:00:00+09:00")
    assert estimated_last_completed_session("KR", now=now) == pd.Timestamp("2026-11-20")
    assert completed_weekday_lag("2026-11-20", "KR", now=now) == 0


def test_weekend_uses_last_weekday():
    now = pd.Timestamp("2026-11-28T13:00:00+09:00")
    assert estimated_last_completed_session("KR", now=now) == pd.Timestamp("2026-11-27")


def test_country_mapping_and_unknown_exchange_are_explicit():
    assert venue_for_country("미국") == "US"
    assert venue_for_country("일본") == "JP"
    assert venue_for_country("독일") == "EU"
    assert venue_for_country("지구") is None
    assert session_notice("2026-11-23", None)["warning"]


def test_future_or_missing_dates_cannot_be_reported_as_fresh():
    now = pd.Timestamp("2026-11-24T22:58:00+09:00")
    assert session_notice("2026-11-25", "KR", now=now)["status"] == "unknown_date"
    assert session_notice(None, "KR", now=now)["status"] == "unknown_date"


def test_naive_time_source_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        estimated_last_completed_session("KR", now=pd.Timestamp("2026-11-24 22:58"))


def test_europe_and_japan_local_close_handling():
    now = pd.Timestamp("2026-11-24T22:58:00+09:00")
    assert estimated_last_completed_session("JP", now=now) == pd.Timestamp("2026-11-24")
    assert estimated_last_completed_session("EU", now=now) == pd.Timestamp("2026-11-23")


def test_local_global_cache_reports_nvda_stale_without_modifying_files(tmp_path):
    from src.market_freshness import find_stale_cached_symbols

    now = pd.Timestamp("2026-11-24T22:58:00+09:00")
    data = {
        "MSFT": "2026-11-23",
        "NVDA": "2026-11-20",
    }
    for sym, date in data.items():
        path = tmp_path / f"{sym}.csv"
        path.write_text("date,close\n" + date + ",100\n", encoding="utf-8")
    universe = [("MSFT", "Microsoft", "미국"), ("NVDA", "NVIDIA", "미국")]
    originals = {sym: (tmp_path / f"{sym}.csv").read_bytes() for sym in data}
    stale = find_stale_cached_symbols(tmp_path, universe, now=now)
    assert len(stale) == 1
    assert stale[0]["symbol"] == "NVDA"
    assert stale[0]["lag"] == 1
    for sym in data:
        assert (tmp_path / f"{sym}.csv").read_bytes() == originals[sym]


def test_missing_cached_symbol_is_warning_not_silent_pass(tmp_path):
    from src.market_freshness import find_stale_cached_symbols

    report = find_stale_cached_symbols(
        tmp_path, [("NVDA", "NVIDIA", "미국")],
        now=pd.Timestamp("2026-11-24T22:58:00+09:00"),
    )
    assert report[0]["symbol"] == "NVDA"
    assert report[0]["date"] == "없음"
