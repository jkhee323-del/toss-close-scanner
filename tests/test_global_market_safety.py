"""Offline regression tests for exchange sessions and crash-safe CSV refresh."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.global_market_safety import (
    PriceDataError, atomic_save_history, merge_price_history, validate_ohlcv,
)
from src.us_session_guard import keep_confirmed_market_daily


def history(n=500):
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = pd.Series(np.arange(n, dtype=float) / 10 + 100)
    return pd.DataFrame({
        "date": dates, "open": close, "high": close + 1,
        "low": close - 1, "close": close, "volume": 100_000,
    })


@pytest.mark.parametrize(
    "country, instant, expected",
    [
        ("미국", "2026-11-02T21:59:00+00:00", 1),
        ("미국", "2026-11-02T22:00:00+00:00", 2),
        ("일본", "2026-11-02T07:29:00+00:00", 1),  # Tokyo 16:29
        ("일본", "2026-11-02T07:30:00+00:00", 2),
        ("프랑스", "2026-11-02T17:29:00+00:00", 1),  # Paris 18:29 CET
        ("프랑스", "2026-11-02T17:30:00+00:00", 2),
    ],
)
def test_local_exchange_close(country, instant, expected):
    frame = pd.DataFrame({"date": ["2026-10-30", "2026-11-02"]})
    result = keep_confirmed_market_daily(frame, country, now=datetime.fromisoformat(instant))
    assert len(result) == expected


def test_tokyo_timestamp_does_not_shift_to_ny_calendar():
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-11-02T09:00:00+09:00"])})
    result = keep_confirmed_market_daily(
        frame, "일본", now=datetime.fromisoformat("2026-11-02T08:00:00+00:00")
    )
    assert result["date"].iloc[0] == pd.Timestamp("2026-11-02")


def test_unsupported_market_fails_closed():
    with pytest.raises(ValueError, match="Unsupported"):
        keep_confirmed_market_daily(pd.DataFrame({"date": ["2026-11-02"]}), "unknown")


def test_reject_malformed_ohlcv():
    frame = history(90)
    frame.loc[5, "high"] = frame.loc[5, "close"] - 2
    with pytest.raises(PriceDataError, match="high"):
        validate_ohlcv(frame)


def test_preserve_full_history_when_fetch_only_contains_one_year(tmp_path):
    path = tmp_path / "AAPL.csv"
    original = history(500)
    original.to_csv(path, index=False)
    fresh = original.tail(90).copy()
    fresh.loc[fresh.index[-1], "close"] += 0.25
    fresh.loc[fresh.index[-1], "high"] += 0.25
    merged = merge_price_history(path, fresh)
    assert len(merged) == len(original)
    assert merged["date"].iloc[0] == original["date"].iloc[0]
    assert merged["close"].iloc[-1] == fresh["close"].iloc[-1]
    assert atomic_save_history(path, merged)
    assert len(pd.read_csv(path)) == 500
    assert not atomic_save_history(path, merged)


def test_reject_large_vendor_disagreement_without_overwriting(tmp_path):
    path = tmp_path / "AAPL.csv"
    original = history(500)
    original.to_csv(path, index=False)
    prior_bytes = path.read_bytes()
    fresh = original.tail(90).copy()
    fresh["open"] *= 0.5
    fresh["high"] *= 0.5
    fresh["low"] *= 0.5
    fresh["close"] *= 0.5
    with pytest.raises(PriceDataError, match="discrepancy"):
        merge_price_history(path, fresh)
    assert path.read_bytes() == prior_bytes


def test_atomic_write_failure_keeps_original(tmp_path, monkeypatch):
    import src.global_market_safety as safety

    path = tmp_path / "MSFT.csv"
    original = history(90)
    original.to_csv(path, index=False)
    prior_bytes = path.read_bytes()
    newer = original.copy()
    newer.loc[newer.index[-1], "close"] += 0.5
    newer.loc[newer.index[-1], "high"] += 0.5

    def fail(*args):
        raise OSError("simulated interruption")

    monkeypatch.setattr(safety.os, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        atomic_save_history(path, newer)
    assert path.read_bytes() == prior_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["MSFT.csv"]
