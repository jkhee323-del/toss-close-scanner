"""KR watchdog must verify newer price dates without mutating saved caches."""
import pandas as pd
import pytest
from src import kr_refresh_watchdog as watchdog


def _bars(dates):
    n = len(dates)
    return pd.DataFrame({
        "date": dates,
        "open": [100.0] * n,
        "high": [103.0] * n,
        "low": [99.0] * n,
        "close": [102.0] * n,
        "volume": [10000] * n,
    })


def _files(tmp_path):
    days = pd.bdate_range("2026-11-10", periods=9)
    cached = _bars(days[:-1])
    for symbol in watchdog.REFERENCE_SYMBOLS:
        cached.to_csv(tmp_path / f"{symbol}.csv", index=False)
    return cached, _bars(days), days


def test_watchdog_detects_new_completed_session_without_writes(tmp_path, monkeypatch):
    cached, live, days = _files(tmp_path)
    monkeypatch.setattr(watchdog, "fetch_naver_ohlcv", lambda *args: live.copy())
    original = (tmp_path / "005930.csv").read_bytes()
    out = watchdog.probe(tmp_path, cutoff=days[-1])
    assert out["needs_refresh"]
    assert all(row["missing"] and row["overlap"] == 8 for row in out["reference_symbols"])
    assert (tmp_path / "005930.csv").read_bytes() == original


def test_watchdog_no_refresh_when_prices_already_current(tmp_path, monkeypatch):
    cached, _, days = _files(tmp_path)
    monkeypatch.setattr(watchdog, "fetch_naver_ohlcv", lambda *args: cached.copy())
    out = watchdog.probe(tmp_path, cutoff=days[-1])
    assert not out["needs_refresh"]


def test_watchdog_rejects_different_adjusted_price_basis(tmp_path, monkeypatch):
    cached, live, days = _files(tmp_path)
    live[["open", "high", "low", "close"]] *= 2
    monkeypatch.setattr(watchdog, "fetch_naver_ohlcv", lambda *args: live.copy())
    with pytest.raises(ValueError, match="adjustment bases disagree"):
        watchdog.probe(tmp_path, cutoff=days[-1])


def test_watchdog_never_treats_future_session_as_completed(tmp_path, monkeypatch):
    cached, live, days = _files(tmp_path)
    monkeypatch.setattr(watchdog, "fetch_naver_ohlcv", lambda *args: live.copy())
    out = watchdog.probe(tmp_path, cutoff=days[-2])
    assert not out["needs_refresh"]
    assert out["provider_latest"] == days[-2].date().isoformat()
