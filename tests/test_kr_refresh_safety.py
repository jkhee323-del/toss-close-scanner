"""KRX data refresh must not treat intraday volume as a completed session."""
import pandas as pd
import pytest
from src import auto_update as updater


def test_kr_cutoff_before_close():
    assert updater._latest_complete_kr_date(pd.Timestamp("2026-11-16 09:30", tz="Asia/Seoul")) == pd.Timestamp("2026-11-15")


def test_kr_cutoff_after_close_buffer():
    assert updater._latest_complete_kr_date(pd.Timestamp("2026-11-16 16:11", tz="Asia/Seoul")) == pd.Timestamp("2026-11-16")


def test_kr_refresh_removes_incomplete_session(tmp_path, monkeypatch):
    folder = tmp_path / "history"
    folder.mkdir()
    path = folder / "005930.csv"
    pd.DataFrame({
        "date": ["2026-11-13", "2026-11-16"],
        "open": [100, 101], "high": [103, 105], "low": [99, 100],
        "close": [102, 104], "volume": [5000, 300],
    }).to_csv(path, index=False)
    monkeypatch.setattr(updater, "KR_HISTORY", folder)
    monkeypatch.setattr(updater, "_latest_complete_kr_date", lambda: pd.Timestamp("2026-11-15"))
    monkeypatch.setattr(updater.time, "sleep", lambda _: None)
    # Simulate provider returning only an unfinished session, which must be filtered.
    raw = pd.DataFrame({
        "시가": [101], "고가": [105], "저가": [100],
        "종가": [104], "거래량": [300],
    }, index=pd.DatetimeIndex(["2026-11-16"], name="날짜"))
    monkeypatch.setattr(updater.stock, "get_market_ohlcv_by_date", lambda *args: raw)
    updater.update_korea()
    result = pd.read_csv(path)
    assert result["date"].tolist() == ["2026-11-13"]
    assert result["volume"].tolist() == [5000]


def test_stale_empty_krx_provider_fails_refresh(tmp_path, monkeypatch):
    folder = tmp_path / "history"
    folder.mkdir()
    pd.DataFrame({
        "date": ["2026-11-13"], "open": [100], "high": [103],
        "low": [99], "close": [102], "volume": [5000],
    }).to_csv(folder / "005930.csv", index=False)
    monkeypatch.setattr(updater, "KR_HISTORY", folder)
    monkeypatch.setattr(updater, "_latest_complete_kr_date", lambda: pd.Timestamp("2026-11-19"))
    monkeypatch.setattr(updater.stock, "get_market_ohlcv_by_date", lambda *args: pd.DataFrame())
    monkeypatch.setattr(updater.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="KRX provider may be unavailable"):
        updater.update_korea()


def test_recent_completed_friday_is_ok_on_weekend(tmp_path, monkeypatch):
    folder = tmp_path / "history"
    folder.mkdir()
    pd.DataFrame({
        "date": ["2026-11-20"], "open": [100], "high": [103],
        "low": [99], "close": [102], "volume": [5000],
    }).to_csv(folder / "005930.csv", index=False)
    monkeypatch.setattr(updater, "KR_HISTORY", folder)
    monkeypatch.setattr(updater, "_latest_complete_kr_date", lambda: pd.Timestamp("2026-11-20"))
    monkeypatch.setattr(updater.stock, "get_market_ohlcv_by_date", lambda *args: pd.DataFrame())
    monkeypatch.setattr(updater.time, "sleep", lambda _: None)
    updater.update_korea()


def test_krx_refresh_requires_broad_recent_coverage(tmp_path, monkeypatch):
    folder = tmp_path / "history"
    folder.mkdir()
    for symbol, date in [("005930", "2026-11-19"), ("000001", "2026-11-13"), ("000002", "2026-11-13")]:
        pd.DataFrame({
            "date": [date], "open": [100], "high": [103],
            "low": [99], "close": [102], "volume": [5000],
        }).to_csv(folder / f"{symbol}.csv", index=False)
    monkeypatch.setattr(updater, "KR_HISTORY", folder)
    monkeypatch.setattr(updater, "_latest_complete_kr_date", lambda: pd.Timestamp("2026-11-19"))
    monkeypatch.setattr(updater.stock, "get_market_ohlcv_by_date", lambda *args: pd.DataFrame())
    monkeypatch.setattr(updater.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="only 33.3%"):
        updater.update_korea()
