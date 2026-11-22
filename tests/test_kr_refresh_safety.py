"""KRX data refresh must not treat intraday volume as a completed session."""
import pandas as pd
import pytest
from src import auto_update as updater


@pytest.fixture(autouse=True)
def avoid_external_naver_request(monkeypatch):
    # Tests are never allowed to fetch a live market provider by accident.
    monkeypatch.setattr(updater, "fetch_naver_ohlcv", lambda *args: pd.DataFrame())


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


def test_no_krx_login_naver_can_append_verified_next_session(tmp_path, monkeypatch):
    folder = tmp_path / "history"
    folder.mkdir()
    days = pd.bdate_range("2026-11-09", periods=9)
    original = pd.DataFrame({
        "date": days[:-1], "open": [100.0] * 8,
        "high": [103.0] * 8, "low": [99.0] * 8,
        "close": [102.0] * 8, "volume": [1000] * 8,
    })
    original.to_csv(folder / "005930.csv", index=False)
    expected = original.copy()
    expected.loc[len(expected)] = [days[-1], 101.0, 105.0, 100.0, 104.0, 1500]
    monkeypatch.setattr(updater, "KR_HISTORY", folder)
    monkeypatch.setattr(updater, "_latest_complete_kr_date", lambda: pd.Timestamp(days[-1]))
    monkeypatch.setattr(updater.stock, "get_market_ohlcv_by_date", lambda *args: pd.DataFrame())
    monkeypatch.setattr(updater, "fetch_naver_ohlcv", lambda *args: expected.copy())
    monkeypatch.setattr(updater.time, "sleep", lambda *_: None)
    updater.update_korea()
    actual = pd.read_csv(folder / "005930.csv")
    assert len(actual) == 9
    assert actual["close"].iloc[-1] == 104
    assert actual["close"].iloc[0] == 102


def test_naver_bad_adjustment_basis_never_overwrites_existing_prices(tmp_path, monkeypatch):
    folder = tmp_path / "history"
    folder.mkdir()
    days = pd.bdate_range("2026-11-09", periods=9)
    original = pd.DataFrame({
        "date": days[:-1], "open": [100.0] * 8,
        "high": [103.0] * 8, "low": [99.0] * 8,
        "close": [102.0] * 8, "volume": [1000] * 8,
    })
    file = folder / "005930.csv"
    original.to_csv(file, index=False)
    before = file.read_bytes()
    wrong = original.copy()
    wrong[["open", "high", "low", "close"]] *= 2
    wrong.loc[len(wrong)] = [days[-1], 201.0, 205.0, 200.0, 204.0, 2000]
    monkeypatch.setattr(updater, "KR_HISTORY", folder)
    monkeypatch.setattr(updater, "_latest_complete_kr_date", lambda: pd.Timestamp(days[-1]))
    monkeypatch.setattr(updater.stock, "get_market_ohlcv_by_date", lambda *args: pd.DataFrame())
    monkeypatch.setattr(updater, "fetch_naver_ohlcv", lambda *args: wrong.copy())
    monkeypatch.setattr(updater.time, "sleep", lambda *_: None)
    # Stored last session is one day behind. The post-refresh fallback must
    # reject the mismatched vendor rather than rewriting the cached rows.
    with pytest.raises(RuntimeError, match="KRX provider may be unavailable"):
        updater.update_korea()
    assert file.read_bytes() == before
