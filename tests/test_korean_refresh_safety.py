"""Offline KRX bar and atomic cache safety checks."""
from datetime import datetime

import numpy as np
import pandas as pd

from src.us_session_guard import keep_confirmed_market_daily


def _prices(n=120):
    close = np.arange(n, dtype=float) * 0.1 + 100
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-02", periods=n),
        "open": close, "high": close + 1,
        "low": close - 1, "close": close,
        "volume": 100000,
    })


def _provider_frame(data):
    result = data.set_index("date").rename(columns={
        "open": "시가", "high": "고가", "low": "저가",
        "close": "종가", "volume": "거래량",
    })
    result.index.name = "날짜"
    return result


def test_korean_session_rejects_intraday_bar():
    frame = pd.DataFrame({"date": ["2026-11-06", "2026-11-09"]})
    before = keep_confirmed_market_daily(
        frame, "한국", now=datetime.fromisoformat("2026-11-09T07:29:00+00:00")
    )
    after = keep_confirmed_market_daily(
        frame, "한국", now=datetime.fromisoformat("2026-11-09T07:30:00+00:00")
    )
    assert len(before) == 1
    assert len(after) == 2


def test_korean_refresh_preserves_old_file_if_vendor_bar_bad(tmp_path, monkeypatch):
    from src import auto_update

    folder = tmp_path / "history"
    folder.mkdir()
    old = _prices()
    path = folder / "005930.csv"
    old.to_csv(path, index=False)
    prior = path.read_bytes()

    wrong = old.tail(5).copy()
    wrong.loc[wrong.index[-1], "high"] = 1.0

    monkeypatch.setattr(auto_update, "KR_HISTORY", folder)
    monkeypatch.setattr(
        auto_update.stock, "get_market_ohlcv_by_date",
        lambda *args, **kwargs: _provider_frame(wrong),
    )
    monkeypatch.setattr(auto_update.time, "sleep", lambda duration: None)
    auto_update.update_korea()
    assert path.read_bytes() == prior


def test_korean_refresh_keeps_full_history_on_overlap(tmp_path, monkeypatch):
    from src import auto_update

    folder = tmp_path / "history"
    folder.mkdir()
    old = _prices()
    path = folder / "005930.csv"
    old.to_csv(path, index=False)
    changed = old.tail(5).copy()
    changed.loc[changed.index[-1], "close"] += 0.2
    changed.loc[changed.index[-1], "high"] += 0.2

    monkeypatch.setattr(auto_update, "KR_HISTORY", folder)
    monkeypatch.setattr(
        auto_update.stock, "get_market_ohlcv_by_date",
        lambda *args, **kwargs: _provider_frame(changed),
    )
    monkeypatch.setattr(auto_update.time, "sleep", lambda duration: None)
    auto_update.update_korea()
    saved = pd.read_csv(path)
    assert len(saved) == len(old)
    assert abs(saved["close"].iloc[-1] - changed["close"].iloc[-1]) < 1e-8
