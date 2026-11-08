"""Offline integration checks for the deployed global refresh path."""
import numpy as np
import pandas as pd
import pytest


def make_history(n=300):
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = np.linspace(100, 120, n)
    return pd.DataFrame({
        "date": dates, "open": close, "high": close + 1,
        "low": close - 1, "close": close, "volume": 100000,
    })


def test_fetch_global_routes_non_us_ticker_to_its_market_guard(monkeypatch):
    import yfinance as yf
    from src import auto_update

    raw = make_history(90).rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }).set_index("date")
    raw.index.name = "Date"
    seen = []
    monkeypatch.setattr(yf, "download", lambda *a, **k: raw.copy())
    def guard(frame, country):
        seen.append(country)
        return frame
    monkeypatch.setattr(auto_update, "keep_confirmed_market_daily", guard)
    result = auto_update._fetch_global("7203.T", country="일본")
    assert seen == ["일본"]
    assert len(result) == 90


def test_refresh_preserves_history_and_writes_candidate_csv(tmp_path, monkeypatch):
    from src import auto_update

    old = make_history(300)
    prices = tmp_path / "global_history"
    prices.mkdir()
    old.to_csv(prices / "7203.T.csv", index=False)
    monkeypatch.setattr(auto_update, "GLOBAL_HISTORY", prices)
    monkeypatch.setattr(auto_update, "GLOBAL_CACHE", tmp_path / "global_candidates.csv")
    monkeypatch.setattr(auto_update, "GLOBAL_SCAN_UNIVERSE", [("7203.T", "Toyota", "일본")])
    monkeypatch.setattr(
        auto_update, "_fetch_global", lambda ticker, *, country: old.tail(90).copy()
    )
    auto_update.update_global()
    assert len(pd.read_csv(prices / "7203.T.csv")) == 300
    candidates = pd.read_csv(tmp_path / "global_candidates.csv")
    assert not candidates.empty
    assert (candidates["symbol"] == "7203.T").all()


def test_many_failed_downloads_keep_previous_candidate_cache(tmp_path, monkeypatch):
    from src import auto_update

    prices = tmp_path / "global_history"
    prices.mkdir()
    cached = tmp_path / "global_candidates.csv"
    cached.write_text("symbol,score\\nPREVIOUS,99\\n", encoding="utf-8")
    original = cached.read_bytes()
    monkeypatch.setattr(auto_update, "GLOBAL_HISTORY", prices)
    monkeypatch.setattr(auto_update, "GLOBAL_CACHE", cached)
    monkeypatch.setattr(auto_update, "GLOBAL_SCAN_UNIVERSE", [
        ("AAPL", "Apple", "미국"), ("MSFT", "Microsoft", "미국"),
        ("NVDA", "Nvidia", "미국"), ("GOOGL", "Alphabet", "미국"),
    ])

    def partially_failing(ticker, *, country):
        if ticker != "AAPL":
            raise OSError("simulated provider outage")
        return make_history(300)
    monkeypatch.setattr(auto_update, "_fetch_global", partially_failing)
    with pytest.raises(RuntimeError, match="Too many global data failures"):
        auto_update.update_global()
    assert cached.read_bytes() == original
