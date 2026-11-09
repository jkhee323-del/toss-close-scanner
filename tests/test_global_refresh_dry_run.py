"""Smoke-mode guarantees: no cache changes and no partial rankings."""
import numpy as np
import pandas as pd
import pytest


def rows(n=100):
    close = np.linspace(100, 120, n)
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-02", periods=n),
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 100000,
    })


def test_global_dry_run_does_not_touch_files(tmp_path, monkeypatch):
    from src import auto_update
    cache = tmp_path / "global_candidates.csv"
    cache.write_text("x,y\nold,1\n")
    before = cache.read_bytes()
    monkeypatch.setattr(auto_update, "GLOBAL_CACHE", cache)
    monkeypatch.setattr(auto_update, "GLOBAL_HISTORY", tmp_path / "history")
    monkeypatch.setattr(auto_update, "GLOBAL_SCAN_UNIVERSE",
                        [("AAPL", "Apple", "미국"), ("MSFT", "Microsoft", "미국")])
    seen = []
    def downloaded(ticker, *, country):
        seen.append(ticker)
        return rows()
    monkeypatch.setattr(auto_update, "_fetch_global", downloaded)
    auto_update.update_global(dry_run=True, max_symbols=1)
    assert seen == ["AAPL"]
    assert not (tmp_path / "history").exists()
    assert cache.read_bytes() == before


def test_limited_live_updates_only_history_not_shared_rankings(tmp_path, monkeypatch):
    from src import auto_update
    cache = tmp_path / "global_candidates.csv"
    cache.write_text("x,y\nold,1\n")
    before = cache.read_bytes()
    monkeypatch.setattr(auto_update, "GLOBAL_CACHE", cache)
    monkeypatch.setattr(auto_update, "GLOBAL_HISTORY", tmp_path / "history")
    monkeypatch.setattr(auto_update, "GLOBAL_SCAN_UNIVERSE",
                        [("AAPL", "Apple", "미국"), ("MSFT", "Microsoft", "미국")])
    monkeypatch.setattr(auto_update, "_fetch_global", lambda ticker, *, country: rows())
    auto_update.update_global(max_symbols=1)
    assert (tmp_path / "history" / "AAPL.csv").exists()
    assert not (tmp_path / "history" / "MSFT.csv").exists()
    assert cache.read_bytes() == before


def test_invalid_limited_refresh_arguments(tmp_path, monkeypatch):
    from src import auto_update
    monkeypatch.setattr(auto_update, "GLOBAL_HISTORY", tmp_path / "history")
    with pytest.raises(ValueError, match="max_symbols"):
        auto_update.update_global(max_symbols=-1)
    with pytest.raises(ValueError, match="period"):
        auto_update.update_global(history_period="100y")
    assert not (tmp_path / "history").exists()
