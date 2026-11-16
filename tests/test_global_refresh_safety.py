"""Global refresh validation must reject corrupt feeds without touching saved caches."""
import pandas as pd
import pytest

from src import auto_update as updater


def valid_frame():
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=100, freq="B")
    return pd.DataFrame({
        "date": dates,
        "open": [100.0] * 100,
        "high": [102.0] * 100,
        "low": [99.0] * 100,
        "close": [101.0] * 100,
        "volume": [1000] * 100,
    })


def test_global_valid_history():
    out = updater._validate_global_history(valid_frame(), "TEST")
    assert len(out) == 100
    assert out["date"].is_monotonic_increasing


@pytest.mark.parametrize("damage", ["duplicate", "negative", "bad_ohlc", "stale", "short", "future"])
def test_global_invalid_history(damage):
    df = valid_frame()
    if damage == "duplicate":
        df.loc[1, "date"] = df.loc[0, "date"]
    elif damage == "negative":
        df.loc[0, "volume"] = -1
    elif damage == "bad_ohlc":
        df.loc[0, "high"] = 90.0
    elif damage == "stale":
        df["date"] = df["date"] - pd.Timedelta(days=30)
    elif damage == "short":
        df = df.tail(10)
    elif damage == "future":
        df.loc[99, "date"] = pd.Timestamp.now() + pd.Timedelta(days=10)
    with pytest.raises(ValueError):
        updater._validate_global_history(df, "TEST")


def test_global_partial_outage_preserves_candidate_cache(tmp_path, monkeypatch):
    history = tmp_path / "history"
    cache = tmp_path / "candidates.csv"
    cache.write_text("original-cache", encoding="utf-8")
    monkeypatch.setattr(updater, "GLOBAL_HISTORY", history)
    monkeypatch.setattr(updater, "GLOBAL_CACHE", cache)
    monkeypatch.setattr(updater, "GLOBAL_SCAN_UNIVERSE", [
        ("AAA", "AAA", "US"), ("BBB", "BBB", "US"),
        ("CCC", "CCC", "US"), ("DDD", "DDD", "US"),
    ])
    monkeypatch.setattr(updater, "_fetch_global", lambda ticker: valid_frame() if ticker == "AAA" else (_ for _ in ()).throw(ValueError("provider unavailable")))
    with pytest.raises(RuntimeError, match="Global feed incomplete"):
        updater.update_global()
    assert cache.read_text(encoding="utf-8") == "original-cache"
