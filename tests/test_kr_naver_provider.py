"""NAVER alternative feed cannot silently mix symbols or adjustment bases."""
import pandas as pd
import pytest

from src.kr_naver_provider import fetch_naver_ohlcv, validate_naver_overlap


def _cached(days=10):
    d = pd.bdate_range("2026-11-02", periods=days)
    return pd.DataFrame({
        "date": d,
        "open": [100.0] * days, "high": [102.0] * days,
        "low": [99.0] * days, "close": [101.0] * days,
        "volume": [1000] * days,
    })


def test_valid_independent_naver_overlap():
    base = _cached()
    assert validate_naver_overlap(base, base.copy()) == len(base)


def test_stale_adjusted_share_price_rejected():
    base = _cached()
    alt = base.copy()
    alt["close"] *= 2
    with pytest.raises(ValueError, match="adjustment bases disagree"):
        validate_naver_overlap(base, alt)


def test_unverified_one_day_overlap_rejected():
    base = _cached()
    with pytest.raises(ValueError, match="Insufficient independent overlap"):
        validate_naver_overlap(base, base.tail(1))


def test_naver_invalid_price_rows_rejected_before_replacement():
    base = _cached()
    alt = base.copy()
    alt.loc[0, "close"] = 0
    with pytest.raises(ValueError, match="Invalid alternative provider feed"):
        validate_naver_overlap(base, alt)


def test_naver_ticker_is_exact_six_digit_code():
    with pytest.raises(ValueError, match="six digits"):
        fetch_naver_ohlcv("005930.KS", "2026-11-01", "2026-11-20")


def test_normalize_naver_data_without_guessing_code(monkeypatch):
    import sys
    from types import SimpleNamespace

    raw = pd.DataFrame({
        "Open": [100, 101], "High": [102, 104], "Low": [99, 100],
        "Close": [101, 103], "Volume": [1000, 2000], "Change": [0, .02],
    }, index=pd.DatetimeIndex(["2026-11-18", "2026-11-19"], name="Date"))
    calls = []
    def reader(symbol, start, end):
        calls.append((symbol, start, end))
        return raw
    monkeypatch.setitem(sys.modules, "FinanceDataReader", SimpleNamespace(DataReader=reader))
    out = fetch_naver_ohlcv("005930", "2026-11-01", "2026-11-21")
    assert calls[0][0] == "NAVER:005930"
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert out["close"].tolist() == [101, 103]
