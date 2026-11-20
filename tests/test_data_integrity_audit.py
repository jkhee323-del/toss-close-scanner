"""Data integrity audit regression tests."""
import pandas as pd

from src.data_integrity_audit import inspect_market, check_ohlcv_quality, recent_valid_segment


def create_history(path, *, dates=None, bad_price=False):
    dates = dates if dates is not None else pd.bdate_range("2026-11-02", periods=10)
    frame = pd.DataFrame({
        "date": dates,
        "open": [100.0] * len(dates),
        "high": [101.0] * len(dates),
        "low": [99.0] * len(dates),
        "close": [100.0] * len(dates),
        "volume": [1000] * len(dates),
    })
    if bad_price:
        frame.loc[0, "high"] = 90.0
    frame.to_csv(path, index=False)


def test_valid_market_cache(tmp_path):
    create_history(tmp_path / "005930.csv")
    create_history(tmp_path / "000660.csv")
    out = inspect_market(tmp_path)
    assert out["status"] == "pass"
    assert out["recent_share"] == 1.0
    assert out["total"] == 2


def test_reject_corrupt_ohlc(tmp_path):
    create_history(tmp_path / "bad.csv", bad_price=True)
    out = inspect_market(tmp_path)
    assert out["status"] == "fail"
    assert out["total_errors"] >= 1


def test_reject_stale_subset(tmp_path):
    create_history(tmp_path / "good.csv")
    create_history(tmp_path / "stale.csv", dates=pd.bdate_range("2026-09-01", periods=10))
    out = inspect_market(tmp_path, min_recent_share=0.80)
    assert out["status"] == "fail"
    assert out["lagged_count"] == 1


def test_adjusted_price_one_unit_roundoff_not_hard_error(tmp_path):
    path = tmp_path / "adjusted.csv"
    create_history(path)
    raw = pd.read_csv(path)
    raw.loc[0, "high"] = 99.0  # maximum open or close is 100: one unit rounding
    reason, roundoffs = check_ohlcv_quality(raw)
    assert reason is None
    assert roundoffs == 1


def test_zero_candle_not_accepted_as_roundoff(tmp_path):
    path = tmp_path / "zero.csv"
    create_history(path)
    raw = pd.read_csv(path)
    raw.loc[0, ["open", "high", "low"]] = 0
    reason, _ = check_ohlcv_quality(raw)
    assert reason == "invalid OHLCV numbers"


def test_recovers_only_unbroken_suffix_after_old_invalid_prices():
    dates = pd.bdate_range("2024-01-01", periods=330)
    frame = pd.DataFrame({
        "date": dates, "open": [100.0] * 330, "high": [101.0] * 330,
        "low": [99.0] * 330, "close": [100.0] * 330,
        "volume": [10000] * 330,
    })
    frame.loc[20, ["open", "high", "low"]] = 0
    recent = recent_valid_segment(frame, minimum_rows=260)
    assert len(recent) == 309
    assert recent["date"].iloc[0] == dates[21]
    assert recent["date"].iloc[-1] == dates[-1]


def test_recent_corrupted_candle_never_silently_bridged():
    dates = pd.bdate_range("2024-01-01", periods=330)
    frame = pd.DataFrame({
        "date": dates, "open": [100.0] * 330, "high": [101.0] * 330,
        "low": [99.0] * 330, "close": [100.0] * 330,
        "volume": [10000] * 330,
    })
    frame.loc[310, "close"] = 0
    assert recent_valid_segment(frame, minimum_rows=260).empty
    frame.loc[329, "open"] = 0
    assert recent_valid_segment(frame, minimum_rows=1).empty


def test_audit_separates_recently_usable_from_blocked_history(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=330)
    template = pd.DataFrame({
        "date": dates, "open": [100.0] * 330,
        "high": [101.0] * 330, "low": [99.0] * 330,
        "close": [100.0] * 330, "volume": [10000] * 330,
    })
    old = template.copy()
    old.loc[5, ["open", "high", "low"]] = 0
    old.to_csv(tmp_path / "recoverable.csv", index=False)
    new = template.copy()
    new.loc[329, "close"] = 0
    new.to_csv(tmp_path / "blocked.csv", index=False)
    template.to_csv(tmp_path / "good.csv", index=False)
    result = inspect_market(tmp_path)
    assert result["status"] == "fail"
    assert result["recoverable_recent_count"] == 1
    assert result["recoverable_recent_files"][0]["file"] == "recoverable.csv"
    assert result["blocked_recent_files"] == ["blocked.csv"]
    assert result["blocked_recent_count"] == 1
