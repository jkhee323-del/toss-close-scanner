"""Data integrity audit regression tests."""
import pandas as pd

from src.data_integrity_audit import inspect_market


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
