"""Korean data repair inventory never modifies source candles."""
import pandas as pd
from src.data_quality_triage import diagnose_file, run


def test_legacy_zero_price_is_identified_and_recently_usable(tmp_path):
    days = pd.bdate_range("2024-01-01", periods=340)
    df = pd.DataFrame({
        "date": days, "open": [100.0] * 340,
        "high": [101.0] * 340, "low": [99.0] * 340,
        "close": [100.0] * 340, "volume": [5000] * 340,
    })
    df.loc[40, ["open", "high", "low", "close"]] = 0.0
    path = tmp_path / "001470.csv"
    df.to_csv(path, index=False)
    original = path.read_bytes()
    report = diagnose_file(path)
    assert report["bad_row_count"] == 1
    assert report["first_bad_date"] == days[40].date().isoformat()
    assert "nonpositive OHLC" in report["issue_types"]
    assert report["safe_short_90_sessions"]
    assert report["safe_long_241_sessions"]
    assert path.read_bytes() == original


def test_recent_source_failure_must_remain_quarantined(tmp_path):
    days = pd.bdate_range("2024-01-01", periods=340)
    df = pd.DataFrame({
        "date": days, "open": [100.0] * 340,
        "high": [101.0] * 340, "low": [99.0] * 340,
        "close": [100.0] * 340, "volume": [5000] * 340,
    })
    df.loc[339, "open"] = 0
    df.to_csv(tmp_path / "000650.csv", index=False)
    summary = run(tmp_path)
    assert summary["file"].tolist() == ["000650.csv"]
    assert not bool(summary.loc[0, "safe_short_90_sessions"])
    assert not bool(summary.loc[0, "safe_long_241_sessions"])
