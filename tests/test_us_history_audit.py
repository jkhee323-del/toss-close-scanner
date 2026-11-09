"""Read-only US price depth and quality audit tests."""
import numpy as np
import pandas as pd

from src.us_history_audit import audit_history


def make_frame(n=800):
    close = np.arange(n, dtype=float) * 0.01 + 50.0
    return pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=n),
        "open": close, "high": close + 1,
        "low": close - 1, "close": close, "volume": 10_000,
    })


def test_audit_reports_model_history_sufficiency(tmp_path):
    make_frame().to_csv(tmp_path / "AAPL.csv", index=False)
    make_frame(300).to_csv(tmp_path / "MSFT.csv", index=False)
    make_frame().to_csv(tmp_path / "7203.T.csv", index=False)  # Japan, exclude
    report = audit_history(tmp_path, as_of=pd.Timestamp("2026-12-01").date())
    assert report["us_tickers"] == 2
    assert report["invalid_histories"] == 0
    assert report["adequate_full_history"]["12m"] == 1
    assert report["adequate_full_history"]["1m"] == 2


def test_audit_flags_duplicate_cross_ticker_history(tmp_path):
    df = make_frame(300)
    df.to_csv(tmp_path / "AAPL.csv", index=False)
    df.to_csv(tmp_path / "MSFT.csv", index=False)
    report = audit_history(tmp_path, as_of=pd.Timestamp("2026-12-01").date())
    assert ["AAPL", "MSFT"] in report["duplicate_groups"]
    assert "IDENTICAL_SERIES_ACROSS_TICKERS" in report["by_ticker"]["MSFT"]["issues"]


def test_audit_keeps_invalid_data_untouched(tmp_path):
    path = tmp_path / "AAPL.csv"
    path.write_text("date,close\n2026-01-01,12\n")
    original = path.read_bytes()
    report = audit_history(tmp_path)
    assert report["invalid_histories"] == 1
    assert path.read_bytes() == original
