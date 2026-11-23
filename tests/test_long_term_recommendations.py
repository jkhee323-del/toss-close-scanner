"""Regression tests for long-term recommendation data dependencies.

Run: python -m pytest tests/test_long_term_recommendations.py -q
"""
from pathlib import Path

import pandas as pd

from src import long_term_scanner as scanner


def test_korean_recommendations_do_not_require_us_universe(tmp_path, monkeypatch):
    """KR scans must not read a US-only CSV."""
    history = tmp_path / "history"
    history.mkdir()
    (history / "005930.csv").write_text("date,close,volume\n", encoding="utf-8")
    monkeypatch.setattr(scanner, "HISTORY_DIR", history)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        scanner,
        "analyze_stock",
        lambda path, horizon: {
            "종목코드": path.stem,
            "기준일": "2026-10-20",
            "점수": 75.0,
            "기간수익률(%)": 5.0,
        },
    )
    result = scanner.get_recommendations("1", market="KR")
    assert result["종목코드"].tolist() == ["005930"]


def test_invalid_horizon_returns_empty_without_reading_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = scanner.get_recommendations("invalid", market="KR")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_rejects_missing_ohlcv_columns(tmp_path):
    path = tmp_path / "000001.csv"
    dates = pd.date_range("2025-01-01", periods=250, freq="B")
    pd.DataFrame({"date": dates, "close": 100.0, "volume": 1000}).to_csv(path, index=False)
    assert scanner.analyze_stock(path, 20) is None


def test_rejects_zero_historical_close_without_infinite_momentum(tmp_path):
    path = tmp_path / "000002.csv"
    dates = pd.date_range("2025-01-01", periods=250, freq="B")
    close = [100.0] * 250
    close[-21] = 0.0
    pd.DataFrame({
        "date": dates,
        "open": [100.0] * 250,
        "high": [101.0] * 250,
        "low": [99.0] * 250,
        "close": close,
        "volume": [1000] * 250,
    }).to_csv(path, index=False)
    assert scanner.analyze_stock(path, 20) is None


def test_rejects_negative_volume(tmp_path):
    path = tmp_path / "000003.csv"
    dates = pd.date_range("2025-01-01", periods=250, freq="B")
    volume = [1000] * 250
    volume[-1] = -10
    pd.DataFrame({
        "date": dates,
        "open": [100.0] * 250,
        "high": [101.0] * 250,
        "low": [99.0] * 250,
        "close": [100.0] * 250,
        "volume": volume,
    }).to_csv(path, index=False)
    assert scanner.analyze_stock(path, 20) is None
