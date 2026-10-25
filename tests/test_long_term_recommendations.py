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
