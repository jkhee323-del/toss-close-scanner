"""Regression tests for medium/long-horizon scanner edge cases.

These tests intentionally document current risks before changing production code.
Run: python -m pytest tests/test_scanner_edge_cases.py -q
"""
import numpy as np
import pandas as pd

from src.indicators import add_indicators
from src.scanner import scan_symbol


def make_history(n=90, increasing=True):
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    close = np.arange(100.0, 100.0 + n) if increasing else 100 + np.sin(np.arange(n) / 3) * 5
    return pd.DataFrame({
        "date": dates,
        "open": close - 0.3,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(n, 100000.0),
    })


def test_rsi_is_100_when_last_14_changes_are_all_gains():
    result = add_indicators(make_history())
    assert result["rsi14"].iloc[-1] == 100.0


def test_scanner_uses_latest_date_even_when_input_is_unsorted():
    history = make_history(increasing=False)
    reversed_history = history.iloc[::-1].reset_index(drop=True)
    expected = scan_symbol("000001", "test", history)
    actual = scan_symbol("000001", "test", reversed_history)
    assert expected is not None
    assert actual is not None
    assert actual["date"] == expected["date"]
    assert actual["close"] == expected["close"]
