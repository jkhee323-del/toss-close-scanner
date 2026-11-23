"""Production OBV integration without changing existing short-term features."""
from pathlib import Path

import pandas as pd

from src.indicators import add_indicators
from src.long_term_scanner import analyze_stock


def _prices(periods=300):
    dates = pd.bdate_range("2025-01-01", periods=periods)
    close = [100.0 + 0.1 * i for i in range(periods)]
    return pd.DataFrame({
        "date": dates,
        "open": close,
        "high": [value + 2 for value in close],
        "low": [value - 2 for value in close],
        "close": close,
        "volume": [1000] * periods,
    })


def test_obv_is_available_with_stable_existing_indicator_features():
    result = add_indicators(_prices())
    for col in ["obv", "obv_change_5d", "obv_change_20d", "ret_20d", "rsi14"]:
        assert col in result
    assert result["obv_change_5d"].iloc[-1] == 1
    assert result["obv_change_20d"].iloc[-1] == 1


def test_long_term_stock_is_scored_without_experimental_probability_model(tmp_path):
    file = tmp_path / "005930.csv"
    _prices().to_csv(file, index=False)
    for horizon in (20, 60, 120, 240):
        ranked = analyze_stock(file, horizon)
        assert ranked is not None
        assert 0 <= ranked["점수"] <= 100
        assert ranked["종목코드"] == "005930"
