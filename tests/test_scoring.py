import pandas as pd
from src.scoring import score_row, display_probability


def test_probability_range():
    assert 0 < display_probability(0) < 100
    assert 0 < display_probability(100) < 100


def test_score_range():
    row = pd.Series({
        "ret_5d": 0.05,
        "ma5": 105,
        "ma20": 100,
        "ma60": 95,
        "volume_ratio": 2.0,
        "close_position": 0.9,
        "near_20d_high": 0.98,
        "rsi14": 60,
    })
    score, parts = score_row(row)
    assert 0 <= score <= 100
    assert set(parts) == {
        "momentum", "trend", "volume",
        "close_strength", "breakout", "rsi"
    }
