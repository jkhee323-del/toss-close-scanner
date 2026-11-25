"""CVD must come from signed execution data, never OHLCV estimates."""
import pandas as pd
import pytest

from src.cvd import compute_cvd


def test_cvd_accumulates_real_aggressor_side_and_sorts():
    df = pd.DataFrame({
        "timestamp": [
            "2026-11-17T09:00:02+09:00",
            "2026-11-17T09:00:00+09:00",
            "2026-11-17T09:00:01+09:00",
        ],
        "volume": [7, 10, 4],
        "aggressor_side": ["SELL", "BUY", "buy"],
    })
    result = compute_cvd(df)
    assert result["signed_volume"].tolist() == [10, 4, -7]
    assert result["cvd"].tolist() == [10, 14, 7]


def test_cvd_can_reset_at_exchange_session_boundary():
    df = pd.DataFrame({
        "timestamp": ["2026-11-16T15:25:00+09:00", "2026-11-17T09:00:00+09:00"],
        "volume": [3, 5],
        "aggressor_side": ["buy", "sell"],
    })
    result = compute_cvd(df, reset_each_session=True)
    assert result["cvd"].tolist() == [3, -5]
    assert result["session_date"].tolist() == ["2026-11-16", "2026-11-17"]


@pytest.mark.parametrize("data", [
    {"timestamp": ["2026-11-17T09:00:00+09:00"], "volume": [3]},
    {"timestamp": ["2026-11-17T09:00:00+09:00"], "volume": [3], "aggressor_side": ["unknown"]},
    {"timestamp": ["2026-11-17T09:00:00"], "volume": [3], "aggressor_side": ["buy"]},
    {"timestamp": ["2026-11-17T09:00:00+09:00"], "volume": [0], "aggressor_side": ["buy"]},
])
def test_cvd_rejects_ambiguous_or_missing_data(data):
    with pytest.raises(ValueError):
        compute_cvd(pd.DataFrame(data))


@pytest.mark.parametrize("volume", [float("inf"), float("-inf"), float("nan")])
def test_cvd_rejects_nonfinite_volume(volume):
    with pytest.raises(ValueError):
        compute_cvd(pd.DataFrame({"timestamp": ["2026-11-17T09:00:00+09:00"], "volume": [volume], "aggressor_side": ["buy"]}))
