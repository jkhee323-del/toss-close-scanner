"""Long-horizon model lab must not train on outcomes maturing after cut-off."""
import pandas as pd
import pytest

from src.long_term_probability_lab import prepare_symbol, split_no_leakage


def test_holdout_targets_and_training_cutoff(tmp_path):
    df = pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=800),
        "open": [100 + i * .1 for i in range(800)],
        "high": [102 + i * .1 for i in range(800)],
        "low": [99 + i * .1 for i in range(800)],
        "close": [101 + i * .1 for i in range(800)],
        "volume": [100000 + i for i in range(800)],
    })
    path = tmp_path / "005930.csv"
    df.to_csv(path, index=False)
    dataset = prepare_symbol(path, 120)
    assert not dataset.empty
    train, holdout = split_no_leakage(dataset)
    assert not train.empty
    assert not holdout.empty
    assert train["future_date"].max() <= pd.Timestamp("2024-12-30")
    assert holdout["date"].min() >= pd.Timestamp("2025-01-01")
    assert dataset["future_date"].max() <= df["date"].max()


def test_unmatured_labels_are_not_created(tmp_path):
    df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=280),
        "open": [100.0] * 280,
        "high": [102.0] * 280,
        "low": [99.0] * 280,
        "close": [101.0] * 280,
        "volume": [10000] * 280,
    })
    path = tmp_path / "000001.csv"
    df.to_csv(path, index=False)
    got = prepare_symbol(path, 240)
    assert not got.empty
    assert got["date"].max() <= df["date"].iloc[-241]
    assert (got["future_date"] <= df["date"].iloc[-1]).all()


def test_model_target_uses_next_open_not_signal_close(tmp_path):
    df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=340),
        "open": [100.0] * 340,
        "high": [103.0] * 340,
        "low": [97.0] * 340,
        "close": [101.0] * 340,
        "volume": [10000] * 340,
    })
    # The signal close is fixed, but the next-session open has a gap up.
    df.loc[261, "open"] = 150.0
    df.loc[261, "high"] = 155.0
    path = tmp_path / "000001.csv"
    df.to_csv(path, index=False)
    result = prepare_symbol(path, 20)
    row = result.loc[result["date"] == df.loc[260, "date"]].iloc[0]
    assert row["future_net_return"] == pytest.approx(101.0 / 150.0 - 1 - 0.002)
    assert row["target"] == 0
