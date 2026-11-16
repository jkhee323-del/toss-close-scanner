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
        "date": pd.bdate_range("2024-01-01", periods=300),
        "open": [100.0] * 300,
        "high": [102.0] * 300,
        "low": [99.0] * 300,
        "close": [101.0] * 300,
        "volume": [10000] * 300,
    })
    path = tmp_path / "000001.csv"
    df.to_csv(path, index=False)
    got = prepare_symbol(path, 240)
    assert got.empty  # 240-day target requires 240 future bars after indicator warmup
