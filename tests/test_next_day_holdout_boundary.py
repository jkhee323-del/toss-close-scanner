"""Temporal boundary regression for the next-day candidate comparison."""
import pandas as pd
import pytest

from src.next_day_candidate_holdout import split_candidate_holdout


def test_train_excludes_next_day_outcome_crossing_holdout_boundary():
    data = pd.DataFrame({
        "date": pd.to_datetime(["2024-12-26", "2024-12-30", "2025-01-02", "2025-01-05"]),
        "next_date": pd.to_datetime(["2024-12-27", "2025-01-02", "2025-01-03", "2025-01-06"]),
        "target": [1, 1, 0, 1],
    })
    train, holdout = split_candidate_holdout(data)
    assert train["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-12-26"]
    assert holdout["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-02", "2025-01-05"]
    assert train["next_date"].max() <= pd.Timestamp("2024-12-30")


def test_missing_next_date_rejected():
    data = pd.DataFrame({"date": ["2024-12-30"], "target": [1]})
    with pytest.raises(ValueError):
        split_candidate_holdout(data)
