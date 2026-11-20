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


def test_executable_return_uses_next_open_not_signal_close():
    from src.next_day_validation_feature_models import build_dataset
    from src import next_day_validation_feature_models as exp
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "005930.csv"
        dates = pd.bdate_range("2024-01-01", periods=270)
        close = [100.0] * 270
        opening = [100.0] * 270
        opening[261] = 120.0
        pd.DataFrame({
            "date": dates,
            "open": opening,
            "high": [max(105, val + 1) for val in opening],
            "low": [95.0] * 270,
            "close": close,
            "volume": [10000] * 270,
        }).to_csv(path, index=False)
        previous = exp.LABEL_DATA_END
        try:
            exp.LABEL_DATA_END = pd.Timestamp("2026-01-01")
            data = build_dataset(Path(temp))
        finally:
            exp.LABEL_DATA_END = previous
        row = data.loc[data["date"] == dates[260]].iloc[0]
        assert row["next_return"] == pytest.approx(0.0)
        assert row["execution_return"] == pytest.approx(100 / 120 - 1)
