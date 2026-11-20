"""Validation train/label embargo at the 2023/2024 model boundary."""
import pandas as pd

from src.next_day_validation_feature_models import get_time_split


def test_validation_training_excludes_maturing_2024_label():
    rows = pd.DataFrame({
        "date": pd.to_datetime(["2023-12-27", "2023-12-28", "2024-01-02", "2024-01-03"]),
        "next_date": pd.to_datetime(["2023-12-28", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "target": [1, 0, 1, 0],
    })
    train, validation = get_time_split(rows)
    assert train["date"].dt.strftime("%Y-%m-%d").tolist() == ["2023-12-27"]
    assert validation["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]
    assert train["next_date"].max() <= pd.Timestamp("2023-12-28")
