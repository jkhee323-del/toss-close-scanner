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


def test_next_day_loader_quarantines_bad_file_without_bridging_days(tmp_path, monkeypatch):
    from src import next_day_validation_feature_models as mod
    dates = pd.bdate_range("2023-01-02", periods=300)
    good = pd.DataFrame({
        "date": dates, "open": [100.0] * 300,
        "high": [101.0] * 300, "low": [99.0] * 300,
        "close": [100.0] * 300, "volume": [5000] * 300,
    })
    good.to_csv(tmp_path / "005930.csv", index=False)
    bad = good.copy()
    bad.loc[150, "close"] = 0.0
    bad.to_csv(tmp_path / "000001.csv", index=False)
    monkeypatch.setattr(mod, "LABEL_DATA_END", pd.Timestamp("2024-12-31"))
    result = mod.load_raw_history(tmp_path)
    assert set(result["symbol"]) == {"005930"}
    assert len(result) == len(good)
