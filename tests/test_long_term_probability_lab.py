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


def test_prob_lab_does_not_bridge_bad_ohlcv(tmp_path):
    frame = pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=800),
        "open": [100.0] * 800,
        "high": [102.0] * 800,
        "low": [99.0] * 800,
        "close": [101.0] * 800,
        "volume": [10000] * 800,
    })
    frame.loc[300, ["open", "high", "low", "close"]] = 0.0
    path = tmp_path / "000001.csv"
    frame.to_csv(path, index=False)
    result = prepare_symbol(path, 20)
    assert not result.empty
    assert result["date"].min() > frame["date"].iloc[300]
    assert (result["future_date"] > result["date"]).all()


def test_prob_lab_rejects_insufficient_recent_history_after_corruption(tmp_path):
    frame = pd.DataFrame({
        "date": pd.bdate_range("2025-01-02", periods=300),
        "open": [100.0] * 300, "high": [102.0] * 300,
        "low": [99.0] * 300, "close": [101.0] * 300,
        "volume": [10000] * 300,
    })
    frame.loc[299, "close"] = 0.0
    path = tmp_path / "000001.csv"
    frame.to_csv(path, index=False)
    assert prepare_symbol(path, 20).empty


def test_walkforward_does_not_bridge_a_future_invalid_bar(tmp_path):
    from src.long_term_walkforward import prepare_history

    frame = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=300),
        "open": 100.0, "high": 102.0, "low": 99.0,
        "close": 101.0, "volume": 10000,
    })
    path = tmp_path / "000001.csv"
    frame.to_csv(path, index=False)
    clean = prepare_history(path).set_index("date")
    frame.loc[100, "close"] = 0
    frame.to_csv(path, index=False)
    broken = prepare_history(path).set_index("date")
    early = frame.loc[90, "date"]
    assert broken.loc[early, "ret_20d"] == clean.loc[early, "ret_20d"]
    assert pd.isna(broken.loc[early, "exit_20"])
    assert pd.notna(broken.loc[frame.loc[75, "date"], "exit_20"])
    assert frame.loc[110, "date"] not in broken.index


def test_walkforward_purges_labels_maturing_on_or_after_refit():
    import numpy as np

    from src.long_term_walkforward import evaluate
    from src.long_term_probability_lab import FEATURES

    count = 1010
    dataset = pd.DataFrame({feature: np.linspace(-1, 1, count) for feature in FEATURES})
    dataset["symbol"] = [f"{i:06}" for i in range(count)]
    dataset["date"] = [pd.Timestamp("2024-12-01")] * 1002 + [pd.Timestamp("2025-01-02")] * 8
    dataset["exit_20"] = [pd.Timestamp("2024-12-31")] * 1000 + [pd.Timestamp("2025-01-01")] * 2 + [pd.Timestamp("2025-01-30")] * 8
    dataset["net_20"] = np.where(np.arange(count) % 2 == 0, 0.01, -0.01)
    dataset.loc[count - 1, ["exit_20", "net_20"]] = [pd.NaT, np.nan]
    [result] = evaluate(dataset, 20)
    assert result["train_rows"] == 1000
    assert result["matured_rows"] == 7
    assert result["top10_selected"] == 8
    assert result["top10_matured"] == 7
    assert result["top10_completed_days"] == 0
    assert pd.isna(result["top10_mean_net"])
    [fallback] = evaluate(dataset, 20, validate_blend=True)
    assert fallback["ranking_abstained"]
    assert fallback["blend"] == 0
    assert fallback["top10_selected"] == 0
    assert pd.isna(fallback["top10_mean_net"])


def test_walkforward_counts_selected_stocks_without_any_mature_outcome():
    import numpy as np
    from src.long_term_walkforward import evaluate
    from src.long_term_probability_lab import FEATURES

    dataset = pd.DataFrame({feature: np.linspace(-1, 1, 1010) for feature in FEATURES})
    dataset["symbol"] = [f"{i:06}" for i in range(1010)]
    dataset["date"] = [pd.Timestamp("2024-12-01")] * 1000 + [pd.Timestamp("2025-01-02")] * 10
    dataset["exit_20"] = [pd.Timestamp("2024-12-31")] * 1000 + [pd.NaT] * 10
    dataset["net_20"] = np.tile([0.01, -0.01], 505)
    dataset.loc[1000:, "net_20"] = np.nan
    [result] = evaluate(dataset, 20)
    assert result["status"] == "pending"
    assert result["top10_selected"] == 10
    assert result["top10_excluded_days"] == 1
    assert result["matured_rows"] == result["top10_matured"] == 0
    for metric in ("model_brier", "model_auc", "top10_mean_net", "top10_hit_rate"):
        assert pd.isna(result[metric])


def test_incomplete_basket_cannot_inflate_returns_and_benchmark_dates_are_matched():
    from src.long_term_walkforward import summarize_portfolios

    days = pd.date_range("2025-01-01", periods=3)
    selected = pd.DataFrame({
        "date": [days[0]] * 10 + [days[1]] * 10 + [days[2]] * 10,
        "net_20": [0.02] * 10 + [0.99] * 9 + [float("nan")] + [0.04] * 10,
    })
    universe = pd.concat([selected, pd.DataFrame({
        "date": days, "net_20": [0.13, 0.88, float("nan")],
    })], ignore_index=True)
    result = summarize_portfolios(selected, universe, "net_20")
    assert result["top10_signal_days"] == 3
    assert result["top10_completed_days"] == 2
    assert result["top10_excluded_days"] == 1
    assert result["top10_mean_net"] == pytest.approx(0.03)
    assert result["paired_days"] == 1
    assert result["paired_top10_mean_net"] == pytest.approx(0.02)
    assert result["universe_mean_net"] == pytest.approx(0.03)
    assert result["paired_excess_net"] == pytest.approx(-0.01)


def test_symbol_holdout_excludes_test_stocks_from_training_history():
    import numpy as np

    from src.long_term_walkforward import evaluate
    from src.long_term_probability_lab import FEATURES

    count = 1006
    dataset = pd.DataFrame({feature: np.linspace(-1, 1, count) for feature in FEATURES})
    dataset["symbol"] = ["000001"] * 1000 + ["000002"] * 6
    dataset["date"] = [pd.Timestamp("2024-12-01")] * 1003 + [pd.Timestamp("2025-01-02")] * 3
    dataset["exit_20"] = [pd.Timestamp("2024-12-31")] * 1003 + [pd.Timestamp("2025-01-30")] * 3
    dataset["net_20"] = np.where(np.arange(count) % 2 == 0, 0.01, -0.01)
    [result] = evaluate(dataset, 20, relative=True, held_out={"000002"})
    assert result["train_rows"] == 1000
    assert result["scored_rows"] == 3


def test_long_range_features_have_identical_past_with_future_prices_appended(tmp_path):
    import numpy as np
    from src.long_term_walkforward import LONG_FEATURES, prepare_history

    count = 600
    close = 100 + np.arange(count) * 0.05 + np.sin(np.arange(count))
    frame = pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=count),
        "open": close, "high": close + 2, "low": close - 2,
        "close": close, "volume": 10000,
    })
    path = tmp_path / "000001.csv"
    frame.iloc[:350].to_csv(path, index=False)
    prefix = prepare_history(path, long_features=True).set_index("date")
    frame.to_csv(path, index=False)
    extended = prepare_history(path, long_features=True).set_index("date")
    pd.testing.assert_frame_equal(prefix[LONG_FEATURES], extended.loc[prefix.index, LONG_FEATURES])
    assert prefix.index.min() == frame.loc[240, "date"]


def test_validation_blend_purges_overlapping_labels(monkeypatch):
    import numpy as np
    from src import long_term_walkforward as lab

    count = 400
    days = pd.bdate_range("2022-01-03", periods=count)
    rows = pd.DataFrame({
        "date": np.repeat(days, 20), "symbol": np.tile(np.arange(20).astype(str), count),
        "feature": np.repeat(np.arange(count), 20),
        "net_20": np.tile([-0.01, 0.01], count * 10),
        "exit_20": np.repeat(days + pd.offsets.BDay(20), 20),
    })
    calls = []
    class Model:
        def fit(self, features, target):
            calls.append(features.copy())
        def predict_proba(self, features):
            return np.tile([0.5, 0.5], (len(features), 1))
    monkeypatch.setattr(lab, "make_pipeline", lambda *args: Model())
    assert lab.validation_blend(rows, ["feature"], 20, relative=True) == 0.0
    [trained] = calls
    expected = rows.loc[rows["exit_20"] < days[-126], ["feature"]]
    pd.testing.assert_frame_equal(trained, expected)


def test_validation_blend_has_no_fit_when_history_is_too_short(monkeypatch):
    from src import long_term_walkforward as lab
    monkeypatch.setattr(lab, "make_pipeline", lambda *args: pytest.fail("unexpected training"))
    assert lab.validation_blend(pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=100)}),
                                ["feature"], 240, relative=True) == 0.0
