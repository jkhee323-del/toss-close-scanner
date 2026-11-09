"""Research-only, leakage-sensitive US horizon model regression tests."""
import numpy as np
import pandas as pd

from src.us_horizon_research import (
    FEATURES, HORIZONS, evaluate_horizon, make_labeled_rows,
    purged_split, run_all,
)


def history(seed=1, n=700):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.8, n))
    return pd.DataFrame({
        "date": pd.bdate_range("2022-01-03", periods=n),
        "open": close.copy(),
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 100_000 + np.arange(n) * 50,
    })


def test_horizon_features_only_depend_on_signal_or_earlier_days():
    old = history()
    rows = make_labeled_rows(old, "AAPL", 20)
    chosen = rows.iloc[50]
    signal_date = chosen["date"]
    # Perturb a future close. This can alter the label but not past features.
    changed = old.copy()
    future = changed.index[changed["date"] == signal_date][0] + 20
    changed.loc[future, ["open", "high", "low", "close"]] += 0.5
    newer = make_labeled_rows(changed, "AAPL", 20)
    new_row = newer.loc[newer["date"] == signal_date].iloc[0]
    for col in FEATURES:
        assert np.isclose(chosen[col], new_row[col])
    assert chosen["future_return"] != new_row["future_return"]


def test_label_uses_exact_horizon_trading_session():
    raw = history()
    rows = make_labeled_rows(raw, "AAPL", 240)
    assert not rows.empty
    first = rows.iloc[0]
    idx = int(raw.index[raw["date"] == first["date"]][0])
    assert first["exit_date"] == raw["date"].iloc[idx + 240]
    expected = raw["close"].iloc[idx + 240] / raw["close"].iloc[idx] - 1
    assert np.isclose(first["future_return"], expected)


def test_asof_excludes_future_exit_labels():
    raw = history()
    cutoff = raw["date"].iloc[500]
    x = make_labeled_rows(raw, "AAPL", 20, as_of=str(cutoff.date()))
    assert x["date"].max() < cutoff
    assert x["exit_date"].max() <= cutoff


def test_train_outcomes_are_known_before_test_period():
    x = pd.concat(
        [make_labeled_rows(history(seed=i), ticker, 60)
         for i, ticker in enumerate(["AAPL", "MSFT", "NVDA"], 1)],
        ignore_index=True,
    )
    cutoff = pd.Timestamp("2023-05-01")
    train, test = purged_split(x, cutoff=cutoff)
    assert len(train) and len(test)
    assert train["exit_date"].max() < cutoff
    assert test["date"].min() >= cutoff
    assert not set(train.index).intersection(test.index) or train["exit_date"].max() < test["date"].min()


def test_evaluation_is_research_only_and_has_baselines():
    rows = pd.concat(
        [make_labeled_rows(history(seed=i), ticker, 20)
         for i, ticker in enumerate(["AAPL", "MSFT", "NVDA", "AMD"], 1)],
        ignore_index=True,
    )
    result, _ = evaluate_horizon(rows, 20, min_train=200, min_test=80)
    assert result["status"] in {"OOS_GATE_PASSED_EXPERIMENTAL", "OOS_GATE_FAILED"}
    assert not result["verified_for_live"]
    assert not result["calibrated_probability"]
    assert "always_up_accuracy" in result
    assert "prior_brier" in result
    assert result["train_last_exit_date"] < result["test_start"]


def test_all_horizons_and_us_only_report(tmp_path):
    directory = tmp_path / "history"
    directory.mkdir()
    history(n=330).to_csv(directory / "AAPL.csv", index=False)
    history(seed=2, n=330).to_csv(directory / "7203.T.csv", index=False)
    report = run_all(directory, tmp_path / "reports", as_of="2025-01-01")
    assert set(report["results"]) == set(HORIZONS)
    assert report["eligible_price_histories"] == 1
    assert all(not result.get("verified_for_live", False)
               for result in report["results"].values())
    assert (tmp_path / "reports" / "us_horizon_research.json").exists()
    assert not list((tmp_path / "reports").glob("*.joblib"))
