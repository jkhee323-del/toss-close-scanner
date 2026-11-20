"""No-lookahead regression tests for the long-term historical audit."""
import pandas as pd
import pytest

from src.long_term_scanner import analyze_stock
from src.long_term_holdout import realized_return, rank_and_grade, summarize_completed_periods


def _history():
    days = pd.bdate_range("2024-01-01", periods=520)
    values = [100 + i * 0.1 for i in range(520)]
    return pd.DataFrame({
        "date": days,
        "open": values, "high": [x + 2 for x in values],
        "low": [x - 2 for x in values], "close": values,
        "volume": [1000 + i for i in range(520)],
    })


def test_historical_snapshot_ignores_future_prices(tmp_path):
    original = _history()
    path = tmp_path / "000001.csv"
    original.to_csv(path, index=False)
    signal_day = original["date"].iloc[299]
    base = analyze_stock(path, 20, as_of_date=str(signal_day.date()))
    assert base is not None
    changed = original.copy()
    changed.loc[changed.index > 299, ["open", "high", "low", "close"]] *= 100
    changed.to_csv(path, index=False)
    after = analyze_stock(path, 20, as_of_date=str(signal_day.date()))
    assert after == base


def test_future_return_uses_next_open_and_later_close():
    df = _history()
    day = df["date"].iloc[300]
    value = realized_return(df, day, 20, cost_rate=0.002)
    expected = df["close"].iloc[320] / df["open"].iloc[301] - 1 - 0.002
    assert value == pytest.approx(expected)


def test_unfinished_horizon_is_not_evaluated():
    df = _history()
    assert realized_return(df, df["date"].iloc[-10], 20) is None


def test_unfinished_top_candidate_is_not_replaced_by_future_complete_stock():
    # Highest-ranked symbol has no observed future. It must remain in TOP2.
    candidates = [
        ("AAA", 99.0, None),
        ("BBB", 90.0, 0.04),
        ("CCC", 80.0, 0.30),
    ]
    result = rank_and_grade(candidates, top_n=2)
    assert result["top_symbols"] == "AAA,BBB"
    assert result["top_evaluated_count"] == 1
    assert result["top_unfinished_count"] == 1
    assert result["top_coverage"] == 0.5
    assert result["top_mean_net_return"] == pytest.approx(0.04)


def test_unfinished_all_top_candidates_never_generate_phantom_returns():
    out = rank_and_grade([("AAA", 99.0, None), ("BBB", 95.0, None), ("CCC", 30.0, 0.99)], 2)
    assert out["top_symbols"] == "AAA,BBB"
    assert out["top_evaluated_count"] == 0
    assert pd.isna(out["top_mean_net_return"])
    assert pd.isna(out["top_win_rate"])


def test_summary_uses_only_fully_matured_portfolios():
    inputs = pd.DataFrame([
        {"horizon_days": 20, "top_n": 2, "top_evaluated_count": 2,
         "top_mean_net_return": 0.02, "top_win_rate": 0.5, "excess_vs_universe": 0.01},
        {"horizon_days": 20, "top_n": 2, "top_evaluated_count": 1,
         "top_mean_net_return": 0.99, "top_win_rate": 1.0, "excess_vs_universe": 0.80},
    ])
    summary = summarize_completed_periods(inputs).iloc[0]
    assert summary["signal_periods"] == 2
    assert summary["fully_matured_periods"] == 1
    assert summary["excluded_incomplete_periods"] == 1
    assert summary["mean_return_completed_only"] == pytest.approx(0.02)


def test_backtest_does_not_jump_over_invalid_holding_day():
    frame = _history()
    signal = frame["date"].iloc[300]
    frame.loc[309, ["open", "high", "low", "close"]] = 0.0
    assert realized_return(frame, signal, 20) is None


def test_future_corruption_does_not_change_past_scanner_signal(tmp_path):
    frame = _history()
    path = tmp_path / "005930.csv"
    cutoff = frame["date"].iloc[299]
    original = analyze_stock(path, 20, as_of_date=str(cutoff.date()), history_frame=frame)
    assert original is not None
    frame.loc[400, ["open", "high", "low", "close"]] = 0.0
    actual = analyze_stock(path, 20, as_of_date=str(cutoff.date()), history_frame=frame)
    assert actual == original


def test_corrupted_date_cannot_be_discarded_from_backtest_history(tmp_path):
    from src.long_term_holdout import _load_history
    frame = _history()
    frame.loc[100, ["open", "high", "low", "close"]] = 0.0
    path = tmp_path / "005930.csv"
    frame.to_csv(path, index=False)
    loaded = _load_history(path)
    assert len(loaded) == len(frame)
    assert loaded.loc[100, "close"] == 0.0
