"""Overseas cache guards must reject mixed ticker columns and mismatched prices."""
import pandas as pd
import pytest

from src.data_integrity_audit import check_ohlcv_quality
from src.global_cache_validation import validate_global_candidates


def _fixture(root):
    folder = root / "global_history"
    folder.mkdir()
    dates = pd.bdate_range("2026-11-12", periods=4)
    history = pd.DataFrame({
        "date": dates,
        "open": [100.0, 102.0, 103.0, 104.0],
        "high": [104.0, 105.0, 106.0, 107.0],
        "low": [99.0, 100.0, 101.0, 102.0],
        "close": [102.0, 103.0, 104.0, 105.0],
        "volume": [1000, 1100, 1200, 1300],
    })
    history.to_csv(folder / "GOOD.csv", index=False)
    candidates = pd.DataFrame({
        "symbol": ["GOOD"] * 4,
        "date": list(reversed(dates)),
        "day_offset": list(range(4)),
        "entry_price": [105.0, 104.0, 103.0, 102.0],
    })
    candidates.to_csv(root / "global_candidates.csv", index=False)
    return folder, candidates


def test_global_candidate_prices_match_own_history(tmp_path):
    _fixture(tmp_path)
    assert validate_global_candidates(tmp_path) == {"candidate_rows_verified": 4, "distinct_symbols": 1}


def test_duplicate_ohlcv_columns_rejected_even_when_pandas_mangles_names(tmp_path):
    folder, _ = _fixture(tmp_path)
    csv = folder / "GOOD.csv"
    header, *rows = csv.read_text().splitlines()
    csv.write_text(header + ",open,close\n" + "\n".join(row + ",999,888" for row in rows) + "\n")
    raw = pd.read_csv(csv)
    reason, _ = check_ohlcv_quality(raw)
    assert reason == "unexpected or duplicated OHLCV columns"
    with pytest.raises(ValueError, match="corrupted history"):
        validate_global_candidates(tmp_path)


def test_candidate_cross_ticker_price_contamination_rejected(tmp_path):
    _, cand = _fixture(tmp_path)
    cand.loc[0, "entry_price"] = 266.66
    cand.to_csv(tmp_path / "global_candidates.csv", index=False)
    with pytest.raises(ValueError, match="differs from own history"):
        validate_global_candidates(tmp_path)


def test_nonfinite_price_cannot_be_silently_accepted(tmp_path):
    _, cand = _fixture(tmp_path)
    cand.loc[0, "entry_price"] = float("nan")
    cand.to_csv(tmp_path / "global_candidates.csv", index=False)
    with pytest.raises(ValueError, match="nonfinite price"):
        validate_global_candidates(tmp_path)
