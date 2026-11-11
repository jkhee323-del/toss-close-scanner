"""No-lookahead cache staleness warning tests."""
from datetime import date

import pandas as pd
import pytest

from src.cache_freshness import candidate_freshness, freshness_message


def test_old_candidates_are_labeled_as_historical_not_today(tmp_path):
    p = tmp_path / "global_candidates.csv"
    pd.DataFrame({"date": ["2026-11-02", "2026-10-30"]}).to_csv(p, index=False)
    result = candidate_freshness(p, as_of=date(2026, 11, 11))
    assert result["state"] == "STALE"
    assert result["age_calendar_days"] == 9
    assert "오늘 추천이 아닙니다" in freshness_message(result, market="해외")


def test_current_and_future_dates(tmp_path):
    p = tmp_path / "global_candidates.csv"
    pd.DataFrame({"date": ["2026-11-11"]}).to_csv(p, index=False)
    assert candidate_freshness(p, as_of=date(2026, 11, 11))["state"] == "RECENT"
    assert candidate_freshness(p, as_of=date(2026, 11, 10))["state"] == "FUTURE"


def test_missing_invalid_and_no_mutation(tmp_path):
    p = tmp_path / "global_candidates.csv"
    assert candidate_freshness(p, as_of=date(2026, 11, 11))["state"] == "MISSING"
    p.write_text("date\ninvalid\n")
    b = p.read_bytes()
    assert candidate_freshness(p, as_of=date(2026, 11, 11))["state"] == "INVALID"
    assert p.read_bytes() == b


def test_bad_age_limit():
    with pytest.raises(ValueError):
        candidate_freshness("whatever.csv", stale_after_days=-1)
