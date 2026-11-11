"""Offline checks for a conservative, fail-closed scanner release gate."""
from datetime import date, timedelta

import pandas as pd
import pytest

from src.release_gate import REQUIRED_PROOFS, check_release, latest_csv_date


def valid_project(tmp_path, last_date: str):
    for directory in ("src", "tests", "data/history", "data/global_history"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "app.py").write_text("# test\n")
    (tmp_path / "requirements.txt").write_text("pandas\n")
    for sub, name in [("history", "005930"), ("global_history", "AAPL")]:
        pd.DataFrame({"date": [last_date], "close": [100]}).to_csv(
            tmp_path / "data" / sub / f"{name}.csv", index=False
        )
    return tmp_path


def test_missing_external_evidence_blocks_even_if_caches_recent(tmp_path):
    project = valid_project(tmp_path, "2026-11-10")
    result = check_release(project, as_of=date(2026, 11, 11))
    assert result["release_decision"] == "BLOCKED"
    assert result["checks"]["kr_cache_fresh"]
    assert result["checks"]["global_cache_fresh"]
    for key in REQUIRED_PROOFS:
        assert key in result["blockers"]


def test_old_cache_and_future_cache_are_not_fresh(tmp_path):
    project = valid_project(tmp_path, "2026-11-02")
    report = check_release(project, as_of=date(2026, 11, 11), evidence={
        key: True for key in REQUIRED_PROOFS
    })
    assert "kr_cache_fresh" in report["blockers"]
    assert "global_cache_fresh" in report["blockers"]
    valid_project(project, "2026-11-12")
    future = check_release(project, as_of=date(2026, 11, 11))
    assert not future["checks"]["kr_cache_fresh"]


def test_proofs_must_be_explicit_true(tmp_path):
    project = valid_project(tmp_path, "2026-11-10")
    proof = {key: True for key in REQUIRED_PROOFS}
    proof["real_cvd_feed_verified"] = "true"  # strings are not evidence
    result = check_release(project, as_of=date(2026, 11, 11), evidence=proof)
    assert result["release_decision"] == "BLOCKED"
    assert "real_cvd_feed_verified" in result["blockers"]
    proof["real_cvd_feed_verified"] = True
    checked = check_release(project, as_of=date(2026, 11, 11), evidence=proof)
    assert checked["release_decision"] == "REQUIRES_HUMAN_RELEASE_REVIEW"


def test_corrupt_date_csv_blocks_release(tmp_path):
    project = valid_project(tmp_path, "2026-11-10")
    (project / "data" / "history" / "005930.csv").write_text(
        "date,close\nnot-a-date,100\n", encoding="utf-8"
    )
    result = check_release(project, as_of=date(2026, 11, 11))
    assert "kr_cache_usable" in result["blockers"]
    assert "kr_cache_fresh" in result["blockers"]
    assert result["cache"]["kr"]["invalid_files"]


def test_bad_freshness_argument_rejected(tmp_path):
    with pytest.raises(ValueError, match="nonnegative"):
        check_release(tmp_path, as_of=date(2026, 11, 11), freshness_days=-1)


def test_does_not_modify_any_csv(tmp_path):
    project = valid_project(tmp_path, "2026-11-10")
    path = project / "data" / "global_history" / "AAPL.csv"
    prior = path.read_bytes()
    _ = check_release(project, as_of=date(2026, 11, 11))
    assert path.read_bytes() == prior
