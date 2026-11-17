"""Preliminary evaluation sampling must not depend on symbol sort order."""
from pathlib import Path
import pytest

from src.lab_sampling import sample_history_paths


def test_reproducible_sample_scattered_across_codes():
    paths = [Path(f"{i:06}.csv") for i in range(500)]
    first = sample_history_paths(paths, 80)
    second = sample_history_paths(list(reversed(paths)), 80)
    assert len(first) == 80
    assert [p.stem for p in first] == [p.stem for p in second]
    assert any(int(p.stem) > 300 for p in first)
    assert any(int(p.stem) < 100 for p in first)


def test_full_sample_and_bad_limit():
    paths = [Path("000002.csv"), Path("000001.csv")]
    assert [p.stem for p in sample_history_paths(paths, None)] == ["000001", "000002"]
    with pytest.raises(ValueError):
        sample_history_paths(paths, 0)
