"""Reproducible broad-market sampling for preliminary laboratory runs.

Never select the first N sorted symbol codes: that overweights one code prefix.
Deterministic SHA-256 ranking yields the same sample for model and backtests.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sample_history_paths(paths: list[Path], max_symbols: int | None, seed: str = "scanner-v1") -> list[Path]:
    unique = {p.stem: p for p in paths}
    ordered = sorted(unique.values(), key=lambda p: p.stem)
    if max_symbols is None:
        return ordered
    if max_symbols <= 0:
        raise ValueError("max_symbols must be positive")
    return sorted(
        ordered,
        key=lambda p: (hashlib.sha256(f"{seed}:{p.stem}".encode("utf-8")).hexdigest(), p.stem),
    )[:max_symbols]
