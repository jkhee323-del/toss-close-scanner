"""Validate Yahoo OHLCV and preserve long-term history during refresh.

No cache write occurs until the downloaded rows and merged history pass validation.
The writes are atomic; overlapping material price discrepancies fail closed.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class PriceDataError(ValueError):
    """A download or cached daily-bar series is not safe to publish."""


def validate_ohlcv(frame: pd.DataFrame, *, min_rows: int = 70) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not set(COLUMNS).issubset(frame):
        raise PriceDataError("OHLCV columns missing")
    out = frame[COLUMNS].copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        raise PriceDataError("expected timezone-naive exchange session dates")
    if dates.isna().any() or not pd.api.types.is_datetime64_any_dtype(dates):
        raise PriceDataError("invalid OHLCV dates")
    out["date"] = dates.dt.normalize()
    for key in COLUMNS[1:]:
        out[key] = pd.to_numeric(out[key], errors="coerce")
    if out.isna().any().any():
        raise PriceDataError("missing or unparseable OHLCV")
    if not np.isfinite(out[COLUMNS[1:]].to_numpy(dtype=float)).all():
        raise PriceDataError("nonfinite OHLCV")
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    prices = out[["open", "high", "low", "close"]]
    if (prices <= 0).any().any() or (out["volume"] < 0).any():
        raise PriceDataError("nonpositive prices or negative volume")
    if (out["high"] < prices[["open", "low", "close"]].max(axis=1)).any():
        raise PriceDataError("daily high below open/low/close")
    if (out["low"] > prices[["open", "high", "close"]].min(axis=1)).any():
        raise PriceDataError("daily low above open/high/close")
    if (out["date"].dt.dayofweek >= 5).any():
        raise PriceDataError("weekend bars are not expected in cash-equity OHLCV")
    if len(out) < min_rows:
        raise PriceDataError(f"insufficient history: {len(out)} < {min_rows}")
    return out


def merge_price_history(path: Path, incoming: pd.DataFrame) -> pd.DataFrame:
    """Combine a one-year refresh with the full existing history, not overwrite it."""
    fresh = validate_ohlcv(incoming, min_rows=70)
    if not path.is_file():
        return fresh
    try:
        old = validate_ohlcv(pd.read_csv(path), min_rows=1)
    except (OSError, ValueError) as exc:
        raise PriceDataError(f"cached data invalid; preserve for investigation: {exc}") from exc
    overlap = old[["date", "close"]].merge(
        fresh[["date", "close"]], on="date", how="inner", suffixes=("_old", "_new")
    )
    if not overlap.empty:
        relative = (overlap["close_new"] / overlap["close_old"] - 1).abs()
        if (relative > 0.25).any():
            raise PriceDataError("large overlapping close discrepancy; split/vendor correction review required")
    joined = pd.concat([old, fresh], ignore_index=True)
    return validate_ohlcv(joined, min_rows=70)


def atomic_save_history(path: Path, frame: pd.DataFrame) -> bool:
    """Replace a CSV only after checking the complete merged data.

    Returns False if unchanged. A failed write preserves the previous CSV.
    """
    checked = validate_ohlcv(frame, min_rows=70)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = validate_ohlcv(pd.read_csv(path), min_rows=1)
        if existing.equals(checked):
            return False
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", suffix=".csv",
            prefix=f".{path.stem}_", dir=path.parent, delete=False
        ) as handle:
            tmp = Path(handle.name)
            checked.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return True
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
