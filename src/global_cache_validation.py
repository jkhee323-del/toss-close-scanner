"""Read-only validation of cached overseas recommendations against their own tickers."""
from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from .data_integrity_audit import check_ohlcv_quality


def validate_global_candidates(root: str | Path = "data") -> dict:
    root = Path(root)
    path = root / "global_candidates.csv"
    candidates = pd.read_csv(path, dtype={"symbol": str})
    mandatory = {"symbol", "date", "day_offset", "entry_price"}
    if candidates.empty or not mandatory.issubset(candidates.columns):
        raise ValueError("Global candidate cache missing required rows or columns")
    if set(candidates["day_offset"].astype(int).unique()) != {0, 1, 2, 3}:
        raise ValueError("Global candidate cache missing one of the four session tabs")
    count = 0
    for symbol, group in candidates.groupby("symbol"):
        price_path = root / "global_history" / f"{str(symbol).replace('/', '_')}.csv"
        history = pd.read_csv(price_path)
        issue, _ = check_ohlcv_quality(history)
        if issue:
            raise ValueError(f"{symbol}: corrupted history ({issue})")
        history["date"] = pd.to_datetime(history["date"])
        close_by_date = history.set_index("date")["close"]
        for row in group.itertuples(index=False):
            date = pd.Timestamp(row.date)
            expected_close = close_by_date.get(date)
            if expected_close is None or pd.isna(expected_close):
                raise ValueError(f"{symbol}: candidate session {date.date()} absent from history")
            actual = float(row.entry_price)
            expected = float(expected_close)
            if not math.isfinite(actual) or not math.isfinite(expected):
                raise ValueError(f"{symbol}: missing or nonfinite price on {date.date()}")
            if abs(actual - expected) > max(1e-5, abs(expected) * 1e-7):
                raise ValueError(
                    f"{symbol}: candidate price {actual} differs from own history close {expected} "
                    f"on {date.date()}"
                )
            count += 1
    return {"candidate_rows_verified": count, "distinct_symbols": int(candidates["symbol"].nunique())}


if __name__ == "__main__":
    print(validate_global_candidates())
