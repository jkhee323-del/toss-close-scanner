"""Real cumulative volume delta from trade-level aggressor data.

CVD requires individual executions with an EXPLICIT aggressor side. Daily OHLCV,
OBV, and inferred candle direction cannot substitute for buy/sell aggressor data.
This module intentionally has no fabricated fallback.
"""
from __future__ import annotations

import pandas as pd


def compute_cvd(
    trades: pd.DataFrame,
    *,
    timezone: str = "Asia/Seoul",
    reset_each_session: bool = False,
) -> pd.DataFrame:
    """Compute signed-volume delta and cumulative delta from execution records.

    Input columns: timestamp (ISO datetime WITH timezone), volume (>0),
    aggressor_side ('buy'/'sell'). Rows must be chronological or sortable.
    Returns timestamp (local time), session_date, volume, aggressor_side,
    signed_volume, cvd. No estimates or tick-rule imputation.
    """
    needed = {"timestamp", "volume", "aggressor_side"}
    if trades is None or not needed.issubset(trades.columns):
        raise ValueError("CVD requires timestamp, volume and aggressor_side")
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "session_date", "volume", "aggressor_side", "signed_volume", "cvd"])
    frame = trades[["timestamp", "volume", "aggressor_side"]].copy()
    # Require timezone-aware timestamps to avoid ambiguous exchange sessions.
    try:
        raw_times = pd.to_datetime(frame["timestamp"], utc=False, errors="raise")
        if raw_times.dt.tz is None:
            raise ValueError("CVD timestamps must include a timezone")
        frame["timestamp"] = raw_times.dt.tz_convert(timezone)
    except (AttributeError, TypeError) as exc:
        raise ValueError("CVD timestamps must have a consistent explicit timezone") from exc
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame["aggressor_side"] = frame["aggressor_side"].astype(str).str.lower().str.strip()
    if frame["timestamp"].isna().any() or frame["volume"].isna().any() or (frame["volume"] <= 0).any():
        raise ValueError("CVD requires valid timestamps and strictly positive trade volume")
    if not frame["aggressor_side"].isin(["buy", "sell"]).all():
        raise ValueError("Unknown aggressor side; cannot calculate actual CVD")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    frame["session_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame["signed_volume"] = frame["volume"].where(frame["aggressor_side"].eq("buy"), -frame["volume"])
    if reset_each_session:
        frame["cvd"] = frame.groupby("session_date")["signed_volume"].cumsum()
    else:
        frame["cvd"] = frame["signed_volume"].cumsum()
    return frame
