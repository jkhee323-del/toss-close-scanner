"""Real cumulative volume delta from trade-level aggressor data.

CVD requires individual executions with an EXPLICIT aggressor side. Daily OHLCV,
OBV, and inferred candle direction cannot substitute for buy/sell aggressor data.
This module intentionally has no fabricated fallback.
"""
from __future__ import annotations

import numpy as np
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
    columns = ["timestamp", "volume", "aggressor_side"]
    if "accumulated_volume" in trades:
        columns.append("accumulated_volume")
    frame = trades[columns].copy()
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
    if frame["timestamp"].isna().any() or not np.isfinite(frame["volume"]).all() or (frame["volume"] <= 0).any():
        raise ValueError("CVD requires valid timestamps and strictly positive trade volume")
    if not frame["aggressor_side"].isin(["buy", "sell"]).all():
        raise ValueError("Unknown aggressor side; cannot calculate actual CVD")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    frame["session_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    if "accumulated_volume" in frame:
        total = pd.to_numeric(frame["accumulated_volume"], errors="coerce")
        if not np.isfinite(total).all() or (total < frame["volume"]).any():
            raise ValueError("Invalid exchange accumulated volume")
        increments = total.groupby(frame["session_date"]).diff()
        repeated = frame["session_date"].duplicated()
        if not np.array_equal(increments.loc[repeated].to_numpy(), frame.loc[repeated, "volume"].to_numpy()):
            raise ValueError("Missing, duplicate or reversed executions in accumulated volume")
        frame["accumulated_volume"] = total
    frame["signed_volume"] = frame["volume"].where(frame["aggressor_side"].eq("buy"), -frame["volume"])
    if reset_each_session:
        frame["cvd"] = frame.groupby("session_date")["signed_volume"].cumsum()
    else:
        frame["cvd"] = frame["signed_volume"].cumsum()
    return frame


def cvd_divergence(trades: pd.DataFrame, *, bar_minutes: int = 5) -> pd.DataFrame:
    """Compare the closing trade price and signed volume over adjacent time bars.

    Results are timestamped at bar close. The last observed bar of each
    session is excluded because the input may end before that bar closes.
    """
    if bar_minutes <= 0:
        raise ValueError("CVD divergence requires a positive bar length")
    delta = compute_cvd(trades, reset_each_session=True)
    price = pd.to_numeric(trades["price"], errors="coerce")
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("CVD divergence requires positive execution prices")
    if delta.empty:
        return pd.DataFrame(columns=["bar_end", "price_change", "cvd_change", "divergence"])
    # Use positional ordering so callers may supply non-unique DataFrame indexes.
    order = pd.to_datetime(trades["timestamp"], utc=True).reset_index(drop=True).sort_values(kind="stable").index
    delta["price"] = price.to_numpy()[order]
    interval = pd.Timedelta(minutes=bar_minutes)
    delta["bar_end"] = delta["timestamp"].dt.floor(interval) + interval
    bars = delta.groupby(["session_date", "bar_end"], as_index=False).agg(
        price=("price", "last"), cvd=("cvd", "last")
    )
    by_session = bars.groupby("session_date")
    bars["price_change"] = by_session["price"].diff()
    bars["cvd_change"] = by_session["cvd"].diff()
    bars["divergence"] = np.select(
        [(bars["price_change"] < 0) & (bars["cvd_change"] > 0),
         (bars["price_change"] > 0) & (bars["cvd_change"] < 0)],
        ["bullish", "bearish"], default="none",
    )
    complete_pair = (by_session["bar_end"].diff() == interval) & (
        bars["bar_end"] < by_session["bar_end"].transform("max")
    )
    return bars.loc[complete_pair, ["bar_end", "price_change", "cvd_change", "divergence"]]
