"""True trade-side cumulative volume delta (CVD).

Requires time-and-sales tick data with *verified aggressor-side direction*.
Daily OHLCV, OBV, and buy/sell-guessing from a candle cannot yield true CVD.
No data => no CVD value; callers must display "체결 데이터 없음".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ("timestamp", "price", "size", "aggressor_side")


class CVDDataError(ValueError):
    """Trade executions are invalid or have no verified side."""


def calculate_cvd(
    trades: pd.DataFrame, *, initial_cvd: float = 0.0,
    reset_daily: bool = False, timezone: str = "Asia/Seoul",
) -> pd.DataFrame:
    """Create tick-level signed volume and cumulative delta.

    aggressor_side = 'buy' means buyer-initiated / lifting ask;
                     'sell' means seller-initiated / hitting bid.
    All timestamps must be timezone-aware. Trade size is shares/contracts,
    not KRW; CVD unit is identical to the trade-size unit.
    """
    if not isinstance(trades, pd.DataFrame) or trades.empty or not set(COLUMNS).issubset(trades):
        raise CVDDataError("verified trade ticks with side are required")
    x = trades[list(COLUMNS)].copy()
    if x.isna().any().any():
        raise CVDDataError("null trade input")
    # Check each record BEFORE coercing its timezone. Mixed aware/naive
    # strings can otherwise be transformed to NaT and obscure the error.
    for value in x["timestamp"]:
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise CVDDataError("invalid timestamps") from exc
        if pd.isna(stamp):
            raise CVDDataError("invalid timestamps")
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise CVDDataError("timestamps must be timezone-aware")
    ts = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise CVDDataError("invalid timestamps")
    x["timestamp"] = ts
    if (x["aggressor_side"].map(lambda s: s in ("buy", "sell")) == False).any():
        raise CVDDataError("aggressor_side must be explicit buy or sell")
    x["price"] = pd.to_numeric(x["price"], errors="coerce")
    x["size"] = pd.to_numeric(x["size"], errors="coerce")
    if x[["price", "size"]].isna().any().any():
        raise CVDDataError("invalid price or trade size")
    if not np.isfinite(x[["price", "size"]].to_numpy(dtype=float)).all():
        raise CVDDataError("nonfinite trade price/size")
    if (x["price"] <= 0).any() or (x["size"] <= 0).any():
        raise CVDDataError("nonpositive trade price/size")
    x = x.sort_values("timestamp", kind="stable").reset_index(drop=True)
    x["signed_volume"] = np.where(x["aggressor_side"].eq("buy"), x["size"], -x["size"])
    if reset_daily:
        local_day = x["timestamp"].dt.tz_convert(timezone).dt.normalize()
        x["cvd"] = x.groupby(local_day, sort=False)["signed_volume"].cumsum() + initial_cvd
    else:
        x["cvd"] = x["signed_volume"].cumsum() + initial_cvd
    return x


def cvd_availability(data: pd.DataFrame | None) -> dict:
    """Client-safe status; never substitute OBV as trade CVD."""
    if not isinstance(data, pd.DataFrame) or data.empty or not set(COLUMNS).issubset(data):
        return {"available": False, "value": None, "label": "실제 체결 데이터 없음"}
    try:
        result = calculate_cvd(data)
        return {
            "available": True,
            "value": float(result["cvd"].iloc[-1]),
            "label": "실제 체결 방향 기반 CVD (주/계약)",
        }
    except (CVDDataError, ValueError):
        return {"available": False, "value": None, "label": "실제 체결 데이터 검증 실패"}
