"""Optional KRX foreign/institutional net-purchase statistics.

The official daily investor breakdown is generally finalized after 18:00 KST.
These are real net-purchase amounts (KRW), NOT estimated CVD.
"""
from __future__ import annotations

import pandas as pd


def latest_published_flow_date(now: pd.Timestamp | None = None) -> pd.Timestamp:
    local_now = pd.Timestamp.now(tz="Asia/Seoul") if now is None else pd.Timestamp(now)
    local_now = local_now.tz_localize("Asia/Seoul") if local_now.tzinfo is None else local_now.tz_convert("Asia/Seoul")
    cutoff = local_now.normalize().tz_localize(None)
    if local_now.hour < 18:
        cutoff -= pd.Timedelta(days=1)
    return cutoff


def summarize_investor_flows(raw: pd.DataFrame, end_date: pd.Timestamp, lookback: int = 20) -> dict | None:
    if raw is None or raw.empty or lookback <= 0:
        return None
    if not {"기관합계", "외국인합계"}.issubset(raw.columns):
        return None
    frame = raw[["기관합계", "외국인합계"]].copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()]
    for column in ["기관합계", "외국인합계"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame.index.normalize() <= pd.Timestamp(end_date).normalize()]
    frame = frame.dropna().sort_index().tail(lookback)
    if len(frame) < 5:
        return None
    return {
        "as_of": frame.index[-1].strftime("%Y-%m-%d"),
        "sessions": len(frame),
        "foreign_net_krw": int(frame["외국인합계"].sum()),
        "institution_net_krw": int(frame["기관합계"].sum()),
    }


def fetch_investor_flows(ticker: str, scan_date: str, lookback: int = 20) -> dict | None:
    """Fetch a bounded range and never use data after the scan's reference date."""
    from pykrx import stock

    if len(ticker) != 6 or not ticker.isdigit():
        raise ValueError("KRX ticker must be six digits")
    end = min(pd.Timestamp(scan_date).normalize(), latest_published_flow_date())
    start = end - pd.Timedelta(days=max(45, 3 * lookback))
    raw = stock.get_market_trading_value_by_date(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), ticker)
    return summarize_investor_flows(raw, end, lookback=lookback)
