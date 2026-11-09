"""Separate, optional Korean investor net-purchase data ingestion.

Source: pykrx daily per-ticker get_market_trading_value_by_date, net amount
in KRW (기관합계, 외국인합계). Never infer these figures from OHLCV.
Missing or malformed provider data is not silently treated as zero demand.
Research-only until live supplier/schema checks have succeeded.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .global_market_safety import atomic_write_table

NEEDED = ("기관합계", "외국인합계")
FIELDS = ("date", "institutional_net_krw", "foreign_net_krw")
DEFAULT_DIRECTORY = Path("data/kr_investor_flows")


class FlowDataError(ValueError):
    """Investor-flow data could not be validated."""


def validate_flow_frame(raw: pd.DataFrame, *, as_of: date | None = None) -> pd.DataFrame:
    """Reject missing columns, duplicate dates and non-finite investor cash flows."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise FlowDataError("investor-flow data unavailable")
    if not set(NEEDED).issubset(raw.columns):
        raise FlowDataError("missing Korean institution/foreign net-purchase fields")
    x = raw.loc[:, list(NEEDED)].copy()
    x["date"] = pd.to_datetime(raw.index, errors="coerce")
    if x["date"].isna().any():
        raise FlowDataError("invalid investor flow date")
    if isinstance(x["date"].dtype, pd.DatetimeTZDtype):
        raise FlowDataError("investor flow dates must be Korean-local dates")
    x["date"] = x["date"].dt.normalize()
    if x["date"].duplicated().any():
        raise FlowDataError("duplicate investor flow session")
    if (x["date"].dt.dayofweek > 4).any():
        raise FlowDataError("investor flow on weekend")
    if as_of is not None and (x["date"].dt.date > as_of).any():
        raise FlowDataError("future-dated investor flow")
    for column in NEEDED:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    if x[list(NEEDED)].isna().any().any():
        raise FlowDataError("non-numeric investor flow")
    if not np.isfinite(x[list(NEEDED)].to_numpy(dtype=float)).all():
        raise FlowDataError("non-finite investor flow")
    result = x.rename(columns={
        "기관합계": "institutional_net_krw",
        "외국인합계": "foreign_net_krw",
    })
    return result[list(FIELDS)].sort_values("date").reset_index(drop=True)


def fetch_korean_net_flows(
    ticker: str, *, from_date: date, to_date: date,
    provider=None,
) -> pd.DataFrame:
    """Fetch real daily net-buy **amounts** for a six-digit KRX ticker."""
    if not isinstance(ticker, str) or len(ticker) != 6 or not ticker.isascii() or not ticker.isdigit():
        raise FlowDataError("ticker must be a six-digit KRX code")
    if from_date > to_date or to_date > date.today():
        raise FlowDataError("invalid or future investor-flow date window")
    if provider is None:
        from pykrx import stock
        provider = stock.get_market_trading_value_by_date
    data = provider(from_date.strftime("%Y%m%d"), to_date.strftime("%Y%m%d"), ticker)
    return validate_flow_frame(data, as_of=to_date)


def merge_flow_history(path: Path, incoming: pd.DataFrame) -> pd.DataFrame:
    """Keep past observations, replacing only overlapping session rows."""
    new = incoming[list(FIELDS)].copy()
    if path.exists():
        old = pd.read_csv(path)
        if not set(FIELDS).issubset(old.columns):
            raise FlowDataError("existing flow cache malformed")
        new = pd.concat([old[list(FIELDS)], new], ignore_index=True)
    new["date"] = pd.to_datetime(new["date"], errors="coerce")
    for column in FIELDS[1:]:
        new[column] = pd.to_numeric(new[column], errors="coerce")
    if new.isna().any().any() or not np.isfinite(new[list(FIELDS[1:])].to_numpy(dtype=float)).all():
        raise FlowDataError("invalid cached/incoming flow values")
    new = new.sort_values("date").drop_duplicates("date", keep="last")
    return new.reset_index(drop=True)


def save_korean_flows(
    ticker: str, flows: pd.DataFrame, *, directory: Path = DEFAULT_DIRECTORY
) -> Path:
    if len(ticker) != 6 or not ticker.isascii() or not ticker.isdigit():
        raise FlowDataError("invalid ticker")
    destination = Path(directory) / f"{ticker}.csv"
    combined = merge_flow_history(destination, flows)
    atomic_write_table(destination, combined)
    return destination


def latest_flow_snapshot(
    ticker: str, *, directory: Path = DEFAULT_DIRECTORY
) -> dict | None:
    """Return actual published flow values only; None is 'unavailable'."""
    if len(ticker) != 6 or not ticker.isascii() or not ticker.isdigit():
        return None
    path = Path(directory) / f"{ticker}.csv"
    if not path.is_file():
        return None
    try:
        x = pd.read_csv(path)
        if not set(FIELDS).issubset(x.columns) or x.empty:
            return None
        x["date"] = pd.to_datetime(x["date"], errors="raise")
        if x["date"].duplicated().any():
            return None
        for column in FIELDS[1:]:
            x[column] = pd.to_numeric(x[column], errors="raise")
        if x[list(FIELDS[1:])].isna().any().any():
            return None
        row = x.sort_values("date").iloc[-1]
        if row["date"].date() > date.today():
            return None
        return {
            "date": row["date"].strftime("%Y-%m-%d"),
            "institutional_net_krw": int(row["institutional_net_krw"]),
            "foreign_net_krw": int(row["foreign_net_krw"]),
            "source": "pykrx:KRX daily trading value, net buys",
            "estimated": False,
        }
    except (OSError, ValueError, TypeError, KeyError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="+", help="Six-digit Korean ticker codes")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    if not 1 <= args.days <= 365:
        parser.error("--days must be between 1 and 365")
    end = date.today()
    start = end - timedelta(days=args.days)
    for symbol in args.tickers:
        try:
            downloaded = fetch_korean_net_flows(symbol, from_date=start, to_date=end)
            path = save_korean_flows(symbol, downloaded, directory=args.output_dir)
            print(f"{symbol}: {len(downloaded)} daily net-purchase rows -> {path}")
        except (FlowDataError, OSError, ValueError) as exc:
            print(f"[UNAVAILABLE] {symbol}: {exc}")


if __name__ == "__main__":
    main()
