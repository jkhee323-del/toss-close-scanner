"""Normalize recorded KIS KRX execution messages for CVD research."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cvd import compute_cvd

MAX_CAPTURE_TRADES = 100_000
CAPTURE_SOURCE = "KIS_KRX_H0STCNT0"

KIS_EXECUTION_COLUMNS = (
    "MKSC_SHRN_ISCD", "BSOP_DATE", "STCK_CNTG_HOUR", "STCK_PRPR",
    "CNTG_VOL", "ACML_VOL", "CCLD_DVSN",
)


def read_kis_capture(path: Path, ticker: str) -> pd.DataFrame:
    """Load a trusted collector file without weakening its sequence checks."""
    columns = ["timestamp", "volume", "price", "aggressor_side", "accumulated_volume", "symbol", "source"]
    trades = pd.read_csv(path, usecols=columns, dtype={"symbol": str}, nrows=MAX_CAPTURE_TRADES + 1)
    if trades.empty or len(trades) > MAX_CAPTURE_TRADES:
        raise ValueError("KIS 저장 파일은 1~100,000건의 체결 기록이어야 합니다.")
    if not trades["symbol"].eq(ticker).all() or not trades["source"].eq(CAPTURE_SOURCE).all():
        raise ValueError("저장 파일의 종목 또는 데이터 공급원이 일치하지 않습니다.")
    values = trades[["price", "volume", "accumulated_volume"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values).all().all() or (values <= 0).any().any() or (values % 1 != 0).any().any():
        raise ValueError("KIS 저장 파일의 가격과 체결 수량은 양의 정수여야 합니다.")
    if not pd.to_datetime(trades["timestamp"], utc=True).is_monotonic_increasing:
        raise ValueError("KIS 저장 파일의 체결 순서가 올바르지 않습니다.")
    delta = compute_cvd(trades)
    if delta["session_date"].nunique() != 1:
        raise ValueError("KIS 저장 파일은 한 거래일만 포함해야 합니다.")
    return delta.assign(price=values["price"].to_numpy(), symbol=ticker, source=CAPTURE_SOURCE)


def normalize_kis_krx_trades(raw: pd.DataFrame) -> pd.DataFrame:
    """Accept one symbol/session in provider order and reject volume gaps.

    Opening auctions or other unsigned executions cannot be assigned a side.
    Coverage starts at the first recorded signed trade, possibly mid-session.
    """
    if raw.empty or not set(KIS_EXECUTION_COLUMNS).issubset(raw.columns):
        raise ValueError("KIS KRX 체결 파일에 필수 열 또는 체결 기록이 없습니다.")
    data = raw[list(KIS_EXECUTION_COLUMNS)].astype("string")
    for column, pattern in (
        ("MKSC_SHRN_ISCD", r"[0-9]{6}"),
        ("BSOP_DATE", r"[0-9]{8}"),
        ("STCK_CNTG_HOUR", r"[0-9]{6}"),
        ("CNTG_VOL", r"[0-9]+"),
        ("ACML_VOL", r"[0-9]+"),
        ("STCK_PRPR", r"[0-9]+"),
    ):
        if not data[column].str.fullmatch(pattern).fillna(False).all():
            raise ValueError(f"잘못된 KIS 체결 필드 {column}")
    if data["MKSC_SHRN_ISCD"].nunique() != 1 or data["BSOP_DATE"].nunique() != 1:
        raise ValueError("KIS 체결 파일은 한 종목과 한 거래일만 포함해야 합니다.")
    timestamp = pd.to_datetime(
        data["BSOP_DATE"] + data["STCK_CNTG_HOUR"], format="%Y%m%d%H%M%S"
    ).dt.tz_localize("Asia/Seoul")
    volume = data["CNTG_VOL"].to_numpy(dtype=np.int64)
    total_volume = data["ACML_VOL"].to_numpy(dtype=np.int64)
    price = data["STCK_PRPR"].to_numpy(dtype=np.int64)
    sides = data["CCLD_DVSN"].map({"1": "buy", "5": "sell"})
    if sides.isna().any():
        raise ValueError("KIS 체결 파일에 매수·매도 방향이 없는 체결이 포함되어 있습니다.")
    if not timestamp.is_monotonic_increasing or (volume <= 0).any() or (price <= 0).any():
        raise ValueError("KIS 체결 순서, 가격 또는 수량을 확인해주세요.")
    if total_volume[0] < volume[0] or not np.array_equal(np.diff(total_volume), volume[1:]):
        raise ValueError("KIS 누적 체결량에 누락·중복·역순 체결이 감지되었습니다.")
    return pd.DataFrame({
        "timestamp": timestamp, "volume": volume, "price": price, "aggressor_side": sides,
        "accumulated_volume": total_volume,
    })
