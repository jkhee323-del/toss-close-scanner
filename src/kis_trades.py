"""Normalize recorded KIS KRX execution messages for CVD research."""
from __future__ import annotations

import numpy as np
import pandas as pd

KIS_EXECUTION_COLUMNS = (
    "MKSC_SHRN_ISCD", "BSOP_DATE", "STCK_CNTG_HOUR", "STCK_PRPR",
    "CNTG_VOL", "ACML_VOL", "CCLD_DVSN",
)


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
    })
