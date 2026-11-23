"""Safely summarize optional user-supplied investor net-purchase CSV exports.

No claim of an authenticated exchange feed is made. The supplied file is not
saved on disk, never changes production scores, and dates after the signal
reference are excluded to avoid look-ahead leakage.

Required columns (KRW units): date, symbol, foreign_net_krw, institution_net_krw
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ("date", "symbol", "foreign_net_krw", "institution_net_krw")


def parse_imported_flows(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or not set(REQUIRED).issubset(raw.columns):
        raise ValueError("필수 CSV 열: date, symbol, foreign_net_krw, institution_net_krw")
    if raw.empty:
        raise ValueError("수급 CSV 데이터가 없습니다.")
    if len(raw) > 200_000:
        raise ValueError("수급 CSV는 20만 행 이하만 사용할 수 있습니다.")
    df = raw[list(REQUIRED)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if not df["symbol"].str.fullmatch(r"\d{1,6}").all():
        raise ValueError("한국 종목 코드는 1~6자리 숫자여야 합니다.")
    df["symbol"] = df["symbol"].str.zfill(6)
    for column in ("foreign_net_krw", "institution_net_krw"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("날짜·종목코드·수급 금액에 빈 값이나 잘못된 값이 있습니다.")
    numeric = df[["foreign_net_krw", "institution_net_krw"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("수급 금액에 무한대 또는 잘못된 값이 있습니다.")
    if df.duplicated(["date", "symbol"]).any():
        raise ValueError("같은 종목과 날짜가 중복되었습니다. 합산 기준을 확인해주세요.")
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def summarize_imported_flows(
    data: pd.DataFrame,
    symbols: list[str],
    as_of: str | pd.Timestamp,
    lookback: int = 20,
) -> pd.DataFrame:
    if not 1 <= lookback <= 250:
        raise ValueError("조회 거래일 수는 1~250이어야 합니다.")
    clean = parse_imported_flows(data)
    reference = pd.Timestamp(as_of).normalize()
    clean = clean.loc[clean["date"] <= reference]
    output = []
    for symbol in symbols:
        ticker = str(symbol).zfill(6)
        frames = clean.loc[clean["symbol"] == ticker].tail(lookback)
        if frames.empty:
            output.append({
                "종목코드": ticker, "수급 기준일": "자료 없음", "확인 거래일": 0,
                "외국인 순매수(억원)": None, "기관 순매수(억원)": None,
            })
        else:
            output.append({
                "종목코드": ticker,
                "수급 기준일": frames["date"].max().date().isoformat(),
                "확인 거래일": len(frames),
                "외국인 순매수(억원)": round(frames["foreign_net_krw"].sum() / 1e8, 2),
                "기관 순매수(억원)": round(frames["institution_net_krw"].sum() / 1e8, 2),
            })
    return pd.DataFrame(output)
