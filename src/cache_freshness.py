"""Data-as-of labels for Streamlit research screens.

Read the candidates' latest session labels rather than calling old cache
"today". Missing sources stay missing. Calendar-day age is only a warning
heuristic, not a verified exchange holiday/calendar calculation.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def candidate_freshness(
    path: Path | str, *, as_of: date | None = None, stale_after_days: int = 5,
) -> dict:
    today = as_of or date.today()
    result = {"source": str(path), "last_session": None,
              "age_calendar_days": None, "state": "MISSING"}
    if stale_after_days < 0:
        raise ValueError("stale_after_days must be >= 0")
    p = Path(path)
    if not p.is_file():
        return result
    try:
        # Candidate rows may contain four previous-day entries, but the
        # freshest published session is always the max valid date.
        frame = pd.read_csv(p, usecols=["date"])
        if frame.empty:
            return {**result, "state": "INVALID"}
        labels = pd.to_datetime(frame["date"], errors="coerce")
        if labels.isna().any():
            return {**result, "state": "INVALID"}
        latest = labels.max().date()
        age = (today - latest).days
        result.update(
            last_session=latest.isoformat(), age_calendar_days=age,
            state=("FUTURE" if age < 0 else
                   "STALE" if age > stale_after_days else "RECENT"),
        )
    except (OSError, ValueError, KeyError, TypeError):
        result["state"] = "INVALID"
    return result


def freshness_message(result: dict, *, market: str) -> str:
    status = result["state"]
    stamp = result.get("last_session")
    if status == "RECENT":
        return f"{market} 저장된 데이터 기준일: {stamp}. 실시간 시세가 아닙니다."
    if status == "STALE":
        return (f"{market} 데이터 주의: 마지막 날짜 {stamp} "
                f"(달력일 {result['age_calendar_days']}일 전). "
                "자동 갱신과 실제 장 마감 여부를 확인하세요. 오늘 추천이 아닙니다.")
    if status == "FUTURE":
        return f"{market} 데이터 날짜가 미래({stamp})입니다. 데이터를 검증하세요."
    return f"{market} 데이터가 없거나 날짜를 확인할 수 없습니다. 추천 신뢰도를 판단할 수 없습니다."
