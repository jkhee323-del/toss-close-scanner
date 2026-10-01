from __future__ import annotations
import math
import pandas as pd


def _safe_float(value, default=0.0):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


def _clip(v, lo=0.0, hi=1.0):
    v = _safe_float(v)
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, v))


def _safe_ratio(numerator, denominator, default=0.0):
    num = _safe_float(numerator)
    den = _safe_float(denominator)
    if den == 0:
        return default if num == 0 else (1.0 if num > 0 else -1.0)
    return num / den


def score_row(row: pd.Series) -> tuple[float, dict[str, float]]:
    parts = {}

    parts["momentum"] = 18 * _clip((_safe_float(row.get("ret_5d")) + 0.02) / 0.10)
    parts["trend"] = 18 * (
        0.5 * _clip((_safe_ratio(row.get("ma5"), row.get("ma20"), default=1.0) - 0.98) / 0.06)
        + 0.5 * _clip((_safe_ratio(row.get("ma20"), row.get("ma60"), default=1.0) - 0.98) / 0.08)
    )
    parts["volume"] = 18 * _clip((_safe_float(row.get("volume_ratio")) - 0.8) / 2.2)
    parts["close_strength"] = 18 * _clip(_safe_float(row.get("close_position")))
    parts["breakout"] = 18 * _clip((_safe_float(row.get("near_20d_high")) - 0.94) / 0.06)
    parts["rsi"] = 10 * _clip((_safe_float(row.get("rsi14")) - 45) / 30)

    score = sum(parts.values())
    score = max(0.0, min(100.0, float(score)))
    return round(score, 2), {k: round(max(0.0, min(100.0, float(v))), 2) for k, v in parts.items()}


def display_probability(score: float) -> float:
    # 실제 확률이 아니라 초기 UI용 점수 변환값.
    s = _safe_float(score, default=0.0)
    s = max(0.0, min(100.0, s))
    p = 1 / (1 + math.exp(-(s - 50) / 12))
    return round(max(0.0, min(100.0, p * 100)), 1)
