from __future__ import annotations
import math
import pandas as pd


def _clip(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


def score_row(row: pd.Series) -> tuple[float, dict[str, float]]:
    parts = {}

    parts["momentum"] = 18 * _clip((row["ret_5d"] + 0.02) / 0.10)
    parts["trend"] = 18 * (
        0.5 * _clip((row["ma5"] / row["ma20"] - 0.98) / 0.06)
        + 0.5 * _clip((row["ma20"] / row["ma60"] - 0.98) / 0.08)
    )
    parts["volume"] = 18 * _clip((row["volume_ratio"] - 0.8) / 2.2)
    parts["close_strength"] = 18 * _clip(row["close_position"])
    parts["breakout"] = 18 * _clip((row["near_20d_high"] - 0.94) / 0.06)
    parts["rsi"] = 10 * _clip((row["rsi14"] - 45) / 30)

    score = sum(parts.values())
    return round(score, 2), {k: round(v, 2) for k, v in parts.items()}


def display_probability(score: float) -> float:
    # 실제 확률이 아니라 초기 UI용 점수 변환값.
    p = 1 / (1 + math.exp(-(score - 50) / 12))
    return round(p * 100, 1)
