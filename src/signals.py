"""오늘/전일/2일전/3일전 신호 계산과 익절가(호가단위 반영) 계산.

- 저장된 일봉(data/history/*.csv)에서 최근 N거래일 각각의 지표를 계산하고
  저장된 두 모델(up_model / strong_model)로 점수를 매깁니다.
- 신호일 종가를 진입가로 보고, 그 뒤 고가가 1차(+10%)/2차(+20%) 가격에 닿았는지 추적합니다.
"""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .indicators import add_indicators
from .model import FEATURES


# ── 호가단위 ────────────────────────────────────────────────────────────────
def tick_size(price: float) -> int:
    """KRX 주식 호가단위(2023년 이후 코스피·코스닥 공통)."""
    if price < 2_000:
        return 1
    if price < 5_000:
        return 5
    if price < 20_000:
        return 10
    if price < 50_000:
        return 50
    if price < 200_000:
        return 100
    if price < 500_000:
        return 500
    return 1_000


def floor_to_tick(price: float) -> float:
    """호가단위로 내림. 실제로 주문·체결 가능한 가격으로 맞춥니다."""
    price = round(float(price), 6)
    t = tick_size(price)
    return float(math.floor(price / t + 1e-9) * t)


def target_prices(entry: float) -> tuple[float, float]:
    """1차(+10%), 2차(+20%) 익절가. 호가단위로 내림하므로 +9.9%처럼 표시될 수 있습니다."""
    return floor_to_tick(entry * 1.10), floor_to_tick(entry * 1.20)


# ── 모델 / 점수 ─────────────────────────────────────────────────────────────
def load_models(data_dir: str | Path = "data"):
    d = Path(data_dir)
    return joblib.load(d / "up_model.joblib"), joblib.load(d / "strong_model.joblib")


def compute_scored_window(data_dir: str | Path = "data", n_days: int = 4):
    """최근 n_days 거래일 × 전 종목의 점수·결과를 한 번에 계산합니다.

    반환: (DataFrame, meta). day_offset 0 = 최신일(오늘), 1 = 전일 …
    """
    data_dir = Path(data_dir)
    up_model, strong_model = load_models(data_dir)

    parts: list[pd.DataFrame] = []
    training_rows = 0
    n_files = 0
    keep = FEATURES + ["trading_value", "date", "open", "high", "low", "close", "volume"]
    for path in sorted((data_dir / "history").glob("*.csv")):
        try:
            df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            x = add_indicators(df)
        except Exception:
            continue
        n_files += 1
        valid = x[FEATURES].notna().all(axis=1)
        training_rows += max(int(valid.sum()) - 1, 0)  # 마지막 행은 다음날 수익률이 없어 학습에서 제외
        t = x[keep].tail(n_days + 3).copy()
        t["symbol"] = path.stem.zfill(6)
        parts.append(t)

    if not parts:
        return pd.DataFrame(), {}

    allrows = pd.concat(parts, ignore_index=True)
    counts = allrows.groupby("date")["symbol"].nunique()
    ok_dates = counts[counts >= 0.5 * counts.max()].index.sort_values()
    dates = list(ok_dates[-n_days:])
    latest = dates[-1]

    w = allrows[allrows["date"].isin(dates)].dropna(subset=FEATURES + ["trading_value"]).copy()
    w = w.sort_values(["symbol", "date"]).reset_index(drop=True)

    # 신호 이후 최고가 / 현재가(최신 종가)
    g = w.groupby("symbol")
    w["max_high_after"] = g["high"].transform(lambda s: s[::-1].shift(1).cummax()[::-1])
    w["last_close"] = g["close"].transform("last")
    w["last_date"] = g["date"].transform("max")
    w = w[w["last_date"] == latest].copy()  # 최신일 데이터가 없는(거래정지 등) 종목 제외

    up = up_model.predict_proba(w[FEATURES])[:, 1]
    strong = strong_model.predict_proba(w[FEATURES])[:, 1]
    w["up_model_score"] = up * 100
    w["strong_model_score"] = strong * 100
    w["scanner_score"] = (up + strong) / 2 * 100

    w["entry_price"] = w["close"]
    w["change_pct"] = w["ret_1d"] * 100
    tg = [target_prices(p) for p in w["entry_price"]]
    w["target_1"] = [a for a, _ in tg]
    w["target_2"] = [b for _, b in tg]
    w["hit_1"] = (w["max_high_after"] >= w["target_1"]).fillna(False)
    w["hit_2"] = (w["max_high_after"] >= w["target_2"]).fillna(False)
    w["ret_since"] = (w["last_close"] / w["entry_price"] - 1) * 100

    offset = {d: len(dates) - 1 - i for i, d in enumerate(dates)}
    w["day_offset"] = w["date"].map(offset)
    w = w.reset_index(drop=True)

    meta = {
        "latest_date": pd.Timestamp(latest),
        "dates": {int(v): pd.Timestamp(k) for k, v in offset.items()},
        "training_symbols": n_files,
        "training_rows": training_rows,
    }
    return w, meta


def pick_signals(window: pd.DataFrame, offset: int, top_n: int, min_trading_value: float):
    """해당 일자의 상위 신호. (표시용 DataFrame, 필터 통과 종목 수)를 돌려줍니다."""
    if window is None or window.empty:
        return pd.DataFrame(), 0
    d = window[(window["day_offset"] == offset) & (window["trading_value"] >= min_trading_value)]
    d = d.sort_values("scanner_score", ascending=False)
    return d.head(top_n).reset_index(drop=True), int(len(d))
