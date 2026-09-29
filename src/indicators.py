from __future__ import annotations
import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["ret_1d"] = x["close"].pct_change()
    x["ret_5d"] = x["close"].pct_change(5)
    x["ret_20d"] = x["close"].pct_change(20)

    x["ma5"] = x["close"].rolling(5).mean()
    x["ma20"] = x["close"].rolling(20).mean()
    x["ma60"] = x["close"].rolling(60).mean()

    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["volume_ratio"] = x["volume"] / x["vol_ma20"]

    day_range = (x["high"] - x["low"]).replace(0, np.nan)
    x["close_position"] = (x["close"] - x["low"]) / day_range

    x["high20"] = x["high"].rolling(20).max()
    x["near_20d_high"] = x["close"] / x["high20"]

    delta = x["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi14"] = 100 - (100 / (1 + rs))
    # 거래대금
    x["trading_value"] = x["close"] * x["volume"]
    x["trading_value_ma20"] = x["trading_value"].rolling(20).mean()

    # 일중 변동폭
    x["range_pct"] = (x["high"] - x["low"]) / x["close"]

    # 이동평균선과 현재가의 거리
    x["ma5_distance"] = x["close"] / x["ma5"] - 1
    x["ma20_distance"] = x["close"] / x["ma20"] - 1

    # 전일 종가 대비 시가 갭
    x["gap"] = x["open"] / x["close"].shift(1) - 1

    # 최근 변동성
    x["volatility_20d"] = x["ret_1d"].rolling(20).std()
    return x
