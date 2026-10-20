
from pathlib import Path

import numpy as np
import pandas as pd

from .indicators import add_indicators

from .search import US_STOCKS



HISTORY_DIR = Path("data/history")

US_HISTORY_DIR = Path("data/global_history")


# 약 1개월, 3개월, 6개월, 1년
HORIZONS = {
    "1": ("1개월", 20),
    "2": ("3개월", 60),
    "3": ("6개월", 120),
    "4": ("1년", 240),
}


def analyze_stock(path: Path, horizon: int):
    try:
        df = pd.read_csv(path)

        required = {"date", "close", "volume"}
        if not required.issubset(df.columns):
            return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        df = (
            df.dropna(subset=["date", "close", "volume"])
            .sort_values("date")
            .drop_duplicates("date")
        )

        if len(df) < max(241, horizon + 1):
            return None


        # 기존 지표 계산 함수를 활용해 OBV와 거래대금 계산
        df = add_indicators(df)

        close = df["close"]
        volume = df["volume"]

        obv_5d = float(df["obv_change_5d"].iloc[-1])
        obv_20d = float(df["obv_change_20d"].iloc[-1])

        if not np.isfinite(obv_5d):
            obv_5d = 0.0

        if not np.isfinite(obv_20d):
            obv_20d = 0.0


        if close.iloc[-1] <= 0:
            return None

        current = float(close.iloc[-1])

        ma20 = close.tail(20).mean()
        ma60 = close.tail(60).mean()
        ma120 = close.tail(120).mean()
        ma240 = close.tail(240).mean()

        momentum = current / close.iloc[-horizon - 1] - 1

        recent_volume = volume.tail(5).mean()
        average_volume = volume.tail(20).mean()

        volume_ratio = (
            recent_volume / average_volume
            if average_volume > 0
            else 0.0
        )

        high_240 = close.tail(240).max()
        low_240 = close.tail(240).min()

        price_position = (
            (current - low_240) / (high_240 - low_240)
            if high_240 > low_240
            else 0.5
        )

        # 점수는 규칙 기반 평가이며 상승 확률이 아님

        # 중·장기 스캐너 v3: 연속형 기술적 점수
        # 점수는 0~100이며 상승 확률이 아님

        def clamp(value, low, high):
            return max(low, min(high, float(value)))

        # 1. 추세 점수: 최대 30점
        trend_score = (
            10 * clamp((current / ma20 - 0.95) / 0.10, 0, 1)
            + 10 * clamp((current / ma60 - 0.95) / 0.15, 0, 1)
            + 10 * clamp((current / ma120 - 0.95) / 0.20, 0, 1)
        )

        # 2. 기간별 모멘텀: 최대 20점
        momentum_score = 20 * clamp(
            (momentum + 0.05) / 0.25, 0, 1
        )

        # 3. 거래량 증가: 최대 15점
        volume_score = 15 * clamp(
            (volume_ratio - 0.7) / 1.3, 0, 1
        )

        # 4. OBV 강도: 최대 20점
        obv_score = (
            8 * clamp((obv_5d + 0.3) / 0.6, 0, 1)
            + 12 * clamp((obv_20d + 0.3) / 0.6, 0, 1)
        )

        # 5. 장기 가격 위치: 최대 15점
        position_score = 15 * clamp(
            (price_position - 0.3) / 0.7, 0, 1
        )

        # 단기 급등에 따른 과열 위험 감점
        overheat_penalty = 15 * clamp(
            (momentum - 0.25) / 0.35, 0, 1
        )

        score = round(
            clamp(
                trend_score
                + momentum_score
                + volume_score
                + obv_score
                + position_score
                - overheat_penalty,
                0,
                100,
            ),
            1,
        )

        return {
            "종목코드": path.stem,
            "기준일": df["date"].iloc[-1].strftime("%Y-%m-%d"),
            "현재가": round(current, 2),
            "점수": score,
            "기간수익률(%)": round(momentum * 100, 2),
            "거래량비율": round(float(volume_ratio), 2),
            "가격위치(%)": round(float(price_position) * 100, 1),
            "OBV_5일": round(obv_5d, 3),
            "OBV_20일": round(obv_20d, 3),
        }

    except (ValueError, KeyError, TypeError, OSError, IndexError):
        return None



def get_recommendations(choice: str, market: str = "KR"):

    """웹사이트에서 사용할 중·장기 추천 종목 반환."""
    if choice not in HORIZONS:
        return pd.DataFrame()

    _, horizon = HORIZONS[choice]
    results = []


    history_dir = US_HISTORY_DIR if market == "US" else HISTORY_DIR


    us_tickers = set(pd.read_csv("data/us_universe.csv", dtype=str)["ticker"])


    for path in sorted(history_dir.glob("*.csv")):

        if market == "US" and path.stem not in us_tickers:
            continue


        result = analyze_stock(path, horizon)
        if result is not None:
            results.append(result)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    latest_date = df["기준일"].max()

    df = df[df["기준일"] == latest_date].copy()

    df = df.sort_values(
        ["점수", "기간수익률(%)"],
        ascending=[False, False],
    )

    return df.reset_index(drop=True)


def run():
    print("\n=== 중·장기 투자 스캐너 v3 ===")

    for key, (name, days) in HORIZONS.items():
        print(f"{key}. {name} ({days}거래일)")

    choice = input("\n투자 기간 선택 (1~4): ").strip()

    if choice not in HORIZONS:
        print("잘못된 선택입니다.")
        return

    name, horizon = HORIZONS[choice]
    files = sorted(HISTORY_DIR.glob("*.csv"))

    if not files:
        print("CSV 파일이 없습니다.")
        return

    results = []

    for path in files:
        result = analyze_stock(path, horizon)
        if result is not None:
            results.append(result)

    if not results:
        print("분석 가능한 종목이 없습니다.")
        return

    result_df = pd.DataFrame(results)
    latest_date = result_df["기준일"].max()

    result_df = result_df[
        result_df["기준일"] == latest_date
    ].copy()

    result_df = result_df.sort_values(
        ["점수", "기간수익률(%)"],
        ascending=[False, False],
    )

    print(f"\n분석 기준일: {latest_date}")
    print(f"투자 기간: {name}")
    print(f"분석 종목 수: {len(result_df)}")
    print("\n=== 상위 10개 종목 ===")
    print(result_df.head(10).to_string(index=False))

    print(
        "\n주의: 점수는 기술적 지표 기반이며 "
        "실제 상승 확률이 아닙니다."
    )


if __name__ == "__main__":
    run()
