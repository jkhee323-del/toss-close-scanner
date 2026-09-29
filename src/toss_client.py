from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock


class TossClient:
    """
    현재는 Toss Open API 대신 pykrx를 사용해
    한국 주식의 일봉 데이터를 가져옵니다.

    나중에 Toss API 사용이 가능해지면
    이 파일만 다시 교체할 수 있습니다.
    """

    def __init__(self, *args, **kwargs):
        pass

    def get_daily_history(self, symbol: str, bars: int = 1200) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=max(bars * 2, 365))

        start = start_date.strftime("%Y%m%d")
        end = end_date.strftime("%Y%m%d")

        df = stock.get_market_ohlcv_by_date(
            start,
            end,
            symbol
        )

        if df.empty:
            raise ValueError(f"{symbol}: 주가 데이터를 가져오지 못했습니다.")

        # pykrx의 한글 컬럼명을 기존 프로그램 형식으로 변경
        df = df.rename(
            columns={
                "시가": "open",
                "고가": "high",
                "저가": "low",
                "종가": "close",
                "거래량": "volume",
            }
        )

        required_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        df = df[required_columns].copy()

        # 날짜를 일반 컬럼으로 변경
        df = df.reset_index()

        date_column = df.columns[0]
        df = df.rename(columns={date_column: "date"})

        df["date"] = pd.to_datetime(df["date"])

        # 최근 bars개만 사용
        df = df.tail(bars).reset_index(drop=True)

        return df