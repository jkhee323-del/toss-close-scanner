from __future__ import annotations

from pathlib import Path
import time

import pandas as pd

from .scanner import get_all_korean_stocks
from .toss_client import TossClient


HISTORY_DIR = Path("data/history")


def download_history(limit: int = 100, bars: int = 1200):
    """
    KOSPI + KOSDAQ 종목의 과거 주가 데이터를 저장합니다.
    이미 충분한 데이터가 있는 파일은 건너뜁니다.
    """

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    universe = get_all_korean_stocks().head(limit)
    client = TossClient()

    success = 0
    failed = 0
    skipped = 0

    total = len(universe)

    for number, item in enumerate(
        universe.itertuples(index=False),
        start=1,
    ):
        symbol = item.symbol
        name = item.name

        output_path = HISTORY_DIR / f"{symbol}.csv"

        # 이미 저장된 종목은 우선 건너뜁니다.
        if output_path.exists():
            try:
                existing_df = pd.read_csv(output_path)

                if len(existing_df) >= bars:
                    print(
                        f"[{number}/{total}] 건너뜀: "
                        f"{symbol} {name} ({len(existing_df)}일)"
                    )
                    skipped += 1
                    continue

                print(
                    f"[{number}/{total}] 다시 받기: "
                    f"{symbol} {name} ({len(existing_df)}일)"
                )

            except Exception:
                print(
                    f"[{number}/{total}] 기존 파일 오류, 다시 받기: "
                    f"{symbol} {name}"
                )

        try:
            df = client.get_daily_history(symbol, bars=bars)

            if len(df) < 100:
                print(
                    f"[{number}/{total}] 데이터 부족: "
                    f"{symbol} {name} ({len(df)}일)"
                )
                failed += 1
                continue

            df.to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
            )

            success += 1

            print(
                f"[{number}/{total}] 저장 완료: "
                f"{symbol} {name} ({len(df)}일)"
            )

        except Exception as exc:
            failed += 1
            print(
                f"[{number}/{total}] 실패: "
                f"{symbol} {name} / {exc}"
            )

        # 데이터 제공처에 너무 빠르게 요청하지 않도록 잠시 대기
        time.sleep(0.15)

    print()
    print("===== 다운로드 완료 =====")
    print("새로 저장:", success)
    print("건너뜀:", skipped)
    print("실패:", failed)


if __name__ == "__main__":
    download_history(limit=500, bars=1200)