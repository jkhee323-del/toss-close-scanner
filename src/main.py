from __future__ import annotations
import argparse
import pandas as pd
from .config import get_settings
from .toss_client import TossClient
from .scanner import scan_universe, get_all_korean_stocks
from .backtest import run_backtest


def cmd_scan(args):
    settings = get_settings()
    universe = get_all_korean_stocks()
    universe = universe.head(20)

    if args.symbols:
        universe = universe[universe["symbol"].isin(args.symbols)].copy()

    client = TossClient(
        settings.client_id,
        settings.client_secret,
        settings.base_url,
    )

    result = scan_universe(client, universe, top_n=args.top)

    if result.empty:
        print("조건을 만족하는 종목이 없습니다.")
        return

    print(f"\n=== 다음 거래일 후보 TOP {len(result)} ===")
    cols = [
        "symbol", "name", "close", "change_1d",
        "volume_ratio", "close_position",
        "near_20d_high", "score", "display_probability"
    ]
    print(result[cols].to_string(index=False))

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out = settings.results_dir / "latest_scan.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}")


def cmd_backtest(args):
    result = run_backtest(args.data_dir, threshold=args.threshold)
    if result.empty:
        print("백테스트 결과가 없습니다.")
        return

    print("\n=== 백테스트 결과 ===")
    print(result.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Toss Securities Close Trading Scanner v1"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="토스증권 API로 현재 후보 스캔")
    scan.add_argument("--top", type=int, default=10)
    scan.add_argument("--symbols", nargs="*")
    scan.set_defaults(func=cmd_scan)

    bt = sub.add_parser("backtest", help="CSV 과거 데이터 백테스트")
    bt.add_argument("--data-dir", default="data/history")
    bt.add_argument("--threshold", type=float, default=70)
    bt.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
