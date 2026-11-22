"""Read-only NAVER fresh daily close smoke, no production file mutation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .kr_naver_provider import fetch_naver_ohlcv, validate_naver_overlap
from .data_integrity_audit import check_ohlcv_quality


def main() -> None:
    symbol = "005930"
    original = pd.read_csv(Path("data/history") / f"{symbol}.csv")
    original["date"] = pd.to_datetime(original["date"])
    latest = pd.Timestamp(original["date"].max())
    start = (latest - pd.Timedelta(days=25)).date().isoformat()
    end = (pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
           + pd.Timedelta(days=1)).date().isoformat()
    alternative = fetch_naver_ohlcv(symbol, start, end)
    issue, _ = check_ohlcv_quality(alternative)
    if issue is not None:
        raise RuntimeError(f"NAVER feed not valid: {issue}")
    matched = validate_naver_overlap(original, alternative, minimum_matches=3)
    print({
        "symbol": symbol,
        "cached_latest": latest.date().isoformat(),
        "naver_latest": alternative["date"].max().date().isoformat(),
        "matched_overlap_sessions": matched,
        "naver_rows": len(alternative),
        "naver_latest_close": float(alternative["close"].iloc[-1]),
        "cached_latest_close": float(original["close"].iloc[-1]),
    })
    if alternative["date"].max() < latest:
        raise RuntimeError("Independent NAVER feed is older than cached current session")


if __name__ == "__main__":
    main()
