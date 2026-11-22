"""Read-only sample-wide NAVER fall-back integration test on 60 Korean symbols."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from . import auto_update
from .lab_sampling import sample_history_paths
from .data_integrity_audit import check_ohlcv_quality

SAMPLE_SIZE = 60


def main() -> None:
    all_paths = sorted(Path("data/history").glob("*.csv"))
    if len(all_paths) < SAMPLE_SIZE:
        raise RuntimeError("Korean history universe too small")
    latest_date = max(pd.Timestamp(pd.read_csv(p, usecols=["date"])["date"].iloc[-1]) for p in all_paths)
    eligible = []
    for path in all_paths:
        frame = pd.read_csv(path)
        if pd.Timestamp(frame["date"].iloc[-1]) != latest_date or len(frame) < 40:
            continue
        # The 20 most recent cached days must be complete before comparing a
        # new independently sourced session.
        issue, _ = check_ohlcv_quality(frame.tail(20).reset_index(drop=True))
        if issue is None:
            eligible.append(path)
    sample = sample_history_paths(eligible, SAMPLE_SIZE, seed="naver-live-safe-fallback-2026")
    if len(sample) != SAMPLE_SIZE:
        raise RuntimeError("Fewer than 60 recently complete Korean symbols")
    expected = {}
    with tempfile.TemporaryDirectory(prefix="kr-broad-fallback-") as tmp:
        root = Path(tmp)
        for path in sample:
            frame = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            expected[path.stem] = frame.iloc[-1]
            frame.iloc[:-1].to_csv(root / path.name, index=False)
        previous_path = auto_update.KR_HISTORY
        previous_cutoff = auto_update._latest_complete_kr_date
        previous_fetch = auto_update.stock.get_market_ohlcv_by_date
        previous_sleep = auto_update.time.sleep
        try:
            auto_update.KR_HISTORY = root
            auto_update._latest_complete_kr_date = lambda: latest_date
            auto_update.stock.get_market_ohlcv_by_date = lambda *_args: pd.DataFrame()
            auto_update.time.sleep = lambda *_args: None
            auto_update.update_korea()
        finally:
            auto_update.KR_HISTORY = previous_path
            auto_update._latest_complete_kr_date = previous_cutoff
            auto_update.stock.get_market_ohlcv_by_date = previous_fetch
            auto_update.time.sleep = previous_sleep
        restored = []
        failed = []
        for sym, last in expected.items():
            after = pd.read_csv(root / f"{sym}.csv", parse_dates=["date"]).iloc[-1]
            if (pd.Timestamp(after["date"]) == latest_date and
                abs(float(after["close"]) - float(last["close"])) <= 0.01):
                restored.append(sym)
            else:
                failed.append(sym)
        print({
            "eligible": len(eligible), "sample": len(sample),
            "source_session": latest_date.date().isoformat(),
            "new_session_restored": len(restored),
            "unrestored_symbols": failed,
        })
        if len(restored) < int(SAMPLE_SIZE * 0.90):
            raise RuntimeError("NAVER fallback did not restore 90% of sampled completed sessions")


if __name__ == "__main__":
    main()
