"""Live, read-only end-to-end smoke of Korea's independently verified NAVER fallback.

Copies stock histories to a disposable temporary directory, deliberately
removes one completed session, disables KRX, and checks that independent
NAVER data restore the exact same original close WITHOUT touching repo data.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from . import auto_update

SYMBOLS = ("005930", "000660", "035420", "005380", "068270", "035720")


def main() -> None:
    cases = {}
    with tempfile.TemporaryDirectory(prefix="kr-price-fallback-") as tmp:
        root = Path(tmp)
        for sym in SYMBOLS:
            source = Path("data/history") / f"{sym}.csv"
            if not source.exists():
                raise RuntimeError(f"Source dataset missing {sym}")
            history = pd.read_csv(source, parse_dates=["date"])
            history = history.sort_values("date").reset_index(drop=True)
            if len(history) < 50:
                raise RuntimeError(f"Insufficient source records for {sym}")
            last = history.iloc[-1].copy()
            cases[sym] = last
            # Emulate a one-day lag. Source file is never modified.
            history.iloc[:-1].to_csv(root / f"{sym}.csv", index=False)
        complete = min(pd.Timestamp(row["date"]) for row in cases.values())
        if any(pd.Timestamp(row["date"]) != complete for row in cases.values()):
            raise RuntimeError("Different final market dates in sample source files")
        original_history = auto_update.KR_HISTORY
        original_cutoff = auto_update._latest_complete_kr_date
        original_krx = auto_update.stock.get_market_ohlcv_by_date
        original_sleep = auto_update.time.sleep
        try:
            auto_update.KR_HISTORY = root
            auto_update._latest_complete_kr_date = lambda: complete
            auto_update.stock.get_market_ohlcv_by_date = lambda *_args: pd.DataFrame()
            auto_update.time.sleep = lambda _: None
            auto_update.update_korea()
        finally:
            auto_update.KR_HISTORY = original_history
            auto_update._latest_complete_kr_date = original_cutoff
            auto_update.stock.get_market_ohlcv_by_date = original_krx
            auto_update.time.sleep = original_sleep
        for sym, last in cases.items():
            after = pd.read_csv(root / f"{sym}.csv", parse_dates=["date"]).iloc[-1]
            if pd.Timestamp(after["date"]) != pd.Timestamp(last["date"]):
                raise RuntimeError(f"{sym}: missed latest completed session in NAVER fallback")
            if abs(float(after["close"]) - float(last["close"])) > 0.01:
                raise RuntimeError(f"{sym}: different last close from verified original")
        print({
            "sample_symbols": list(SYMBOLS),
            "source_date": complete.date().isoformat(),
            "provider": "NAVER (KRX unavailable simulated)",
            "sample_new_session_append_success": len(cases),
        })


if __name__ == "__main__":
    main()
