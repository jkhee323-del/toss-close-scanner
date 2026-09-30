"""Scheduled market-data refresh for the deployed scanner.

Designed for GitHub Actions. It updates only the local CSV cache; the workflow
commits changed CSVs, and Streamlit Community Cloud then sees the repository
update automatically.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pykrx import stock

from .indicators import add_indicators
from .search import GLOBAL_SCAN_UNIVERSE

DATA = Path("data")
KR_HISTORY = DATA / "history"
GLOBAL_HISTORY = DATA / "global_history"
GLOBAL_CACHE = DATA / "global_candidates.csv"


def _normalize_kr(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    x = raw.rename(columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}).copy()
    needed = ["open", "high", "low", "close", "volume"]
    if not all(c in x.columns for c in needed):
        return pd.DataFrame()
    x = x[needed].reset_index()
    x = x.rename(columns={x.columns[0]: "date"})
    x["date"] = pd.to_datetime(x["date"]).dt.tz_localize(None)
    return x.dropna(subset=needed).sort_values("date").reset_index(drop=True)


def update_korea() -> None:
    """Incrementally append the latest KRX trading days to every stored symbol."""
    KR_HISTORY.mkdir(parents=True, exist_ok=True)
    paths = sorted(KR_HISTORY.glob("*.csv"))
    if not paths:
        raise RuntimeError("data/history has no CSV files")

    end = datetime.now() + timedelta(days=1)
    ok = fail = unchanged = 0
    for i, path in enumerate(paths, 1):
        symbol = path.stem.zfill(6)
        try:
            old = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
            last = pd.Timestamp(old["date"].max())
            # Re-fetch a small overlap to safely handle revisions/holidays.
            start = (last - pd.Timedelta(days=10)).strftime("%Y%m%d")
            raw = stock.get_market_ohlcv_by_date(start, end.strftime("%Y%m%d"), symbol)
            fresh = _normalize_kr(raw)
            if fresh.empty:
                unchanged += 1
                continue
            merged = pd.concat([old, fresh], ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None)
            merged = merged.drop_duplicates("date", keep="last").sort_values("date").tail(1200).reset_index(drop=True)
            before = last.normalize()
            after = pd.Timestamp(merged["date"].max()).normalize()
            # Always rewrite overlap only when values or date count changed.
            old_cmp = old.tail(1200).reset_index(drop=True)
            same = len(old_cmp) == len(merged) and old_cmp.equals(merged)
            if same:
                unchanged += 1
            else:
                merged.to_csv(path, index=False, encoding="utf-8-sig")
                ok += 1
                print(f"[{i}/{len(paths)}] {symbol}: {before.date()} -> {after.date()}")
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[WARN] {symbol}: {exc}")
        time.sleep(0.06)

    print(f"KR refresh: changed={ok}, unchanged={unchanged}, failed={fail}")
    if fail > max(30, len(paths) // 3):
        raise RuntimeError("Too many KRX refresh failures")


def _fetch_global(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError("empty Yahoo response")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    need = ["date", "open", "high", "low", "close", "volume"]
    out = raw[need].copy()
    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_localize(None)
    return out.dropna(subset=need[1:]).sort_values("date").reset_index(drop=True)


def _technical_score(row: pd.Series) -> float:
    score = 50.0
    score += max(-10, min(10, float(row["ret_5d"]) * 100 * 1.5))
    score += max(-8, min(8, float(row["ret_20d"]) * 100 * 0.5))
    score += max(-6, min(8, (float(row["volume_ratio"]) - 1.0) * 8))
    rsi = float(row["rsi14"])
    if 45 <= rsi <= 65:
        score += 7
    elif rsi >= 75:
        score -= 7
    elif rsi < 35:
        score -= 3
    score += max(-5, min(7, (float(row["near_20d_high"]) - 0.90) * 50))
    return max(0.0, min(100.0, score))


def _global_rows(ticker: str, name: str, country: str, df: pd.DataFrame) -> list[dict]:
    x = add_indicators(df).dropna(subset=["ret_5d", "ret_20d", "volume_ratio", "rsi14", "near_20d_high"]).copy()
    if len(x) < 5:
        return []
    rows: list[dict] = []
    latest = pd.Timestamp(x.iloc[-1]["date"])
    last_close = float(x.iloc[-1]["close"])
    for off in range(4):
        idx = len(x) - 1 - off
        row = x.iloc[idx]
        prev = x.iloc[max(0, idx - 1)]
        close = float(row["close"])
        future = x.iloc[idx + 1:]
        max_high = float(future["high"].max()) if not future.empty else None
        t1, t2 = round(close * 1.10, 2), round(close * 1.20, 2)
        rows.append({
            "symbol": ticker, "name": name, "country": country,
            "date": pd.Timestamp(row["date"]), "latest_date": latest, "day_offset": off,
            "entry_price": close, "last_close": last_close,
            "change_pct": (close / float(prev["close"]) - 1) * 100 if float(prev["close"]) else 0.0,
            "trading_value": float(row["close"] * row["volume"]), "volume_ratio": float(row["volume_ratio"]),
            "rsi14": float(row["rsi14"]), "ret_5d": float(row["ret_5d"]), "near_20d_high": float(row["near_20d_high"]),
            "candidate_score": _technical_score(row), "target_1": t1, "target_2": t2,
            "ret_since": (last_close / close - 1) * 100,
            "hit_1": bool(max_high is not None and max_high >= t1), "hit_2": bool(max_high is not None and max_high >= t2),
            "max_high_after": max_high,
        })
    return rows


def update_global() -> None:
    """Refresh global price cache and precompute today/previous-3-day candidates."""
    GLOBAL_HISTORY.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def one(item):
        ticker, name, country = item
        df = _fetch_global(ticker)
        safe = ticker.replace("/", "_")
        df.to_csv(GLOBAL_HISTORY / f"{safe}.csv", index=False, encoding="utf-8-sig")
        return _global_rows(ticker, name, country, df)

    failures = 0
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(one, item): item for item in GLOBAL_SCAN_UNIVERSE}
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                rows.extend(fut.result())
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"[WARN] {item[0]}: {exc}")

    if not rows:
        raise RuntimeError("No global candidates could be refreshed")
    z = pd.DataFrame(rows)
    out = (z.sort_values(["day_offset", "candidate_score", "trading_value"], ascending=[True, False, False])
             .groupby("day_offset", group_keys=False).head(14).reset_index(drop=True))
    out.to_csv(GLOBAL_CACHE, index=False, encoding="utf-8-sig")
    print(f"Global refresh: candidates={len(out)}, failed={failures}/{len(GLOBAL_SCAN_UNIVERSE)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["kr", "global", "all"], default="all")
    args = ap.parse_args()
    if args.market in {"kr", "all"}:
        update_korea()
    if args.market in {"global", "all"}:
        update_global()


if __name__ == "__main__":
    main()
