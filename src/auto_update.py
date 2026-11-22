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
from .kr_naver_provider import fetch_naver_ohlcv
from .kr_recent_refresh_guard import verified_new_candles

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


def _latest_complete_kr_date(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """KRX daily bars are final only after the closing session settles."""
    local_now = pd.Timestamp.now(tz="Asia/Seoul") if now is None else pd.Timestamp(now)
    if local_now.tzinfo is None:
        local_now = local_now.tz_localize("Asia/Seoul")
    else:
        local_now = local_now.tz_convert("Asia/Seoul")
    cutoff = local_now.normalize().tz_localize(None)
    if (local_now.hour, local_now.minute) < (16, 10):
        cutoff -= pd.Timedelta(days=1)
    return cutoff


def update_korea() -> None:
    """Incrementally append the latest KRX trading days to every stored symbol."""
    KR_HISTORY.mkdir(parents=True, exist_ok=True)
    paths = sorted(KR_HISTORY.glob("*.csv"))
    if not paths:
        raise RuntimeError("data/history has no CSV files")

    cutoff = _latest_complete_kr_date()
    end = cutoff + pd.Timedelta(days=1)
    ok = fail = unchanged = 0
    latest_saved_sessions: list[pd.Timestamp] = []
    for i, path in enumerate(paths, 1):
        symbol = path.stem.zfill(6)
        try:
            original = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
            old = original.loc[original["date"].dt.normalize() <= cutoff].copy()
            if old.empty:
                raise ValueError("No completed KRX sessions in stored history")
            last = pd.Timestamp(old["date"].max())
            # Verify the provider against actual saved closing sessions BEFORE
            # accepting any update. Append new sessions only; do not overwrite
            # stored historical adjustments using unaudited vendor revisions.
            start_day = last - pd.Timedelta(days=21)
            source = "KRX"
            try:
                raw = stock.get_market_ohlcv_by_date(
                    start_day.strftime("%Y%m%d"), end.strftime("%Y%m%d"), symbol
                )
                candidate = _normalize_kr(raw)
                additions = verified_new_candles(old, candidate, cutoff)
            except Exception as krx_error:
                # Authorized-independent NAVER source when KRX login is absent.
                # No data is accepted without price-basis overlap verification.
                source = "NAVER"
                try:
                    candidate = fetch_naver_ohlcv(
                        symbol, start_day.date().isoformat(), end.date().isoformat()
                    )
                    additions = verified_new_candles(old, candidate, cutoff)
                except Exception as alt_error:
                    raise RuntimeError(
                        f"KRX unavailable ({type(krx_error).__name__}); "
                        f"NAVER unverified ({type(alt_error).__name__}: {alt_error})"
                    ) from alt_error
            merged = pd.concat([old, additions], ignore_index=True) if not additions.empty else old
            merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None)
            merged = merged.drop_duplicates("date", keep="last").sort_values("date").tail(1200).reset_index(drop=True)
            before = pd.Timestamp(original["date"].max()).normalize()
            after = pd.Timestamp(merged["date"].max()).normalize()
            latest_saved_sessions.append(after)
            old_cmp = original.tail(1200).reset_index(drop=True)
            if len(old_cmp) == len(merged) and old_cmp.equals(merged):
                unchanged += 1
            else:
                merged.to_csv(path, index=False, encoding="utf-8-sig")
                ok += 1
                print(f"[{i}/{len(paths)}] {symbol}: {before.date()} -> {after.date()} ({source})")
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[WARN] {symbol}: {exc}")
        time.sleep(0.06)

    print(f"KR refresh: changed={ok}, unchanged={unchanged}, failed={fail}")
    if fail > max(30, len(paths) // 3):
        raise RuntimeError("Too many KRX refresh failures")
    # A silently empty/unauthenticated provider must not be counted as an
    # ongoing success indefinitely. Allow one business-day lag for holidays
    # or data settlement, and require broad universe coverage.
    freshness_floor = cutoff - pd.offsets.BDay(1)
    recent_count = sum(day >= freshness_floor for day in latest_saved_sessions)
    coverage = recent_count / len(paths)
    print(f"KR refresh freshness: latest_cutoff={cutoff.date()}, "
          f"minimum_accepted={freshness_floor.date()}, "
          f"recent_symbols={recent_count}/{len(paths)} ({coverage:.1%})")
    if coverage < 0.70:
        raise RuntimeError(
            f"KRX provider may be unavailable: only {coverage:.1%} of symbols "
            f"have a session at or after {freshness_floor.date()}"
        )


def _fetch_global(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError("empty Yahoo response")

    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(-1):
            raw = raw.xs(ticker, axis=1, level=-1)
        elif len(raw.columns.get_level_values(-1).unique()) == 1:
            raw = raw.droplevel(-1, axis=1)
        else:
            raise ValueError(f"종목 구분 불가: {ticker}")

    if raw.columns.duplicated().any():
        raise ValueError(f"중복된 가격 열 발견: {ticker}")

    raw = raw.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    need = ["date", "open", "high", "low", "close", "volume"]
    out = raw[need].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out.dropna(subset=need[1:]).sort_values("date").reset_index(drop=True)


def _validate_global_history(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Reject broken or stale provider responses before overwriting cached prices."""
    needed = ["date", "open", "high", "low", "close", "volume"]
    if df is None or not all(c in df.columns for c in needed):
        raise ValueError(f"{ticker}: missing required OHLCV columns")
    out = df[needed].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for col in needed[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[needed].isna().any().any():
        raise ValueError(f"{ticker}: missing OHLCV values")
    out = out.sort_values("date")
    if out["date"].duplicated().any():
        raise ValueError(f"{ticker}: duplicate session dates")
    if len(out) < 70:
        raise ValueError(f"{ticker}: too few price observations")
    if (out[["open", "high", "low", "close"]] <= 0).any().any() or (out["volume"] < 0).any():
        raise ValueError(f"{ticker}: invalid prices or volume")
    if ((out["high"] < out[["open", "close", "low"]].max(axis=1)) |
        (out["low"] > out[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError(f"{ticker}: inconsistent OHLC")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    latest = out["date"].iloc[-1]
    if latest > now + pd.Timedelta(days=1):
        raise ValueError(f"{ticker}: future-dated prices")
    if now - latest > pd.Timedelta(days=7):
        raise ValueError(f"{ticker}: stale prices ending {latest.date()}")
    return out.reset_index(drop=True)


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
        df = _validate_global_history(_fetch_global(ticker), ticker)
        candidates = _global_rows(ticker, name, country, df)
        if len(candidates) != 4:
            raise ValueError(f"{ticker}: incomplete candidate history")
        safe = ticker.replace("/", "_")
        df.to_csv(GLOBAL_HISTORY / f"{safe}.csv", index=False, encoding="utf-8-sig")
        return candidates

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

    expected = len(GLOBAL_SCAN_UNIVERSE)
    succeeded = expected - failures
    # Do not replace a previously usable candidate cache with a partial outage.
    if not rows or succeeded < max(1, int(expected * 0.7)):
        raise RuntimeError(f"Global feed incomplete: {succeeded}/{expected} symbols refreshed")
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
