"""Read-only health gate for committed KRX and overseas OHLCV caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def check_ohlcv_quality(raw: pd.DataFrame, *, rounding_tolerance: float = 1.0) -> tuple[str | None, int]:
    """Return hard error and count of harmless 1-unit adjusted-price roundoffs.

    Never adjust, rewrite or invent a price. Suspended zero-price candles fail.
    """
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(raw.columns):
        return "missing OHLCV columns", 0
    dates = pd.to_datetime(raw["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    numeric = raw[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    if dates.empty or dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        return "invalid, duplicate or unsorted dates", 0
    if numeric.isna().any().any() or (numeric[["open", "high", "low", "close"]] <= 0).any().any() or (numeric["volume"] < 0).any():
        return "invalid OHLCV numbers", 0
    upper = numeric[["open", "low", "close"]].max(axis=1) - numeric["high"]
    lower = numeric["low"] - numeric[["open", "high", "close"]].min(axis=1)
    if (upper > rounding_tolerance).any() or (lower > rounding_tolerance).any():
        return "inconsistent OHLC", 0
    small_mismatches = int(((upper > 0) | (lower > 0)).sum())
    return None, small_mismatches


def inspect_market(history_dir: Path, *, min_recent_share: float = 0.80, max_lag_business_days: int = 5) -> dict:
    """Inspect all cached histories, including schema, chronology and recent coverage."""
    files = sorted(history_dir.glob("*.csv"))
    if not files:
        return {"status": "fail", "total": 0, "errors": ["no CSV histories found"]}
    issues = []
    last_dates = []
    rounding_adjustment_rows = 0
    for path in files:
        try:
            raw = pd.read_csv(path)
            reason, roundoffs = check_ohlcv_quality(raw)
            if reason is not None:
                issues.append(f"{path.name}: {reason}")
                continue
            rounding_adjustment_rows += roundoffs
            dates = pd.to_datetime(raw["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
            last_dates.append((path.name, dates.iloc[-1]))
        except (OSError, ValueError, TypeError, pd.errors.ParserError) as exc:
            issues.append(f"{path.name}: unreadable ({type(exc).__name__})")
    if not last_dates:
        return {"status": "fail", "total": len(files), "valid": 0, "errors": issues[:20], "total_errors": len(issues)}
    # Date freshness measured against market's own latest observed session,
    # avoiding time-of-day and timezone assumptions about exchanges.
    latest = max(date for _, date in last_dates)
    threshold = latest - pd.offsets.BDay(max_lag_business_days)
    lagged = [name for name, date in last_dates if date < threshold]
    recent_share = (len(last_dates) - len(lagged)) / len(files)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if latest > now + pd.Timedelta(days=1):
        issues.append(f"Future-dated cache: {latest.date()}")
    if recent_share < min_recent_share:
        issues.append(f"Only {recent_share:.1%} recent histories against {latest.date()}")
    return {
        "status": "pass" if not issues else "fail",
        "total": len(files),
        "valid": len(last_dates),
        "minor_roundoff_rows": rounding_adjustment_rows,
        "latest": latest.strftime("%Y-%m-%d"),
        "recent_share": round(recent_share, 4),
        "lagged_count": len(lagged),
        "lagged_examples": lagged[:10],
        "errors": issues[:20],
        "total_errors": len(issues),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/data_integrity_audit.json"))
    args = parser.parse_args()
    results = {
        "korea": inspect_market(args.root / "history"),
        "global": inspect_market(args.root / "global_history"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if any(market["status"] != "pass" for market in results.values()):
        raise SystemExit("At least one data cache failed integrity audit")


if __name__ == "__main__":
    main()
