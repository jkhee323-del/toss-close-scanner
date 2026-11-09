"""Read-only USD/US-equity price cache quality audit.

Detects stale, short, non-monotonic, invalid or possible duplicate series.
No live prices, inferred close prices, or automatic cache deletions.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .global_market_safety import validate_ohlcv
from .search import US_NAME_MAP

HORIZON_MIN_ROWS = {"1m": 300, "3m": 400, "6m": 550, "12m": 800}


def audit_history(
    folder: Path, *, as_of: date | None = None,
    required_days_old: int = 7,
) -> dict:
    """Emit per-US-ticker issues, usable rows and coverage thresholds.

    7 calendar days is a warning heuristic, not an exchange trading calendar.
    """
    now = pd.Timestamp(as_of or date.today())
    by_ticker = {}
    fingerprints = {}
    for path in sorted(Path(folder).glob("*.csv")):
        ticker = path.stem
        if US_NAME_MAP.get(ticker, ("", ""))[1] != "미국":
            continue
        item = {"status": "INVALID", "issues": [], "rows": 0}
        by_ticker[ticker] = item
        try:
            raw = pd.read_csv(path)
            required = {"date", "open", "high", "low", "close", "volume"}
            if not required.issubset(raw):
                raise ValueError("missing OHLCV columns")
            parsed_dates = pd.to_datetime(raw["date"], errors="coerce")
            if parsed_dates.isna().any():
                raise ValueError("unparseable dates")
            if parsed_dates.duplicated().any():
                item["issues"].append("DUPLICATE_SESSION_DATE")
            if not parsed_dates.is_monotonic_increasing:
                item["issues"].append("UNSORTED_SESSION_DATE")
            frame = validate_ohlcv(raw, min_rows=70)
            item["rows"] = len(frame)
            last = frame["date"].max()
            first = frame["date"].min()
            item["first_date"] = str(first.date())
            item["last_date"] = str(last.date())
            if last > now:
                item["issues"].append("FUTURE_BAR")
            if (now - last).days > required_days_old:
                item["issues"].append("STALE_WARNING")
            if frame["volume"].eq(0).sum():
                item["issues"].append("ZERO_VOLUME_ROWS")
            item["zero_volume_rows"] = int(frame["volume"].eq(0).sum())
            item["sufficient_rows"] = {
                h: len(frame) >= minrows for h, minrows in HORIZON_MIN_ROWS.items()
            }
            # Identical entire close/date/volume series for two tickers
            # is suspicious but requires investigation, not deletion.
            import hashlib

            packed = frame[["date", "close", "volume"]].to_csv(index=False).encode()
            digest = hashlib.sha256(packed).hexdigest()
            fingerprints.setdefault(digest, []).append(ticker)
            item["status"] = "OK" if not item["issues"] else "WARNING"
        except (ValueError, OSError, TypeError, KeyError) as exc:
            item["issues"].append(str(exc))
    duplicates = [v for v in fingerprints.values() if len(v) > 1]
    for group in duplicates:
        for ticker in group:
            by_ticker[ticker]["issues"].append("IDENTICAL_SERIES_ACROSS_TICKERS")
            by_ticker[ticker]["status"] = "WARNING"
    return {
        "audit_as_of": now.date().isoformat(),
        "us_tickers": len(by_ticker),
        "valid_histories": sum(row["status"] != "INVALID" for row in by_ticker.values()),
        "invalid_histories": sum(row["status"] == "INVALID" for row in by_ticker.values()),
        "duplicate_groups": duplicates,
        "adequate_full_history": {
            h: sum(row.get("sufficient_rows", {}).get(h, False) for row in by_ticker.values())
            for h in HORIZON_MIN_ROWS
        },
        "by_ticker": by_ticker,
        "warning": "Historical capture time, corporate actions and survivorship bias cannot be audited from daily CSV alone.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=Path("data/global_history"))
    parser.add_argument("--output", type=Path, default=Path("reports/us_history_audit.json"))
    args = parser.parse_args()
    result = audit_history(args.history)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("US price histories:", result["us_tickers"])
    print("Valid:", result["valid_histories"], "Invalid:", result["invalid_histories"])
    print("Enough rows by horizon:", result["adequate_full_history"])
    print("Report:", args.output)


if __name__ == "__main__":
    main()
