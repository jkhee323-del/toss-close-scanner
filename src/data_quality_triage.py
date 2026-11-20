"""Read-only, actionable triage report for historical OHLCV source repairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data_integrity_audit import check_ohlcv_quality, recent_valid_segment

NEEDED = ["date", "open", "high", "low", "close", "volume"]


def diagnose_file(path: Path) -> dict | None:
    """Describe bad source rows; never amend historical prices."""
    try:
        raw = pd.read_csv(path)
        problem, _ = check_ohlcv_quality(raw)
        if problem is None:
            return None
        result = {
            "file": path.name,
            "problem": problem,
            "bad_row_count": 0,
            "first_bad_date": "",
            "last_bad_date": "",
            "issue_types": "",
            "safe_short_90_sessions": False,
            "safe_long_241_sessions": False,
        }
        short = recent_valid_segment(raw, minimum_rows=90)
        long = recent_valid_segment(raw, minimum_rows=241)
        result["safe_short_90_sessions"] = not short.empty
        result["safe_long_241_sessions"] = not long.empty
        if not set(NEEDED).issubset(raw.columns):
            result["issue_types"] = "missing columns"
            return result
        data = raw[NEEDED].copy()
        dates = pd.to_datetime(data["date"], errors="coerce", utc=True)
        vals = data[NEEDED[1:]].apply(pd.to_numeric, errors="coerce")
        positive = vals[["open", "high", "low", "close"]].gt(0).all(axis=1)
        volume_ok = vals["volume"].ge(0)
        numeric_ok = vals.notna().all(axis=1)
        high_ok = vals["high"] + 1 >= vals[["open", "low", "close"]].max(axis=1)
        low_ok = vals["low"] - 1 <= vals[["open", "high", "close"]].min(axis=1)
        broken = dates.isna() | ~positive | ~volume_ok | ~numeric_ok | ~high_ok | ~low_ok
        issues = []
        if (dates.isna()).any():
            issues.append("invalid date")
        if (dates.duplicated()).any() or not dates.is_monotonic_increasing:
            issues.append("date order/duplicate")
        if (~positive).any():
            issues.append("nonpositive OHLC")
        if (~volume_ok).any():
            issues.append("negative volume")
        if (~numeric_ok).any():
            issues.append("missing/nonfinite number")
        if (~(high_ok & low_ok)).any():
            issues.append("OHLC contradiction")
        if set(raw.columns) != set(NEEDED) or len(raw.columns) != len(NEEDED):
            issues.append("unexpected/mixed ticker columns")
        if not issues:
            issues.append(problem)
        result["issue_types"] = "; ".join(issues)
        result["bad_row_count"] = int(broken.sum())
        bad_dates = dates[broken].dropna()
        if not bad_dates.empty:
            result["first_bad_date"] = bad_dates.min().date().isoformat()
            result["last_bad_date"] = bad_dates.max().date().isoformat()
        return result
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as exc:
        return {
            "file": path.name, "problem": f"read failure: {type(exc).__name__}",
            "bad_row_count": -1, "first_bad_date": "", "last_bad_date": "",
            "issue_types": "unreadable", "safe_short_90_sessions": False,
            "safe_long_241_sessions": False,
        }


def run(directory: Path) -> pd.DataFrame:
    reports = [diagnose_file(path) for path in sorted(directory.glob("*.csv"))]
    return pd.DataFrame(
        [report for report in reports if report is not None],
        columns=[
            "file", "problem", "bad_row_count", "first_bad_date",
            "last_bad_date", "issue_types", "safe_short_90_sessions",
            "safe_long_241_sessions",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("data/history"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/kr_source_quality_triage.csv"))
    args = parser.parse_args()
    report = run(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    stats = {
        "flagged": len(report),
        "recent_short_usable": int(report["safe_short_90_sessions"].sum()),
        "recent_long_usable": int(report["safe_long_241_sessions"].sum()),
        "cannot_scan_short": int((~report["safe_short_90_sessions"]).sum()),
        "source_file": str(args.output),
    }
    print(json.dumps(stats, ensure_ascii=False))
    print(report.to_string(index=False))
    print("No historical source rows modified.")


if __name__ == "__main__":
    main()
