"""Read-only production release gate.

This module NEVER infers approval from unit tests, historical hit rates, or
the presence of model files. It checks dated source data and explicitly
provided evidence. Missing proof blocks release; lab CI expects that block.

Evidence is a review checklist, not an authentication scheme. Independent
manual review of every external source and deployed site is still required.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import pandas as pd

REQUIRED_PROOFS = (
    "pc_version_reconciled",
    "provider_end_to_end_verified",
    "kr_investor_flow_live_verified",
    "real_cvd_feed_verified",
    "model_walkforward_independent_passed",
    "next_day_costs_independent_passed",
    "deployed_browser_verified",
    "scheduled_refresh_end_to_end_verified",
)


def latest_csv_date(directory: Path) -> dict:
    paths = sorted(directory.glob("*.csv")) if directory.is_dir() else []
    dates = []
    issues = []
    for path in paths:
        try:
            date_column = pd.read_csv(path, usecols=["date"])["date"]
            parsed = pd.to_datetime(date_column, errors="coerce")
            if parsed.isna().any() or parsed.empty:
                raise ValueError("missing or invalid date")
            if isinstance(parsed.dtype, pd.DatetimeTZDtype):
                raise ValueError("timezone-aware session labels")
            dates.append(pd.Timestamp(parsed.max()).date())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"{path.name}: {exc}")
    return {
        "files": len(paths),
        "valid_files": len(dates),
        "latest_date": max(dates).isoformat() if dates else None,
        "invalid_files": issues,
    }


def check_release(
    root: Path, *, as_of: date, evidence: dict | None = None,
    freshness_days: int = 7,
) -> dict:
    if freshness_days < 0:
        raise ValueError("freshness_days must be nonnegative")
    root = Path(root)
    proof = evidence if isinstance(evidence, dict) else {}
    checks = {}
    needed = ("app.py", "requirements.txt", "src", "tests", "data")
    checks["project_structure"] = all((root / p).exists() for p in needed)
    caches = {}
    for code, relpath in (
        ("kr", "data/history"),
        ("global", "data/global_history"),
    ):
        record = latest_csv_date(root / relpath)
        caches[code] = record
        last = date.fromisoformat(record["latest_date"]) if record["latest_date"] else None
        checks[f"{code}_cache_usable"] = (
            record["files"] > 0 and record["files"] == record["valid_files"]
            and not record["invalid_files"]
        )
        checks[f"{code}_cache_fresh"] = (
            last is not None and 0 <= (as_of - last).days <= freshness_days
        )
    for key in REQUIRED_PROOFS:
        checks[key] = proof.get(key) is True
    blockers = [key for key, success in checks.items() if not success]
    return {
        "release_decision": "BLOCKED" if blockers else "REQUIRES_HUMAN_RELEASE_REVIEW",
        "evaluated_on": as_of.isoformat(),
        "checks": checks,
        "blockers": blockers,
        "cache": caches,
        "note": (
            "A green checklist is not automatic production certification. "
            "Exchange holidays, full vendor integrity, current UI and "
            "investment outcomes require independent verification."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path("."))
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--evidence", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=Path("release_preflight.json"))
    ap.add_argument("--freshness-days", type=int, default=7)
    args = ap.parse_args()

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    evidence = None
    if args.evidence is not None:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    results = check_release(
        args.project, as_of=as_of, evidence=evidence,
        freshness_days=args.freshness_days,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Release decision: {results['release_decision']}")
    print("Blocking checks:", ", ".join(results["blockers"]) or "(none)")
    print(f"Report: {args.output}")
    if results["blockers"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
