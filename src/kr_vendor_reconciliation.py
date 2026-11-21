"""Read-only reconciliation of archived invalid KRX bars against fresh provider responses.

NEVER edits historical CSVs. Vendor responses may themselves be invalid or on
a different corporate-action adjustment basis; any proposed replacement
requires manual review of price basis, splits and surrounding valid sessions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .auto_update import _normalize_kr
from .data_quality_triage import NEEDED, diagnose_file

FIELDS = ["open", "high", "low", "close", "volume"]


def invalid_row_mask(frame: pd.DataFrame) -> pd.Series:
    if not set(NEEDED).issubset(frame.columns):
        raise ValueError("History missing OHLCV fields")
    values = frame[FIELDS].apply(pd.to_numeric, errors="coerce")
    high_gap = values[["open", "low", "close"]].max(axis=1) - values["high"]
    low_gap = values["low"] - values[["open", "high", "close"]].min(axis=1)
    bad = values.isna().any(axis=1) | values[["open", "high", "low", "close"]].le(0).any(axis=1)
    bad |= values["volume"].lt(0) | high_gap.gt(1.0) | low_gap.gt(1.0)
    return bad


def validate_vendor_row(row: pd.Series) -> bool:
    try:
        a = [float(row[k]) for k in FIELDS]
    except (KeyError, ValueError, TypeError):
        return False
    op, hi, lo, close, vol = a
    return (all(np.isfinite(a)) and min(op, hi, lo, close) > 0 and vol >= 0
            and hi + 1 >= max(op, lo, close) and lo - 1 <= min(op, hi, close))


def compare_history(
    symbol: str,
    original: pd.DataFrame,
    vendor: pd.DataFrame,
) -> list[dict]:
    """Classify invalid rows without ever filling or overwriting a source bar."""
    data = original.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if data["date"].isna().any() or data["date"].duplicated().any():
        return [{"symbol": symbol, "date": "", "status": "invalid_original_date",
                 "original": "", "provider": "", "neighbor_mismatch": ""}]
    data = data.sort_values("date")
    broken = data.loc[invalid_row_mask(data)]
    provider = vendor.copy()
    if not provider.empty and set(NEEDED).issubset(provider.columns):
        provider["date"] = pd.to_datetime(provider["date"], errors="coerce").dt.normalize()
        provider = provider.dropna(subset=["date"]).drop_duplicates("date", keep=False).set_index("date")
    else:
        provider = pd.DataFrame()
    indexed = data.set_index("date")
    findings = []
    for day, row in broken.set_index("date").iterrows():
        base = {k: row[k] for k in FIELDS}
        item = {"symbol": symbol, "date": day.date().isoformat(),
                "original": json.dumps({k: float(v) if pd.notna(v) else None for k, v in base.items()}),
                "provider": "", "neighbor_mismatch": ""}
        if day not in provider.index:
            item["status"] = "provider_missing_session"
        else:
            replacement = provider.loc[day]
            item["provider"] = json.dumps({k: float(replacement[k]) for k in FIELDS})
            if not validate_vendor_row(replacement):
                item["status"] = "provider_also_invalid"
            else:
                # Compare two nearby, *unbroken* original sessions to the same
                # provider price basis. Disagreement flags corporate-action risk.
                neighbors = indexed.loc[
                    (indexed.index >= day - pd.Timedelta(days=10))
                    & (indexed.index <= day + pd.Timedelta(days=10))
                ]
                mismatch = False
                matched_neighbors = 0
                for other_day, other in neighbors.iterrows():
                    if other_day == day or other_day not in provider.index:
                        continue
                    if not validate_vendor_row(other):
                        continue
                    new_other = provider.loc[other_day]
                    if not validate_vendor_row(new_other):
                        continue
                    matched_neighbors += 1
                    orig_close, new_close = float(other["close"]), float(new_other["close"])
                    if abs(orig_close - new_close) > max(1.0, orig_close * 0.01):
                        mismatch = True
                item["neighbor_mismatch"] = mismatch
                item["neighbor_sessions_checked"] = matched_neighbors
                item["status"] = (
                    "price_basis_review_required" if mismatch or matched_neighbors < 2
                    else "vendor_valid_pending_manual_approval"
                )
        findings.append(item)
    return findings


def vendor_is_trustworthy(vendor: pd.DataFrame) -> bool:
    """Reject blank/zero-only feeds (e.g., failed KRX login) before comparison."""
    if vendor is None or vendor.empty or not set(NEEDED).issubset(vendor.columns):
        return False
    valid = vendor.apply(validate_vendor_row, axis=1)
    # At least two genuine neighboring observations and most rows credible.
    return bool(valid.sum() >= 2 and valid.mean() >= 0.80)


def fetch_kr_vendor(symbol: str, first_bad: pd.Timestamp, last_bad: pd.Timestamp) -> pd.DataFrame:
    from pykrx import stock
    start = (first_bad - pd.Timedelta(days=12)).strftime("%Y%m%d")
    end = (last_bad + pd.Timedelta(days=12)).strftime("%Y%m%d")
    return _normalize_kr(stock.get_market_ohlcv_by_date(start, end, symbol))


def audit(
    directory: Path,
    max_symbols: int | None = None,
    fetcher=fetch_kr_vendor,
) -> pd.DataFrame:
    paths = [p for p in sorted(directory.glob("*.csv")) if diagnose_file(p) is not None]
    if max_symbols is not None:
        if max_symbols <= 0:
            raise ValueError("max_symbols must be positive")
        paths = paths[:max_symbols]
    results = []
    for i, path in enumerate(paths, 1):
        symbol = path.stem
        try:
            original = pd.read_csv(path)
            if original["date"].isna().any():
                raise ValueError("invalid original dates")
            invalid = invalid_row_mask(original)
            days = pd.to_datetime(original.loc[invalid, "date"], errors="coerce").dropna()
            if days.empty:
                results.append({"symbol": symbol, "date": "", "status": "no_repairable_dates"})
                continue
            provider = fetcher(symbol, days.min(), days.max())
            if not vendor_is_trustworthy(provider):
                results.append({
                    "symbol": symbol,
                    "date": "",
                    "first_bad_date": str(days.min().date()),
                    "last_bad_date": str(days.max().date()),
                    "affected_rows": int(invalid.sum()),
                    "status": "vendor_unavailable_or_untrusted",
                    "error": "No sufficiently valid independent KRX bars; authentication or provider may be unavailable",
                })
                continue
            results.extend(compare_history(symbol, original, provider))
        except Exception as exc:
            results.append({"symbol": symbol, "date": "", "status": "provider_fetch_or_parse_error",
                            "error": f"{type(exc).__name__}: {str(exc)[:180]}"})
        if i % 10 == 0:
            print(f"Analyzed {i}/{len(paths)} flagged symbols", flush=True)
        time.sleep(0.12)
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("data/history"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/kr_vendor_reconciliation.csv"))
    parser.add_argument("--max-symbols", type=int, default=None)
    args = parser.parse_args()
    report = audit(args.directory, max_symbols=args.max_symbols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    status = report["status"].value_counts().to_dict() if not report.empty else {}
    print(f"Vendor reconciliation: {len(report)} rows, statuses={json.dumps(status, ensure_ascii=False)}")
    print("No production or archived market prices were changed.")
    if report.empty:
        raise SystemExit("No invalid histories found or no diagnostic rows produced")


if __name__ == "__main__":
    main()
