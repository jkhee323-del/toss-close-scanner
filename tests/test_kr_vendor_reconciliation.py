"""Read-only historical Korean vendor reconciliation guards."""
from __future__ import annotations

import pandas as pd

from src.kr_vendor_reconciliation import (
    audit, compare_history, invalid_row_mask, validate_vendor_row,
)


def _bars():
    days = pd.bdate_range("2023-05-01", periods=26)
    original = pd.DataFrame({
        "date": days,
        "open": [100.0] * 26,
        "high": [105.0] * 26,
        "low": [98.0] * 26,
        "close": [102.0] * 26,
        "volume": [10000] * 26,
    })
    return original


def test_valid_provider_can_be_proposed_but_never_rewrites_original(tmp_path):
    good = _bars()
    corrupted = good.copy()
    corrupted.loc[10, ["open", "high", "low", "close"]] = 0
    original_file = tmp_path / "000100.csv"
    corrupted.to_csv(original_file, index=False)
    before = original_file.read_bytes()
    result = audit(tmp_path, fetcher=lambda symbol, start, end: good.copy())
    assert len(result) == 1
    assert result.iloc[0]["status"] == "vendor_valid_pending_manual_approval"
    assert result.iloc[0]["neighbor_sessions_checked"] >= 2
    assert original_file.read_bytes() == before


def test_price_basis_divergence_blocks_automatic_replacement():
    original = _bars()
    original.loc[10, ["open", "high", "low", "close"]] = 0
    vendor = _bars()
    vendor.loc[9, ["open", "high", "low", "close"]] = [200, 205, 198, 202]
    result = compare_history("000100", original, vendor)
    assert result[0]["status"] == "price_basis_review_required"
    assert bool(result[0]["neighbor_mismatch"])


def test_provider_zero_prices_must_not_be_accepted():
    original = _bars()
    original.loc[10, ["open", "high", "low", "close"]] = 0
    vendor = _bars()
    vendor.loc[10, ["open", "high", "low", "close"]] = 0
    result = compare_history("000100", original, vendor)
    assert result[0]["status"] == "provider_also_invalid"
    assert not validate_vendor_row(vendor.iloc[10])


def test_missing_vendor_session_is_recorded():
    original = _bars()
    original.loc[10, ["open", "high", "low", "close"]] = 0
    vendor = _bars().drop(index=10)
    result = compare_history("000100", original, vendor)
    assert result[0]["status"] == "provider_missing_session"


def test_negative_volume_and_invalid_high_are_detected():
    data = _bars()
    data.loc[2, "volume"] = -1
    data.loc[3, "high"] = 80
    assert invalid_row_mask(data).sum() == 2
