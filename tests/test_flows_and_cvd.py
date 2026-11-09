"""Investor-flow and tick CVD contract tests with no live-provider dependency."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.kr_investor_flows import (
    FlowDataError, fetch_korean_net_flows, latest_flow_snapshot,
    save_korean_flows, validate_flow_frame,
)
from src.real_cvd import CVDDataError, calculate_cvd, cvd_availability


def sample_flows():
    return pd.DataFrame({
        "기관합계": [1000, -200],
        "외국인합계": [3000, -500],
    }, index=pd.to_datetime(["2026-10-29", "2026-10-30"]))


def ticks():
    return pd.DataFrame({
        "timestamp": ["2026-11-02T09:00:00+09:00",
                      "2026-11-02T09:00:01+09:00",
                      "2026-11-02T09:00:02+09:00"],
        "price": [100, 101, 100],
        "size": [10, 3, 5],
        "aggressor_side": ["buy", "buy", "sell"],
    })


def test_kr_net_amount_signs_and_source_contract():
    raw = sample_flows()
    def provider(start, end, ticker):
        assert (start, end, ticker) == ("20261029", "20261030", "005930")
        return raw
    x = fetch_korean_net_flows(
        "005930", from_date=date(2026, 10, 29), to_date=date(2026, 10, 30),
        provider=provider,
    )
    assert x["institutional_net_krw"].tolist() == [1000, -200]
    assert x["foreign_net_krw"].tolist() == [3000, -500]


def test_flow_cache_merge_and_missing_is_not_zero(tmp_path):
    assert latest_flow_snapshot("005930", directory=tmp_path) is None
    first = validate_flow_frame(sample_flows())
    save_korean_flows("005930", first, directory=tmp_path)
    snap = latest_flow_snapshot("005930", directory=tmp_path)
    assert snap["foreign_net_krw"] == -500
    assert snap["estimated"] is False
    replacement = first.tail(1).copy()
    replacement.loc[replacement.index[-1], "foreign_net_krw"] = -1000
    save_korean_flows("005930", replacement, directory=tmp_path)
    assert latest_flow_snapshot("005930", directory=tmp_path)["foreign_net_krw"] == -1000
    assert len(pd.read_csv(tmp_path / "005930.csv")) == 2


def test_missing_or_bad_provider_flow_is_rejected():
    with pytest.raises(FlowDataError, match="missing"):
        validate_flow_frame(pd.DataFrame({"기관합계": [1]}, index=pd.to_datetime(["2026-10-30"])))
    wrong = sample_flows()
    wrong.loc[wrong.index[-1], "외국인합계"] = np.nan
    with pytest.raises(FlowDataError, match="non-numeric"):
        validate_flow_frame(wrong)
    with pytest.raises(FlowDataError, match="ticker"):
        fetch_korean_net_flows("AAPL", from_date=date(2026, 10, 1), to_date=date(2026, 10, 2))


def test_cvd_is_signed_trade_volume_only():
    result = calculate_cvd(ticks())
    assert result["signed_volume"].tolist() == [10, 3, -5]
    assert result["cvd"].tolist() == [10, 13, 8]
    assert cvd_availability(ticks())["value"] == 8.0


def test_daily_ohlcv_is_not_trade_cvd():
    daily = pd.DataFrame({"date": ["2026-11-02"], "close": [100], "volume": [12345]})
    assert cvd_availability(daily)["value"] is None
    with pytest.raises(CVDDataError, match="verified trade ticks"):
        calculate_cvd(daily)


def test_cvd_rejects_missing_trade_side_and_naive_time():
    bad = ticks().drop(columns=["aggressor_side"])
    with pytest.raises(CVDDataError):
        calculate_cvd(bad)
    bad = ticks()
    bad.loc[0, "aggressor_side"] = "unknown"
    with pytest.raises(CVDDataError, match="aggressor_side"):
        calculate_cvd(bad)
    bad = ticks()
    bad.loc[0, "timestamp"] = "2026-11-02T09:00:00"
    with pytest.raises(CVDDataError, match="timezone-aware"):
        calculate_cvd(bad)


def test_cvd_daily_reset():
    t = ticks()
    next_day = t.iloc[[0]].copy()
    next_day["timestamp"] = ["2026-11-03T09:00:00+09:00"]
    combined = pd.concat([t, next_day], ignore_index=True)
    result = calculate_cvd(combined, reset_daily=True)
    assert result["cvd"].tolist() == [10, 13, 8, 10]
