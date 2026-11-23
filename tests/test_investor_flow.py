"""KR investor flow is date-bounded and missing data is not silently zeroed."""
import pandas as pd
from src.investor_flow import latest_published_flow_date, summarize_investor_flows


def test_published_flow_cutoff():
    assert latest_published_flow_date(pd.Timestamp("2026-11-16 17:59", tz="Asia/Seoul")) == pd.Timestamp("2026-11-15")
    assert latest_published_flow_date(pd.Timestamp("2026-11-16 18:00", tz="Asia/Seoul")) == pd.Timestamp("2026-11-16")


def test_flow_excludes_future_observations():
    df = pd.DataFrame({
        "기관합계": [100, 100, 100, 100, 100, 999999],
        "외국인합계": [200, 200, 200, 200, 200, 999999],
    }, index=pd.bdate_range("2026-11-06", periods=6))
    cutoff = df.index[-2]
    out = summarize_investor_flows(df, cutoff)
    assert out["sessions"] == 5
    assert out["institution_net_krw"] == 500
    assert out["foreign_net_krw"] == 1000
    assert out["as_of"] == cutoff.strftime("%Y-%m-%d")


def test_flow_missing_data_is_unavailable():
    assert summarize_investor_flows(pd.DataFrame(), pd.Timestamp("2026-11-16")) is None
    partial = pd.DataFrame({"기관합계": [1] * 8}, index=pd.bdate_range("2026-11-01", periods=8))
    assert summarize_investor_flows(partial, pd.Timestamp("2026-11-16")) is None


def test_live_flow_fails_explicitly_without_secrets(monkeypatch):
    import pytest
    from src.investor_flow import fetch_investor_flows
    monkeypatch.delenv("KRX_ID", raising=False)
    monkeypatch.delenv("KRX_PW", raising=False)
    with pytest.raises(RuntimeError, match="KRX_ID and KRX_PW"):
        fetch_investor_flows("005930", "2026-11-13")
