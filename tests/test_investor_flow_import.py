"""Date-safe optional investor net-purchase CSV import."""
import pandas as pd
import pytest

from src.investor_flow_import import parse_imported_flows, summarize_imported_flows


def _flows():
    return pd.DataFrame({
        "date": ["2026-11-19", "2026-11-20", "2026-11-23", "2026-11-20"],
        "symbol": ["5930", "005930", "005930", "000660"],
        "foreign_net_krw": [100_000_000, -50_000_000, 999_000_000, 20_000_000],
        "institution_net_krw": [-200_000_000, 100_000_000, 999_000_000, -50_000_000],
    })


def test_imported_investor_values_only_through_signal_date():
    out = summarize_imported_flows(_flows(), ["005930", "000660"], "2026-11-20")
    samsung = out.loc[out["종목코드"] == "005930"].iloc[0]
    assert samsung["수급 기준일"] == "2026-11-20"
    assert samsung["확인 거래일"] == 2
    assert samsung["외국인 순매수(억원)"] == pytest.approx(0.5)
    assert samsung["기관 순매수(억원)"] == pytest.approx(-1.0)


def test_uploaded_flow_missing_symbol_not_faked_as_zero():
    out = summarize_imported_flows(_flows(), ["123456"], "2026-11-20")
    assert out.iloc[0]["수급 기준일"] == "자료 없음"
    assert pd.isna(out.iloc[0]["외국인 순매수(억원)"])


def test_duplicate_date_and_symbol_is_rejected():
    raw = _flows()
    raw.loc[1, "date"] = raw.loc[0, "date"]
    with pytest.raises(ValueError, match="중복"):
        parse_imported_flows(raw)


@pytest.mark.parametrize("corrupt", ["bad_code", "missing_money", "infinite", "missing_column"])
def test_bad_import_never_generates_pseudo_flows(corrupt):
    df = _flows()
    if corrupt == "bad_code":
        df.loc[0, "symbol"] = "USD"
    elif corrupt == "missing_money":
        df.loc[0, "foreign_net_krw"] = None
    elif corrupt == "infinite":
        df.loc[0, "institution_net_krw"] = float("inf")
    else:
        df = df.drop(columns=["institution_net_krw"])
    with pytest.raises(ValueError):
        parse_imported_flows(df)


def test_import_preserves_negative_net_sell_and_date_window():
    result = summarize_imported_flows(_flows(), ["000660"], "2026-11-20", lookback=1)
    assert result.iloc[0]["기관 순매수(억원)"] == pytest.approx(-0.5)
    assert result.iloc[0]["외국인 순매수(억원)"] == pytest.approx(0.2)
