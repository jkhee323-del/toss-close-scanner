"""Validate research backfill never modifies the production price directory."""
import numpy as np
import pandas as pd

from src.us_research_backfill import download_research_histories


def stock_data(n=800):
    close = np.linspace(100, 120, n)
    return pd.DataFrame({
        "date": pd.bdate_range("2023-01-02", periods=n),
        "open": close, "high": close + 1,
        "low": close - 1, "close": close,
        "volume": np.full(n, 100000),
    })


def test_research_download_isolated_and_requires_us_catalog(tmp_path):
    requested = []
    def get(ticker, *, country, period):
        requested.append((ticker, country, period))
        return stock_data()
    dest = tmp_path / "research"
    result = download_research_histories(
        dest, tickers=("AAPL", "AAPL", "7203.T", "MSFT"),
        pause_seconds=0, fetcher=get,
    )
    assert requested == [("AAPL", "미국", "5y"), ("MSFT", "미국", "5y")]
    assert result["written"] == ["AAPL", "MSFT"]
    assert "7203.T" in result["failed"]
    assert sorted(x.name for x in dest.iterdir()) == ["AAPL.csv", "MSFT.csv"]


def test_short_series_fails_without_writing(tmp_path):
    result = download_research_histories(
        tmp_path / "study", tickers=("AAPL",), pause_seconds=0,
        fetcher=lambda symbol, *, country, period: stock_data(100),
    )
    assert "AAPL" in result["failed"]
    assert not (tmp_path / "study").exists()
