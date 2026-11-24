"""Offline public overseas Streamlit smoke: no shared cache writes or fresh-price claims."""
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_overseas_recommendations_page_does_not_rewrite_shared_csv(monkeypatch):
    import FinanceDataReader as fdr
    import yfinance as yf

    # Render a known daily chart deterministically; do not call finance sites.
    source = pd.read_csv(ROOT / "data/global_history/AAPL.csv", parse_dates=["date"]).tail(120)
    raw = source.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }).set_index("date")
    raw.index.name = "Date"
    monkeypatch.setattr(yf, "download", lambda *_args, **_kwargs: raw.copy())
    monkeypatch.setattr(
        fdr, "StockListing",
        lambda *args, **kwargs: pd.DataFrame(columns=["Code", "Name"])
    )

    cache = ROOT / "data/global_candidates.csv"
    original_bytes = cache.read_bytes()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.query_params["m"] = "us"
    at.run(timeout=120)

    assert not at.exception, [str(getattr(exc, "message", exc)) for exc in at.exception]
    assert cache.read_bytes() == original_bytes
    assert all("해외 추천 새로 분석" not in button.label for button in at.button)
