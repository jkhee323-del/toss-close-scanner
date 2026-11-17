"""Live read-only smoke test of KRX investor flow provider (no production writes)."""
from __future__ import annotations

from .investor_flow import fetch_investor_flows, latest_published_flow_date


def main() -> None:
    date = latest_published_flow_date().strftime("%Y-%m-%d")
    result = fetch_investor_flows("005930", date, lookback=20)
    if result is None:
        raise RuntimeError("KRX flow provider returned no usable published data")
    if not result["as_of"] <= date or result["sessions"] < 5:
        raise RuntimeError(f"Invalid KRX flow date/coverage: {result}")
    print(
        "Confirmed KRX flow feed: ticker=005930, "
        f"as_of={result['as_of']}, sessions={result['sessions']}, "
        f"foreign_net_krw={result['foreign_net_krw']}, "
        f"institution_net_krw={result['institution_net_krw']}"
    )


if __name__ == "__main__":
    main()
