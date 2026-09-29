from __future__ import annotations
import pandas as pd
import FinanceDataReader as fdr
from .indicators import add_indicators
from .scoring import score_row, display_probability

def scan_symbol(symbol: str, name: str, df: pd.DataFrame) -> dict | None:
    if len(df) < 70:
        return None

    x = add_indicators(df)
    row = x.iloc[-1]

    required = [
        "ret_5d", "ma5", "ma20", "ma60",
        "volume_ratio", "close_position",
        "near_20d_high", "rsi14"
    ]
    if row[required].isna().any():
        return None

    score, components = score_row(row)

    return {
        "symbol": symbol,
        "name": name,
        "date": row["date"].date().isoformat(),
        "close": float(row["close"]),
        "change_1d": round(float(row["ret_1d"]) * 100, 2),
        "volume_ratio": round(float(row["volume_ratio"]), 2),
        "close_position": round(float(row["close_position"]) * 100, 1),
        "near_20d_high": round(float(row["near_20d_high"]) * 100, 1),
        "rsi14": round(float(row["rsi14"]), 1),
        "score": score,
        "display_probability": display_probability(score),

        **{f"score_{k}": v for k, v in components.items()},
    }


def scan_universe(client, universe: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    results = []

    for item in universe.itertuples(index=False):
        try:
            df = client.get_daily_history(item.symbol, bars=200)
            result = scan_symbol(item.symbol, item.name, df)
            if result:
                results.append(result)
        except Exception as exc:
            print(f"[WARN] {item.symbol} {item.name}: {exc}")

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    return out.sort_values(
        ["score", "volume_ratio"], ascending=[False, False]
    ).head(top_n).reset_index(drop=True)

def get_all_korean_stocks() -> pd.DataFrame:
    """KOSPI + KOSDAQ 전체 종목 목록을 가져옵니다."""

    kospi = fdr.StockListing("KOSPI")[["Code", "Name"]].copy()
    kosdaq = fdr.StockListing("KOSDAQ")[["Code", "Name"]].copy()

    kospi["market"] = "KOSPI"
    kosdaq["market"] = "KOSDAQ"

    universe = pd.concat([kospi, kosdaq], ignore_index=True)

    universe = universe.rename(
        columns={
            "Code": "symbol",
            "Name": "name",
        }
    )

    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)

    return universe[["symbol", "name", "market"]]