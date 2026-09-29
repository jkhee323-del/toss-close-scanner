from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .indicators import add_indicators
from .model import FEATURES, make_training_data


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )


def train_dual_models(data_dir: str | Path = "data/history"):
    data = make_training_data(data_dir).copy()
    data["strong_target"] = (data["next_return"] >= 0.03).astype(int)

    up_model = _new_model()
    strong_model = _new_model()
    up_model.fit(data[FEATURES], data["target"])
    strong_model.fit(data[FEATURES], data["strong_target"])
    return up_model, strong_model, data


def latest_feature_rows(data_dir: str | Path = "data/history") -> pd.DataFrame:
    data_dir = Path(data_dir)
    rows = []
    for path in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            x = add_indicators(df)
            if x.empty:
                continue
            row = x.iloc[-1].copy()
            if row[FEATURES + ["trading_value"]].isna().any():
                continue
            row["symbol"] = path.stem
            rows.append(row)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def load_name_map(path: str | Path = "data/universe.csv") -> dict[str, str]:
    names: dict[str, str] = {}
    path = Path(path)
    if path.exists():
        try:
            u = pd.read_csv(path, dtype={"symbol": str})
            u["symbol"] = u["symbol"].str.zfill(6)
            names.update(dict(zip(u["symbol"], u["name"])))
        except Exception:
            pass

    # 인터넷 연결이 가능하면 현재 KRX 목록으로 종목명을 보완합니다.
    try:
        import FinanceDataReader as fdr
        krx = fdr.StockListing("KRX")[["Code", "Name"]].copy()
        krx["Code"] = krx["Code"].astype(str).str.zfill(6)
        names.update(dict(zip(krx["Code"], krx["Name"])))
    except Exception:
        pass
    return names


def predict_latest(
    data_dir: str | Path = "data/history",
    top_n: int = 20,
    min_trading_value: float = 10_000_000_000,
    up_weight: float = 0.50,
    strong_weight: float = 0.50,
) -> tuple[pd.DataFrame, dict]:
    """Return latest scanner results.

    v3.1 ships with a precomputed latest-feature snapshot so the dashboard opens
    immediately instead of retraining two models on 560k+ rows at every launch.
    The historical CSVs and training code are still included for later refreshes.
    """
    snapshot_path = Path("data/latest_snapshot.csv")
    if snapshot_path.exists():
        latest = pd.read_csv(snapshot_path, dtype={"symbol": str}, parse_dates=["date"])
        latest["symbol"] = latest["symbol"].astype(str).str.zfill(6)
        latest = latest[latest["trading_value"] >= min_trading_value].copy()
        if latest.empty:
            return latest, {}
        # v3.2: packaged snapshot may contain code-only names. Refresh names from KRX when online.
        names = load_name_map()
        if names:
            mapped = latest["symbol"].map(names)
            latest["name"] = mapped.fillna(latest.get("name", latest["symbol"]))
        latest["combined_score"] = latest["up_score"] * up_weight + latest["strong_score"] * strong_weight
        latest["scanner_score"] = latest["combined_score"] * 100
        out = latest.sort_values("combined_score", ascending=False).head(top_n).reset_index(drop=True)
        meta = {
            "training_rows": 563_007,
            "training_symbols": 499,
            "latest_date": pd.to_datetime(out["date"]).max(),
            "candidate_count": len(latest),
            "min_trading_value": min_trading_value,
        }
        return out, meta

    # Fallback for projects without the packaged snapshot.
    up_model, strong_model, train_data = train_dual_models(data_dir)
    latest = latest_feature_rows(data_dir)
    if latest.empty:
        return latest, {}
    latest = latest[latest["trading_value"] >= min_trading_value].copy()
    if latest.empty:
        return latest, {}
    latest["up_score"] = up_model.predict_proba(latest[FEATURES])[:, 1]
    latest["strong_score"] = strong_model.predict_proba(latest[FEATURES])[:, 1]
    latest["combined_score"] = latest["up_score"] * up_weight + latest["strong_score"] * strong_weight
    names = load_name_map()
    latest["name"] = latest["symbol"].map(names).fillna(latest["symbol"])
    latest["entry_price"] = latest["close"]
    latest["target_1"] = latest["entry_price"] * 1.10
    latest["target_2"] = latest["entry_price"] * 1.20
    latest["change_pct"] = latest["ret_1d"] * 100
    latest["up_model_score"] = latest["up_score"] * 100
    latest["strong_model_score"] = latest["strong_score"] * 100
    latest["scanner_score"] = latest["combined_score"] * 100
    out = latest.sort_values("combined_score", ascending=False).head(top_n).reset_index(drop=True)
    meta = {
        "training_rows": len(train_data),
        "training_symbols": int(train_data["symbol"].nunique()),
        "latest_date": pd.to_datetime(out["date"]).max(),
        "candidate_count": len(latest),
        "min_trading_value": min_trading_value,
    }
    return out, meta
