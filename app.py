"""종가매매 AI 스캐너 v9 - 신호표 + 차트 + 종목 리포트 화면 (오늘/전일/2일전/3일전 탭, 통합 검색)."""
from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import urlencode

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.indicators import add_indicators
from src.model import FEATURES
from src.search import GLOBAL_SCAN_UNIVERSE, US_NAME_MAP, load_kr_listing, search_all, search_yahoo
from src.signals import compute_scored_window, floor_to_tick, pick_signals, target_prices
from src.toss_client import TossClient

st.set_page_config(page_title="종가매매 AI 스캐너 v9", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

RED, BLUE = "#e0383e", "#2b59d1"          # 한국식: 상승 = 빨강, 하락 = 파랑
DAY_LABELS = ["오늘", "전일", "2일전", "3일전"]
TV_OPTIONS = [0, 10, 30, 50, 100, 200, 500]

st.markdown(r"""
<style>
html, body, [class*="css"] { font-family: "Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif; }
.stApp { background:#f4f5f8; }
.block-container { padding-top:.9rem; padding-bottom:2rem; max-width:1680px; }
header[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid #e7ebf0; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-radius:8px; border-color:#e3e7ee !important; }
a { text-decoration:none !important; }
.logo { font-size:22px; font-weight:900; color:#111827; letter-spacing:-.5px; }
.ver { font-size:12px; color:#8a94a3; margin-left:8px; }
.mkt { display:inline-flex; border:1px solid #d9dee6; border-radius:6px; overflow:hidden; background:#fff; }
.mkt a, .seg a { padding:6px 13px; font-size:13px; font-weight:600; color:#4b5563 !important; background:#fff; border-right:1px solid #e3e7ee; }
.mkt a:last-child, .seg a:last-child { border-right:none; }
.mkt a.on, .seg a.on { background:#1f2937; color:#fff !important; }
.seg { display:inline-flex; border:1px solid #d9dee6; border-radius:5px; overflow:hidden; }
.seg a { padding:4px 10px; font-size:12px; }
.ph { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:2px 2px 8px; }
.pt { font-size:15px; font-weight:800; color:#111827; }
.pt .mut { color:#6b7280; font-weight:600; margin-left:4px; }
.pt .cnt { color:#111827; }
.hr { font-size:12px; color:#4b5563; white-space:nowrap; }
.hr b { color:#111827; }
.badge { font-size:11px; background:#eaf0fd; color:#2f5bd0; padding:2px 8px; border-radius:10px; margin-left:6px; font-weight:700; }
.tbl-wrap { max-height:505px; overflow-y:auto; border-top:1px solid #e9edf2; }
table.sig { width:100%; border-collapse:collapse; font-size:13px; color:#111827; }
table.sig th { position:sticky; top:0; z-index:1; background:#f5f6f8; color:#4b5563; font-weight:600; padding:9px 8px; text-align:left; border-bottom:1px solid #e3e7ee; white-space:nowrap; }
table.sig td { padding:8px; border-bottom:1px solid #f0f2f5; white-space:nowrap; }
table.sig th.num, table.sig td.num { text-align:right; }
table.sig tr:hover td { background:#f6f9ff; }
table.sig tr.sel td { background:#dce8fc; }
table.sig a.nm { color:#111827 !important; font-weight:700; }
table.sig a.cd { color:#374151 !important; }
.tag { font-size:10px; font-weight:700; color:#0f766e; background:#dff5f0; border:1px solid #b7e6dc; padding:0 5px; border-radius:3px; margin-left:5px; vertical-align:1px; }
.tp { color:#c0262d; font-weight:700; }
.tpct { color:#8a94a3; font-size:11px; }
.ck { display:inline-block; width:14px; height:14px; line-height:14px; text-align:center; background:#22b14c; color:#fff; border-radius:3px; font-size:10px; margin-left:4px; vertical-align:1px; }
.pos { color:#d1343a; font-weight:700; } .neg { color:#2b59d1; font-weight:700; }
.rbtn { display:inline-block; padding:3px 11px; border:1px solid #c7d5f4; background:#eaf0fd; color:#2f5bd0 !important; border-radius:4px; font-size:12px; font-weight:600; }
.sumline { font-size:12px; color:#374151; background:#f8fafc; border:1px solid #e9edf2; border-radius:6px; padding:7px 10px; margin:0 0 8px; }
.foot { font-size:11px; color:#8a94a3; line-height:1.5; margin-top:8px; }
.empty { text-align:center; color:#8a94a3; font-size:13px; padding:38px 0; }
.legend { display:flex; gap:18px; justify-content:center; font-size:12px; color:#4b5563; margin-top:2px; }
.legend i { display:inline-block; width:20px; border-top:2px dashed #c0262d; vertical-align:middle; margin-right:5px; }
.legend i.dot { border-top-style:dotted; }
.legend .tri { color:#d1343a; margin-right:3px; }
.chips { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:2px 0 8px; font-size:12px; color:#6b7280; }
.chip { display:inline-block; padding:4px 11px; border:1px solid #d9dee6; border-radius:14px; background:#fff; color:#111827 !important; font-size:13px; font-weight:600; }
.chip small { color:#8a94a3; font-weight:500; margin-left:4px; }
.chip.on { background:#1f2937; border-color:#1f2937; color:#fff !important; } .chip.on small { color:#cbd5e1; }
.chip.x { color:#6b7280 !important; font-weight:500; }
.card { background:#fff; border:1px solid #e9edf2; border-radius:8px; padding:12px 14px; height:100%; }
.ctitle { font-size:14px; font-weight:800; color:#111827; margin:0 0 6px; }
.reason { padding:6px 0; border-bottom:1px solid #f0f2f5; font-size:13px; }
.score-row { display:flex; gap:8px; margin:6px 0 10px; }
.score { flex:1; border-radius:8px; padding:10px; text-align:center; font-weight:700; font-size:12px; }
.score b { display:block; font-size:22px; margin-top:3px; }
.score-red { background:#fff0f2; color:#d1343a; } .score-blue { background:#edf5ff; color:#2b59d1; } .score-purple { background:#f5efff; color:#7a3ee6; }
.guide { border-radius:7px; padding:9px 11px; margin:6px 0; font-weight:700; font-size:13px; display:flex; justify-content:space-between; }
.guide-entry { background:#edf5ff; color:#2b59d1; } .guide-target { background:#fff0f2; color:#c0262d; } .guide-stop { background:#eef3ff; color:#2b59d1; }
.note { background:#fff8e6; border:1px solid #fde7aa; color:#7a5b00; border-radius:8px; padding:9px 10px; font-size:12px; line-height:1.55; margin-top:8px; }
.rep-empty { text-align:center; color:#8a94a3; font-size:13px; padding:26px 0 30px; }
</style>
""", unsafe_allow_html=True)


# ── URL 상태 ────────────────────────────────────────────────────────────────
def qget(key: str, default: str = "") -> str:
    v = st.query_params.get(key, default)
    if isinstance(v, (list, tuple)):
        v = v[-1] if v else default
    return str(v)


def to_int(v: str, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


market = qget("m", "kr")
market = market if market in ("kr", "us") else "kr"
day = min(max(to_int(qget("d", "0"), 0), 0), 3)
sym = qget("sym").strip()
rep = qget("rep") == "1"
q_url = qget("q").strip()

S = {"m": market, "d": day, "n": 14, "tv": 100}


def make_link(**over) -> str:
    p = dict(S)
    p.update(over)
    clean = {k: v for k, v in p.items() if v not in (None, "")}
    return escape("?" + urlencode(clean))


# ── 데이터 로딩 (캐시) ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def get_listing():
    return load_kr_listing("data", online=True)


@st.cache_data(show_spinner=False)
def get_known_codes():
    return sorted(p.stem for p in Path("data/history").glob("*.csv"))


@st.cache_data(show_spinner=False)
def get_window():
    return compute_scored_window("data", 4)


@st.cache_data(show_spinner=False)
def load_kr_chart(symbol: str, bars: int = 120):
    p = Path("data/history") / f"{symbol}.csv"
    return pd.read_csv(p, parse_dates=["date"]).sort_values("date").tail(bars).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_saved_models():
    return joblib.load("data/up_model.joblib"), joblib.load("data/strong_model.joblib")


def fetch_kr_history(symbol: str, bars: int = 260) -> pd.DataFrame:
    local = Path("data/history") / f"{symbol}.csv"
    if local.exists():
        return pd.read_csv(local, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    errors = []
    try:
        return TossClient().get_daily_history(symbol, bars=bars)
    except Exception as e:  # noqa: BLE001
        errors.append(f"pykrx: {e}")
    try:
        import FinanceDataReader as fdr

        raw = fdr.DataReader(symbol).reset_index()
        raw = raw.rename(columns={raw.columns[0]: "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        raw = raw[["date", "open", "high", "low", "close", "volume"]].dropna()
        raw["date"] = pd.to_datetime(raw["date"])
        if not raw.empty:
            return raw.tail(bars).reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        errors.append(f"FinanceDataReader: {e}")
    raise ValueError("한국 종목 일봉을 받지 못했습니다. 종목코드와 인터넷 연결을 확인해주세요. " + " | ".join(errors[-2:]))


@st.cache_data(show_spinner=False, ttl=1800)
def analyze_kr(symbol: str, name: str):
    df = fetch_kr_history(symbol)
    x = add_indicators(df).dropna(subset=FEATURES + ["trading_value"]).copy()
    if x.empty:
        raise ValueError("분석에 필요한 일봉 데이터가 부족합니다.")
    row = x.iloc[-1]
    up_model, strong_model = load_saved_models()
    ff = pd.DataFrame([{f: float(row[f]) for f in FEATURES}])
    up = float(up_model.predict_proba(ff)[:, 1][0])
    strong = float(strong_model.predict_proba(ff)[:, 1][0])
    entry = float(row["close"])
    t1, t2 = target_prices(entry)
    d = pd.Timestamp(row["date"])
    sel = {
        "market": "kr", "symbol": symbol, "name": name, "date": d, "latest_date": d,
        "entry_price": entry, "last_close": entry, "change_pct": float(row["ret_1d"] * 100),
        "trading_value": float(row["trading_value"]), "volume_ratio": float(row["volume_ratio"]),
        "rsi14": float(row["rsi14"]), "ret_5d": float(row["ret_5d"]), "near_20d_high": float(row["near_20d_high"]),
        "up_model_score": up * 100, "strong_model_score": strong * 100, "scanner_score": (up + strong) / 2 * 100,
        "target_1": t1, "target_2": t2, "ret_since": 0.0, "hit_1": False, "hit_2": False,
        "max_high_after": None, "past": False, "from_table": False,
    }
    return sel, df.tail(120).reset_index(drop=True)


def sel_from_kr_row(r, latest_date) -> dict:
    d = pd.Timestamp(r["date"])
    mh = r["max_high_after"]
    return {
        "market": "kr", "symbol": str(r["symbol"]), "name": str(r["name"]), "date": d, "latest_date": latest_date,
        "entry_price": float(r["entry_price"]), "last_close": float(r["last_close"]), "change_pct": float(r["change_pct"]),
        "trading_value": float(r["trading_value"]), "volume_ratio": float(r["volume_ratio"]), "rsi14": float(r["rsi14"]),
        "ret_5d": float(r["ret_5d"]), "near_20d_high": float(r["near_20d_high"]),
        "up_model_score": float(r["up_model_score"]), "strong_model_score": float(r["strong_model_score"]),
        "scanner_score": float(r["scanner_score"]), "target_1": float(r["target_1"]), "target_2": float(r["target_2"]),
        "ret_since": float(r["ret_since"]), "hit_1": bool(r["hit_1"]), "hit_2": bool(r["hit_2"]),
        "max_high_after": None if pd.isna(mh) else float(mh), "past": d < latest_date, "from_table": True,
    }


# ── 해외 ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=900)
def load_us_chart(symbol: str, bars: int = 180):
    """미국·해외 일봉: yfinance 우선, Yahoo chart JSON을 보조 경로로 사용합니다."""
    symbol = symbol.strip().upper()
    errors = []
    try:
        import yfinance as yf

        raw = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, progress=False, threads=False)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            raw = raw.reset_index()
            raw = raw.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            need = ["date", "open", "high", "low", "close", "volume"]
            if all(c in raw.columns for c in need):
                out = raw[need].copy()
                out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
                out = out.dropna(subset=["open", "high", "low", "close", "volume"]).sort_values("date")
                if not out.empty:
                    return out.tail(bars).reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        errors.append(f"yfinance: {e}")
    try:
        import requests

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        r = requests.get(url, params={"range": "1y", "interval": "1d", "events": "history"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        r.raise_for_status()
        result = r.json()["chart"]["result"][0]
        ts = result.get("timestamp", [])
        qd = result["indicators"]["quote"][0]
        out = pd.DataFrame({
            "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
            "open": qd.get("open", []), "high": qd.get("high", []), "low": qd.get("low", []),
            "close": qd.get("close", []), "volume": qd.get("volume", []),
        })
        out = out.dropna(subset=["open", "high", "low", "close", "volume"]).sort_values("date")
        if not out.empty:
            return out.tail(bars).reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        errors.append(f"Yahoo: {e}")
    raise ValueError("해외 가격 데이터를 받지 못했습니다. 티커를 확인해주세요. " + " | ".join(errors[-2:]))


def technical_candidate_score(row) -> float:
    score = 50.0
    score += max(-10, min(10, float(row["ret_5d"]) * 100 * 1.5))
    score += max(-8, min(8, float(row["ret_20d"]) * 100 * 0.5))
    score += max(-6, min(8, (float(row["volume_ratio"]) - 1.0) * 8))
    rsi = float(row["rsi14"])
    if 45 <= rsi <= 65:
        score += 7
    elif rsi >= 75:
        score -= 7
    elif rsi < 35:
        score -= 3
    score += max(-5, min(7, (float(row["near_20d_high"]) - .90) * 50))
    return max(0.0, min(100.0, score))


def _us_snapshot(symbol: str, name: str, country: str, df: pd.DataFrame, day_offset: int = 0):
    x = add_indicators(df).dropna(subset=["ret_5d", "ret_20d", "volume_ratio", "rsi14", "near_20d_high"]).copy()
    if len(x) <= day_offset:
        raise ValueError("기술지표 계산에 필요한 데이터가 부족합니다.")
    idx = len(x) - 1 - day_offset
    row = x.iloc[idx]
    prev = x.iloc[max(0, idx - 1)]
    close = float(row["close"])
    d = pd.Timestamp(row["date"])
    latest = pd.Timestamp(x.iloc[-1]["date"])
    future = x.iloc[idx + 1:]
    max_high = float(future["high"].max()) if not future.empty else None
    last_close = float(x.iloc[-1]["close"])
    t1, t2 = round(close * 1.10, 2), round(close * 1.20, 2)
    change = (close / float(prev["close"]) - 1) * 100 if float(prev["close"]) else 0.0
    return {
        "market": "us", "symbol": symbol, "name": name, "country": country, "date": d, "latest_date": latest,
        "day_offset": day_offset, "entry_price": close, "last_close": last_close, "change_pct": change,
        "trading_value": float(row["close"] * row["volume"]), "volume_ratio": float(row["volume_ratio"]),
        "rsi14": float(row["rsi14"]), "ret_5d": float(row["ret_5d"]), "near_20d_high": float(row["near_20d_high"]),
        "candidate_score": technical_candidate_score(row), "target_1": t1, "target_2": t2,
        "ret_since": (last_close / close - 1) * 100, "hit_1": bool(max_high is not None and max_high >= t1),
        "hit_2": bool(max_high is not None and max_high >= t2), "max_high_after": max_high,
        "past": day_offset > 0, "from_table": False,
    }


@st.cache_data(show_spinner=False, ttl=1800)
def scan_global_candidates(top_n: int = 14):
    """해외 유니버스를 한 번만 내려받아 오늘~3일전 후보를 함께 만듭니다."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows = []
    def one(item):
        ticker, name, country = item
        df = load_us_chart(ticker, 180)
        got = []
        for off in range(4):
            snap = _us_snapshot(ticker, name, country, df, off)
            got.append({k: snap[k] for k in ["symbol", "name", "country", "date", "latest_date", "day_offset",
                "entry_price", "last_close", "change_pct", "trading_value", "volume_ratio", "rsi14", "ret_5d",
                "near_20d_high", "candidate_score", "target_1", "target_2", "ret_since", "hit_1", "hit_2", "max_high_after"]})
        return got
    # 여러 종목을 동시에 받되 워커 수를 제한해 Yahoo 과부하/차단 위험을 낮춥니다.
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(one, item) for item in GLOBAL_SCAN_UNIVERSE]
        for fut in as_completed(futures):
            try:
                rows.extend(fut.result())
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()
    z = pd.DataFrame(rows)
    return (z.sort_values(["day_offset", "candidate_score", "trading_value"], ascending=[True, False, False])
             .groupby("day_offset", group_keys=False).head(top_n).reset_index(drop=True))


def analyze_global_stock(symbol: str):
    symbol = symbol.strip().upper()
    df = load_us_chart(symbol, 180)
    name, country = US_NAME_MAP.get(symbol, (symbol, "해외"))
    return _us_snapshot(symbol, name, country, df), df.tail(120).reset_index(drop=True)


def sel_from_us_row(r) -> dict:
    d = pd.Timestamp(r["date"])
    close = float(r["entry_price"])
    return {
        "market": "us", "symbol": str(r["symbol"]), "name": str(r["name"]), "country": str(r["country"]), "date": d, "latest_date": d,
        "entry_price": close, "last_close": close, "change_pct": float(r["change_pct"]), "trading_value": float(r["trading_value"]),
        "volume_ratio": float(r["volume_ratio"]), "rsi14": float(r["rsi14"]), "ret_5d": float(r["ret_5d"]),
        "near_20d_high": float(r["near_20d_high"]), "candidate_score": float(r["candidate_score"]),
        "target_1": float(r["target_1"]), "target_2": float(r["target_2"]),
        "ret_since": float(r.get("ret_since", 0.0)), "hit_1": bool(r.get("hit_1", False)), "hit_2": bool(r.get("hit_2", False)),
        "max_high_after": r.get("max_high_after", None), "past": int(r.get("day_offset", 0)) > 0, "from_table": True,
        "latest_date": pd.Timestamp(r.get("latest_date", d)),
    }


GLOBAL_CACHE = Path("data/global_candidates.csv")


def save_global_candidates(df):
    if df is not None and not df.empty:
        GLOBAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(GLOBAL_CACHE, index=False, encoding="utf-8-sig")


def load_saved_global_candidates():
    if not GLOBAL_CACHE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(GLOBAL_CACHE, parse_dates=["date"])
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


# ── HTML 조각 ───────────────────────────────────────────────────────────────
def fmt_price(v, mk: str) -> str:
    if v is None or pd.isna(v):
        return "-"
    return f"{v:,.0f}" if (mk == "kr" or v >= 1000) else f"{v:,.2f}"


def sign_cls(v: float) -> str:
    return "pos" if v > 0 else "neg" if v < 0 else ""


def seg(items, css="seg") -> str:
    inner = "".join(f'<a class="{"on" if on else ""}" href="{href}" target="_self">{label}</a>' for label, href, on in items)
    return f'<div class="{css}">{inner}</div>'


def target_cell(t, base, hit, mk, check) -> str:
    ck = '<span class="ck">✓</span>' if (check and hit) else ""
    return f'<td class="num"><span class="tp">{fmt_price(t, mk)}</span> <span class="tpct">({(t / base - 1) * 100:+.1f}%)</span>{ck}</td>'


def signals_table(rows: list[dict], mk: str, past: bool, selected: str) -> str:
    if not rows:
        return '<div class="empty">조건을 만족하는 종목이 없습니다.</div>'
    code_h = "코드" if mk == "kr" else "티커"
    head = f'<th>{code_h}</th><th>종목명</th>'
    head += '<th class="num">진입가</th><th class="num">현재가</th><th class="num">수익률</th>' if past else '<th class="num">현재가</th>'
    head += '<th class="num">1차익절</th><th class="num">2차익절</th><th></th>'
    body = []
    for r in rows:
        href = make_link(m=mk, sym=r["symbol"], rep="", q="")
        rhref = make_link(m=mk, sym=r["symbol"], rep="1", q="")
        tag = f'<span class="tag">{escape(r["tag"])}</span>' if r.get("tag") else ""
        cells = f'<td><a class="cd" href="{href}" target="_self">{escape(r["symbol"])}</a></td>'
        cells += f'<td><a class="nm" href="{href}" target="_self">{escape(r["name"])}</a>{tag}</td>'
        if past:
            cells += f'<td class="num">{fmt_price(r["entry"], mk)}</td><td class="num"><b>{fmt_price(r["price"], mk)}</b></td>'
            cells += f'<td class="num {sign_cls(r["ret"])}">{r["ret"]:+.1f}%</td>'
        else:
            cells += f'<td class="num"><b>{fmt_price(r["price"], mk)}</b></td>'
        cells += target_cell(r["t1"], r["entry"], r["hit1"], mk, past) + target_cell(r["t2"], r["entry"], r["hit2"], mk, past)
        cells += f'<td class="num"><a class="rbtn" href="{rhref}" target="_self">리포트</a></td>'
        body.append(f'<tr class="{"sel" if r["symbol"] == selected else ""}">{cells}</tr>')
    return f'<div class="tbl-wrap"><table class="sig"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def make_chart(chart: pd.DataFrame, sel: dict) -> go.Figure:
    c = chart.copy().reset_index(drop=True)
    xs = pd.to_datetime(c["date"]).dt.strftime("%Y-%m-%d")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=xs, open=c["open"], high=c["high"], low=c["low"], close=c["close"], name="일봉",
        increasing_line_color=RED, increasing_fillcolor=RED, decreasing_line_color=BLUE, decreasing_fillcolor=BLUE,
    ))
    t1, t2, entry = float(sel["target_1"]), float(sel["target_2"]), float(sel["entry_price"])
    fig.add_hline(y=t1, line_dash="dash", line_color=RED, line_width=1.2, annotation_text="1차 +10%",
                  annotation_position="top right", annotation_font=dict(size=11, color=RED))
    fig.add_hline(y=t2, line_dash="dot", line_color="#9f1d22", line_width=1.4, annotation_text="2차 +20%",
                  annotation_position="top right", annotation_font=dict(size=11, color="#9f1d22"))
    ed = pd.Timestamp(sel["date"]).strftime("%Y-%m-%d")
    hit = c[xs == ed]
    if not hit.empty:
        fig.add_trace(go.Scatter(x=[ed], y=[float(hit.iloc[0]["low"]) * 0.985], mode="markers", name="진입",
                                 marker=dict(symbol="triangle-up", size=10, color=RED), hoverinfo="skip"))
    lo = float(c["low"].min()) * 0.97
    hi = max(float(c["high"].max()), t2) * 1.04
    idx = np.unique(np.linspace(0, len(c) - 1, 6).astype(int))
    fig.update_layout(height=470, margin=dict(l=6, r=6, t=8, b=8), paper_bgcolor="white", plot_bgcolor="white",
                      showlegend=False, hovermode="x unified", xaxis_rangeslider_visible=False)
    fig.update_xaxes(type="category", tickmode="array", tickvals=[xs.iloc[i] for i in idx],
                     ticktext=[pd.Timestamp(xs.iloc[i]).strftime("%m/%d") for i in idx],
                     showgrid=False, tickfont=dict(size=11, color="#8a94a3"), linecolor="#e3e7ee")
    fig.update_yaxes(side="right", range=[lo, hi], showgrid=True, gridcolor="#f0f2f5",
                     tickformat=",.0f" if (sel["market"] == "kr" or entry >= 1000) else ",.2f",
                     tickfont=dict(size=11, color="#8a94a3"))
    return fig


LEGEND = '<div class="legend"><span><i></i>1차익절 +10%</span><span><i class="dot"></i>2차익절 +20%</span><span><span class="tri">▲</span>진입</span></div>'


def kv(label: str, value: str) -> str:
    return f'<div class="reason"><b>{label}</b><span style="float:right">{value}</span></div>'


def report_kr(sel: dict, meta: dict, cand_count: int):
    entry, t1, t2 = sel["entry_price"], sel["target_1"], sel["target_2"]
    won = lambda v: f"{v:,.0f}원"  # noqa: E731
    a, b, c, d = st.columns([1.05, 1.2, 1.0, .95], gap="small")
    with a:
        st.markdown('<div class="card"><div class="ctitle">📋 종목 기본 정보</div>' +
                    kv("종목명", f'{escape(sel["name"])} ({escape(sel["symbol"])})') +
                    kv("현재가", f'<b>{won(sel["last_close"])}</b>') +
                    kv("신호일 등락", f'<span class="{sign_cls(sel["change_pct"])}">{sel["change_pct"]:+.2f}%</span>') +
                    kv("거래대금", f'{sel["trading_value"] / 1e8:,.0f}억원') +
                    kv("거래량 / 20일 평균", f'{sel["volume_ratio"]:.2f}배') +
                    kv("RSI(14)", f'{sel["rsi14"]:.1f}') + '</div>', unsafe_allow_html=True)
    with b:
        ok = lambda cond: "✅" if cond else "▫️"  # noqa: E731
        st.markdown(
            '<div class="card"><div class="ctitle">🧠 AI 분석 결과</div><div class="score-row">'
            f'<div class="score score-red">종합점수<b>{sel["scanner_score"]:.1f}</b></div>'
            f'<div class="score score-blue">상승 모델<b>{sel["up_model_score"]:.1f}</b></div>'
            f'<div class="score score-purple">강한 상승<b>{sel["strong_model_score"]:.1f}</b></div></div>'
            f'<div class="reason">{ok(sel["volume_ratio"] >= 1)} 20일 평균 대비 거래량 {sel["volume_ratio"]:.2f}배</div>'
            f'<div class="reason">{ok(sel["ret_5d"] > 0)} 최근 5일 수익률 {sel["ret_5d"] * 100:+.2f}%</div>'
            f'<div class="reason">{ok(sel["near_20d_high"] > .9)} 20일 고점 대비 위치 {sel["near_20d_high"] * 100:.1f}%</div>'
            f'<div class="reason">▫️ RSI(14) {sel["rsi14"]:.1f}</div></div>', unsafe_allow_html=True)
    with c:
        if sel["past"]:
            mh = sel["max_high_after"]
            st.markdown('<div class="card"><div class="ctitle">📈 신호 이후 결과</div>' +
                        kv("신호일(진입)", f'{sel["date"]:%Y-%m-%d}') + kv("진입가", won(entry)) + kv("현재가", won(sel["last_close"])) +
                        kv("수익률", f'<span class="{sign_cls(sel["ret_since"])}">{sel["ret_since"]:+.1f}%</span>') +
                        kv("신호 이후 최고가", won(mh) if mh else "-") +
                        kv("1차 / 2차 도달", f'{"✅" if sel["hit_1"] else "▫️"} / {"✅" if sel["hit_2"] else "▫️"}') +
                        '<div class="note">고가 기준(장중)으로 목표가에 닿았는지만 표시합니다. 실제 체결을 보장하지 않습니다.</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="ctitle">📊 스캐너 상태</div>' +
                        kv("최신 데이터", f'{meta["latest_date"]:%Y-%m-%d}') +
                        kv("학습 종목", f'{meta["training_symbols"]:,}개') +
                        kv("학습 사례", f'{meta["training_rows"]:,}건') +
                        kv("필터 통과", f'{cand_count:,}개') +
                        '<div class="note">과거 신호의 실제 적중률은 위 신호표의 전일·2일전·3일전 탭에서 확인할 수 있습니다.</div></div>',
                        unsafe_allow_html=True)
    with d:
        stop = floor_to_tick(entry * .97)
        st.markdown('<div class="card"><div class="ctitle">🎯 매매 참고 가이드</div>'
                    f'<div class="guide guide-entry"><span>진입 기준</span><span>{won(entry)}</span></div>'
                    f'<div class="guide guide-target"><span>1차 참고 +10%</span><span>{won(t1)}</span></div>'
                    f'<div class="guide guide-target"><span>2차 참고 +20%</span><span>{won(t2)}</span></div>'
                    f'<div class="guide guide-stop"><span>손절 참고 -3%</span><span>{won(stop)}</span></div>'
                    '<div class="note">목표가와 손절가는 모델이 예측한 가격이 아니라 화면용 기계적 참고선입니다(호가단위 내림). '
                    '모델 점수 역시 실제 상승확률이 아니며 투자 수익을 보장하지 않습니다.</div></div>', unsafe_allow_html=True)


def report_us(sel: dict):
    a, b, c = st.columns(3, gap="small")
    with a:
        st.markdown('<div class="card"><div class="ctitle">📋 종목 정보</div>' +
                    kv("시장", escape(str(sel.get("country", "해외")))) + kv("RSI(14)", f'{sel["rsi14"]:.1f}') +
                    kv("최근 5일", f'<span class="{sign_cls(sel["ret_5d"])}">{sel["ret_5d"] * 100:+.2f}%</span>') +
                    kv("거래량 / 20일 평균", f'{sel["volume_ratio"]:.2f}배') + '</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="ctitle">🧭 해외 후보 분석</div>'
                    f'<div class="score-row"><div class="score score-red">기술 후보점수<b>{sel["candidate_score"]:.1f}</b></div></div>'
                    f'<div class="reason">최근 5일 수익률 {sel["ret_5d"] * 100:+.2f}%</div>'
                    f'<div class="reason">20일 고점 대비 위치 {sel["near_20d_high"] * 100:.1f}%</div>'
                    '<div class="note">이 점수는 상승확률이 아니며 해외 전용 AI 모델의 예측값도 아닙니다. '
                    'RSI·추세·거래량·20일 고점 위치를 합친 기술지표 후보 점수입니다.</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="card"><div class="ctitle">🎯 참고 가격</div>'
                    f'<div class="guide guide-entry"><span>현재 기준</span><span>{fmt_price(sel["entry_price"], "us")}</span></div>'
                    f'<div class="guide guide-target"><span>+10% 참고</span><span>{fmt_price(sel["target_1"], "us")}</span></div>'
                    f'<div class="guide guide-target"><span>+20% 참고</span><span>{fmt_price(sel["target_2"], "us")}</span></div>'
                    '<div class="note">+10%/+20%는 예측 목표가가 아닌 기계적 참고선입니다.</div></div>', unsafe_allow_html=True)


# ── 사이드바 (설정) ─────────────────────────────────────────────────────────
n0 = min(30, max(5, to_int(qget("n"), 14)))
tv0 = to_int(qget("tv"), 100)
tv0 = tv0 if tv0 in TV_OPTIONS else 100
with st.sidebar:
    st.markdown("### ⚙️ 스캐너 설정")
    top_n = st.slider("표시 종목 수", 5, 30, n0, key="top_n_w")
    min_value_100m = st.select_slider("최소 당일 거래대금", options=TV_OPTIONS, value=tv0, key="tv_w",
                                      format_func=lambda x: "필터 없음" if x == 0 else f"{x}억원 이상")
    if st.button("🔔 오늘 신호 다시 분석", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("표시 종목 수와 거래대금 필터는 한국 신호표에 적용됩니다.")
    st.markdown("---")
    st.caption("※ 모델 점수는 검증된 실제 상승확률이 아닙니다.")
S["n"], S["tv"] = top_n, min_value_100m

# ── 상단 바: 로고 · 통합 검색 · 시장 전환 ───────────────────────────────────
c_logo, c_search, c_mkt, c_act = st.columns([2.2, 3.3, 1.5, 1.5], gap="small", vertical_alignment="center")
q_in = c_search.text_input("종목 검색", value=q_url, placeholder="종목명·코드·티커 검색 (예: 삼성전자, 엔비디아, NVDA)",
                           label_visibility="collapsed", key="q_input")
q = q_in.strip()
if q != q_url:
    sym = ""  # 새 검색어면 선택 종목 초기화

listing = get_listing()
cands = search_all(q, listing, known_codes=get_known_codes()) if q else []
if q:
    # 로컬 사전에 없는 회사명도 Yahoo 검색으로 보완합니다. 단, 정확한 티커 직접입력 결과는 유지합니다.
    need_web = (not cands) or (len(cands) == 1 and cands[0].get("rank") == 9 and len(q) > 5)
    if need_web:
        web_hits = search_yahoo(q)
        if web_hits:
            direct = [c for c in cands if c.get("rank") == 9]
            cands = web_hits + [c for c in direct if c["symbol"] not in {w["symbol"] for w in web_hits}]
            cands = cands[:8]
cand_keys = {(c["market"], c["symbol"]) for c in cands}
search_mode = False
if q and cands:
    if not (sym and (market, sym) in cand_keys):
        sym, market = cands[0]["symbol"], cands[0]["market"]
    search_mode = True
S["m"] = market

refresh_us = False
if market == "us":
    refresh_us = c_act.button("🌎 해외 추천 새로 분석", use_container_width=True)

# ── 데이터/선택 종목 결정 ───────────────────────────────────────────────────
sel, chart, err = None, None, None
rows: list[dict] = []
meta: dict = {}
cand_count = 0
panel_title, panel_tabs, sumline, foot = "", "", "", ""

if market == "kr":
    with st.spinner("저장된 한국주식 신호를 불러오는 중입니다..."):
        window, meta = get_window()
    if window is None or window.empty:
        st.error("data/history 폴더에서 일봉 CSV를 찾지 못했습니다.")
        st.stop()
    names = dict(zip(listing["Code"], listing["Name"])) if not listing.empty else {}
    latest_date = meta["latest_date"]
    sig, cand_count = pick_signals(window, day, top_n, min_value_100m * 100_000_000)
    day_date = meta["dates"].get(day, latest_date)
    if not sig.empty:
        sig["name"] = sig["symbol"].map(names).fillna(sig["symbol"])
    past = day > 0
    for r in sig.itertuples(index=False):
        rows.append({"symbol": r.symbol, "name": r.name, "price": r.last_close, "entry": r.entry_price, "ret": r.ret_since,
                     "t1": r.target_1, "t2": r.target_2, "hit1": bool(r.hit_1), "hit2": bool(r.hit_2)})
    panel_title = (f'{DAY_LABELS[day]} 진입 신호 <span class="mut">{day_date:%Y%m%d}</span> '
                   f'<span class="cnt">{len(sig)}종목</span>')
    panel_tabs = seg([(DAY_LABELS[i], make_link(d=i, sym="", q="", rep=""), i == day) for i in range(4)])
    if past and not sig.empty:
        k1, k2, n_sig = int(sig["hit_1"].sum()), int(sig["hit_2"].sum()), len(sig)
        avg = float(sig["ret_since"].mean())
        sumline = (f'<div class="sumline">신호일 종가 진입 가정 · 1차 도달 <b>{k1}/{n_sig}</b> · 2차 도달 <b>{k2}/{n_sig}</b> · '
                   f'평균 수익률 <b class="{sign_cls(avg)}">{avg:+.1f}%</b></div>')
        foot = ('<div class="foot">※ 과거 신호는 저장된 모델로 다시 계산한 값이며, 모델 학습 데이터와 겹치는 구간이라 실제 성적보다 좋게 보일 수 있습니다. '
                '1차/2차 도달은 신호 이후 고가가 목표가에 닿았는지(장중 기준)만 표시합니다.</div>')
    if sym:
        hit = sig[sig["symbol"] == sym] if not sig.empty else sig
        try:
            if not hit.empty:
                sel, chart = sel_from_kr_row(hit.iloc[0], latest_date), load_kr_chart(sym)
            else:
                sel, chart = analyze_kr(sym, names.get(sym, sym))
        except Exception as e:  # noqa: BLE001
            err = str(e)
    elif not sig.empty:
        first = sig.iloc[0]
        sym = str(first["symbol"])
        sel, chart = sel_from_kr_row(first, latest_date), load_kr_chart(sym)
else:
    past = day > 0
    if refresh_us:
        with st.spinner("해외 종목을 분석하고 오늘~3일전 결과를 저장하고 있습니다..."):
            fresh = scan_global_candidates(top_n)
        if fresh.empty:
            st.error("해외 추천 데이터를 받지 못했습니다.")
        else:
            save_global_candidates(fresh)
    all_cand = load_saved_global_candidates()
    if not all_cand.empty and "day_offset" not in all_cand.columns:
        all_cand["day_offset"] = 0
    cand_df = all_cand[all_cand["day_offset"].astype(int) == day].copy() if not all_cand.empty else pd.DataFrame()
    if not cand_df.empty:
        cand_df = cand_df.head(top_n).reset_index(drop=True)
    for r in cand_df.itertuples(index=False):
        rows.append({"symbol": r.symbol, "name": r.name, "price": r.last_close if past else r.entry_price,
                     "entry": r.entry_price, "ret": r.ret_since if past else 0.0,
                     "t1": r.target_1, "t2": r.target_2, "hit1": bool(r.hit_1), "hit2": bool(r.hit_2),
                     "tag": r.country if r.country != "미국" else ""})
    day_date = pd.Timestamp(cand_df.iloc[0]["date"]) if not cand_df.empty else None
    date_txt = f' <span class="mut">{day_date:%Y%m%d}</span>' if day_date is not None else ""
    panel_title = f'{DAY_LABELS[day]} 해외 후보{date_txt} <span class="cnt">{len(rows)}종목</span>'
    panel_tabs = seg([(DAY_LABELS[i], make_link(m="us", d=i, sym="", q="", rep=""), i == day) for i in range(4)])
    if rows:
        if past:
            k1, k2, n_sig = int(cand_df["hit_1"].sum()), int(cand_df["hit_2"].sum()), len(cand_df)
            avg = float(cand_df["ret_since"].mean())
            sumline = (f'<div class="sumline">신호일 종가 진입 가정 · 1차 도달 <b>{k1}/{n_sig}</b> · 2차 도달 <b>{k2}/{n_sig}</b> · '
                       f'현재까지 평균 <b class="{sign_cls(avg)}">{avg:+.1f}%</b></div>')
        else:
            sumline = '<div class="sumline">미국·일본·유럽 및 테마 종목 · 기술지표 기반 후보 점수 순 (AI 상승확률이 아닙니다)</div>'
    if sym:
        hit = cand_df[cand_df["symbol"] == sym] if not cand_df.empty else cand_df
        try:
            if not hit.empty:
                sel, chart = sel_from_us_row(hit.iloc[0]), load_us_chart(sym, 120)
            else:
                sel, chart = analyze_global_stock(sym)
        except Exception as e:
            err = str(e)
    elif rows:
        first = cand_df.iloc[0]
        sym = str(first["symbol"])
        try:
            sel, chart = sel_from_us_row(first), load_us_chart(sym, 120)
        except Exception as e:
            err = str(e)

# ── 상단 바 마무리 ──────────────────────────────────────────────────────────
ver_txt = f'v9 · 데이터 기준 {meta["latest_date"]:%Y-%m-%d}' if meta else "v9"
c_logo.markdown(f'<span class="logo">📈 종가매매 AI 스캐너</span><span class="ver">{ver_txt}</span>', unsafe_allow_html=True)
c_mkt.markdown(seg([("🇰🇷 한국", make_link(m="kr", d=0, sym="", q="", rep=""), market == "kr"),
                    ("🌎 해외", make_link(m="us", d=0, sym="", q="", rep=""), market == "us")], css="mkt"), unsafe_allow_html=True)

if q:
    if cands:
        chips = "".join(
            f'<a class="chip {"on" if (c["market"] == market and c["symbol"] == sym) else ""}" target="_self" '
            f'href="{make_link(m=c["market"], sym=c["symbol"], q=q, rep="")}">{escape(c["name"])}<small>{escape(c["symbol"])} · {escape(c["sub"])}</small></a>'
            for c in cands)
        st.markdown(f'<div class="chips"><span>검색 결과</span>{chips}<a class="chip x" target="_self" href="{make_link(q="", sym="", rep="")}">✕ 검색 해제</a></div>',
                    unsafe_allow_html=True)
    else:
        st.warning(f"'{q}' 검색 결과가 없습니다. 종목명, 6자리 코드, 해외 티커(예: NVDA)로 다시 검색해보세요.")

# ── 본문: 신호표 | 차트 ─────────────────────────────────────────────────────
left, right = st.columns([1.0, 1.0], gap="small")
with left:
    with st.container(border=True):
        st.markdown(f'<div class="ph"><div class="pt">{panel_title}</div>{panel_tabs}</div>{sumline}', unsafe_allow_html=True)
        if market == "us" and not rows:
            st.markdown('<div class="empty">이 날짜의 저장된 해외 추천이 없습니다.<br>오른쪽 위 <b>🌎 해외 추천 새로 분석</b>을 한 번 누르면 오늘·전일·2일전·3일전 결과가 함께 저장됩니다.<br>검색은 추천 목록과 관계없이 티커·회사명으로 사용할 수 있습니다.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(signals_table(rows, market, past, sym) + foot, unsafe_allow_html=True)

with right:
    with st.container(border=True):
        if sel is None:
            if err:
                st.error(f"{sym} 데이터를 불러오지 못했습니다: {err}")
            else:
                st.markdown('<div class="empty" style="padding:150px 0">종목을 선택하거나 검색해주세요.</div>', unsafe_allow_html=True)
        else:
            badge = "검색 결과" if search_mode else ("신호" if sel["from_table"] else "조회")
            entry_note = f' · 진입일 {sel["date"]:%Y%m%d}' if sel["past"] else ""
            st.markdown(
                f'<div class="ph"><div class="pt">{escape(sel["name"])} <span class="mut">({escape(sel["symbol"])})</span><span class="badge">{badge}</span></div>'
                f'<div class="hr">{sel["latest_date"]:%Y%m%d} · 종가 <b>{fmt_price(sel["last_close"], sel["market"])}</b> · '
                f'진입 <b>{fmt_price(sel["entry_price"], sel["market"])}</b>{entry_note}</div></div>', unsafe_allow_html=True)
            st.plotly_chart(make_chart(chart, sel), use_container_width=True, config={"displayModeBar": False})
            st.markdown(LEGEND, unsafe_allow_html=True)

# ── 하단: 종목 리포트 ───────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="ph"><div class="pt">종목 리포트</div></div>', unsafe_allow_html=True)
    if sel is not None and (rep or search_mode):
        if sel["market"] == "kr":
            report_kr(sel, meta, cand_count)
        else:
            report_us(sel)
    else:
        st.markdown('<div class="rep-empty">목록에서 <b>리포트</b> 버튼을 누르면 이 자리에 표시됩니다.</div>', unsafe_allow_html=True)

# ── URL 동기화 (새로고침해도 같은 화면) ─────────────────────────────────────
params = {"m": market, "d": str(day), "n": str(top_n), "tv": str(min_value_100m)}
if sym:
    params["sym"] = sym
if q:
    params["q"] = q
if rep:
    params["rep"] = "1"
st.query_params.from_dict(params)
