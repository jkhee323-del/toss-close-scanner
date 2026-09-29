"""한국·해외 통합 종목 검색.

- 한국: 종목명/코드. 온라인이면 KOSPI·KOSDAQ 전체 목록을 받아 data/kr_names.csv 로 저장해 두므로
  다음부터는 인터넷이 없어도 종목명이 표시됩니다.
- 해외: 티커, 영문명, 한글명(엔비디아 → NVDA). 목록에 없는 티커도 직접 입력하면 조회를 시도합니다.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ── 해외 종목 ──────────────────────────────────────────────────────────────
# 추천 스캔에 쓰는 기본 유니버스 (ticker, 이름, 국가)
GLOBAL_UNIVERSE = [
    ("AAPL", "Apple", "미국"), ("MSFT", "Microsoft", "미국"), ("NVDA", "NVIDIA", "미국"), ("AMZN", "Amazon", "미국"),
    ("META", "Meta Platforms", "미국"), ("GOOGL", "Alphabet", "미국"), ("TSLA", "Tesla", "미국"), ("AVGO", "Broadcom", "미국"),
    ("AMD", "AMD", "미국"), ("NFLX", "Netflix", "미국"), ("PLTR", "Palantir", "미국"), ("JPM", "JPMorgan Chase", "미국"),
    ("V", "Visa", "미국"), ("COST", "Costco", "미국"), ("LLY", "Eli Lilly", "미국"), ("WMT", "Walmart", "미국"),
    ("7203.T", "Toyota Motor", "일본"), ("6758.T", "Sony Group", "일본"), ("9984.T", "SoftBank Group", "일본"),
    ("8306.T", "Mitsubishi UFJ", "일본"), ("8035.T", "Tokyo Electron", "일본"), ("6861.T", "Keyence", "일본"),
    ("SAP.DE", "SAP", "독일"), ("SIE.DE", "Siemens", "독일"), ("DTE.DE", "Deutsche Telekom", "독일"),
    ("ASML.AS", "ASML", "네덜란드"), ("MC.PA", "LVMH", "프랑스"), ("AIR.PA", "Airbus", "프랑스"),
    ("OR.PA", "L'Oreal", "프랑스"), ("NESN.SW", "Nestle", "스위스"), ("NOVN.SW", "Novartis", "스위스"),
]

# 검색 전용(추천 스캔에는 넣지 않음)
EXTRA_US = [
    ("GOOG", "Alphabet (C)", "미국"), ("INTC", "Intel", "미국"), ("MU", "Micron Technology", "미국"),
    ("QCOM", "Qualcomm", "미국"), ("TSM", "TSMC (ADR)", "미국"), ("ORCL", "Oracle", "미국"),
    ("CRM", "Salesforce", "미국"), ("ADBE", "Adobe", "미국"), ("DIS", "Walt Disney", "미국"),
    ("NKE", "Nike", "미국"), ("SBUX", "Starbucks", "미국"), ("KO", "Coca-Cola", "미국"),
    ("PEP", "PepsiCo", "미국"), ("MCD", "McDonald's", "미국"), ("BA", "Boeing", "미국"),
    ("XOM", "Exxon Mobil", "미국"), ("BRK-B", "Berkshire Hathaway B", "미국"), ("UNH", "UnitedHealth", "미국"),
    ("PFE", "Pfizer", "미국"), ("COIN", "Coinbase", "미국"), ("HOOD", "Robinhood", "미국"),
    ("SOFI", "SoFi Technologies", "미국"), ("IONQ", "IonQ", "미국"), ("RKLB", "Rocket Lab", "미국"),
    ("SMCI", "Super Micro Computer", "미국"), ("ARM", "Arm Holdings", "미국"), ("UBER", "Uber", "미국"),
    ("ABNB", "Airbnb", "미국"), ("SHOP", "Shopify", "미국"), ("SNOW", "Snowflake", "미국"),
    ("PANW", "Palo Alto Networks", "미국"), ("CRWD", "CrowdStrike", "미국"),
    ("CRCL", "Circle Internet Group", "미국"), ("EVGN", "Evogene", "미국"), ("MARA", "MARA Holdings", "미국"),
    ("RIOT", "Riot Platforms", "미국"), ("CLSK", "CleanSpark", "미국"), ("MSTR", "Strategy", "미국"),
    ("PYPL", "PayPal", "미국"), ("XYZ", "Block", "미국"), ("NU", "Nu Holdings", "미국"),
    ("RBLX", "Roblox", "미국"), ("HIMS", "Hims & Hers Health", "미국"), ("ASTS", "AST SpaceMobile", "미국"),
    ("OKLO", "Oklo", "미국"), ("SMR", "NuScale Power", "미국"), ("QBTS", "D-Wave Quantum", "미국"),
    ("RGTI", "Rigetti Computing", "미국"), ("TEM", "Tempus AI", "미국"), ("SOUN", "SoundHound AI", "미국"),
    ("SPY", "SPDR S&P 500 ETF", "미국"), ("QQQ", "Invesco QQQ", "미국"), ("SCHD", "Schwab US Dividend ETF", "미국"),
    ("SOXL", "Direxion Semiconductor Bull 3X", "미국"), ("TQQQ", "ProShares UltraPro QQQ", "미국"),
]

KOREAN_ALIASES = {
    "AAPL": ("애플",), "MSFT": ("마이크로소프트", "마소"), "NVDA": ("엔비디아",), "AMZN": ("아마존",),
    "META": ("메타", "페이스북"), "GOOGL": ("구글", "알파벳"), "GOOG": ("구글C", "알파벳C"), "TSLA": ("테슬라",),
    "AVGO": ("브로드컴",), "AMD": ("에이엠디",), "NFLX": ("넷플릭스",), "PLTR": ("팔란티어",),
    "JPM": ("제이피모건", "jp모건", "제이피모간"), "V": ("비자",), "COST": ("코스트코",), "LLY": ("일라이릴리", "릴리"),
    "WMT": ("월마트",), "7203.T": ("도요타",), "6758.T": ("소니",), "9984.T": ("소프트뱅크",),
    "8306.T": ("미쓰비시UFJ", "미쓰비시"), "8035.T": ("도쿄일렉트론",), "6861.T": ("키엔스",),
    "SAP.DE": ("에스에이피",), "SIE.DE": ("지멘스",), "DTE.DE": ("도이치텔레콤",), "ASML.AS": ("에이에스엠엘",),
    "MC.PA": ("루이비통", "엘브이엠에이치"), "AIR.PA": ("에어버스",), "OR.PA": ("로레알",),
    "NESN.SW": ("네슬레",), "NOVN.SW": ("노바티스",),
    "INTC": ("인텔",), "MU": ("마이크론",), "QCOM": ("퀄컴",), "TSM": ("티에스엠씨", "대만반도체", "tsmc"),
    "ORCL": ("오라클",), "CRM": ("세일즈포스",), "ADBE": ("어도비",), "DIS": ("디즈니",), "NKE": ("나이키",),
    "SBUX": ("스타벅스",), "KO": ("코카콜라",), "PEP": ("펩시",), "MCD": ("맥도날드",), "BA": ("보잉",),
    "XOM": ("엑슨모빌",), "BRK-B": ("버크셔해서웨이", "버크셔"), "UNH": ("유나이티드헬스",), "PFE": ("화이자",),
    "COIN": ("코인베이스",), "HOOD": ("로빈후드",), "SOFI": ("소파이",), "IONQ": ("아이온큐",),
    "RKLB": ("로켓랩",), "SMCI": ("슈퍼마이크로", "슈마컴"), "ARM": ("암홀딩스", "암"), "UBER": ("우버",),
    "ABNB": ("에어비앤비",), "SHOP": ("쇼피파이",), "SNOW": ("스노우플레이크",), "PANW": ("팔로알토",),
    "CRWD": ("크라우드스트라이크",),
    "CRCL": ("서클", "서클인터넷", "스테이블코인", "stablecoin"), "EVGN": ("에보진", "이보진", "evogene"),
    "MARA": ("마라", "마라홀딩스", "비트코인채굴"), "RIOT": ("라이엇", "라이엇플랫폼"), "CLSK": ("클린스파크",),
    "MSTR": ("스트래티지", "마이크로스트래티지", "비트코인"), "PYPL": ("페이팔", "스테이블코인"),
    "XYZ": ("블록", "스퀘어"), "NU": ("누홀딩스",), "RBLX": ("로블록스",), "HIMS": ("힘스",),
    "ASTS": ("AST스페이스모바일", "에이에스티에스"), "OKLO": ("오클로",), "SMR": ("뉴스케일",),
    "QBTS": ("디웨이브",), "RGTI": ("리게티",), "TEM": ("템퍼스", "템퍼스AI"), "SOUN": ("사운드하운드",),
}

GLOBAL_SCAN_UNIVERSE = GLOBAL_UNIVERSE + [x for x in EXTRA_US if x[0] not in {"SPY", "QQQ", "SCHD", "SOXL", "TQQQ"}]
US_STOCKS = GLOBAL_UNIVERSE + EXTRA_US
US_NAME_MAP = {t: (n, c) for t, n, c in US_STOCKS}

_TICKER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,11}$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).casefold()


def _rank(q: str, *fields: str) -> int | None:
    """0 = 완전 일치, 1 = 앞부분 일치, 2 = 포함, None = 불일치."""
    best = None
    for f in fields:
        f = _norm(f)
        if not f:
            continue
        if f == q:
            r = 0
        elif f.startswith(q):
            r = 1
        elif q in f:
            r = 2
        else:
            continue
        best = r if best is None else min(best, r)
    return best


# ── 한국 종목 목록 ─────────────────────────────────────────────────────────
def load_kr_listing(data_dir: str | Path = "data", online: bool = True) -> pd.DataFrame:
    """Code, Name, market 목록. 우선순위: 온라인 KRX > 저장된 캐시 > 스냅샷 > universe.csv"""
    d = Path(data_dir)
    frames: list[pd.DataFrame] = []

    if online:
        try:
            import FinanceDataReader as fdr

            fresh = []
            for market in ("KOSPI", "KOSDAQ"):
                x = fdr.StockListing(market)[["Code", "Name"]].copy()
                x["market"] = market
                fresh.append(x)
            fresh = pd.concat(fresh, ignore_index=True)
            fresh["Code"] = fresh["Code"].astype(str).str.zfill(6)
            frames.append(fresh)
            try:  # 다음에 오프라인이어도 종목명이 보이도록 저장
                fresh.to_csv(d / "kr_names.csv", index=False, encoding="utf-8-sig")
            except Exception:
                pass
        except Exception:
            pass

    cache = d / "kr_names.csv"
    if cache.exists():
        try:
            frames.append(pd.read_csv(cache, dtype={"Code": str}))
        except Exception:
            pass

    snap = d / "latest_snapshot.csv"
    if snap.exists():
        try:
            s = pd.read_csv(snap, dtype={"symbol": str})
            frames.append(pd.DataFrame({"Code": s["symbol"].astype(str).str.zfill(6),
                                        "Name": s["name"].astype(str), "market": "저장 데이터"}))
        except Exception:
            pass

    uni = d / "universe.csv"
    if uni.exists():
        try:
            u = pd.read_csv(uni, dtype={"symbol": str})
            frames.append(pd.DataFrame({"Code": u["symbol"].astype(str).str.zfill(6),
                                        "Name": u["name"].astype(str), "market": "로컬"}))
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=["Code", "Name", "market"])
    out = pd.concat(frames, ignore_index=True)
    out["Code"] = out["Code"].astype(str).str.zfill(6)
    out = out[out["Name"].astype(str) != out["Code"]]  # 코드가 이름으로 저장된 행 제외
    return out.drop_duplicates("Code", keep="first").reset_index(drop=True)


# ── 검색 ───────────────────────────────────────────────────────────────────
def search_kr(query: str, listing: pd.DataFrame, limit: int = 8) -> list[dict]:
    q = _norm(query)
    if not q or listing is None or listing.empty:
        return []
    hits = []
    for code, name, market in zip(listing["Code"], listing["Name"], listing["market"]):
        r = _rank(q, name, code)
        if r is not None:
            hits.append((r, len(str(name)), code, name, market))
    hits.sort(key=lambda t: (t[0], t[1], t[2]))
    return [{"symbol": c, "name": n, "market": "kr", "sub": m if m in ("KOSPI", "KOSDAQ") else "한국", "rank": r}
            for r, _, c, n, m in hits[:limit]]


def search_us(query: str, limit: int = 8) -> list[dict]:
    q = _norm(query)
    if not q:
        return []
    hits = []
    for ticker, name, country in US_STOCKS:
        r = _rank(q, ticker, name, *KOREAN_ALIASES.get(ticker, ()))
        if r is not None:
            hits.append((r, len(name), ticker, name, country))
    hits.sort(key=lambda t: (t[0], t[1], t[2]))
    return [{"symbol": t, "name": n, "market": "us", "sub": c, "rank": r} for r, _, t, n, c in hits[:limit]]


def search_all(query: str, listing: pd.DataFrame, limit: int = 8, known_codes=()) -> list[dict]:
    """한국·해외 통합 검색. 완전 일치 → 앞부분 일치 → 포함 순, 같은 순위면 한국 먼저."""
    q = query.strip()
    if not q:
        return []
    res = search_kr(q, listing, limit) + search_us(q, limit)
    res.sort(key=lambda c: c["rank"])  # 안정 정렬이라 같은 순위에서는 한국이 앞에 옵니다
    seen, out = set(), []
    for c in res:
        key = (c["market"], c["symbol"])
        if key not in seen:
            seen.add(key)
            out.append(c)

    # 종목명을 아직 모르는 한국 코드(6자리 숫자, 또는 저장된 일봉이 있는 코드)도 코드로 조회할 수 있게 함
    code = q.upper()
    is_kr_code = bool(re.fullmatch(r"\d{6}", q)) or code in set(known_codes)
    if is_kr_code and ("kr", code) not in seen:
        out.append({"symbol": code, "name": code, "market": "kr", "sub": "코드로 직접 조회", "rank": 8})
        seen.add(("kr", code))

    # 목록에 없는 티커(예: 새로 상장한 종목)도 직접 조회해 볼 수 있게 함
    if _TICKER_RE.match(q) and not is_kr_code:
        tk = q.upper()
        if (not out) or (q == tk and ("us", tk) not in seen):
            name, country = US_NAME_MAP.get(tk, (tk, "해외"))
            out.append({"symbol": tk, "name": name, "market": "us", "sub": f"{country} · 티커로 직접 조회", "rank": 9})
    return out[:limit]


def search_yahoo(query: str, limit: int = 6) -> list[dict]:
    """Yahoo Finance 검색 API로 로컬 목록에 없는 해외 종목명/티커를 찾습니다."""
    q = str(query).strip()
    if not q:
        return []
    try:
        import requests
        r = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": q, "quotesCount": limit, "newsCount": 0, "enableFuzzyQuery": "true"},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
        )
        r.raise_for_status()
        out = []
        allowed = {"EQUITY", "ETF"}
        for x in r.json().get("quotes", []):
            if str(x.get("quoteType", "")).upper() not in allowed:
                continue
            sym = str(x.get("symbol", "")).upper().strip()
            if not sym:
                continue
            name = str(x.get("longname") or x.get("shortname") or sym)
            exch = str(x.get("exchange") or x.get("exchDisp") or "해외")
            out.append({"symbol": sym, "name": name, "market": "us", "sub": f"{exch} · Yahoo 검색", "rank": 7})
        return out[:limit]
    except Exception:
        return []
