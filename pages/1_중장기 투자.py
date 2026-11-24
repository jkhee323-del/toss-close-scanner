
import os

import pandas as pd
import streamlit as st

from src.long_term_scanner import HORIZONS, get_recommendations
from src.market_freshness import session_notice
from src.investor_flow import fetch_investor_flows
from src.investor_flow_import import summarize_imported_flows
from src.search import load_kr_listing, US_NAME_MAP


st.set_page_config(
    page_title="중·장기 스캐너",
    layout="wide",
)

nav_short, nav_long = st.columns(2)

with nav_short:
    st.page_link(
    "app.py",
        label="📈 단기 스캐너",
        width="stretch",
    )

with nav_long:
    st.page_link(
    "pages/1_중장기 투자.py",
        label="📊 중·장기 스캐너",
        width="stretch",
    )

st.title("📊 중·장기 투자 스캐너")
st.caption("한국·미국 저장 일봉 기반 기술적 분석 · 실시간 시세가 아닙니다.")

market = st.radio("투자 시장 선택", ["🇰🇷 한국", "🇺🇸 미국"], horizontal=True)
market_code = "US" if market == "🇺🇸 미국" else "KR"

choice = st.selectbox(
    "투자 기간 선택",
    options=list(HORIZONS.keys()),
    format_func=lambda x: HORIZONS[x][0],
)

name, days = HORIZONS[choice]

top_n = st.slider(
    "추천 종목 수",
    min_value=5,
    max_value=30,
    value=10,
    step=1,
)
st.caption("가격은 분석 기준일의 저장 종가이며 실시간 현재가가 아닙니다. 기술적 점수는 상승 확률이 아닙니다.")

run_analysis = st.button("🔍 선택한 조건으로 분석 실행", type="primary")
if run_analysis:
    st.session_state["long_term_scan_active"] = True



@st.cache_data(ttl=3600, show_spinner=False)

def load_recommendations(period, market):
    return get_recommendations(period, market=market)





@st.cache_data(ttl=21600, show_spinner=False)
def load_company_names():
    listing = load_kr_listing("data", online=True)

    if listing is None or listing.empty:
        return {}

    listing = listing.copy()
    listing["Code"] = listing["Code"].astype(str).str.zfill(6)

    return dict(zip(listing["Code"], listing["Name"]))

if not st.session_state.get("long_term_scan_active", False):
    st.stop()


with st.spinner("중·장기 투자 종목을 분석하고 있습니다..."):

    result = load_recommendations(choice, market_code)


if result.empty:
    st.warning("분석 가능한 종목 데이터가 없습니다.")
else:
    result = result.copy()

    names = load_company_names()


    if market_code == "US":
        names = {ticker: name for ticker, (name, country) in US_NAME_MAP.items() if country == "미국"}

        korean_names = {
            "AAPL": "애플",
            "MSFT": "마이크로소프트",
            "NVDA": "엔비디아",
            "LLY": "일라이 릴리",
            "COST": "코스트코",
            "UNH": "유나이티드헬스",
            "PEP": "펩시코",
            "TEM": "템퍼스 AI",
            "PANW": "팔로알토 네트웍스",
            "MSTR": "스트래티지",
            "ARM": "ARM 홀딩스",
            "ORCL": "오라클",
            "QCOM": "퀄컴",
            "RGTI": "리게티 컴퓨팅",
            "SMR": "뉴스케일 파워",
            "SOUN": "사운드하운드 AI",
            "EVGN": "에보진",

            "XOM": "엑슨모빌",
            "V": "비자",
            "SHOP": "쇼피파이",
            "KO": "코카콜라",
            "PFE": "화이자",
            "BRK-B": "버크셔 해서웨이",
            "WMT": "월마트",
            "ABNB": "에어비앤비",
            "PLTR": "팔란티어",
            "DIS": "월트 디즈니",
            "CRM": "세일즈포스",
            "PYPL": "페이팔",
            "AMZN": "아마존",
            "SBUX": "스타벅스",
            "XYZ": "블록",
            "CRWD": "크라우드스트라이크",
            "RBLX": "로블록스",
            "JPM": "JP모건 체이스",
            "META": "메타",
            "ADBE": "어도비",
            "GOOGL": "알파벳",
            "GOOG": "알파벳",
            "AMD": "AMD",
            "NU": "누 홀딩스",

        }
        names.update(korean_names)

        us_df = pd.read_csv("data/us_universe.csv", dtype=str)
        csv_names = dict(zip(us_df["ticker"], us_df["name"]))
        names = {**csv_names, **names}

        ko_df = pd.read_csv("data/us_names_ko.csv", dtype=str)
        ko_names = dict(zip(ko_df["ticker"], ko_df["name_ko"]))
        names.update(ko_names)



        result["종목코드"] = result["종목코드"].astype(str)
    else:
        names = load_company_names()
        result["종목코드"] = result["종목코드"].astype(str).str.zfill(6)



    result.insert(
        0,
        "기업명",
        result["종목코드"].map(names).fillna("기업명 확인 필요"),
    )

    result.insert(
        0,
        "순위",
        range(1, len(result) + 1),
    )

    a, b, c = st.columns(3)

    with a:
        st.metric("분석 기준일", str(result["기준일"].iloc[0]))

    with b:
        st.metric("선택한 투자 기간", name)

    with c:
        st.metric("분석 가능 종목", f"{len(result)}개")




    st.subheader(f"{HORIZONS[choice][0]} 추천 종목 TOP {top_n}")



    latest_session = pd.to_datetime(result["기준일"].iloc[0], errors="coerce")
    freshness = session_notice(latest_session, market_code)
    if freshness["warning"]:
        st.warning(
            f"시장별 마지막 저장 거래일은 {freshness['date']}입니다. "
            "최근 장 마감 데이터가 아직 반영되지 않았을 수 있습니다. "
            "현지 휴장일과 자동갱신 상태를 확인하세요."
        )
    st.caption(f"기술적 점수와 가격의 기준일: {freshness['date']} · 실시간 가격 아님")

    result = result.head(top_n).copy()


    display_columns = [
        "순위",
        "기업명",
        "현재가",
        "점수",
        "기간수익률(%)",
        "거래량비율",
        "가격위치(%)",
        "OBV_5일",
        "OBV_20일",
    ]

    display_columns = [
        col for col in display_columns if col in result.columns
    ]

    st.dataframe(
        result[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "순위": st.column_config.NumberColumn(
                "순위", format="%d위"
            ),
            "기업명": st.column_config.TextColumn("기업명"),
            "종목코드": st.column_config.TextColumn("종목코드"),
            "현재가": st.column_config.NumberColumn(

"기준일 종가", format="$%,.2f" if market_code == "US" else "%,.0f원"

            ),
            "점수": st.column_config.ProgressColumn(
                "분석 점수", min_value=0, max_value=100,
                format="%.1f",
            ),
            "기간수익률(%)": st.column_config.NumberColumn(
                "기간 수익률", format="%.2f%%"
            ),
            "거래량비율": st.column_config.NumberColumn(
                "거래량 비율", format="%.2f배"
            ),
        },
    )

    st.caption(
        "거래량 비율은 평균 거래량 대비 수준이며, "
        "OBV는 거래량 기반 추세 지표입니다."
    )

    if market_code == "KR":
        with st.expander("외국인·기관 순매수 (최근 20거래일, 실제 거래대금)"):
            st.caption(
                "수급 데이터는 기술적 추천 점수에 미반영됩니다. "
                "인증된 KRX 조회와 직접 가져온 CSV는 별개입니다."
            )
            st.markdown("**CSV 파일로 직접 확인 (KRX 로그인 없이 사용 가능)**")
            st.caption(
                "필수 열: date, symbol, foreign_net_krw, institution_net_krw. "
                "금액 단위는 원(KRW), 종목코드는 6자리입니다. "
                "CSV 제공처와 정확성은 사용자가 확인해야 하며 자동 수급 데이터가 아닙니다."
            )
            imported_file = st.file_uploader(
                "수급 데이터 CSV 선택", type=["csv"], key="imported_kr_investor_flows"
            )
            if imported_file is not None:
                if imported_file.size > 3 * 1024 * 1024:
                    st.error("수급 CSV 파일 크기는 3MB 이하로 제한합니다.")
                else:
                    try:
                        uploaded = pd.read_csv(imported_file, dtype={"symbol": str})
                        imported_summary = summarize_imported_flows(
                            uploaded, result["종목코드"].head(10).tolist(),
                            result["기준일"].iloc[0], lookback=20,
                        )
                        by_code = dict(zip(result["종목코드"], result["기업명"]))
                        imported_summary.insert(
                            0, "기업명",
                            imported_summary["종목코드"].map(by_code).fillna("이름 미확인"),
                        )
                        st.dataframe(imported_summary, hide_index=True, width="stretch")
                        st.caption(
                            "CSV 내부의 미래 날짜는 기준일 이후라면 계산에서 제외됩니다. "
                            "자료 없음은 0원 순매수로 취급하지 않습니다."
                        )
                    except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
                        st.error(f"수급 CSV를 확인해주세요: {exc}")

            if not (os.getenv("KRX_ID") and os.getenv("KRX_PW")):
                st.warning("KRX 수급 조회에 필요한 인증 설정이 없습니다. 배포 환경의 비밀 설정에 KRX_ID·KRX_PW를 등록한 뒤 이용할 수 있습니다.")
            else:
                if st.checkbox("상위 10종목의 순매수 금액 조회", key="show_kr_investor_flow"):
                    flow_rows = []
                    for _, selected in result.head(10).iterrows():
                        entry = {
                            "기업명": selected["기업명"],
                            "종목코드": selected["종목코드"],
                            "수급 기준일": "조회 실패/자료 없음",
                            "외국인 순매수(억원)": None,
                            "기관 순매수(억원)": None,
                        }
                        try:
                            data = fetch_investor_flows(selected["종목코드"], selected["기준일"])
                            if data is not None:
                                entry["수급 기준일"] = data["as_of"]
                                entry["외국인 순매수(억원)"] = round(data["foreign_net_krw"] / 100_000_000, 2)
                                entry["기관 순매수(억원)"] = round(data["institution_net_krw"] / 100_000_000, 2)
                        except Exception:
                            pass  # Keep unavailable values null, never silently report zero.
                        flow_rows.append(entry)
                    st.dataframe(pd.DataFrame(flow_rows), hide_index=True, width="stretch")
                    st.caption("양수는 순매수, 음수는 순매도입니다. 거래소 제공 오류 시 자료 없음으로 표시합니다.")

st.info(
    "이 점수는 기술적 분석 점수입니다. "
    "실제 상승 확률이나 투자 수익을 보장하지 않습니다."
)
