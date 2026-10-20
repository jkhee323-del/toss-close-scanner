
import streamlit as st

st.set_page_config(
    page_title="주식 투자 스캐너",
    page_icon="📈",
    layout="wide",
)

pages = [
    st.Page("pages/0_종가매매.py", title="종가매매", icon="📈", default=True),
    st.Page("pages/1_중장기 투자.py", title="중장기 투자", icon="📊"),
]

navigation = st.navigation(pages)
navigation.run()