"""Public Streamlit screens must never change the shared overseas signal CSV."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_public_rebuild_button_or_global_cache_writes():
    for relative in ("app.py", "pages/0_종가매매.py"):
        file = ROOT / relative
        if not file.exists():
            continue  # production only uses app.py
        source = file.read_text(encoding="utf-8")
        assert "save_global_candidates" not in source
        assert "refresh_us" not in source
        assert "to_csv(GLOBAL_CACHE" not in source
        assert "해외 추천 새로 분석" not in source
        assert "last_close" in source
        assert "find_stale_cached_symbols" in source
        assert "마지막 저장 종가" in source


def test_long_term_page_discloses_reference_date_and_no_realtime_price():
    page = (ROOT / "pages/1_중장기 투자.py").read_text(encoding="utf-8")
    assert "session_notice" in page
    assert "기준일 종가" in page
    assert "실시간 가격" in page
