"""Native Streamlit smoke test for the deployed two-page scanner.

This runs the actual app scripts in Streamlit's supported test harness,
not only Python import/compile checks. It does not place any orders.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _assert_no_errors(app, label):
    errors = [str(getattr(x, "message", x)) for x in app.exception]
    assert not errors, f"{label}: {errors}"


def test_main_and_long_term_page_start_and_analyze():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    _assert_no_errors(at, "short-term scanner")
    at.switch_page("pages/1_중장기 투자.py").run()
    _assert_no_errors(at, "long-term page initial load")
    at.session_state["long_term_scan_active"] = True
    at.run(timeout=150)
    _assert_no_errors(at, "long-term analysis")
