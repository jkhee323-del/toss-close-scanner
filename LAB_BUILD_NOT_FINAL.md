# TOSS CLOSE SCANNER — EXPERIMENTAL LAB BUILD ONLY

This is NOT the user's newest Windows V3.1 project and is NOT an approved
production scanner. This ZIP is a reproducible GitHub laboratory snapshot.

- Live/prod branch main is intentionally unchanged.
- The 1m, 3m, 6m, and 12m AI research models FAILED their out-of-sample
  research qualification tests on the 12-stock five-year sample.
- Do not present experimental model score as a probability of profit.
- No calibrated live AI recommendation model is included.
- Korean institutional/foreign flow source integration requires live KRX
  verification; a missing data source is displayed as unavailable.
- True CVD needs verified aggressor-side trade ticks; daily candles do not
  contain them.
- The current ZIP is from a DIFFERENT repository version than the user's
  local V3.1 folder; do not replace the V3.1 working folder with this archive.
- Historical price caches in this branch may be stale. Do not trade off them.
- Real world transaction costs, calendar, liquidity and live execution are
  not certified.

To inspect without modifying production:
  python -m pip install -r requirements.txt pytest
  python -m pytest tests -q
  python -m src.auto_update --market global --history-period 5y --max-symbols 1 --dry-run

See US_SESSION_LAB_STATUS.md for verified dates and outstanding release blocks.
