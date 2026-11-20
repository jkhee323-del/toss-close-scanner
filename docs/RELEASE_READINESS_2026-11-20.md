# Release readiness — 2026-11-20

**Decision: NOT READY for unconditional promotion from `accuracy-lab` to `main`.**

This report records checks performed on experimental code without altering production
trading models or rewriting historical market prices.

## Verified

- `main` Korean and overseas daily price refresh workflows reported successful
  runs on 2026-11-19 (UTC timestamps of runs can differ from exchange session dates).
- `accuracy-lab` test suite compiles Python sources and checks scanner regression tests.
- International history integrity: **81/81 valid** in the new file-level audit.
- Korean history integrity: **450/499 accepted**, **49 requiring review or exclusion**.
  The original stricter audit flagged 82; 33 were reclassified after allowing
  single-unit adjusted-price rounding. Do not substitute fabricated OHLC for
  zero-price/suspended historical entries.
- The short-term scanner in `accuracy-lab` now excludes invalid price-history files
  and warns users when records are excluded.
- The short-term research holdout now excludes labels that cross the training
  cutoff. It additionally reports next-session open-to-close returns separately
  from the close-to-close prediction target.
- Experimental true CVD calculation requires explicit aggressor-side trade prints.
  Optional user CSV analysis is isolated from automated production scans.

## Blocking items

1. **KR historical remediation:** audit still fails on 49 files; inspect
   `market-data-integrity-report` artifacts, review affected source rows,
   validate against an authoritative data provider, and re-run the gate.
   Do not silently forward-fill or overwrite disputed prices.
2. **KRX investor flow:** live API smoke tests failed because KRX credentials
   were not provided to the job. Configure `KRX_ID` and `KRX_PW` only as
   GitHub Actions/deployment secrets and confirm authorized provider access.
   Avoid entering account secrets in issue comments or chat.
3. **Trade-level CVD:** no licensed, signed trade feed has been attached.
   CSV-based manual CVD is experimental, not automatic live-market CVD.
4. **Model quality:** prior 80-symbol holdout did not establish long-term model
   improvement. Do not label model scores as calibrated rise probabilities.
   Compare executable next-open entry net returns before adopting short-term
   strategy changes.
5. **Deployment isolation:** `main` and `accuracy-lab` have substantial
   independent commits and differing cached market files; do not blindly merge
   full branches. Promote reviewed code changes selectively after checks.

## Required acceptance checks for later release

- All regression tests pass on the exact candidate commit.
- All market data integrity errors are resolved, explicitly quarantined with
  documented scope, or backed by verified vendor corrections.
- US/KR final dates match completed exchange sessions, not unfinished candles.
- Experimental trade metrics survive independent chronological out-of-sample
  checking with transaction costs and next-open execution.
- KRX flow retrieval and CVD feature visibility are disabled unless trustworthy
  inputs and permissions exist.
- Production deployment passes a smoke test for both market views, four date
  tabs, stock selection, and price-chart rendering.

## GitHub Actions links

- [Market data integrity](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/data-integrity.yml)
- [Regression tests](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/accuracy-lab-tests.yml)
- [Next-day experimental validation](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/next-day-accuracy-lab.yml)
- [Long-term experimental validation](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/long-term-evaluation.yml)
- [KRX investor feed smoke test](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/investor-flow-smoke.yml)

This is engineering research and data quality assurance, not financial advice
or evidence of guaranteed trading performance.
