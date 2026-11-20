# Toss Close Scanner — release readiness, 2026-11-20

**Status: targeted safety fixes released to `main`; full eight-phase experimental project NOT yet approved for general promotion.**

This report distinguishes actual GitHub deployments, validated historical results,
experiment-only features and external dependencies. No trading strategy is guaranteed
profitable.

## Production fixes completed

- Korean close-scanner now rejects malformed history, uses only a contiguous clean
  recent slice where possible, and shows the number of excluded stocks in the UI.
  **90 uninterrupted sessions** are required for the short-term scoring path;
  **241+ sessions** are required for long-horizon indicators.
- Korean market latest saved completed session at last check: **2026-11-19**.
- Live KR short scanner sample after the safe-history adjustment: **495 usable of
  499 stored**, **45 historical series recovered**, **4 excluded**; all four signal
  day tabs work. This does NOT imply 499 historical files are clean.
- The overseas Yahoo price ingestion now selects the requested ticker explicitly
  from multi-index columns and rejects duplicate/malformed OHLCV, stale provider
  data and widespread provider outages. The prior concatenated ticker columns
  observed in `META.csv` and `MSFT.csv` were successfully re-fetched and
  repaired **without synthetic/fabricated prices**.
- `global_candidates.csv` is now cross-checked against each ticker's own saved
  close on each recommendation date. The latest verified batch contains **81
  overseas histories and 56 cached recommendation rows**, all matching.
  The last US history session at verification was **2026-11-18**; newer sessions
  should not be assumed until the scheduled provider refresh actually obtains them.
- The production interface disables cached overseas recommendations with an
  explicit error if history/candidate correspondence fails.
- Main and lab both have executable release smoke workflows; the production
  workflow now tests global provider guards, foreign candidate prices, Korean
  model inference and four date tabs.

## Verified test links

- [Production scanner smoke tests](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/production-smoke.yml)
- [Experimental scanner smoke tests](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/release-smoke.yml)
- [Accuracy-lab regression tests](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/accuracy-lab-tests.yml)
- [Global data refresh and sync](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/update-global.yml)
- [Market data integrity audit](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/data-integrity.yml)

## Known blockers / not completed

1. **Archived Korean price history:** strict full-history audit still reports
   **49 legacy files with invalid OHLCV** (including zero-price suspension
   records and inconsistent adjusted prices). A valid recent suffix may be
   tradable even though the archived source contains bad rows. These raw
   historic records are NOT corrected or silently invented. Full-history
   backtests and AI retraining need provider-level remediation or documented
   exclusions.
2. **KRX foreign/institutional net purchases:** current provider requires
   authorized `KRX_ID` and `KRX_PW` secrets. Live smoke test failed without
   credentials; the optional research UI remains feature-gated. Do not
   enter secrets in chat or source control.
3. **True intraday CVD:** calculation module and user-supplied signed-trade CSV
   visualization exist in `accuracy-lab`, but no authorized automatic
   aggressor-side trade feed has been supplied.
4. **Predictive model quality:** 1/3/6/12-month experiments had weak AUCs and
   did not outperform constant Brier-score baselines. Do not replace the
   operational scoring model or call research scores calibrated probabilities.
5. **Short-term trading outcome research:** on 2025-2026 retrospective signals,
   HGB baseline outperformed two alternatives under the assumed next-session
   open-to-close execution. These are hypothetical historical results with
   simplified round-trip trading costs, not brokerage fills or an unseen
   holdout. Further independent forward validation is necessary.
6. **Site hosting:** passing GitHub tests establishes repository code/model
   execution, not a visually inspected Streamlit deployment or a validated
   broker execution environment.

## Operational safeguards

- Never merge all of `accuracy-lab` into diverged `main`; copy only reviewed
  and CI-tested fixes.
- Never fix invalid OHLCV by automatic price interpolation or by treating
  a suspended zero-price candle as a real transaction.
- Require **both** global history schema validation and recommendation-to-ticker
  price match after every refresh. The foreign refresh workflow now rebases
  its data-only commit before pushing so concurrent main changes are preserved.
- Keep experimental features opt-in, retain audit reports and ensure
  non-synchronous upstream data is visibly labeled with its reference date.
- Full release signoff remains blocked until the unresolved research, source
  quality and authorized feed dependencies are satisfied.

This document describes code checks and data hygiene, not investment advice.
