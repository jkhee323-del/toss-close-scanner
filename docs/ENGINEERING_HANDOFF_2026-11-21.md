# Toss Close Scanner — 2026-11-21 engineering handoff

## Scope and release decision

Continue the eight-stage project without merging the diverged `accuracy-lab`
and `main` branches wholesale. Experimental models, optional KRX flows and
true trade-level CVD are **not** approved for automatic production promotion.
All GitHub Actions comparisons are research validation, not trading advice.

## Market data status checked on November 21

- The latest stored `005930` Korean session was **2026-11-20**.
- The latest stored `MSFT` overseas session was **2026-11-20**.
- Main's 2026-11-21 overseas data refresh completed successfully.
- Prior production smoke validated 495 Korean symbols, four day tabs,
  81 overseas history files and 56 global candidate-price rows.
- A vendor-confirmed source has **not** repaired the archived Korean
  historical anomaly records. Original archived prices are left untouched.

## Historical Korean anomaly reconciliation

- **49** archived symbols contain flagged OHLCV anomalies.
- Read-only source inventory identified **1,847** flagged rows across these
  symbols at initial vendor comparison. That is the count of suspect cached
  records, not independently confirmed vendor errors.
- Recent uninterrupted data remain usable for **45** short-term and **42**
  longer-term analyses, while **4** remain excluded by the short-term gate.
- `src/kr_vendor_reconciliation.py` and
  `.github/workflows/kr-vendor-reconciliation.yml` were added to audit
  missing/invalid KRX bars without writing to `data/history`.
- The first live reconciliation returned **KRX login failed**, plus
  zero/invalid vendor prices; these cannot count as independent verification.
- The code now reports a missing-credential or unavailable-provider status,
  and the workflow reports an unresolved blocker rather than 'repaired'.
- **Required external dependency:** authorized KRX historical-price access,
  with `KRX_ID` and `KRX_PW` kept only in GitHub Actions secrets.
  Never commit credentials to source or send them in chat.
- After valid access is configured, manually dispatch the
  **Archived Korean Vendor Reconciliation** GitHub Actions workflow. Review
  the uploaded `kr-vendor-review-candidates` report. Any candidate
  amendment must be reviewed for corporate actions, adjustment basis and
  neighboring prices; no automatic historical price overwrite is authorized.

## Automated pre-publication checks

- `main` overseas and Korean refresh workflows now execute
  `python -m src.production_smoke` immediately after price refresh and
  **before** committing cached data. This avoids relying on a separate push
  workflow, which GitHub does not trigger for some bot-made commits.
- Smoke includes actual Korean model inference/four tabs, overseas OHLCV
  schema and per-symbol candidate-price correspondence.
- Failed pre-publication smoke prevents the workflow from publishing bad
  refreshed prices.
- Old successful runs and new feature code need distinct final checks.
  Inspect the latest workflow conclusion for each market before signoff.

## Model holdout decision

- Next-day backtest on ~2025-2026 historical signals completed successfully.
  With 49 corrupted symbol histories excluded, baseline HGB was superior
  to the researched RandomForest and LogisticRegression models under the
  next-session open-to-close simulation and 0.2% assumed round-trip cost.
- The 1/3/6/12-month models still failed to beat the constant-rate Brier
  score benchmark in the latest checked experiments. They remain lab-only.
- No real brokerage fills, unbiased historical delistings, licensed signed
  trade flow, live real-money calibration or independently verified net
  alpha has been established. Do not claim guaranteed returns.

## Further blockers

1. Archived KRX prices need a reliable authenticated source or manually
   reviewed, correctly adjusted alternatives.
2. Foreign/institutional investor flow requires authorized KRX access.
3. Automatically refreshed true CVD needs licensed
   aggressor-side trade data; OHLCV/OBV cannot stand in for it.
4. A hosted Streamlit browser/UI smoke test was not performed by these
   GitHub Actions checks. GitHub script tests are not visual deployment QA.

## Verification

- [KR source anomalies inventory](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/kr-data-triage.yml)
- [Archived KRX source reconciliation](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/kr-vendor-reconciliation.yml)
- [Production scanner checks](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/production-smoke.yml)
- [Korea auto-refresh](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/update-korea.yml)
- [Overseas auto-refresh](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/update-global.yml)
- [Experimental regression suite](https://github.com/jkhee323-del/toss-close-scanner/actions/workflows/accuracy-lab-tests.yml)

No historic bars or production prediction models were overwritten as part
of this handoff.
