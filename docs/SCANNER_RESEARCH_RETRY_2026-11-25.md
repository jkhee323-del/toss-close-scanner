# Scanner research retry — 2026-11-25

## Prediction

Run `python -m src.long_term_walkforward --relative`. The candidate ranks each
indicator relative to stocks observed at the same close. Its daily average
forecast is anchored around the training positive rate. It uses the same
quarter boundaries, mature-label filter and round-trip cost of 0.2% as the
prior run. The prior probability method remains selectable by omitting
`--relative`.

| Sessions | Candidate Brier | Constant Brier | AUC | Accuracy | Top-ten hit rate | Top-ten mean net return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.251281 | 0.251438 | 0.5141 | 51.13% | 50.43% | +0.61% |
| 60 | 0.258605 | 0.259190 | 0.5353 | 45.30% | 55.99% | +3.31% |
| 120 | 0.268170 | 0.268449 | 0.5296 | 37.54% | 55.93% | +7.67% |
| 240 | 0.268221 | 0.268443 | 0.5278 | 35.91% | 62.30% | +17.82% |

The inputs are the 499 stored history files available at branch base
`68ed0b1`. The evaluation aggregates eligible observations by horizon.
Forecast classification uses a fixed 0.5 threshold. Top-ten statistics
describe only observed outcomes and overlapping trades. The new candidate
clears the constant Brier benchmark in this retrospective, with small margins,
and trails the prior candidate's Brier at 60 and 120 sessions. The prior period
informed the decision to investigate a second variant. Keep these settings
fixed and evaluate against future untouched quarters before promotion.

The deterministic 80-symbol sample used in CI provides a sensitivity check
on overlapping data:

| Sessions | Candidate Brier | Constant Brier |
| --- | ---: | ---: |
| 20 | 0.251743 | 0.251795 |
| 60 | 0.258665 | 0.258782 |
| 120 | 0.273960 | 0.273004 |
| 240 | 0.283716 | 0.281112 |

The 120- and 240-session improvements do not hold on the smaller universe.

## KRX investor flows

`python -m src.investor_flow_refresh 005930 000660 --output /private/lab/krx-flow.json`
collects verified 20-session flows using the KRX authentication environment.
The reference calendar should already include the most recently published
market session. The output records the actual reference date and replaces the
previous snapshot atomically only when every symbol passes verification.
Run the command after publication and the price refresh on an authorized
host for unattended use. The lab workflow also accepts manual dispatch on a
repository where the workflow is registered. This work does not schedule a
workflow on the default branch. Real collection still needs authorized
KRX credentials and independent reconciliation of the collected values.

## KIS CVD

`python -m src.kis_cvd_capture 005930 --seconds 120 --output /private/lab/cvd-005930.csv`
uses a current KIS WebSocket approval key from the environment to subscribe to
[H0STCNT0](https://github.com/koreainvestment/open-trading-api/blob/main/examples_llm/domestic_stock/ccnl_krx/ccnl_krx.py).
It writes a bounded, validated capture with actual execution sides, running
CVD and prices. The broker documents a plaintext WebSocket for this feed, so
use a trusted network and a short-lived approval key. Separate capture
windows are separate CVD series. A failed or disconnected capture leaves the
previous file intact. Use the CSV in the lab scanner to inspect divergences.

Local integration tests simulate provider messages and unavailable data.
They do not establish live source access or validate the provider's scope,
accuracy or subscription coverage. Production authorization, continuous
collection and matched scanner-performance attribution remain unverified.
