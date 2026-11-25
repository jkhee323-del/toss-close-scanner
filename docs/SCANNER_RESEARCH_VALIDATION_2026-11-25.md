# Scanner research validation — 2026-11-25

Research runs used the Korean history available on `accuracy-lab` at
`e5fde6ab229017ec2c5839843607278fd94f960c`. Samsung Electronics' stored
reference session was 2026-11-24.

## Long-horizon walk-forward

Run `python -m src.long_term_walkforward` from the repository root. At each
quarter boundary beginning in 2025, train on trades with exits strictly before
the boundary. Compare a fixed regularized logistic model, shrunk halfway
toward the training positive rate, with that rate as a constant forecast.
Signals use data through the signal close. Entry uses the next open and exit
uses the horizon close, with 0.2% round-trip cost deducted. Invalid bars split
histories. A signal whose outcome is unavailable can still be selected in the
daily top ten and counts toward selection coverage.

Brier, AUC and accuracy below are quarterly averages weighted by evaluated
observations. Top-ten figures are weighted by selections with observed
outcomes and show mean returns per trade. Positions can overlap.

| Sessions | Quarters | Model Brier | Constant Brier | AUC | Accuracy | Top-ten hit rate | Top-ten mean net return | Outcomes / selections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 8 | 0.251695 | 0.251438 | 0.5358 | 50.43% | 46.35% | 1.47% | 4375 / 4610 |
| 60 | 7 | 0.258010 | 0.259190 | 0.5572 | 46.33% | 53.56% | 3.91% | 3964 / 4240 |
| 120 | 6 | 0.267550 | 0.268449 | 0.5420 | 39.76% | 54.27% | 7.47% | 3357 / 3620 |
| 240 | 4 | 0.269216 | 0.268443 | 0.5281 | 36.05% | 58.26% | 13.56% | 2142 / 2420 |

The prior HGB holdout on 80 sampled symbols again lost to the constant Brier
benchmark at every horizon. The walk-forward candidate improved aggregate
Brier at 60 and 120 sessions in this retrospective. The history universe is
the repository's current universe, with incomplete point-in-time membership
and delisting coverage. The period was previously inspected, the outcome
windows overlap and unavailable outcomes are excluded from realized metrics.
Independent forward evidence is still needed for promotion.

## Investor flow

The optional Korean scanner lookup now requires all 20 expected market
sessions from its stored Samsung Electronics calendar proxy, bounded by the
signal date and KRX publication cutoff. Missing, duplicate and non-finite
observations cannot fill that window. A 15-minute lookup cache refreshes on
demand and invalidates when the published-date cutoff or calendar changes.

The KRX live smoke stopped at missing `KRX_ID` and `KRX_PW`. No authenticated
response or independent reconciliation was available. KIS documents a
[stock-specific daily investor API](https://github.com/koreainvestment/open-trading-api/blob/main/examples_llm/domestic_stock/investor_trade_by_stock_daily/investor_trade_by_stock_daily.py)
with a KRX market selector. Before activation, compare authorized samples
with the KRX series for identical dates, market scope, investor classification,
units and net values. No unattended collection or scanner lift is established.

## CVD

The lab can read signed-trade CSVs and recorded
[KIS KRX execution messages](https://github.com/koreainvestment/open-trading-api/blob/main/examples_llm/domestic_stock/ccnl_krx/ccnl_krx.py).
The KIS adapter checks single-symbol/session scope, signed sides, order and
continuous cumulative volume from the first recorded signed execution.
The optional price column supports opposite-direction price/CVD moves across
adjacent completed five-minute bars. It provides no predictive score.

KIS offers a live WebSocket, while
[KRX and Koscom document contracted feed access](https://openapi.krx.co.kr/contents/OPP/DATA/OPPDATA003.jsp).
Continuous authorized collection, archival signed data, source completeness
validation and scanner lift remain unverified. Exchange and vendor terms
govern use and redistribution.

## Regression and release

Local regression suite: 124 passed. Lab smoke: 495 usable Korean short-term
symbols, four day tabs, 489 Korean and 66 overseas long-term symbols,
81 overseas histories and 56 overseas candidate rows verified.
