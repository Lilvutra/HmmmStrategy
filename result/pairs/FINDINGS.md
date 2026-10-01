# Pairs screen — cointegration, out-of-sample and the multiple-testing haircut

Analysed **126** of **164** candidate pairs (top-6 correlated peers per ticker at r >= 0.50; the rest lacked the 500 formation observations). Formation 2016-01-01 -> 2021-12-31, test 2022-01-01 -> 2026-12-31.

**10** are cointegrated in-sample (p < 0.05); **35** also have a tradeable half-life; **0** survive the Benjamini-Hochberg haircut at FDR 0.05; **0** are selected after the out-of-sample stability check.

## Selected pairs (ranked by OOS net Sharpe)

| pair | beta | coint p | half-life | OOS p | OOS half-life | net Sharpe | OOS net cum | trades | profit link |
|---|---|---|---|---|---|---|---|---|---|

## Reading this

The screen is deliberately sequential and strict. Correlation is only a candidate filter; cointegration is the actual claim; the half-life band keeps only spreads a strategy can act on; the OOS check reuses the formation hedge ratio with no refitting, so it cannot quietly re-optimise; and the Benjamini-Hochberg step is what separates a real stationarity from the luckiest of tests. A pair that clears all five steps has earned a backtest — not a trade.

Two honest caveats. (1) BH is applied over the analysed pairs, not every pair in the market — the correlation pre-filter is itself a selection, so the FDR here is conditional on it. (2) The number of in-sample 'hits' near p=0.05 is close to the false-positive rate the test count implies, which is exactly the pattern the haircut is meant to expose.

The OOS PnL is a log-spread series per unit notional, net of a flat 60 bps charge per position change. It ignores borrow cost, financing and market impact, so treat net Sharpe as an upper bound.
