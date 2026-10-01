# project4 — pairs trading: do VN equity spreads actually mean-revert?

A plutus-style quant hypothesis project, DB-backed. It moves off project3's single-name
intraday grid and onto **relative value**: long the cheap leg, short the rich leg of two
economically related stocks, and profit when their spread snaps back. Same database
(`algotradeDB`), same conventions — the question is new.

## Hypothesis

> Two VN stocks tied by the same demand pool, input costs or risk factors drift apart
> for liquidity and sentiment reasons and are pulled back together. If enough pairs are
> genuinely cointegrated — not merely correlated — a market-neutral spread book earns
> on the reversion, independent of the market's direction.

The load-bearing word is **cointegrated**. Correlation is cheap and misleading: two
rising random walks can be 0.95-correlated and still diverge forever. The screen below
does not confuse the two.

## Why this is not a re-run of project2 / project3

project2 (intraday grid, VN banks) and project3 (broad-universe intraday grid) both
concluded the same thing: the binding constraint was **execution cost**, not signal or
selection. This project deliberately steps away from both of their failure modes:

- **Daily, not intraday.** Turnover is orders of magnitude lower, so the ~60 bps
  round-trip that killed the grid is paid rarely instead of constantly.
- **Relative, not directional.** The position is long/short, so market beta cancels.
  The trade is paid for the *spread*, not the market.
- **Structural, not statistical-only.** Candidates are peers first; the statistics
  confirm them. A statistical pair with no economic link is a correlation that will
  break, and the fundamentals overlay is there to catch that.

If nothing survives the gates, **that is the finding** — reported, not worked around.

## Pipeline

```
sql/query_daily_panel.sql
        │  broad VN stock universe, daily split-ADJUSTED close + volume
        ▼
scripts/download_data.py         → data/raw/vn_daily_panel.parquet
        │
sql/fundamentals.sql             → data/raw/vn_fundamentals.parquet   (optional)
        │  annual net revenue / net profit for the durability check
        ▼
scripts/pairs_screen.py          → result/pairs/
           correlation → cointegration → half-life
           → out-of-sample stability → z-score cost-gated backtest
           → Benjamini-Hochberg haircut
```

### Data

- **Adjusted prices, always.** `sql/query_daily_panel.sql` reads `quote.adjclose`, not
  `quote.close`. A cash dividend or split moves the raw level without changing anything
  economic; on a spread that opens a fake divergence. Adjusted prices remove it.
- **Universe**: common stocks on HSX / HNX / UPCOM, with a coarse server-side
  pre-screen (≥ 1000 sessions since 2015, ≥ ~300m VND average daily turnover). The
  real screening is in Python.
- **Fundamentals** (`financial.info`, `financial.item`): code `20` = Net Revenue,
  code `70` = Net Profit After Tax; `quarter = 0` is the annual figure.

### The screen (`scripts/pairs_screen.py`)

Five gates, each stricter than the last:

1. **Candidate** — daily log-return correlation ≥ `--min-corr` (default 0.50) within
   the formation window; keep each ticker's top-`--top-k` peers. Deliberately
   permissive: VN single-name daily returns are idiosyncratic (max pairwise
   correlation in the liquid universe is only ~0.80), so a high bar here would
   discard real sector peers before cointegration ever gets to judge them.
2. **Cointegrated** — ADF on the hedge residual (`statsmodels.coint`) with
   `p < --pvalue` (0.05). The actual claim.
3. **Tradeable** — OU half-life in `[--min-half-life, --max-half-life]` days
   (default 2–60). Too fast = cost; too slow = capital tied up while the tie can break.
4. **Robust** — the *same* formation hedge ratio must yield a stationary spread
   out-of-sample (`--oos-pvalue`, 0.10) with a finite OOS half-life, and the z-score
   trade on the test window must clear `--cost-bps`.
5. **Not luck** — Benjamini-Hochberg at FDR `--bh-alpha` across every tested pair.
   Testing thousands of pairs guarantees spurious "discoveries"; BH is the haircut.

### Backtest caveat (explicit)

OOS PnL is the change in the **log spread** while a position is held — the return of a
long-A / short-`beta`·B book per unit notional — net of a flat cost charged on each
position change. It ignores borrow cost, financing, and slippage beyond that flat cost,
so OOS net Sharpe is an **upper bound**, not a forecast.

## Layout

```
project4/
  .plutus/manifest.yaml          # plutus v2.0 manifest (Tier 3, DB-backed)
  sql/
    query_daily_panel.sql        # adjusted daily close + volume, coarse pre-screen
    fundamentals.sql             # annual/quarterly revenue + net profit
  scripts/
    download_data.py             # DB -> parquet (from project3)
    pairs_screen.py              # the screen: corr -> coint -> half-life -> OOS -> BH
  data/raw/                      # gitignored — re-fetchable via the pipeline
  result/pairs/                  # pairs_metrics.csv, selected_pairs.csv, findings, figures
```

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env         # fill in algotradeDB credentials; .env is gitignored
set -a; source .env; set +a
```

## Run

```bash
python scripts/download_data.py --query-file sql/query_daily_panel.sql \
    -o data/raw/vn_daily_panel.parquet

# optional durability diagnostic
python scripts/download_data.py --query-file sql/fundamentals.sql \
    -o data/raw/vn_fundamentals.parquet

python scripts/pairs_screen.py --input data/raw/vn_daily_panel.parquet \
    --fundamentals data/raw/vn_fundamentals.parquet \
    --outdir result/pairs
```

## Outputs (`result/pairs/`)

- `pairs_metrics.csv` — every tested pair: correlation, hedge ratio, cointegration
  p-value, half-life, OOS p-value/half-life, gross/net Sharpe, trades, drawdown,
  BH flag, profit-linkage.
- `selected_pairs.csv` — pairs clearing **all five gates**.
- `screen.svg` — cointegration p-value vs half-life; green = fully selected.
- `equity.svg` — OOS net equity of the selected pairs (if any).
- `FINDINGS.md` — ranked table and what it implies.

## First run — the result so far

Run on 2026-10-01 against `algotradeDB` (678 liquid stocks, formation 2016–2021,
test 2022–2026, cost 60 bps, FDR 0.05):

- Candidate stage: 126 analysable pairs.
- Cointegrated in-sample (p < 0.05): **10**, and they are economically sensible —
  PVS–PVT (oil & gas services), PLX–PVS (petrol), ACB–VIB and ACB–VPB and
  BID–VCB (banks), HCM–MBS (brokers), PHR–SZC and D2D–SZC (industrial parks).
- Benjamini-Hochberg: **0** survive. 10 hits against 126 tests is almost exactly the
  count expected by chance (126 × 0.05 ≈ 6), which is the false-positive pattern the
  haircut exists to expose.
- Two pairs pass the out-of-sample ADF on their own (PVS–PVT p=0.011, BID–VCB
  p=0.038) but neither is significant once the in-sample tests are corrected.

**So the honest headline is: no statistically robust pairs survive multiple
testing on this universe at this granularity.** That is a finding, not a failed run —
and it fits project2/project3's pattern of reporting the null rather than tuning past
it. Levers worth trying next, in order of how much they change the hypothesis:
restrict to a sector where the economic tie is strongest (e.g. banks only), lengthen
the formation window, relax `--cost-bps`, or move to intraday spreads where the
reversion is faster (at the cost project2/3 already measured).

## What would falsify the hypothesis

- **No pairs survive** cointegration + half-life + OOS + BH → the VN daily universe has
  no stable, tradeable relative-value structure at this granularity. Report it.
- **Pairs survive in-sample but not OOS** → the relationships were fitted, not real.
- **Survivors cluster in one or two sectors** → it may be a single factor trade in
  disguise (e.g. all banks), not a diversified spread book. Check the pair list.
- **Profit-linkage is low for survivors** → the spread reverts statistically but the
  businesses have decoupled; the reversion is fragile. Treat with suspicion.
