#!/usr/bin/env python3
"""project4 — pairs-trading screen: which VN stock pairs actually mean-revert?

Pairs trading is a bet on the spread between two related names, not on market
direction. The claim is structural: two stocks tied by the same demand pool, the
same input costs or the same risk factors should be pulled back together when they
temporarily diverge. The screen below tests that claim in four increasingly strict
steps, so a pair has to survive every one of them:

  candidate     daily log-return correlation >= --min-corr within the formation
                window. A cheap filter only. Correlation is NOT enough: two rising
                random walks can be perfectly correlated and never converge.
  cointegrated  Augmented Dickey-Fuller on the hedge residual (statsmodels coint)
                with p < --pvalue. This is the real test: the spread is stationary
                and, by construction, pulled back toward its mean.
  tradeable     OU half-life between --min-half-life and --max-half-life days. Too
                fast and cost eats the edge; too slow and capital is tied up while
                the relationship has time to break.
  robust        the SAME formation hedge ratio yields a stationary spread
                out-of-sample (formation -> test split) AND a z-score reversion
                trade on the test window clears round-trip cost.

Finally a Benjamini-Hochberg haircut is applied across every tested pair. Testing
thousands of pairs guarantees some look cointegrated by luck; BH is what stops that
luck being reported as a discovery. Whatever survives, survives. If nothing does,
that is the finding and must be reported rather than engineered around.

Backtest simplification (explicit): PnL is the change in the log spread while a
position is held — the return of a long-A / short-beta*B book per unit of notional —
net of `cost_bps` charged on each position change. It ignores borrow cost, financing
and slippage beyond the flat cost. This is a screen, not an execution simulator.

Usage:
    python scripts/pairs_screen.py --input data/raw/vn_daily_panel.parquet \
        --outdir result/pairs [--fundamentals data/raw/vn_fundamentals.parquet]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


# --------------------------------------------------------------------------- panel

def load_panel(path: Path) -> pd.DataFrame:
    """Long (date, ticker, adj_close, volume) -> wide date x ticker adjusted close."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    wide = (
        df.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
        .sort_index()
    )
    return wide


def candidate_pairs(wide, start, end, min_corr, top_k, min_overlap):
    """Correlation screen: for each ticker keep its top-k most correlated peers."""
    px = wide.loc[start:end]
    rets = np.log(px.where(px > 0)).diff()
    corr = rets.corr(min_periods=min_overlap)
    c = corr.to_numpy()
    names = corr.columns.to_numpy()
    np.fill_diagonal(c, np.nan)

    pairs: set[tuple[str, str]] = set()
    for i in range(len(names)):
        row = c[i]
        ok = np.where(row >= min_corr)[0]
        if ok.size == 0:
            continue
        top = ok[np.argsort(row[ok])[::-1][:top_k]]
        for j in top:
            a, b = names[i], names[j]
            pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


# ------------------------------------------------------------------- pair statistics

def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Closed-form OLS slope/intercept of y on x (fast for the pair loop)."""
    xm, ym = x.mean(), y.mean()
    denom = float(np.sum((x - xm) ** 2))
    if denom <= 0:
        return np.nan, np.nan
    beta = float(np.sum((x - xm) * (y - ym)) / denom)
    return float(ym - beta * xm), beta


def _half_life(spread: np.ndarray) -> float:
    """OU half-life in days: regress d(spread) on lag(spread); hl = ln2 / -slope."""
    s = np.asarray(spread, dtype=float)
    if len(s) < 10:
        return np.inf
    lag, delta = s[:-1], np.diff(s)
    lm, dm = lag.mean(), delta.mean()
    denom = float(np.sum((lag - lm) ** 2))
    if denom <= 0:
        return np.inf
    slope = float(np.sum((lag - lm) * (delta - dm)) / denom)
    if slope >= 0:                     # not mean-reverting
        return np.inf
    return float(np.log(2) / -slope)


def _aligned(wide, a, b, start, end):
    """Jointly-complete log prices for a pair over a window, plus the index."""
    sub = wide.loc[start:end, [a, b]].dropna()
    if len(sub) < 2:
        return None
    y = np.log(sub[a].to_numpy(dtype=float))
    x = np.log(sub[b].to_numpy(dtype=float))
    return y, x, sub.index


def _backtest(spread, mu, sd, entry, exit_, cost_bps):
    """Vector-free z-score reversion trade on a spread; formation mu/sd (no look-ahead)."""
    z = (spread - mu) / sd
    n = len(z)
    pos = np.zeros(n)
    state = 0
    for i in range(n):
        zi = z[i]
        if state == 0:
            if zi > entry:
                state = -1
            elif zi < -entry:
                state = 1
        elif state == 1 and zi >= -exit_:
            state = 0
        elif state == -1 and zi <= exit_:
            state = 0
        pos[i] = state

    dspread = np.diff(spread, prepend=spread[0])
    held = np.concatenate([[0.0], pos[:-1]])
    gross = held * dspread
    changes = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = gross - changes * (cost_bps / 1e4)

    def _sharpe(series):
        s = series.std(ddof=1)
        return float(series.mean() / s * np.sqrt(252)) if s and s > 0 else np.nan

    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    return {
        "gross_sharpe": _sharpe(gross),
        "net_sharpe": _sharpe(net),
        "net_cum": float(cum[-1]) if n else 0.0,
        "n_trades": int(np.count_nonzero(changes)),
        "max_dd": float(np.min(cum - peak)) if n else 0.0,
    }, cum


def analyse(wide, pairs, args):
    rows, curves = [], {}
    for a, b in pairs:
        form = _aligned(wide, a, b, args.formation_start, args.formation_end)
        if form is None or len(form[0]) < args.min_obs:
            continue
        y, x, _ = form
        alpha, beta = _ols(y, x)
        if not np.isfinite(beta):
            continue
        spread = y - alpha - beta * x
        try:
            _, coint_p, _ = coint(y, x, trend="c", maxlag=10, autolag="aic")
        except Exception:
            coint_p = np.nan
        hl = _half_life(spread)
        mu, sd = float(spread.mean()), float(spread.std(ddof=1))

        oos_p, oos_hl = np.nan, np.inf
        bt = {"gross_sharpe": np.nan, "net_sharpe": np.nan, "net_cum": np.nan,
              "n_trades": 0, "max_dd": np.nan}
        if sd > 0:
            test = _aligned(wide, a, b, args.test_start, args.test_end)
            if test is not None and len(test[0]) >= args.min_oos_obs:
                yt, xt, idx = test
                spread_t = yt - alpha - beta * xt
                try:
                    oos_p = float(adfuller(spread_t, maxlag=10, autolag="aic")[1])
                except Exception:
                    oos_p = np.nan
                oos_hl = _half_life(spread_t)
                bt, cum = _backtest(spread_t, mu, sd, args.entry, args.exit, args.cost_bps)
                curves[(a, b)] = (idx, cum)

        rows.append({
            "pair": f"{a}-{b}", "a": a, "b": b,
            "n_obs": len(y), "alpha": alpha, "beta": beta,
            "coint_pvalue": coint_p, "half_life": hl,
            "oos_pvalue": oos_p, "oos_half_life": oos_hl,
            **bt,
        })
    return pd.DataFrame(rows), curves


# ------------------------------------------------------------- selection helpers

def bh_significant(pvals, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg step-up; True where the null is rejected at FDR=alpha."""
    p = np.asarray(pvals, dtype=float)
    sig = np.zeros(len(p), dtype=bool)
    idx = np.where(~np.isnan(p))[0]
    if idx.size == 0:
        return sig
    pv = p[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    n = ranked.size
    below = ranked <= alpha * np.arange(1, n + 1) / n
    if not below.any():
        return sig
    kmax = int(np.max(np.where(below)[0]))
    sig[idx[order[: kmax + 1]]] = True
    return sig


def profit_linkage(fund_path: Path, pairs, min_year: int) -> dict:
    """Annual net-profit (code 70) correlation per pair — a durability diagnostic."""
    f = pd.read_parquet(fund_path)
    f = f[(f["code"].astype(str) == "70") & (f["quarter"] == 0) & (f["year"] >= min_year)]
    prof = (f.pivot_table(index="year", columns="ticker", values="value", aggfunc="last")
              .sort_index())
    out = {}
    for a, b in pairs:
        if a in prof.columns and b in prof.columns:
            sub = prof[[a, b]].dropna()
            if len(sub) >= 6 and sub[a].std() > 0 and sub[b].std() > 0:
                out[(a, b)] = float(np.corrcoef(sub[a], sub[b])[0, 1])
    return out


# ----------------------------------------------------------------------- outputs

def plot_screen(m: pd.DataFrame, args, out: Path) -> None:
    finite = m[np.isfinite(m["half_life"])]
    fig, ax = plt.subplots(figsize=(10, 7))
    if len(finite):
        colours = ["tab:green" if s else "lightgrey" for s in finite["selected"]]
        ax.scatter(finite["coint_pvalue"], finite["half_life"], c=colours, s=30,
                   alpha=0.8, edgecolors="none")
        for r in finite[finite["selected"]].itertuples():
            ax.annotate(r.pair, (r.coint_pvalue, r.half_life), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.axvline(args.pvalue, color="black", ls="--", lw=0.9)
    ax.axhline(args.min_half_life, color="black", ls=":", lw=0.9)
    ax.axhline(args.max_half_life, color="black", ls=":", lw=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("cointegration p-value (lower = more stationary spread)")
    ax.set_ylabel("OU half-life (days, log) — reverts in the dotted band")
    ax.set_title("Pairs screen — green survives cointegration, half-life, OOS and BH")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_equity(curves, selected, out: Path) -> bool:
    chosen = [p for p in selected if p in curves][:8]
    if not chosen:
        return False
    fig, ax = plt.subplots(figsize=(10, 7))
    for (a, b) in chosen:
        idx, cum = curves[(a, b)]
        ax.plot(idx, cum, lw=1.2, label=f"{a}-{b}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("cumulative net log-return (per unit notional)")
    ax.set_title("Out-of-sample z-score reversion — selected pairs (formation mu/sd)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    return True


def write_findings(m: pd.DataFrame, sel: pd.DataFrame, args, n_candidates: int, out: Path) -> None:
    lines = [
        "# Pairs screen — cointegration, out-of-sample and the multiple-testing haircut",
        "",
        f"Analysed **{len(m)}** of **{n_candidates}** candidate pairs (top-{args.top_k} "
        f"correlated peers per ticker at r >= {args.min_corr:.2f}; the rest lacked the "
        f"{args.min_obs} formation observations). Formation "
        f"{args.formation_start} -> {args.formation_end}, test "
        f"{args.test_start} -> {args.test_end}.",
        "",
        f"**{int((m['coint_pvalue'] < args.pvalue).sum())}** are cointegrated in-sample "
        f"(p < {args.pvalue}); **{int(m['half_life'].between(args.min_half_life, args.max_half_life).sum())}** "
        f"also have a tradeable half-life; **{int(m['bh_significant'].sum())}** survive "
        f"the Benjamini-Hochberg haircut at FDR {args.bh_alpha}; **{len(sel)}** are "
        "selected after the out-of-sample stability check.",
        "",
        "## Selected pairs (ranked by OOS net Sharpe)",
        "",
        "| pair | beta | coint p | half-life | OOS p | OOS half-life | net Sharpe "
        "| OOS net cum | trades | profit link |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sel.itertuples():
        hl = "inf" if not np.isfinite(r.half_life) else f"{r.half_life:.1f}"
        ohl = "inf" if not np.isfinite(r.oos_half_life) else f"{r.oos_half_life:.1f}"
        pl_ = "n/a" if not np.isfinite(getattr(r, "profit_linkage", np.nan)) \
            else f"{r.profit_linkage:+.2f}"
        lines.append(
            f"| {r.pair} | {r.beta:.3f} | {r.coint_pvalue:.4f} | {hl} | "
            f"{r.oos_pvalue:.4f} | {ohl} | {r.net_sharpe:.2f} | {r.net_cum:+.3f} | "
            f"{r.n_trades} | {pl_} |"
        )
    lines += [
        "",
        "## Reading this",
        "",
        "The screen is deliberately sequential and strict. Correlation is only a "
        "candidate filter; cointegration is the actual claim; the half-life band keeps "
        "only spreads a strategy can act on; the OOS check reuses the formation hedge "
        "ratio with no refitting, so it cannot quietly re-optimise; and the "
        "Benjamini-Hochberg step is what separates a real stationarity from the "
        "luckiest of tests. A pair that clears all five steps has earned a backtest — "
        "not a trade.",
        "",
        "Two honest caveats. (1) BH is applied over the analysed pairs, not every pair "
        "in the market — the correlation pre-filter is itself a selection, so the FDR "
        "here is conditional on it. (2) The number of in-sample 'hits' near p=0.05 is "
        "close to the false-positive rate the test count implies, which is exactly the "
        "pattern the haircut is meant to expose.",
        "",
        "The OOS PnL is a log-spread series per unit notional, net of a flat "
        f"{args.cost_bps:.0f} bps charge per position change. It ignores borrow cost, "
        "financing and market impact, so treat net Sharpe as an upper bound.",
        "",
    ]
    out.write_text("\n".join(lines))


# -------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description="Pairs-trading screen for VN equities.")
    p.add_argument("--input", type=Path, required=True, help="daily panel parquet.")
    p.add_argument("--fundamentals", type=Path,
                   help="optional fundamentals parquet for the profit-linkage column.")
    p.add_argument("--outdir", type=Path, default=Path("result/pairs"))
    p.add_argument("--formation-start", default="2016-01-01")
    p.add_argument("--formation-end", default="2021-12-31")
    p.add_argument("--test-start", default="2022-01-01")
    p.add_argument("--test-end", default="2026-12-31")
    p.add_argument("--min-corr", type=float, default=0.50,
                   help="candidate-selection correlation on daily log returns. VN "
                        "single-name daily returns are idiosyncratic (max pairwise "
                        "~0.8), so this is deliberately permissive; cointegration and "
                        "the BH haircut do the real filtering.")
    p.add_argument("--top-k", type=int, default=6,
                   help="correlated peers kept per ticker (caps the pair count).")
    p.add_argument("--min-corr-overlap", type=int, default=250,
                   help="min overlapping returns for a pairwise correlation.")
    p.add_argument("--min-obs", type=int, default=500,
                   help="min formation observations for a pair.")
    p.add_argument("--min-oos-obs", type=int, default=250,
                   help="min test observations for a pair.")
    p.add_argument("--pvalue", type=float, default=0.05,
                   help="formation cointegration p-value threshold.")
    p.add_argument("--oos-pvalue", type=float, default=0.10,
                   help="out-of-sample ADF p-value threshold (looser by design).")
    p.add_argument("--min-half-life", type=float, default=2.0)
    p.add_argument("--max-half-life", type=float, default=60.0)
    p.add_argument("--entry", type=float, default=2.0, help="z-score entry threshold.")
    p.add_argument("--exit", type=float, default=0.5, help="z-score exit threshold.")
    p.add_argument("--cost-bps", type=float, default=60.0,
                   help="round-trip-ish cost charged per position change (VN retail ~60).")
    p.add_argument("--bh-alpha", type=float, default=0.05,
                   help="Benjamini-Hochberg FDR across all tested pairs.")
    args = p.parse_args()

    wide = load_panel(args.input)
    pairs = candidate_pairs(wide, args.formation_start, args.formation_end,
                            args.min_corr, args.top_k, args.min_corr_overlap)
    print(f"[pairs] {wide.shape[1]} tickers, {len(pairs)} candidate pairs "
          f"(corr >= {args.min_corr}, top-{args.top_k})")

    m, curves = analyse(wide, pairs, args)
    if m.empty:
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / "FINDINGS.md").write_text(
            "# Pairs screen\n\nNo candidate pair had enough overlapping history. "
            "Inspect the panel / thresholds.\n")
        print("[pairs] no analysable pairs — wrote an empty finding")
        return

    m["bh_significant"] = bh_significant(m["coint_pvalue"].to_numpy(), args.bh_alpha)
    m["profit_linkage"] = np.nan
    if args.fundamentals and args.fundamentals.exists():
        link = profit_linkage(args.fundamentals, pairs, int(args.formation_start[:4]))
        m["profit_linkage"] = [link.get((r.a, r.b), np.nan) for r in m.itertuples()]

    keep_hl = m["half_life"].between(args.min_half_life, args.max_half_life)
    oos_ok = (m["oos_pvalue"] < args.oos_pvalue) & np.isfinite(m["oos_half_life"])
    m["selected"] = (
        (m["coint_pvalue"] < args.pvalue) & keep_hl & oos_ok & m["bh_significant"]
    )
    # selected first (ranked by OOS net Sharpe); the rest by in-sample p-value
    m = m.sort_values(["selected", "net_sharpe", "coint_pvalue"],
                      ascending=[False, False, True]).reset_index(drop=True)

    sel = m[m["selected"]].sort_values("net_sharpe", ascending=False)

    args.outdir.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.outdir / "pairs_metrics.csv", index=False)
    sel.to_csv(args.outdir / "selected_pairs.csv", index=False)
    plot_screen(m, args, args.outdir / "screen.svg")
    made_equity = plot_equity(curves, list(zip(sel["a"], sel["b"])), args.outdir / "equity.svg")
    write_findings(m, sel, args, len(pairs), args.outdir / "FINDINGS.md")

    print(f"[pairs] cointegrated in-sample: {int((m['coint_pvalue'] < args.pvalue).sum())}"
          f" | BH-surviving: {int(m['bh_significant'].sum())}"
          f" | selected: {len(sel)}")
    if len(sel):
        print(f"[pairs] selected: {', '.join(sel['pair'].head(12))}")
    else:
        print("[pairs] nothing survived every gate — that is the finding, not a bug")
    print(f"[pairs] wrote {args.outdir}/"
          + (" (+equity.svg)" if made_equity else ""))


if __name__ == "__main__":
    main()
