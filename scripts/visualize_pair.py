#!/usr/bin/env python3
"""project4 — visualise one candidate pair and *why* it looked good.

Reads the daily panel and redraws, for a single pair, the exact objects the screen
measures: the two normalised price paths, the hedge-ratio spread with its formation
mean and sigma bands, and the z-score with the entry/exit levels the backtest used.
The point is to show the mechanism, not just the score:

  - the two legs visibly move together (the candidate correlation),
  - the spread wanders around a flat mean instead of trending (cointegration),
  - it crosses the +/-2 sigma bands and comes back within the half-life window
    (the reversion the trade harvests),
  - and it still does so in the out-of-sample period with the FORMATION hedge
    ratio, which is why it looked robust rather than fitted.

Usage:
    python scripts/visualize_pair.py --input data/raw/vn_daily_panel.parquet \
        --pair PVS-PVT --outdir result/pairs
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


def load(wide_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(wide_path)
    df["date"] = pd.to_datetime(df["date"])
    return (df.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
              .sort_index())


def ols(y, x):
    xm, ym = x.mean(), y.mean()
    beta = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    return float(ym - beta * xm), beta


def half_life(spread):
    s = np.asarray(spread, float)
    lag, delta = s[:-1], np.diff(s)
    lm, dm = lag.mean(), delta.mean()
    b = float(np.sum((lag - lm) * (delta - dm)) / np.sum((lag - lm) ** 2))
    return float(np.log(2) / -b) if b < 0 else np.inf


def positions(z, entry, exit_):
    pos = np.zeros(len(z))
    state = 0
    for i, zi in enumerate(z):
        if state == 0:
            state = -1 if zi > entry else (1 if zi < -entry else 0)
        elif state == 1 and zi >= -exit_:
            state = 0
        elif state == -1 and zi <= exit_:
            state = 0
        pos[i] = state
    return pos


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualise one pairs-screen candidate.")
    ap.add_argument("--input", type=Path, required=True, help="daily panel parquet.")
    ap.add_argument("--pair", required=True, help="e.g. PVS-PVT")
    ap.add_argument("--outdir", type=Path, default=Path("result/pairs"))
    ap.add_argument("--formation-start", default="2016-01-01")
    ap.add_argument("--formation-end", default="2021-12-31")
    ap.add_argument("--test-start", default="2022-01-01")
    ap.add_argument("--test-end", default="2026-12-31")
    ap.add_argument("--entry", type=float, default=2.0)
    ap.add_argument("--exit", type=float, default=0.5)
    ap.add_argument("--cost-bps", type=float, default=60.0)
    args = ap.parse_args()

    a, b = args.pair.split("-")
    wide = load(args.input)

    full = wide.loc[args.formation_start:args.test_end, [a, b]].dropna()
    form = wide.loc[args.formation_start:args.formation_end, [a, b]].dropna()
    if len(form) < 50 or len(full) < 50:
        raise SystemExit(f"not enough overlapping history for {args.pair}")

    yf = np.log(form[a].to_numpy(float)); xf = np.log(form[b].to_numpy(float))
    alpha, beta = ols(yf, xf)
    _, coint_p, _ = coint(yf, xf, trend="c", maxlag=10, autolag="aic")
    hl = half_life(yf - alpha - beta * xf)

    y = np.log(full[a].to_numpy(float)); x = np.log(full[b].to_numpy(float))
    spread = y - alpha - beta * x
    mu, sd = float(np.mean(yf - alpha - beta * xf)), float(np.std(yf - alpha - beta * xf, ddof=1))
    z = (spread - mu) / sd
    idx = full.index

    split = pd.Timestamp(args.test_start)
    test_mask = idx >= split
    oos_z = z[test_mask]
    oos_p = float(adfuller(spread[test_mask], maxlag=10, autolag="aic")[1])
    oos_hl = half_life(spread[test_mask])
    pos = positions(oos_z, args.entry, args.exit)
    dspread = np.diff(spread[test_mask], prepend=spread[test_mask][0])
    held = np.concatenate([[0.0], pos[:-1]])
    net = held * dspread - np.abs(np.diff(np.concatenate([[0.0], pos]))) * args.cost_bps / 1e4
    sharpe = (net.mean() / net.std(ddof=1) * np.sqrt(252)) if net.std(ddof=1) > 0 else np.nan
    n_tr = int(np.count_nonzero(np.diff(np.concatenate([[0.0], pos]))))

    lw = 1.2
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2]})
    norm = 100.0
    for tk, col in ((a, "tab:blue"), (b, "tab:orange")):
        s = full[tk] / full[tk].iloc[0] * norm
        axes[0].plot(idx, s, lw=lw, color=col, label=tk)
    axes[0].axvline(split, color="black", ls="--", lw=1.0)
    axes[0].text(split, axes[0].get_ylim()[1], "  out-of-sample ->", va="top", fontsize=9)
    axes[0].set_ylabel(f"price, rebased to {norm:.0f}")
    axes[0].set_title(
        f"{a} vs {b}   |   beta={beta:.3f}   coint p={coint_p:.3f}   half-life={hl:.0f}d"
        f"   |   OOS p={oos_p:.3f}   OOS half-life={oos_hl:.0f}d   "
        f"OOS net Sharpe={sharpe:.2f}   ({n_tr} trades)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)
    axes[0].set_xlim(idx[0], idx[-1])

    axes[1].plot(idx, spread, lw=lw, color="tab:purple")
    for k, style in ((1, ":"), (2, "--")):
        axes[1].axhline(mu + k * sd, color="grey", ls=style, lw=0.9)
        axes[1].axhline(mu - k * sd, color="grey", ls=style, lw=0.9)
    axes[1].axhline(mu, color="black", lw=0.9)
    axes[1].axvline(split, color="black", ls="--", lw=1.0)
    axes[1].set_ylabel(f"spread = log({a}) - {beta:.2f}*log({b})")
    axes[1].grid(alpha=0.25)

    axes[2].plot(idx, z, lw=lw, color="tab:green")
    for lvl in (args.entry, -args.entry):
        axes[2].axhline(lvl, color="red", ls="--", lw=0.9)
    for lvl in (args.exit, -args.exit):
        axes[2].axhline(lvl, color="grey", ls=":", lw=0.9)
    axes[2].axhline(0, color="black", lw=0.9)
    axes[2].axvline(split, color="black", ls="--", lw=1.0)
    axes[2].axvspan(split, idx[-1], color="grey", alpha=0.07)
    axes[2].fill_between(idx, -4, 4, where=(np.abs(z) > args.entry) & ~test_mask,
                         color="tab:red", alpha=0.12)
    axes[2].set_ylabel("z-score")
    axes[2].set_xlabel("date")
    axes[2].set_ylim(-4, 4)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"{args.pair}.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    print(f"[viz] {args.pair}: beta={beta:.3f} coint_p={coint_p:.4f} hl={hl:.1f}d "
          f"oos_p={oos_p:.4f} oos_hl={oos_hl:.1f}d sharpe={sharpe:.2f} trades={n_tr}")
    print(f"[viz] wrote {out}")


if __name__ == "__main__":
    main()
