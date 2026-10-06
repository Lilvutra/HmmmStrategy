#!/usr/bin/env python3
"""project4 — small-multiples view of EVERY analysed pair.

scripts/visualize_pair.py shows one pair in detail. This shows them all at once:
one mini-panel per candidate pair, each the z-score of its hedge spread with the
formation/test boundary, the +/-entry lines and colour-coded by outcome. Sorted by
in-sample cointegration p-value, it is the whole screen as a picture — you can see
at a glance how many "look" reversionary and how few stay so out of sample.

Outputs (to --outdir, default result/pairs/):
  - all_pairs.svg           every analysed pair, sorted by coint p-value
  - cointegrated_pairs.svg  only the pairs with coint p < --pvalue, larger cells

Usage:
    python scripts/visualize_all_pairs.py --input data/raw/vn_daily_panel.parquet \
        --metrics result/pairs/pairs_metrics.csv --outdir result/pairs
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_wide(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return (df.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
              .sort_index())


def spread_z(wide, a, b, alpha, beta, start, end, fend):
    full = wide.loc[start:end, [a, b]].dropna()
    if len(full) < 50:
        return None, None
    y = np.log(full[a].to_numpy(float))
    x = np.log(full[b].to_numpy(float))
    spread = y - alpha - beta * x
    idx = full.index
    fmask = idx <= pd.Timestamp(fend)
    mu, sd = float(spread[fmask].mean()), float(spread[fmask].std(ddof=1))
    if sd <= 0:
        return None, None
    return idx, (spread - mu) / sd


def cell(ax, idx, z, split, entry, colour, title):
    ax.plot(idx, z, lw=0.7, color="tab:green")
    ax.axvline(split, color="black", ls="--", lw=0.6)
    ax.axhline(0, color="black", lw=0.6)
    for lvl in (entry, -entry):
        ax.axhline(lvl, color="red", ls=":", lw=0.6)
    ax.set_ylim(-4, 4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(colour)
    ax.set_title(title, fontsize=5.5, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid view of all pairs.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, default=Path("result/pairs/pairs_metrics.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("result/pairs"))
    ap.add_argument("--formation-start", default="2016-01-01")
    ap.add_argument("--formation-end", default="2021-12-31")
    ap.add_argument("--test-start", default="2022-01-01")
    ap.add_argument("--test-end", default="2026-12-31")
    ap.add_argument("--entry", type=float, default=2.0)
    ap.add_argument("--pvalue", type=float, default=0.05)
    ap.add_argument("--ncols", type=int, default=12)
    args = ap.parse_args()

    wide = load_wide(args.input)
    m = pd.read_csv(args.metrics).sort_values("coint_pvalue").reset_index(drop=True)

    panels = []
    for r in m.itertuples():
        idx, z = spread_z(wide, r.a, r.b, r.alpha, r.beta,
                          args.formation_start, args.test_end, args.formation_end)
        if idx is None:
            continue
        if r.selected:
            colour = "#c8f7c5"          # green: passed every gate
        elif r.coint_pvalue < args.pvalue:
            colour = "#ffe9b0"          # amber: cointegrated in-sample only
        else:
            colour = "#f2f2f2"          # grey: did not clear cointegration
        panels.append((r.pair, r.coint_pvalue, r.bh_significant, idx, z, colour))
    print(f"[grid] plotting {len(panels)} pairs")

    def build(rows, ncols, fname, title):
        n = len(rows)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(1.85 * ncols, 1.35 * nrows + 1.2),
                                 squeeze=False)
        split = pd.Timestamp(args.test_start)
        for i, (pair, pval, bh, idx, z, colour) in enumerate(rows):
            ax = axes[i // ncols][i % ncols]
            tag = f"{pair}\np={pval:.3f}" + ("  BH" if bh else "")
            cell(ax, idx, z, split, args.entry, colour, tag)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(title, fontsize=12)
        fig.text(0.5, 0.005,
                 "grey = not cointegrated   amber = coint p<0.05 in-sample only   "
                 "green = selected;  dashed = formation | test split;  red dotted = +/-2 sigma",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.015, 1, 0.985))
        out = args.outdir / fname
        fig.savefig(out, format="svg")
        plt.close(fig)
        print(f"[grid] wrote {out}")

    build(panels, args.ncols, "all_pairs.svg",
          f"All {len(panels)} analysed pairs — z-score of the hedge spread, sorted by in-sample cointegration p-value")

    coint_only = [p for p in panels if p[1] < args.pvalue]
    if coint_only:
        build(coint_only, min(5, len(coint_only)), "cointegrated_pairs.svg",
              f"The {len(coint_only)} pairs cointegrated in-sample (p < {args.pvalue}) — none survive BH")
    else:
        print("[grid] no in-sample cointegrated pairs — skipping zoom grid")


if __name__ == "__main__":
    main()
