#!/usr/bin/env python3
"""
V_M2_6 — Add BH-FDR alongside the existing Bonferroni column.

FDR (Benjamini-Hochberg) is more powerful than Bonferroni for related tests
(news-editorial across same series, near-event lags). At alpha=0.05, FDR
controls expected proportion of false positives, not family-wise error rate.

Outputs:
  data/structural_break/chow_tests_fdr.csv  — adds p_fdr_bh, sig_fdr
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN = PROJECT_ROOT / "data" / "structural_break" / "chow_tests.csv"
OUT = PROJECT_ROOT / "data" / "structural_break" / "chow_tests_fdr.csv"
ALPHA = 0.05


def bh_fdr(pvals: np.ndarray, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up. Returns (sig_mask, adjusted_p)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    sorted_p = p[order]
    # Adjusted p = min over k >= i of (p_(k) * n / k)
    adj_sorted = np.minimum.accumulate(sorted_p[::-1] * n / ranks[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    # restore original order
    adj = np.empty(n)
    adj[order] = adj_sorted
    sig = adj <= alpha
    return sig, adj


def main() -> None:
    df = pd.read_csv(IN)
    mask = df["p_value"].notna()
    sub = df[mask].copy()

    sig_fdr, adj = bh_fdr(sub["p_value"].values)
    sub["p_fdr_bh"] = adj
    sub["sig_fdr"] = sig_fdr

    out = df.merge(sub[["series", "event", "p_fdr_bh", "sig_fdr"]],
                   on=["series", "event"], how="left")
    out.to_csv(OUT, index=False)

    print(f"Total Chow tests: {len(df)}  (non-null p-values: {mask.sum()})")
    print()
    print("Method        : sig count / total")
    print(f"  raw p<0.05  : {(df['p_value'] < 0.05).sum():>3d} / {mask.sum()}")
    print(f"  Bonferroni  : {df['sig'].sum():>3d} / {mask.sum()}")
    print(f"  BH-FDR      : {sub['sig_fdr'].sum():>3d} / {mask.sum()}")

    print()
    print("Discordant: passes Bonferroni but NOT FDR? (impossible — FDR ≥ Bonf)")
    only_bonf = df[(df["sig"] == True) & (~out["sig_fdr"].fillna(False))]
    print(f"  count: {len(only_bonf)}")
    print()
    print("New under FDR (passes FDR, fails Bonferroni):")
    new_fdr = out[(out["sig_fdr"] == True) & (out["sig"] == False)]
    if len(new_fdr) == 0:
        print("  (none)")
    else:
        print(new_fdr[["series","event","p_value","p_bonferroni","p_fdr_bh","shift"]].to_string(index=False))
    print()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
