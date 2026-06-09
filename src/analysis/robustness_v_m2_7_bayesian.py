#!/usr/bin/env python3
"""
V_M2_7 — Bayesian-style posterior probability over break locations.

For each month in the series, estimates Pr(month is a break) via residual
bootstrap of PELT. Unlike point estimates, this produces a full distribution
over candidate break dates — narrow peaks mean high confidence in date;
broad spread means uncertain dating.

Also compares PELT-L2 (used in main analysis) to alternative cost functions:
  - PELT-L1 (robust to outliers)
  - BinSeg-L2 (different algorithm, same cost)
  - PELT-rbf (kernel-based, captures distributional shifts)

Outputs:
  data/structural_break/posterior_breaks.csv
  data/structural_break/alt_methods_breaks.csv
  figures/structural_break/posterior_breaks.png
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERIES = PROJECT_ROOT / "data" / "structural_break" / "monthly_series_v2.csv"
OUT_DIR = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR = PROJECT_ROOT / "figures" / "structural_break"

ACCENT, COOL, CHAR, MUTED = "#B85042", "#2F5F5D", "#363A3E", "#707070"
BREAK_COLOR = "#E08D17"

MIN_SEG = 12
BIC_PEN_FACTOR = 2.0
N_BOOT = 500
RNG = np.random.default_rng(42)

# Series to analyze — use the C1-corrected series (gap_*_mean_C1)
SERIES_SPECS = [
    ("gap_threat_HL_FT_mean_C1", "Headline−body THREAT gap (C1)",    ACCENT),
    ("gap_diplo_HL_FT_mean_C1",  "Headline−body DIPLOMACY gap (C1)", COOL),
    ("gap_human_HL_FT_mean_C1",  "Headline−body HUMANIZING gap (C1)", CHAR),
]

EVENTS = [
    ("2002-01-29", "Axis of Evil"),
    ("2013-06-15", "Rouhani"),
    ("2015-07-14", "JCPOA signed"),
    ("2016-01-16", "JCPOA Imp Day"),
    ("2018-05-08", "JCPOA exit"),
    ("2020-01-03", "Soleimani"),
    ("2022-09-16", "Mahsa Amini"),
]


def pelt_l2(values: np.ndarray) -> list[int]:
    sigma = float(np.std(values))
    pen = BIC_PEN_FACTOR * sigma**2 * np.log(len(values))
    algo = rpt.Pelt(model="l2", min_size=MIN_SEG).fit(values.reshape(-1, 1))
    breaks = algo.predict(pen=pen)
    return [b for b in breaks if b < len(values)]


def pelt_l1(values: np.ndarray) -> list[int]:
    sigma = float(np.std(values))
    pen = BIC_PEN_FACTOR * sigma * np.log(len(values))  # L1 uses sigma not sigma^2
    algo = rpt.Pelt(model="l1", min_size=MIN_SEG).fit(values.reshape(-1, 1))
    breaks = algo.predict(pen=pen)
    return [b for b in breaks if b < len(values)]


def binseg_l2(values: np.ndarray) -> list[int]:
    sigma = float(np.std(values))
    pen = BIC_PEN_FACTOR * sigma**2 * np.log(len(values))
    algo = rpt.Binseg(model="l2", min_size=MIN_SEG).fit(values.reshape(-1, 1))
    breaks = algo.predict(pen=pen)
    return [b for b in breaks if b < len(values)]


def pelt_rbf(values: np.ndarray) -> list[int]:
    # rbf cost — captures distributional shifts, not just mean shifts
    sigma = float(np.std(values))
    pen = BIC_PEN_FACTOR * np.log(len(values))
    algo = rpt.Pelt(model="rbf", min_size=MIN_SEG).fit(values.reshape(-1, 1))
    breaks = algo.predict(pen=pen)
    return [b for b in breaks if b < len(values)]


def posterior_break_counts(values: np.ndarray, n_boot: int = N_BOOT) -> np.ndarray:
    """For each index, count how often it gets picked as a break under bootstrap."""
    # Build piecewise-constant baseline from PELT-L2 fit
    detected = pelt_l2(values)
    if not detected:
        return np.zeros(len(values))
    segs = [0] + detected + [len(values)]
    baseline = np.zeros_like(values, dtype=float)
    for a, b in zip(segs[:-1], segs[1:]):
        baseline[a:b] = values[a:b].mean()
    residuals = values - baseline

    counts = np.zeros(len(values))
    successes = 0
    for _ in range(n_boot):
        resampled = baseline + RNG.choice(residuals, size=len(residuals), replace=True)
        try:
            bks = pelt_l2(resampled)
        except Exception:
            continue
        successes += 1
        for b in bks:
            # Count with ±2-month tolerance (smooth out tiny bootstrap jitter)
            for offset in [-2, -1, 0, 1, 2]:
                if 0 <= b + offset < len(counts):
                    counts[b + offset] += 0.2  # split mass over the window
    return counts / max(successes, 1)


def main():
    df = pd.read_csv(SERIES, parse_dates=["pub_month"]).set_index("pub_month")

    all_alt_breaks = []
    posteriors = {}

    for col, label, color in SERIES_SPECS:
        s = df[col].dropna()
        if len(s) < 2 * MIN_SEG:
            continue
        v = s.values
        d = s.index

        print(f"\n=== {label} ===")

        # Alt methods
        for method_name, fn in [("PELT_L2",  pelt_l2),
                                 ("PELT_L1",  pelt_l1),
                                 ("BinSeg_L2", binseg_l2),
                                 ("PELT_rbf", pelt_rbf)]:
            try:
                bks = fn(v)
            except Exception as e:
                print(f"  {method_name:11s}: error — {e}")
                continue
            dates = [d[b].date() for b in bks]
            print(f"  {method_name:11s}: {dates}")
            for b in bks:
                all_alt_breaks.append(dict(
                    series=col, method=method_name,
                    break_date=d[b].date(), break_idx=int(b),
                ))

        # Posterior
        post = posterior_break_counts(v)
        posteriors[col] = (d, post)
        # Top 5 modes
        top_idx = np.argsort(post)[::-1][:5]
        top_dates = [(d[i].date(), float(post[i])) for i in sorted(top_idx)]
        print(f"  Top-5 posterior peaks: " + ", ".join(f"{dd}({p:.2f})" for dd, p in top_dates))

    pd.DataFrame(all_alt_breaks).to_csv(OUT_DIR / "alt_methods_breaks.csv", index=False)

    # Posterior CSV
    post_rows = []
    for col, (d, post) in posteriors.items():
        for di, p in zip(d, post):
            post_rows.append(dict(series=col, pub_month=di.date(), posterior=float(p)))
    pd.DataFrame(post_rows).to_csv(OUT_DIR / "posterior_breaks.csv", index=False)

    # Figure: 3 panels, each shows series + posterior bars
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=150, sharex=True)
    for ax, (col, label, color) in zip(axes, SERIES_SPECS):
        s = df[col].dropna()
        if s.empty:
            continue
        d, post = posteriors[col]
        ax_post = ax.twinx()
        # Bars: posterior probability
        ax_post.fill_between(d, 0, post, color=BREAK_COLOR, alpha=0.45, lw=0,
                             label="Bootstrap posterior")
        ax_post.set_ylim(0, max(1.0, float(post.max()) * 1.05))
        ax_post.tick_params(axis="y", labelsize=7, colors=BREAK_COLOR)
        ax_post.set_ylabel("Pr(break)", fontsize=8, color=BREAK_COLOR)
        # Line: 12mo MA of series
        ax.plot(s.index, s.rolling(12, min_periods=1).mean(),
                color=CHAR, lw=1.3, label="C1 series (12mo MA)")
        # Event markers
        for ed, el in EVENTS:
            ax.axvline(pd.Timestamp(ed), color=MUTED, lw=0.6, alpha=0.5, ls=":")
        ax.set_title(label, fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=7, frameon=False)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Year", fontsize=9)
    fig.suptitle("Bootstrap posterior probability of a break per month  (N=500, ±2mo window)",
                 fontsize=11, y=0.995)
    plt.tight_layout()
    fig_path = FIG_DIR / "posterior_breaks.png"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"\nSaved: {fig_path}")
    print(f"Saved: {OUT_DIR / 'alt_methods_breaks.csv'}")
    print(f"Saved: {OUT_DIR / 'posterior_breaks.csv'}")


if __name__ == "__main__":
    main()
