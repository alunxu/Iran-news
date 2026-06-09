#!/usr/bin/env python3
r"""
Module 7 -- Joint multi-series PELT.

Single-series PELT (used in \S 1--\S 2 of the notes) flags breaks where one
tension series shifts.  It misses moments where several series shift together
with small per-series magnitude.  Multivariate PELT pools the evidence across
the (4 layers $\times$ 3 axes = 12) cell series and asks whether a single
joint break date better explains the data.

We use \texttt{ruptures.Pelt(model='l2')} on the (T $\times$ 12) detrended-and-
z-scored panel.  Detrending per series first removes the trend confound that
\S 1's Module 2.5 identified; z-scoring puts all 12 cells on the same scale so
no single cell dominates.

Outputs:
  data/structural_break/joint_breaks.csv
  figures/structural_break/joint_breaks.png
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYER_CSV = PROJECT_ROOT / "data" / "structural_break" / "layer_series.csv"
OUT_DIR = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR = PROJECT_ROOT / "figures" / "structural_break"

ACCENT, COOL, CHAR, MUTED = "#B85042", "#2F5F5D", "#363A3E", "#707070"

LAYERS = ["hl", "ab", "ld", "ft"]
AXES = ["threat", "diplo", "human"]
MIN_SEG = 12
BIC_PEN_FACTOR = 2.0

EVENTS = [
    ("1979-02-01", "Khomeini returns"),
    ("1979-11-04", "Embassy seized"),
    ("1980-09-22", "Iran-Iraq war"),
    ("1988-07-03", "IR655"),
    ("1989-06-03", "Khomeini dies"),
    ("2002-01-29", "Axis of Evil"),
    ("2009-06-13", "Green Movement"),
    ("2013-06-15", "Rouhani"),
    ("2015-07-14", "JCPOA signed"),
    ("2016-01-16", "JCPOA Imp Day"),
    ("2018-05-08", "JCPOA exit"),
    ("2020-01-03", "Soleimani"),
    ("2022-09-16", "Mahsa Amini"),
]

LABEL_EVENTS = [
    ("2002-01-29", "Axis"),
    ("2013-06-15", "Rouhani"),
    ("2016-01-16", "JCPOA"),
    ("2018-05-08", "Exit"),
    ("2020-01-03", "Soleimani"),
    ("2022-09-16", "Mahsa"),
    ("2023-10-07", "Gaza"),
]


def detrend_series(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if len(s) < 24:
        return s
    t = np.arange(len(s), dtype=float)
    slope, intercept, *_ = stats.linregress(t, s.values)
    return s - (slope * t + intercept)


def zscore_series(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std() if s.std() > 1e-6 else s * 0


def nearest_event(date: pd.Timestamp, window_months: int = 12):
    deltas = [(lbl, (date - pd.Timestamp(d)).days / 30.4) for d, lbl in EVENTS]
    deltas.sort(key=lambda x: abs(x[1]))
    label, delta = deltas[0]
    if abs(delta) <= window_months:
        return label, round(delta)
    return None, None


def main():
    df = pd.read_csv(LAYER_CSV, parse_dates=["pub_month"]).set_index("pub_month").sort_index()

    # Build (T x 12) panel of detrended, z-scored series.
    cells = [f"{l}_{a}" for a in AXES for l in LAYERS]
    panel = pd.DataFrame(index=df.index)
    for c in cells:
        s = df[c].dropna()
        s = detrend_series(s)
        s = zscore_series(s)
        panel[c] = s
    panel = panel.dropna()

    print(f"Joint panel: {panel.shape[0]} months × {panel.shape[1]} cells")

    # Run multivariate PELT
    X = panel.values  # (T, d)
    T, d = X.shape
    # Penalty: scale BIC by dimensionality. Per-cell variance is 1 after z-score.
    pen = BIC_PEN_FACTOR * d * np.log(T)
    algo = rpt.Pelt(model="l2", min_size=MIN_SEG).fit(X)
    raw_breaks = algo.predict(pen=pen)
    break_idx = [b for b in raw_breaks if b < T]
    print(f"Joint PELT detected {len(break_idx)} break(s) "
          f"(penalty = {pen:.1f}, BIC × dim × log T)")

    # Compute per-segment means per cell, and shift magnitude at each break
    segs = [0] + break_idx + [T]
    break_rows = []
    for i, b in enumerate(break_idx):
        bdate = panel.index[b]
        ev_lbl, ev_delta = nearest_event(bdate)
        # Pre / post: 12 months on each side (or the segment's bounds)
        pre_start = max(segs[i], b - 12)
        post_end  = min(segs[i + 2], b + 12)
        pre = X[pre_start:b].mean(axis=0)
        post = X[b:post_end].mean(axis=0)
        shifts = post - pre
        # Top 5 cells by absolute shift
        top_idx = np.argsort(np.abs(shifts))[::-1][:5]
        top_cells = [(cells[k], float(shifts[k])) for k in top_idx]
        break_rows.append(dict(
            break_date=bdate.date(),
            n_segments_pre=int(b - pre_start),
            n_segments_post=int(post_end - b),
            nearest_event=ev_lbl,
            event_delta_months=ev_delta,
            top_5_cells=", ".join(f"{c}={s:+.2f}" for c, s in top_cells),
            total_l2_shift=float(np.linalg.norm(shifts)),
        ))
        print(f"\nJoint break {i+1}: {bdate.date()}  (near={ev_lbl}, Δ={ev_delta}mo)")
        for c, s in top_cells:
            print(f"    {c:14s} z-shift = {s:+.2f}")

    pd.DataFrame(break_rows).to_csv(OUT_DIR / "joint_breaks.csv", index=False)

    # ────────────────────────────────────────────────────────────────────
    # Figure: each of 12 cells stacked as a small panel, with joint breaks
    # marked. The reader can see which cells contribute to each break.
    # ────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(AXES), len(LAYERS), figsize=(13, 6.5), dpi=130, sharex=True)
    for r, a in enumerate(AXES):
        for c, l in enumerate(LAYERS):
            ax = axes[r, c]
            cell = f"{l}_{a}"
            s = panel[cell].rolling(12, min_periods=1).mean()
            ax.axhline(0, color="black", lw=0.4, alpha=0.5)
            ax.plot(s.index, s.values,
                    color={"threat": ACCENT, "diplo": COOL, "human": CHAR}[a],
                    lw=0.9, alpha=0.9)
            for ed, _ in LABEL_EVENTS:
                ax.axvline(pd.Timestamp(ed), color=MUTED, lw=0.55, alpha=0.36, ls="-", zorder=0)
            for b in break_idx:
                ax.axvline(panel.index[b], color="#E08D17", lw=1.0, alpha=0.85, ls="--")
            ax.set_ylim(-2.5, 2.5)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.15)
            if r == 0:
                ax.set_title(l.upper(), fontsize=9)
            if c == 0:
                ax.set_ylabel(a.upper(), fontsize=8,
                              color={"threat": ACCENT, "diplo": COOL, "human": CHAR}[a])
    top_ax = axes[0, 0]
    for idx, (ed, label) in enumerate(LABEL_EVENTS):
        d = mdates.date2num(pd.Timestamp(ed).to_pydatetime())
        top_ax.annotate(
            label,
            xy=(d, 1.01),
            xycoords=top_ax.get_xaxis_transform(),
            xytext=(d, 1.18 + (idx % 2) * 0.10),
            textcoords=top_ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
            rotation=42,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.65, alpha=0.75),
            annotation_clip=False,
        )
    for ax in axes[-1, :]:
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"Joint multivariate PELT on 12-cell panel (detrended + z-scored, "
                 f"12mo MA shown). {len(break_idx)} break(s) detected, marked in orange.",
                 fontsize=10, y=0.995)
    plt.tight_layout()
    fig_path = FIG_DIR / "joint_breaks.png"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"\nSaved: {fig_path}")
    print(f"Saved: {OUT_DIR / 'joint_breaks.csv'}")


if __name__ == "__main__":
    main()
