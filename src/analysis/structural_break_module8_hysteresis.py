#!/usr/bin/env python3
r"""
Module 8 -- Hysteresis: transient shock vs persistent baseline shift.

\S 2 of the notes measured each event's pre/post shift in a +/-12-month window.
That window cannot distinguish "transient shock that returned to baseline" from
"persistent baseline shift that stayed at the new level."  This module extends the
post-event window to +36 months and measures persistence directly.

For each event we compute the shift relative to a pre-event 12-month mean at
four post-event horizons (months 1-6, 1-12, 13-24, 25-36), then form the
persistence ratio Δ_36 / Δ_6.  Ratio ~ 1.0 = persistent shift; ratio ~ 0 = full
return to baseline; in between = partial decay.

Outputs:
  data/structural_break/hysteresis_shifts.csv
  data/structural_break/hysteresis_persistence.csv
  figures/structural_break/hysteresis_curves.png
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYER_CSV = PROJECT_ROOT / "data" / "structural_break" / "layer_series.csv"
OUT_DIR = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR = PROJECT_ROOT / "figures" / "structural_break"

ACCENT, COOL, CHAR, MUTED = "#B85042", "#2F5F5D", "#363A3E", "#707070"

EVENTS = [
    ("1979-02-01", "Khomeini returns",  "domestic"),
    ("1979-11-04", "Embassy seized",    "conflict"),
    ("1980-09-22", "Iran-Iraq war",     "conflict"),
    ("1988-07-03", "IR655",             "conflict"),
    ("1989-06-03", "Khomeini dies",     "domestic"),
    ("2002-01-29", "Axis of Evil",      "policy"),
    ("2009-06-13", "Green Movement",    "domestic"),
    ("2013-06-15", "Rouhani elected",   "domestic"),
    ("2015-07-14", "JCPOA signed",      "diplomacy"),
    ("2016-01-16", "JCPOA Imp Day",     "diplomacy"),
    ("2018-05-08", "JCPOA exit",        "diplomacy"),
    ("2020-01-03", "Soleimani",         "conflict"),
    ("2022-09-16", "Mahsa Amini",       "domestic"),
]

LAYERS = [("hl", "Headline", ACCENT),
          ("ab", "Abstract", "#9B6A4F"),
          ("ld", "Lead",     COOL),
          ("ft", "Body",     CHAR)]
AXES = [("threat", "THREAT", ACCENT),
        ("diplo",  "DIPLO",  COOL),
        ("human",  "HUMAN",  CHAR)]
SUBTYPE_ORDER = ["conflict", "diplomacy", "domestic", "policy"]

PRE_WINDOW = 12      # months before event for baseline
HORIZONS = [(1, 6), (1, 12), (13, 24), (25, 36)]   # (start, end) inclusive, months after event
H_LABELS = ["Δ_6", "Δ_12", "Δ_24", "Δ_36"]
PLOT_RANGE = (-12, 36)  # trajectory window in figure


def window_mean(df: pd.DataFrame, anchor: pd.Timestamp, start_mo: int, end_mo: int,
                col: str) -> float:
    """Mean of col over months [anchor+start_mo, anchor+end_mo] inclusive."""
    a = anchor.replace(day=1)
    vals = []
    for k in range(start_mo, end_mo + 1):
        m = a + pd.DateOffset(months=int(k))
        if m in df.index:
            v = df.loc[m, col]
            if not pd.isna(v):
                vals.append(float(v))
    return float(np.mean(vals)) if vals else np.nan


def main():
    df = pd.read_csv(LAYER_CSV, parse_dates=["pub_month"]).set_index("pub_month").sort_index()

    # Per-event per-cell shifts at each horizon
    shift_rows = []
    for event_str, label, subtype in EVENTS:
        ev = pd.Timestamp(event_str)
        for axis_key, axis_label, _ in AXES:
            for layer_key, layer_name, _ in LAYERS:
                col = f"{layer_key}_{axis_key}"
                pre = window_mean(df, ev, -PRE_WINDOW, -2, col)
                row = dict(event=label, event_date=ev.date(), subtype=subtype,
                           layer=layer_key, layer_name=layer_name,
                           axis=axis_key, pre_mean=pre)
                for (start, end), h_label in zip(HORIZONS, H_LABELS):
                    post = window_mean(df, ev, start, end, col)
                    row[f"post_mean_{h_label}"] = post
                    row[f"shift_{h_label}"] = post - pre if not (np.isnan(post) or np.isnan(pre)) else np.nan
                shift_rows.append(row)
    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv(OUT_DIR / "hysteresis_shifts.csv", index=False)

    # Persistence ratio: Δ_36 / Δ_6  (only meaningful when |Δ_6| > 0.5/1k tokens)
    persist_rows = []
    for _, r in shift_df.iterrows():
        d6 = r["shift_Δ_6"]
        d36 = r["shift_Δ_36"]
        if not pd.isna(d6) and abs(d6) > 0.5 and not pd.isna(d36):
            pratio = d36 / d6
            verdict = ("regime change" if pratio > 0.7 else
                       "partial decay" if pratio > 0.3 else
                       "transient (decayed)" if pratio > -0.3 else
                       "reverted (overshoot)")
        else:
            pratio = np.nan
            verdict = "insufficient data"
        persist_rows.append(dict(event=r["event"], subtype=r["subtype"],
                                 axis=r["axis"], layer=r["layer"],
                                 layer_name=r["layer_name"],
                                 shift_6mo=d6, shift_36mo=d36,
                                 persistence_ratio=pratio, verdict=verdict))
    persist_df = pd.DataFrame(persist_rows)
    persist_df.to_csv(OUT_DIR / "hysteresis_persistence.csv", index=False)

    # Print summary: regime-change cells per event
    print("=== Cells with |Δ_6| > 1.0 and known persistence ratio ===")
    big = persist_df[(persist_df["shift_6mo"].abs() > 1.0) & persist_df["persistence_ratio"].notna()].copy()
    big = big.sort_values(["subtype", "event", "axis"])
    print(f"{'Event':<22s}{'subtype':<12s}{'axis':<8s}{'layer':<5s}{'Δ_6':>8s}{'Δ_36':>8s}{'ratio':>8s}  verdict")
    for _, r in big.iterrows():
        print(f"{r['event']:<22s}{r['subtype']:<12s}{r['axis']:<8s}{r['layer']:<5s}"
              f"{r['shift_6mo']:>+8.2f}{r['shift_36mo']:>+8.2f}{r['persistence_ratio']:>+8.2f}  {r['verdict']}")

    # Subtype-level summary: how persistent are each subtype's responses on average?
    print("\n=== Mean persistence ratio by subtype × axis (cells with |Δ_6| > 1.0 only) ===")
    bsub = (big.groupby(["subtype", "axis"])["persistence_ratio"]
                .agg(["mean", "count"]).reset_index())
    print(bsub.to_string(index=False))

    # ────────────────────────────────────────────────────────────────────
    # Figure: extended-window response curves for each subtype × axis
    # Pre/post trajectory averaged across events within subtype, layers shown
    # ────────────────────────────────────────────────────────────────────
    curves = {}  # (subtype, axis, layer) -> 2D array (events × months)
    rel_months = np.arange(PLOT_RANGE[0], PLOT_RANGE[1] + 1)
    for event_str, label, subtype in EVENTS:
        ev = pd.Timestamp(event_str)
        for axis_key, _, _ in AXES:
            for layer_key, _, _ in LAYERS:
                col = f"{layer_key}_{axis_key}"
                pre = window_mean(df, ev, -PRE_WINDOW, -2, col)
                if np.isnan(pre):
                    continue
                vals = []
                for k in rel_months:
                    m = ev.replace(day=1) + pd.DateOffset(months=int(k))
                    if m in df.index and not pd.isna(df.loc[m, col]):
                        vals.append(df.loc[m, col] - pre)
                    else:
                        vals.append(np.nan)
                curves.setdefault((subtype, axis_key, layer_key), []).append(vals)

    fig, axes = plt.subplots(len(SUBTYPE_ORDER), len(AXES),
                             figsize=(13, 8.5), dpi=150, sharex=True)
    for r, subtype in enumerate(SUBTYPE_ORDER):
        # Column-wise common y-range for visual comparability
        for c, (axis_key, axis_label, _) in enumerate(AXES):
            ax = axes[r, c]
            col_max = 0.5
            for layer_key, layer_name, lcolor in LAYERS:
                key = (subtype, axis_key, layer_key)
                if key not in curves:
                    continue
                stack = np.full((len(curves[key]), len(rel_months)), np.nan)
                for i, vals in enumerate(curves[key]):
                    stack[i] = vals
                mean_curve = np.nanmean(stack, axis=0)
                col_max = max(col_max, np.nanmax(np.abs(mean_curve)))
                ax.plot(rel_months, mean_curve, color=lcolor, lw=1.3,
                        alpha=0.9, label=layer_name)
            ax.axhline(0, color="black", lw=0.5, alpha=0.5)
            ax.axvline(0, color=MUTED, lw=1.0, alpha=0.6, ls=":")
            # Shade the horizon zones
            for (start, end), shade in zip(HORIZONS, [0.04, 0.06, 0.08, 0.10]):
                ax.axvspan(start, end, color="grey", alpha=shade)
            ax.set_ylim(-col_max * 1.1, col_max * 1.1)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            if r == 0:
                ax.set_title(axis_label, fontsize=10,
                             color={"threat": ACCENT, "diplo": COOL, "human": CHAR}[axis_key])
            if c == 0:
                ax.set_ylabel(subtype.upper(), fontsize=9, color=CHAR)
            if r == 0 and c == 0:
                ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
    for ax in axes[-1, :]:
        ax.set_xlabel("Months from event  (shaded bands: Δ_6, Δ_12, Δ_24, Δ_36)",
                      fontsize=8)
        ax.set_xticks([-12, 0, 6, 12, 24, 36])
    fig.suptitle("Hysteresis: extended response curves out to +36 months "
                 "(deviation from pre-event baseline)",
                 fontsize=11, y=0.995)
    plt.tight_layout()
    fig_path = FIG_DIR / "hysteresis_curves.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_path}")
    print(f"Saved: {OUT_DIR / 'hysteresis_shifts.csv'}")
    print(f"Saved: {OUT_DIR / 'hysteresis_persistence.csv'}")


if __name__ == "__main__":
    main()
