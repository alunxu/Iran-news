#!/usr/bin/env python3
r"""
Module 5 -- Within-event temporal resolution.

\S 2 of the notes measured the cascade by comparing pre- and post-event 12-month
means. That tells us magnitude, not timing. Here we zoom into the monthly
resolution: for each event, normalize each (layer, axis) series to its pre-event
12-month mean, then average across events within subtype. This gives a
"subtype response function" -- a curve from t=-12 to t=+12 months relative to
the event -- per (subtype, axis, layer).

Questions:
  - At t=0, do all 4 layers shift simultaneously, or does headline shift first?
  - Does the cascade structure (HL > Body in magnitude) develop instantly or
    over months?
  - For CONFLICT (where Body responds more than HL), when does Body actually
    start responding?

Outputs:
  data/structural_break/within_event_curves.csv
  figures/structural_break/within_event_curves.png
  data/structural_break/within_event_timing.csv  (months-to-halfway per layer)
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYER_CSV = PROJECT_ROOT / "data" / "structural_break" / "layer_series.csv"
OUT_DIR = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR = PROJECT_ROOT / "figures" / "structural_break"

ACCENT, COOL, CHAR, MUTED = "#B85042", "#2F5F5D", "#363A3E", "#707070"

WINDOW = 12  # ±months

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


def event_window(df: pd.DataFrame, event_date: pd.Timestamp, col: str) -> pd.Series:
    """Return monthly series in ±WINDOW with relative-month index (-12..+12)."""
    # Anchor to month-start of the event so the relative offsets are exact
    anchor = event_date.replace(day=1)
    rel_idx = np.arange(-WINDOW, WINDOW + 1)
    vals = []
    for k in rel_idx:
        m = anchor + pd.DateOffset(months=int(k))
        v = df.loc[m, col] if m in df.index else np.nan
        vals.append(v)
    return pd.Series(vals, index=rel_idx)


def main():
    df = pd.read_csv(LAYER_CSV, parse_dates=["pub_month"]).set_index("pub_month").sort_index()

    # Build curves: per (event, axis, layer), the 25-month window centered on event,
    # normalized by subtracting the pre-event (months -12..-2) mean.
    rows = []
    curves = {}  # (subtype, axis, layer) -> list of normalized series
    for event_str, label, subtype in EVENTS:
        ev = pd.Timestamp(event_str)
        for axis, axis_label, _ in AXES:
            for layer, layer_name, _ in LAYERS:
                col = f"{layer}_{axis}"
                s = event_window(df, ev, col)
                pre = s.loc[-WINDOW:-2].mean()
                if np.isnan(pre):
                    continue
                normed = s - pre  # deviation from pre-event mean
                curves.setdefault((subtype, axis, layer), []).append(normed)
                for rel_m, v in normed.items():
                    rows.append(dict(
                        event=label, subtype=subtype, axis=axis,
                        layer=layer, layer_name=layer_name,
                        rel_month=int(rel_m), deviation=float(v) if not np.isnan(v) else None,
                    ))
    pd.DataFrame(rows).to_csv(OUT_DIR / "within_event_curves.csv", index=False)

    # Average per (subtype, axis, layer)
    subtype_means = {}
    for key, series_list in curves.items():
        # Stack into 2D array (events × months), nanmean over events
        stack = np.full((len(series_list), 2 * WINDOW + 1), np.nan)
        for i, s in enumerate(series_list):
            stack[i] = s.values
        subtype_means[key] = np.nanmean(stack, axis=0)

    # ────────────────────────────────────────────────────────────────────
    # Figure: 4 rows (subtype) × 3 cols (axis), each shows 4 layer curves
    # ────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(SUBTYPE_ORDER), len(AXES),
                             figsize=(12, 8.5), dpi=150, sharex=True)
    rel_x = np.arange(-WINDOW, WINDOW + 1)

    # Compute global y-range per axis column for visual comparison
    for c, (axis, axis_label, _) in enumerate(AXES):
        col_max = 0
        for r, subtype in enumerate(SUBTYPE_ORDER):
            for layer, _, _ in LAYERS:
                key = (subtype, axis, layer)
                if key in subtype_means:
                    col_max = max(col_max, np.nanmax(np.abs(subtype_means[key])))
        col_max = max(col_max, 1.0) * 1.1

        for r, subtype in enumerate(SUBTYPE_ORDER):
            ax = axes[r, c]
            ax.axhline(0, color="black", lw=0.5, alpha=0.5)
            ax.axvline(0, color=MUTED, lw=1.0, alpha=0.6, ls=":")
            for layer, layer_name, lcolor in LAYERS:
                key = (subtype, axis, layer)
                if key not in subtype_means:
                    continue
                vals = subtype_means[key]
                ax.plot(rel_x, vals, color=lcolor, lw=1.3, alpha=0.9, label=layer_name)
            ax.set_ylim(-col_max, col_max)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            if r == 0:
                ax.set_title(axis_label, fontsize=10, color={"threat": ACCENT,
                                                            "diplo": COOL,
                                                            "human": CHAR}[axis])
            if c == 0:
                ax.set_ylabel(subtype.upper(), fontsize=9, color=CHAR)
            if r == 0 and c == 0:
                ax.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Months from event", fontsize=8)
        ax.set_xticks([-12, -6, 0, 6, 12])

    fig.suptitle("Within-event response curves: mean deviation from pre-event "
                 "baseline, by subtype × axis (lines = 4 layers)",
                 fontsize=11, y=0.995)
    plt.tight_layout()
    fig_path = FIG_DIR / "within_event_curves.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # ────────────────────────────────────────────────────────────────────
    # Timing analysis: for each (subtype, axis, layer), find first month
    # at which the deviation crosses 50% of the final shift (mean of months +6..+12).
    # If the layer never crosses, return NaN.
    # ────────────────────────────────────────────────────────────────────
    timing_rows = []
    for subtype in SUBTYPE_ORDER:
        for axis, axis_label, _ in AXES:
            for layer, layer_name, _ in LAYERS:
                key = (subtype, axis, layer)
                if key not in subtype_means:
                    continue
                vals = subtype_means[key]
                # Final shift = mean of post-event window (+2..+12)
                final = np.nanmean(vals[WINDOW + 2:])
                if np.isnan(final) or abs(final) < 0.1:
                    timing_rows.append(dict(
                        subtype=subtype, axis=axis, layer=layer,
                        layer_name=layer_name, final_shift=final,
                        first_50pct_month=None,
                    ))
                    continue
                # Find first month >= 0 where vals reaches 50% of final
                target = 0.5 * final
                sign = np.sign(final)
                first_month = None
                for i, m in enumerate(rel_x):
                    if m < 0:
                        continue
                    if not np.isnan(vals[i]) and sign * vals[i] >= sign * target:
                        first_month = int(m)
                        break
                timing_rows.append(dict(
                    subtype=subtype, axis=axis, layer=layer,
                    layer_name=layer_name, final_shift=float(final),
                    first_50pct_month=first_month,
                ))
    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(OUT_DIR / "within_event_timing.csv", index=False)

    print("\n=== First month at which the response curve crosses 50% of its final shift ===")
    print("(Negative month = layer was already shifted before the event.")
    print(" None = layer never crossed (no meaningful shift).)")
    for subtype in SUBTYPE_ORDER:
        print(f"\n{subtype.upper()}")
        for axis_key, axis_label, _ in AXES:
            line = f"  {axis_label:7s}"
            for layer_key, layer_name, _ in LAYERS:
                row = timing_df[(timing_df["subtype"] == subtype) &
                                (timing_df["axis"] == axis_key) &
                                (timing_df["layer"] == layer_key)]
                if len(row):
                    m = row["first_50pct_month"].iloc[0]
                    shift = row["final_shift"].iloc[0]
                    cell = f"{layer_name[:3]}: " + (f"+{m}mo " if m is not None and not pd.isna(m)
                                                    else " --  ")
                    cell += f"(Δ={shift:+.1f})"
                    line += f"   {cell}"
            print(line)

    # Build a compact summary: per (subtype, axis), HL timing vs Body timing
    print("\n=== HL vs Body timing summary (months to halfway) ===")
    print(f"{'Subtype':<12s}{'Axis':<10s}{'HL':<10s}{'Body':<10s}{'HL leads?':<10s}")
    for subtype in SUBTYPE_ORDER:
        for axis_key, axis_label, _ in AXES:
            hl = timing_df[(timing_df["subtype"] == subtype) & (timing_df["axis"] == axis_key) &
                           (timing_df["layer"] == "hl")]
            ft = timing_df[(timing_df["subtype"] == subtype) & (timing_df["axis"] == axis_key) &
                           (timing_df["layer"] == "ft")]
            if not len(hl) or not len(ft):
                continue
            hl_m = hl["first_50pct_month"].iloc[0]
            ft_m = ft["first_50pct_month"].iloc[0]
            hl_s = f"{int(hl_m):+d}mo" if hl_m is not None and not pd.isna(hl_m) else "n/a"
            ft_s = f"{int(ft_m):+d}mo" if ft_m is not None and not pd.isna(ft_m) else "n/a"
            verdict = ""
            if hl_m is not None and ft_m is not None and not pd.isna(hl_m) and not pd.isna(ft_m):
                hl_i, ft_i = int(hl_m), int(ft_m)
                if hl_i < ft_i:
                    verdict = f"yes (by {ft_i - hl_i}mo)"
                elif hl_i > ft_i:
                    verdict = f"no (Body leads by {hl_i - ft_i}mo)"
                else:
                    verdict = "tied"
            print(f"{subtype:<12s}{axis_label:<10s}{hl_s:<10s}{ft_s:<10s}{verdict:<10s}")


if __name__ == "__main__":
    main()
