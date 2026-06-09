#!/usr/bin/env python3
"""
Module 4 — Event taxonomy.

Module 3 showed the layer cascade pattern (HL > Abstract > Lead > Body) at
JCPOA Implementation Day. The natural follow-up: is this cascade structure
event-specific (only at JCPOA) or general (every newsworthy moment triggers
the same editor-leads-reporter pattern)?

We classify the 12 historical anchors into three subtypes:
  - CONFLICT          (Embassy seized, Iran-Iraq war, IR655, Soleimani)
  - DIPLOMACY         (JCPOA signed, JCPOA Imp Day, JCPOA exit)
  - DOMESTIC / CIVIL  (Khomeini returns, Khomeini dies, Green Movement,
                       Rouhani elected, Mahsa Amini)
  - POLICY-DECLARATION (Axis of Evil) — kept separate

For each event we compute the pre/post 12-month mean shift at every
(layer × axis) cell — 4 layers × 3 axes = 12 cells per event. Then we
average within subtype to get a 'signature' per subtype.

Outputs:
  data/structural_break/event_shifts.csv
  data/structural_break/subtype_signatures.csv
  figures/structural_break/event_signature_heatmap.png
  figures/structural_break/subtype_signature_grid.png
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

WINDOW_MONTHS = 12
EXCLUDE_CENTRAL_MONTHS = 2  # exclude ±2 mo around event to avoid ramp

EVENTS = [
    # (date, short label, subtype)
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

LAYERS = ["hl", "ab", "ld", "ft"]
LAYER_NAMES = {"hl": "Headline", "ab": "Abstract", "ld": "Lead", "ft": "Body"}
AXES = ["threat", "diplo", "human"]

SUBTYPE_ORDER = ["conflict", "diplomacy", "domestic", "policy"]
SUBTYPE_COLOR = {
    "conflict":  ACCENT,
    "diplomacy": COOL,
    "domestic":  CHAR,
    "policy":    "#8A6F4E",
}


def shift_at_event(df: pd.DataFrame, event_date: pd.Timestamp,
                   col: str) -> tuple[float, float, float]:
    """Return (pre_mean, post_mean, shift) computed on ±WINDOW excluding center."""
    pre_start = event_date - pd.DateOffset(months=WINDOW_MONTHS)
    pre_end   = event_date - pd.DateOffset(months=EXCLUDE_CENTRAL_MONTHS)
    post_start = event_date + pd.DateOffset(months=EXCLUDE_CENTRAL_MONTHS)
    post_end   = event_date + pd.DateOffset(months=WINDOW_MONTHS)

    pre  = df.loc[pre_start:pre_end, col].dropna()
    post = df.loc[post_start:post_end, col].dropna()
    if len(pre) < 3 or len(post) < 3:
        return (np.nan, np.nan, np.nan)
    return (float(pre.mean()), float(post.mean()), float(post.mean() - pre.mean()))


def main():
    df = pd.read_csv(LAYER_CSV, parse_dates=["pub_month"]).set_index("pub_month").sort_index()

    rows = []
    for event_date_str, label, subtype in EVENTS:
        ev = pd.Timestamp(event_date_str)
        for axis in AXES:
            for layer in LAYERS:
                col = f"{layer}_{axis}"
                pre, post, shift = shift_at_event(df, ev, col)
                rows.append(dict(
                    event=label, event_date=ev.date(), subtype=subtype,
                    layer=layer, layer_name=LAYER_NAMES[layer], axis=axis,
                    pre_mean=pre, post_mean=post, shift=shift,
                ))
    event_df = pd.DataFrame(rows)
    event_df.to_csv(OUT_DIR / "event_shifts.csv", index=False)
    print(f"Saved: {OUT_DIR / 'event_shifts.csv'}")
    print(f"Events: {event_df['event'].nunique()}; cells/event: {12}")

    # Subtype signature: mean shift per (axis, layer) within subtype
    sig = (event_df.groupby(["subtype", "axis", "layer"])["shift"]
                  .agg(["mean", "std", "count"]).reset_index())
    sig.to_csv(OUT_DIR / "subtype_signatures.csv", index=False)
    print(f"Saved: {OUT_DIR / 'subtype_signatures.csv'}")

    # ────────────────────────────────────────────────────────────────────
    # Figure 1: event × (layer × axis) heatmap, grouped by subtype
    # ────────────────────────────────────────────────────────────────────
    # Build a wide matrix: rows = events (ordered by subtype, then date),
    # columns = (axis, layer) cell labels.
    event_df["cell"] = event_df["layer_name"] + "·" + event_df["axis"].str.upper()
    cell_order = [f"{LAYER_NAMES[l]}·{a.upper()}" for a in AXES for l in LAYERS]
    pivot = event_df.pivot(index=["subtype", "event_date", "event"],
                          columns="cell", values="shift")[cell_order]
    # Sort by subtype then date
    pivot = pivot.sort_index(level=["subtype", "event_date"])

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    vmax = np.nanmax(np.abs(pivot.values))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r[2]}  ({r[1]})" for r in pivot.index], fontsize=8)
    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6,
                        color="black" if abs(v) < vmax * 0.5 else "white")
    # Vertical separators between axis groups
    for x in (3.5, 7.5):
        ax.axvline(x, color="black", lw=1.2)
    # Horizontal separators between subtypes
    subtype_changes = []
    prev = None
    for idx, st in enumerate([r[0] for r in pivot.index]):
        if st != prev and idx > 0:
            subtype_changes.append(idx)
        prev = st
    for y in subtype_changes:
        ax.axhline(y - 0.5, color="black", lw=1.2)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Post − Pre shift  (per 1000 words)", fontsize=8)
    ax.set_title("Pre/post-event shift per event × (layer · axis)  "
                 "[$\\pm$12mo window, excl. central $\\pm$2mo]",
                 fontsize=11, loc="left")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "event_signature_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'event_signature_heatmap.png'}")

    # ────────────────────────────────────────────────────────────────────
    # Figure 2: subtype signature grid (4 subtypes × 3 axes), each cell
    # shows the 4-layer profile as a small bar chart
    # ────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(SUBTYPE_ORDER), len(AXES),
                             figsize=(11, 7.5), dpi=150, sharey="col")
    for r, subtype in enumerate(SUBTYPE_ORDER):
        sub = event_df[event_df["subtype"] == subtype]
        n_events = sub["event"].nunique()
        for c, axis in enumerate(AXES):
            ax = axes[r, c]
            means = []
            for layer in LAYERS:
                vals = sub[(sub["axis"] == axis) & (sub["layer"] == layer)]["shift"].dropna()
                means.append(float(vals.mean()) if len(vals) else np.nan)
            x = np.arange(len(LAYERS))
            bar_color = SUBTYPE_COLOR[subtype]
            ax.bar(x, means, color=bar_color, alpha=0.85, edgecolor="white", lw=0.5)
            ax.axhline(0, color="black", lw=0.5, alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([LAYER_NAMES[l].replace("Headline", "HL")
                                              .replace("Abstract", "Abs")
                                              .replace("Body", "Body") for l in LAYERS],
                               fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, alpha=0.2, axis="y")
            if r == 0:
                ax.set_title(axis.upper(), fontsize=10, color=ACCENT if axis=="threat"
                             else (COOL if axis=="diplo" else CHAR))
            if c == 0:
                ax.set_ylabel(f"{subtype.upper()}\n(n={n_events})", fontsize=9,
                              color=bar_color)
    fig.suptitle("Subtype signatures: mean pre/post-event shift per (subtype × axis × layer)",
                 fontsize=11, y=0.995)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "subtype_signature_grid.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'subtype_signature_grid.png'}")

    # ────────────────────────────────────────────────────────────────────
    # Print summary
    # ────────────────────────────────────────────────────────────────────
    print("\n=== Subtype signature: mean shift per (subtype, axis, layer) ===")
    for subtype in SUBTYPE_ORDER:
        n = event_df[event_df["subtype"] == subtype]["event"].nunique()
        print(f"\n{subtype.upper()}  (n={n} events)")
        for axis in AXES:
            line = f"  {axis:8s}"
            for layer in LAYERS:
                m = sig[(sig["subtype"] == subtype) & (sig["axis"] == axis) &
                        (sig["layer"] == layer)]["mean"]
                v = float(m.iloc[0]) if len(m) else np.nan
                line += f"  {LAYER_NAMES[layer][:3]} {v:+5.2f}"
            print(line)

    # Cascade ratio test: HL/Body magnitude per subtype × axis
    print("\n=== Cascade ratio test (|HL shift| / |Body shift|) per subtype × axis ===")
    print("(>1 means HL responds more than Body — editor leads reporter)")
    for subtype in SUBTYPE_ORDER:
        print(f"  {subtype.upper():12s}", end="")
        for axis in AXES:
            hl = sig[(sig["subtype"] == subtype) & (sig["axis"] == axis) &
                     (sig["layer"] == "hl")]["mean"]
            ft = sig[(sig["subtype"] == subtype) & (sig["axis"] == axis) &
                     (sig["layer"] == "ft")]["mean"]
            if len(hl) and len(ft) and abs(float(ft.iloc[0])) > 0.01:
                ratio = abs(float(hl.iloc[0])) / abs(float(ft.iloc[0]))
                print(f"   {axis}: {ratio:4.1f}×", end="")
            else:
                print(f"   {axis}:  n/a", end="")
        print()


if __name__ == "__main__":
    main()
