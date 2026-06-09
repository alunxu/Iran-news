#!/usr/bin/env python3
"""
Structural Break Detection — Module 2: Endogenous detection + Confirmatory tests
================================================================================

Two complementary procedures on each tension time series:

  (a) **Bai-Perron-style endogenous detection** via the ruptures `Pelt`
      change-point algorithm with L2 cost and BIC-style penalty. This is
      *data-driven* — finds whatever break dates best partition the series
      into piecewise-constant segments, with no preset event list.

  (b) **Chow F-tests at known candidate event dates**. For each of the 12
      historical anchors from Module 1, test H0 (no mean shift at that
      date) against H1 (mean shift). Reports F-stat, p-value, pre/post
      means, and shift magnitude. Bonferroni-corrected significance.

Cross-reference: each detected break gets matched to the nearest
candidate event within ±12 months. Detected breaks with no near event
are flagged as *novel* (interesting historiographic outliers).

Inputs
------
  data/structural_break/monthly_series.csv (from Module 1)

Outputs
-------
  data/structural_break/baiperron_breaks.csv
  data/structural_break/chow_tests.csv
  figures/structural_break/breakpoints.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERIES_CSV   = PROJECT_ROOT / "data" / "structural_break" / "monthly_series.csv"
OUT_DIR      = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR      = PROJECT_ROOT / "figures" / "structural_break"

# Project palette
ACCENT = "#B85042"; COOL = "#2F5F5D"; CHAR = "#363A3E"; MUTED = "#707070"
BREAK_COLOR = "#E08D17"  # warm orange for detected breaks

# Same anchors as Module 1 — short labels
HISTORICAL_EVENTS = [
    ("1979-02-01", "Khomeini returns"),
    ("1979-11-04", "Embassy seized"),
    ("1980-09-22", "Iran-Iraq war"),
    ("1988-07-03", "IR655 / Vincennes"),
    ("1989-06-03", "Khomeini dies"),
    ("2002-01-29", '"Axis of Evil"'),
    ("2009-06-13", "Green Movement"),
    ("2013-06-15", "Rouhani elected"),
    ("2015-07-14", "JCPOA"),
    ("2018-05-08", "JCPOA exit"),
    ("2020-01-03", "Soleimani"),
    ("2022-09-16", "Mahsa Amini"),
]
EVENT_DATES = [pd.Timestamp(d) for d, _ in HISTORICAL_EVENTS]
EVENT_LABELS = [lbl for _, lbl in HISTORICAL_EVENTS]

# Five tension series to analyse (skip volume — that's covariate, not a tension)
SERIES_SPECS = [
    ("gap_threat_HL_FT",   "Headline−body THREAT gap",    ACCENT),
    ("gap_diplo_HL_FT",    "Headline−body DIPLOMACY gap", COOL),
    ("gap_human_HL_FT",    "Headline−body HUMANIZING gap", CHAR),
    ("gap_threat_news_ed", "News−editorial THREAT gap",   ACCENT),
    ("gap_diplo_news_ed",  "News−editorial DIPLOMACY gap", COOL),
]

# Bai-Perron parameters
MIN_SEGMENT_MONTHS = 12   # smallest allowed segment
BIC_PEN_FACTOR     = 2.0  # higher → fewer breaks
EVENT_MATCH_WINDOW = 12   # ±months for "near event" matching
N_BOOTSTRAP        = 200


# ────────────────────────────────────────────────────────────────────────
# Bai-Perron detection (PELT with BIC penalty)
# ────────────────────────────────────────────────────────────────────────
def detect_breaks(values: np.ndarray, min_size: int = MIN_SEGMENT_MONTHS) -> list[int]:
    """Returns integer index positions of detected breakpoints.
    Excludes the trailing endpoint that ruptures returns by convention.
    """
    if len(values) < 2 * min_size:
        return []
    sigma = np.std(values)
    penalty = BIC_PEN_FACTOR * (sigma ** 2) * np.log(len(values))
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(values.reshape(-1, 1))
    breaks = algo.predict(pen=penalty)
    # ruptures appends n as last element; drop it
    return [b for b in breaks if b < len(values)]


def bootstrap_break_dates(
    series: pd.Series, detected: list[int], n_boot: int = N_BOOTSTRAP
) -> list[tuple[int, int]]:
    """Residual bootstrap to get ±CI on each detected break (in months).

    Builds piecewise-constant baseline, resamples residuals around it,
    re-runs PELT, and returns the median absolute deviation of break
    locations as a coarse CI.
    """
    if not detected:
        return []
    values = series.values
    # Build piecewise-constant baseline
    segs = [0] + detected + [len(values)]
    baseline = np.zeros_like(values, dtype=float)
    for a, b in zip(segs[:-1], segs[1:]):
        baseline[a:b] = values[a:b].mean()
    residuals = values - baseline
    cis: list[tuple[int, int]] = []
    rng = np.random.default_rng(42)
    for true_break in detected:
        offsets = []
        for _ in range(n_boot):
            resampled = baseline + rng.choice(residuals, size=len(residuals), replace=True)
            try:
                boot_breaks = detect_breaks(resampled)
            except Exception:
                continue
            if not boot_breaks:
                continue
            nearest = min(boot_breaks, key=lambda b: abs(b - true_break))
            offsets.append(nearest - true_break)
        if offsets:
            offsets = np.array(offsets)
            cis.append((int(np.percentile(offsets, 2.5)),
                        int(np.percentile(offsets, 97.5))))
        else:
            cis.append((0, 0))
    return cis


# ────────────────────────────────────────────────────────────────────────
# Chow F-test at a known break date
# ────────────────────────────────────────────────────────────────────────
def chow_test(values: np.ndarray, break_idx: int, min_segment: int = 6) -> dict:
    """F-test for a mean shift at break_idx (1-indexed).

    Reports F-stat, p-value (df = 1, n-2), pre- and post-means, shift.
    Returns NaNs if either side is shorter than min_segment.
    """
    n = len(values)
    pre  = values[:break_idx]
    post = values[break_idx:]
    if len(pre) < min_segment or len(post) < min_segment:
        return dict(f_stat=np.nan, p_value=np.nan,
                    pre_mean=np.nan, post_mean=np.nan,
                    shift=np.nan, n_pre=len(pre), n_post=len(post))
    rss_full     = np.sum((values - values.mean()) ** 2)
    rss_unrest   = np.sum((pre - pre.mean()) ** 2) + np.sum((post - post.mean()) ** 2)
    f_stat = ((rss_full - rss_unrest) / 1) / (rss_unrest / (n - 2))
    p_value = 1.0 - stats.f.cdf(f_stat, 1, n - 2)
    return dict(
        f_stat=float(f_stat),
        p_value=float(p_value),
        pre_mean=float(pre.mean()),
        post_mean=float(post.mean()),
        shift=float(post.mean() - pre.mean()),
        n_pre=len(pre),
        n_post=len(post),
    )


# ────────────────────────────────────────────────────────────────────────
# Match detected break → nearest known event
# ────────────────────────────────────────────────────────────────────────
def nearest_event(break_date: pd.Timestamp,
                  window_months: int = EVENT_MATCH_WINDOW
                  ) -> tuple[str | None, int | None]:
    deltas = [(lbl, (break_date - d).days / 30.4)
              for d, lbl in zip(EVENT_DATES, EVENT_LABELS)]
    deltas.sort(key=lambda x: abs(x[1]))
    label, delta = deltas[0]
    if abs(delta) <= window_months:
        return label, round(delta)
    return None, None


# ────────────────────────────────────────────────────────────────────────
# Main pipeline per series
# ────────────────────────────────────────────────────────────────────────
def analyse_series(col: str, label: str, monthly: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = monthly[col].dropna()
    if len(raw) < 2 * MIN_SEGMENT_MONTHS:
        print(f"  [{col}] insufficient data ({len(raw)} months)")
        return pd.DataFrame(), pd.DataFrame()

    values = raw.values
    dates  = raw.index

    # 1. Bai-Perron / PELT detection
    detected_idx = detect_breaks(values)
    bp_records = []
    if detected_idx:
        cis = bootstrap_break_dates(raw, detected_idx)
        segs = [0] + detected_idx + [len(values)]
        for i, (idx, ci) in enumerate(zip(detected_idx, cis)):
            seg_pre  = values[segs[i]:idx]
            seg_post = values[idx:segs[i + 2]]
            br_date = dates[idx]
            ev_label, ev_delta = nearest_event(br_date)
            bp_records.append(dict(
                series=col,
                break_date=br_date.date(),
                break_idx=int(idx),
                ci_low_months=int(ci[0]),
                ci_high_months=int(ci[1]),
                pre_mean=float(seg_pre.mean()),
                post_mean=float(seg_post.mean()),
                shift=float(seg_post.mean() - seg_pre.mean()),
                n_pre=len(seg_pre),
                n_post=len(seg_post),
                nearest_event=ev_label,
                event_delta_months=ev_delta,
            ))
    bp_df = pd.DataFrame(bp_records)

    # 2. Chow tests at each candidate event
    chow_records = []
    for ev_date, ev_label in zip(EVENT_DATES, EVENT_LABELS):
        # Map event date to nearest index in this series' (NaN-dropped) dates
        try:
            idx = dates.get_indexer([ev_date], method="nearest")[0]
        except Exception:
            continue
        if idx < 0 or idx >= len(values):
            continue
        # Skip if event is outside the date range of this series
        if ev_date < dates[0] or ev_date > dates[-1]:
            continue
        res = chow_test(values, idx)
        chow_records.append(dict(
            series=col,
            event=ev_label,
            event_date=ev_date.date(),
            **res,
        ))
    chow_df = pd.DataFrame(chow_records)

    # Bonferroni correction: 12 events × 5 series = 60 tests
    n_tests = len(EVENT_DATES) * len(SERIES_SPECS)
    if not chow_df.empty:
        chow_df["p_bonferroni"] = (chow_df["p_value"] * n_tests).clip(upper=1.0)
        chow_df["sig"] = chow_df["p_bonferroni"] < 0.05

    return bp_df, chow_df


# ────────────────────────────────────────────────────────────────────────
# Figure: 5 panels with detected breaks + Chow-significant events
# ────────────────────────────────────────────────────────────────────────
def plot_breakpoints(monthly: pd.DataFrame, bp_all: pd.DataFrame,
                     chow_all: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=len(SERIES_SPECS), ncols=1, sharex=True,
        figsize=(13, 11.5), dpi=150,
        gridspec_kw={"hspace": 0.38, "top": 0.945, "bottom": 0.05,
                     "left": 0.07, "right": 0.985},
    )

    fig.text(0.07, 0.985,
        "Structural Break Detection — endogenous (PELT) + confirmatory (Chow)",
        fontsize=14.5, fontweight="bold", ha="left", va="top", color=CHAR)
    fig.text(0.07, 0.964,
        "Series: smoothed (12-mo rolling) · ORANGE dashed = endogenously detected break ·"
        " ▲ = Chow-significant candidate event (Bonferroni p<0.05)",
        fontsize=9.5, color=MUTED, ha="left", va="top")

    for i, (ax, (col, title, color)) in enumerate(zip(axes, SERIES_SPECS)):
        raw = monthly[col]
        smoothed = raw.rolling(12, center=True, min_periods=4).mean()
        ax.plot(raw.index, raw.values, color=color, linewidth=0.35, alpha=0.3)
        ax.plot(smoothed.index, smoothed.values, color=color, linewidth=1.8)
        ax.axhline(0, color=MUTED, linewidth=0.4, linestyle=":", zorder=0)

        # Faint historical anchors
        for d in EVENT_DATES:
            ax.axvline(d, color=MUTED, linewidth=0.4, linestyle="-",
                       alpha=0.25, zorder=0)

        # Detected breaks for this series + segment means
        srs_breaks = bp_all[bp_all["series"] == col]
        if not srs_breaks.empty:
            # Build segment x-ranges from break_dates
            dates_clean = monthly[col].dropna().index
            seg_starts = [dates_clean[0]] + [pd.Timestamp(d) for d in srs_breaks["break_date"]]
            seg_ends = [pd.Timestamp(d) for d in srs_breaks["break_date"]] + [dates_clean[-1]]
            for j, (_, row) in enumerate(srs_breaks.iterrows()):
                br_date = pd.Timestamp(row["break_date"])
                ax.axvline(br_date, color=BREAK_COLOR, linewidth=1.4,
                           linestyle="--", alpha=0.9)
                # Tag with shift magnitude
                ax.annotate(
                    f"Δ={row['shift']:+.1f}",
                    xy=(br_date, ax.get_ylim()[1]), xytext=(2, -3),
                    textcoords="offset points",
                    fontsize=7, color=BREAK_COLOR, ha="left", va="top",
                )
            # Plot piecewise-constant means as bold horizontal bars
            seg_means = list(srs_breaks["pre_mean"]) + [srs_breaks["post_mean"].iloc[-1]]
            for s, e, m in zip(seg_starts, seg_ends, seg_means):
                ax.hlines(m, s, e, colors=BREAK_COLOR, linewidth=2.2, alpha=0.55)

        # Chow-significant markers
        srs_chow_sig = chow_all[(chow_all["series"] == col) & (chow_all["sig"])]
        if not srs_chow_sig.empty:
            for _, row in srs_chow_sig.iterrows():
                ev_date = pd.Timestamp(row["event_date"])
                ax.scatter([ev_date], [ax.get_ylim()[0]],
                           marker="^", s=40, c=CHAR, zorder=5,
                           edgecolors="white", linewidth=0.5)

        ax.set_title(title, fontsize=10.5, loc="left", color=CHAR, pad=4)
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=9, colors=CHAR)
        ax.margins(x=0.005)
        if i != len(SERIES_SPECS) - 1:
            ax.tick_params(axis="x", labelbottom=False)

    bottom = axes[-1]
    bottom.xaxis.set_major_locator(mdates.YearLocator(5))
    bottom.xaxis.set_minor_locator(mdates.YearLocator(1))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    bottom.set_xlabel("Year", fontsize=10, color=CHAR)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main():
    if not SERIES_CSV.exists():
        sys.exit(f"FATAL: {SERIES_CSV} missing — run Module 1 first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {SERIES_CSV} ...")
    monthly = pd.read_csv(SERIES_CSV, parse_dates=["pub_month"]).set_index("pub_month")
    print(f"  {len(monthly)} months from {monthly.index.min().date()} to {monthly.index.max().date()}")

    bp_frames, chow_frames = [], []
    for col, label, _ in SERIES_SPECS:
        print(f"\nAnalysing {col} ...")
        bp, chow = analyse_series(col, label, monthly)
        if not bp.empty:
            print(f"  {len(bp)} detected breaks (PELT)")
            for _, r in bp.iterrows():
                tag = f" → {r['nearest_event']}" if r['nearest_event'] else ""
                print(f"    {r['break_date']}  shift={r['shift']:+.2f}{tag}")
        if not chow.empty:
            sig = chow[chow["sig"]]
            print(f"  {len(sig)}/{len(chow)} Chow tests significant (Bonferroni)")
        bp_frames.append(bp)
        chow_frames.append(chow)

    bp_all   = pd.concat(bp_frames, ignore_index=True)   if bp_frames   else pd.DataFrame()
    chow_all = pd.concat(chow_frames, ignore_index=True) if chow_frames else pd.DataFrame()

    bp_path   = OUT_DIR / "baiperron_breaks.csv"
    chow_path = OUT_DIR / "chow_tests.csv"
    bp_all.to_csv(bp_path, index=False)
    chow_all.to_csv(chow_path, index=False)
    print(f"\nSaved → {bp_path}")
    print(f"Saved → {chow_path}")

    print("\nPlotting breakpoints ...")
    plot_breakpoints(monthly, bp_all, chow_all, FIG_DIR / "breakpoints.png")

    # ── Summary table for stdout ──
    print("\n" + "=" * 70)
    print("ENDOGENOUS BREAKS (PELT) per series")
    print("=" * 70)
    if not bp_all.empty:
        view = bp_all[["series", "break_date", "shift", "nearest_event",
                       "event_delta_months"]]
        print(view.to_string(index=False))
    print("\n" + "=" * 70)
    print("CHOW TESTS — significant at Bonferroni p<0.05")
    print("=" * 70)
    if not chow_all.empty:
        sig = chow_all[chow_all["sig"]].copy()
        sig = sig[["series", "event", "event_date", "f_stat", "p_value",
                   "p_bonferroni", "shift"]]
        if len(sig):
            print(sig.to_string(index=False))
        else:
            print("(none — try less conservative threshold)")


if __name__ == "__main__":
    main()
