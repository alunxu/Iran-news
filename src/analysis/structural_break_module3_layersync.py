#!/usr/bin/env python3
"""
Module 3 — Layer synchronization.

Question:  when the editorial side pushes a frame, do reporters pick it up
straight away, or does the body lag (or lead)?  Two complementary views:

  (A) Cross-correlation over the whole 50-year period.  For each framing axis
      (THREAT / DIPLO / HUMAN) and each layer pair, compute the Pearson
      correlation between layer-A at time t and layer-B at time t+k for
      k = -24 ... +24 months.  The peak lag tells us who leads.

  (B) Event study at JCPOA Implementation Day (2016-02-01, the one regime
      shift that survived every robustness check in Module 2.5).  Plot all
      four layers' trajectories in the ±24-month window around the event and
      identify when each layer crosses the post-event mean.

Both views work on linearly-detrended series — non-stationary trends would
otherwise dominate the correlation.

Outputs:
  data/structural_break/layer_xcorr.csv
  data/structural_break/jcpoa_event_study.csv
  figures/structural_break/layer_xcorr.png
  figures/structural_break/jcpoa_event_study.png
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYER_CSV = PROJECT_ROOT / "data" / "structural_break" / "layer_series.csv"
OUT_DIR = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR = PROJECT_ROOT / "figures" / "structural_break"

ACCENT, COOL, CHAR, MUTED = "#B85042", "#2F5F5D", "#363A3E", "#707070"
HIGHLIGHT = "#E08D17"

EVENT_DATE = pd.Timestamp("2016-02-01")  # JCPOA Implementation Day
WINDOW_MONTHS = 24
MAX_LAG = 24

# Three axes × four layers
AXES = [("threat", "THREAT", ACCENT),
        ("diplo",  "DIPLO",  COOL),
        ("human",  "HUMAN",  CHAR)]
LAYERS = [("hl", "Headline", ACCENT),
          ("ab", "Abstract", "#9B6A4F"),
          ("ld", "Lead",     COOL),
          ("ft", "Body",     CHAR)]

# Layer pairs to compute cross-correlation on — focus on editor-reporter axis
LAYER_PAIRS = [
    ("hl", "ft", "Headline → Body"),
    ("hl", "ld", "Headline → Lead"),
    ("ab", "ld", "Abstract → Lead"),
    ("ld", "ft", "Lead → Body"),
]


def detrend_linear(s: pd.Series) -> pd.Series:
    """Subtract OLS linear trend, return residuals indexed by same dates."""
    s = s.dropna()
    if len(s) < 24:
        return s
    t = np.arange(len(s), dtype=float)
    slope, intercept, *_ = stats.linregress(t, s.values)
    return s - (slope * t + intercept)


def lagged_corr(a: pd.Series, b: pd.Series, max_lag: int = MAX_LAG):
    """Compute Pearson correlation between a[t] and b[t+k] for k = -max_lag..+max_lag.

    Positive k means b is shifted later -- so peak at positive k means a leads b.
    Returns (lags, corrs).
    """
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.full(len(lags), np.nan)
    for i, k in enumerate(lags):
        if k >= 0:
            aa = a.iloc[:len(a) - k]
            bb = b.iloc[k:]
        else:
            aa = a.iloc[-k:]
            bb = b.iloc[:len(b) + k]
        # align on dates
        joined = pd.concat([aa.reset_index(drop=True), bb.reset_index(drop=True)],
                           axis=1).dropna()
        if len(joined) < 24:
            continue
        x, y = joined.iloc[:, 0].values, joined.iloc[:, 1].values
        if x.std() < 1e-6 or y.std() < 1e-6:
            continue
        corrs[i] = np.corrcoef(x, y)[0, 1]
    return lags, corrs


def main():
    df = pd.read_csv(LAYER_CSV, parse_dates=["pub_month"]).set_index("pub_month")
    df = df.sort_index()

    # ────────────────────────────────────────────────────────────────────
    # (A) Cross-correlation on detrended series
    # ────────────────────────────────────────────────────────────────────
    print("\n=== (A) Cross-correlation on detrended series ===")
    xcorr_rows = []
    xcorr_arrays = {}

    for axis_key, axis_label, axis_color in AXES:
        for la, lb, pair_label in LAYER_PAIRS:
            col_a = f"{la}_{axis_key}"
            col_b = f"{lb}_{axis_key}"
            sa = detrend_linear(df[col_a])
            sb = detrend_linear(df[col_b])
            lags, corrs = lagged_corr(sa, sb)
            peak_idx = int(np.nanargmax(corrs))
            peak_lag = int(lags[peak_idx])
            peak_corr = float(corrs[peak_idx])
            # corr at lag 0 (synchronous)
            zero_corr = float(corrs[lags.tolist().index(0)])
            xcorr_arrays[(axis_key, la, lb)] = (lags, corrs)
            xcorr_rows.append(dict(
                axis=axis_key, pair=pair_label, layer_a=la, layer_b=lb,
                peak_lag_months=peak_lag,
                peak_corr=peak_corr,
                synchronous_corr=zero_corr,
            ))
            print(f"  {axis_label:8s} {pair_label:25s}  peak lag = {peak_lag:+3d}mo  "
                  f"r = {peak_corr:+.3f}  (sync r = {zero_corr:+.3f})")

    xcorr_df = pd.DataFrame(xcorr_rows)
    xcorr_df.to_csv(OUT_DIR / "layer_xcorr.csv", index=False)

    # Figure: 3 axes (rows) × 4 layer pairs (cols), each cell shows lag profile
    fig, axes = plt.subplots(len(AXES), len(LAYER_PAIRS), figsize=(12, 7), dpi=150,
                             sharex=True)
    for r, (axis_key, axis_label, axis_color) in enumerate(AXES):
        for c, (la, lb, pair_label) in enumerate(LAYER_PAIRS):
            ax = axes[r, c]
            lags, corrs = xcorr_arrays[(axis_key, la, lb)]
            ax.axhline(0, color="black", lw=0.5, alpha=0.4)
            ax.axvline(0, color="black", lw=0.5, alpha=0.4)
            ax.plot(lags, corrs, color=axis_color, lw=1.4)
            peak_idx = int(np.nanargmax(corrs))
            ax.scatter([lags[peak_idx]], [corrs[peak_idx]], color=HIGHLIGHT,
                       s=30, zorder=3)
            ax.text(lags[peak_idx], corrs[peak_idx] + 0.04,
                    f"lag={lags[peak_idx]:+d}\nr={corrs[peak_idx]:.2f}",
                    ha="center", fontsize=7, color=CHAR)
            if r == 0:
                ax.set_title(pair_label, fontsize=9, loc="left")
            if c == 0:
                ax.set_ylabel(axis_label, fontsize=9, color=axis_color)
            ax.set_ylim(-0.2, 1.05)
            ax.set_xticks([-24, -12, 0, 12, 24])
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("Lag (months)\n←  B leads A     A leads B  →", fontsize=8)
    fig.suptitle("Cross-correlation of layer pairs on detrended series  "
                 "(orange dot = peak)",
                 fontsize=11, y=0.995)
    plt.tight_layout()
    xcorr_fig = FIG_DIR / "layer_xcorr.png"
    fig.savefig(xcorr_fig, bbox_inches="tight")
    print(f"\nSaved: {xcorr_fig}")

    # ────────────────────────────────────────────────────────────────────
    # (B) Event study around JCPOA 2016-02
    # ────────────────────────────────────────────────────────────────────
    print(f"\n=== (B) Event study around JCPOA Implementation Day {EVENT_DATE.date()} ===")
    win_start = EVENT_DATE - pd.DateOffset(months=WINDOW_MONTHS)
    win_end = EVENT_DATE + pd.DateOffset(months=WINDOW_MONTHS)
    win = df.loc[win_start:win_end].copy()

    event_rows = []
    fig, axes = plt.subplots(len(AXES), 1, figsize=(11, 7.5), dpi=150, sharex=True)
    for ax, (axis_key, axis_label, axis_color) in zip(axes, AXES):
        for lk, lname, lcolor in LAYERS:
            col = f"{lk}_{axis_key}"
            s = win[col].rolling(3, min_periods=1, center=True).mean()
            ax.plot(s.index, s.values, color=lcolor, lw=1.5, label=lname, alpha=0.9)
            # pre and post means (excluding the 3 months centered on event)
            pre = win.loc[:EVENT_DATE - pd.DateOffset(months=2), col].mean()
            post = win.loc[EVENT_DATE + pd.DateOffset(months=2):, col].mean()
            event_rows.append(dict(
                axis=axis_key, layer=lk, layer_name=lname,
                pre_mean=float(pre), post_mean=float(post),
                shift=float(post - pre),
            ))
        ax.axvline(EVENT_DATE, color=HIGHLIGHT, lw=1.5, ls="--", alpha=0.9)
        ax.set_title(f"{axis_label} density by layer  (vertical = JCPOA Imp Day)",
                     fontsize=10, loc="left")
        ax.set_ylabel("density / 1k tokens", fontsize=8)
        ax.legend(loc="upper right" if axis_key == "diplo" else "best",
                  fontsize=7, frameon=False, ncol=4)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlabel("Date  (3-month centered moving average)", fontsize=9)
    fig.suptitle(f"Event study at JCPOA Implementation Day  ({EVENT_DATE.date()}, ±{WINDOW_MONTHS} months)",
                 fontsize=11, y=0.995)
    plt.tight_layout()
    event_fig = FIG_DIR / "jcpoa_event_study.png"
    fig.savefig(event_fig, bbox_inches="tight")
    print(f"  Saved: {event_fig}")

    event_df = pd.DataFrame(event_rows)
    event_df.to_csv(OUT_DIR / "jcpoa_event_study.csv", index=False)

    print("\n  Pre/post means around JCPOA 2016-02 (per 1000 tokens):")
    pv = event_df.pivot(index="layer_name", columns="axis", values="shift").round(2)
    pv = pv.reindex([l[1] for l in LAYERS])
    print(pv.to_string())

    # ────────────────────────────────────────────────────────────────────
    # (C) Per-layer first-mover detection at JCPOA
    # Compute when each layer's 3-month rolling mean crosses the
    # midpoint between pre and post mean.
    # ────────────────────────────────────────────────────────────────────
    print("\n  First-mover analysis (months until layer reaches midpoint of pre/post means):")
    cross_rows = []
    for axis_key, axis_label, _ in AXES:
        for lk, lname, _ in LAYERS:
            col = f"{lk}_{axis_key}"
            s = win[col].rolling(3, min_periods=1, center=True).mean()
            pre = win.loc[:EVENT_DATE - pd.DateOffset(months=2), col].mean()
            post = win.loc[EVENT_DATE + pd.DateOffset(months=2):, col].mean()
            midpoint = (pre + post) / 2
            # Find first month where series crosses midpoint (in direction of shift)
            sign = np.sign(post - pre)
            if sign == 0 or np.isnan(midpoint):
                cross_dates = None
            else:
                target_func = (lambda v: v <= midpoint) if sign < 0 else (lambda v: v >= midpoint)
                crossing = s[s.index >= EVENT_DATE - pd.DateOffset(months=12)].apply(target_func)
                first_crossing = crossing[crossing].index.min() if crossing.any() else None
                cross_dates = first_crossing
            if cross_dates is not None:
                delta = (cross_dates - EVENT_DATE).days / 30.4
            else:
                delta = np.nan
            cross_rows.append(dict(
                axis=axis_key, layer=lk, layer_name=lname,
                pre_mean=float(pre), post_mean=float(post), shift=float(post - pre),
                crossing_date=str(cross_dates.date()) if cross_dates is not None else None,
                months_from_event=float(delta) if not np.isnan(delta) else None,
            ))
    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(OUT_DIR / "jcpoa_first_mover.csv", index=False)
    print("    " + cross_df[["axis","layer_name","shift","months_from_event"]]
          .to_string(index=False).replace("\n","\n    "))

    print(f"\nSaved: {OUT_DIR/'jcpoa_first_mover.csv'}")


if __name__ == "__main__":
    main()
