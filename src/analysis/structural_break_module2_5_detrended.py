#!/usr/bin/env python3
"""
Module 2.5 — Detrended PELT.

V3 found a +2.0/decade trend in the THREAT HL-body series (p < 1e-4). PELT on a
trending series finds the trend's midpoint as a "break", which is not a regime
change. This module:

  1. Fits a linear trend per series (OLS, time index in months).
  2. Computes residuals (detrended series).
  3. Runs PELT on residuals with the same BIC-style penalty.
  4. Compares detrended breaks to the original C1-series breaks.

Prediction:
  - THREAT 2003-03 (PELT-L2 break in trending series) should disappear or weaken.
  - DIPLO 2016-02 (JCPOA Implementation Day) should survive: that drop is
    discontinuous, not a trend midpoint.

Outputs:
  data/structural_break/detrended_breaks.csv
  data/structural_break/trend_estimates.csv
  figures/structural_break/detrended_breaks.png
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
SERIES = PROJECT_ROOT / "data" / "structural_break" / "monthly_series_v2.csv"
OUT_DIR = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR = PROJECT_ROOT / "figures" / "structural_break"

ACCENT, COOL, CHAR, MUTED = "#B85042", "#2F5F5D", "#363A3E", "#707070"
ORIG_BREAK = "#E08D17"   # orange — original C1 breaks
DETR_BREAK = "#1f77b4"   # blue — detrended breaks
TREND = "#888"

MIN_SEG = 12
BIC_PEN_FACTOR = 2.0

SERIES_SPECS = [
    ("gap_threat_HL_FT_mean_C1", "Headline−body THREAT gap",   ACCENT),
    ("gap_diplo_HL_FT_mean_C1",  "Headline−body DIPLO gap",    COOL),
    ("gap_human_HL_FT_mean_C1",  "Headline−body HUMAN gap",    CHAR),
    ("gap_threat_news_ed_mean_C1", "News−editorial THREAT gap", ACCENT),
    ("gap_diplo_news_ed_mean_C1",  "News−editorial DIPLO gap",  COOL),
]

EVENTS = [
    ("2002-01-29", "Axis of Evil"),
    ("2013-06-15", "Rouhani"),
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
]


def pelt_l2(values: np.ndarray) -> list[int]:
    if len(values) < 2 * MIN_SEG:
        return []
    sigma = float(np.std(values))
    pen = BIC_PEN_FACTOR * sigma**2 * np.log(len(values))
    algo = rpt.Pelt(model="l2", min_size=MIN_SEG).fit(values.reshape(-1, 1))
    breaks = algo.predict(pen=pen)
    return [b for b in breaks if b < len(values)]


def main():
    df = pd.read_csv(SERIES, parse_dates=["pub_month"]).set_index("pub_month")

    trend_rows = []
    break_rows = []
    series_data = {}

    for col, label, color in SERIES_SPECS:
        s = df[col].dropna()
        if len(s) < 2 * MIN_SEG:
            continue
        dates = s.index
        y = s.values.astype(float)
        # Time index in years from first observation (slope unit: per year)
        t_years = np.array([(d - dates[0]).days / 365.25 for d in dates])

        # OLS linear fit
        slope, intercept, r_value, p_value, _ = stats.linregress(t_years, y)
        trend = slope * t_years + intercept
        residuals = y - trend
        trend_rows.append(dict(
            series=col, label=label,
            slope_per_year=float(slope),
            slope_per_decade=float(slope * 10),
            intercept=float(intercept),
            r_squared=float(r_value**2),
            p_value=float(p_value),
            n=len(y),
        ))

        # Breaks on original (C1) and detrended residuals
        orig_breaks = pelt_l2(y)
        detr_breaks = pelt_l2(residuals)

        # Map to dates
        orig_dates = [dates[b].date() for b in orig_breaks]
        detr_dates = [dates[b].date() for b in detr_breaks]

        for b in orig_breaks:
            break_rows.append(dict(series=col, kind="original_C1",
                                   break_date=dates[b].date(), break_idx=int(b)))
        for b in detr_breaks:
            break_rows.append(dict(series=col, kind="detrended",
                                   break_date=dates[b].date(), break_idx=int(b)))

        series_data[col] = dict(
            label=label, color=color, dates=dates, y=y, trend=trend,
            residuals=residuals, orig_breaks=orig_breaks, detr_breaks=detr_breaks,
            slope_decade=slope * 10,
        )

        print(f"\n{label}")
        print(f"  Linear trend: {slope*10:+.3f}/decade (R²={r_value**2:.3f}, p={p_value:.2e})")
        print(f"  Original C1 breaks: {orig_dates if orig_dates else '(none)'}")
        print(f"  Detrended breaks:   {detr_dates if detr_dates else '(none)'}")

    pd.DataFrame(trend_rows).to_csv(OUT_DIR / "trend_estimates.csv", index=False)
    pd.DataFrame(break_rows).to_csv(OUT_DIR / "detrended_breaks.csv", index=False)

    # Figure: 5 rows × 1 column, each shows detrended residuals + both break sets
    fig, axes = plt.subplots(len(series_data), 1, figsize=(11, 8.5), dpi=150, sharex=True)
    for ax, (col, d) in zip(axes, series_data.items()):
        # Plot residuals (detrended series, centered on zero)
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        smoothed = pd.Series(d["residuals"], index=d["dates"]).rolling(12, min_periods=1).mean()
        ax.plot(d["dates"], smoothed, color=CHAR, lw=1.2, label="Detrended residual (12mo MA)")
        # Original C1 breaks
        for b in d["orig_breaks"]:
            ax.axvline(d["dates"][b], color=ORIG_BREAK, lw=1.2, ls="--", alpha=0.85,
                       label="_nolegend_")
        # Detrended breaks
        for b in d["detr_breaks"]:
            ax.axvline(d["dates"][b], color=DETR_BREAK, lw=1.4, alpha=0.85,
                       label="_nolegend_")
        # Event markers: solid vertical pointers, lightly drawn behind the data.
        for ed, _ in EVENTS:
            ax.axvline(pd.Timestamp(ed), color=MUTED, lw=0.65, alpha=0.42, ls="-", zorder=0)

        ax.set_title(f"{d['label']}  (trend $=$ {d['slope_decade']:+.2f}/decade)",
                     fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)

    # Legend on the top axis
    axes[0].plot([], [], color=ORIG_BREAK, lw=1.2, ls="--", label="Original PELT break (trending series)")
    axes[0].plot([], [], color=DETR_BREAK, lw=1.4, label="Detrended PELT break")
    axes[0].legend(loc="upper left", fontsize=7, frameon=False, ncol=3)
    for idx, (ed, label) in enumerate(LABEL_EVENTS):
        d = mdates.date2num(pd.Timestamp(ed).to_pydatetime())
        axes[0].annotate(
            label,
            xy=(d, 1.01),
            xycoords=axes[0].get_xaxis_transform(),
            xytext=(d, 1.17 + (idx % 2) * 0.10),
            textcoords=axes[0].get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7,
            color=MUTED,
            rotation=42,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, alpha=0.75),
            annotation_clip=False,
        )
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Year", fontsize=9)
    fig.suptitle("Detrended PELT: linear-trend residuals with original-series vs.\\ detrended breaks",
                 fontsize=11, y=0.995)
    plt.tight_layout()
    fig_path = FIG_DIR / "detrended_breaks.png"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"\nSaved: {fig_path}")
    print(f"Saved: {OUT_DIR / 'trend_estimates.csv'}")
    print(f"Saved: {OUT_DIR / 'detrended_breaks.csv'}")


if __name__ == "__main__":
    main()
