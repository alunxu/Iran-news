#!/usr/bin/env python3
"""
Re-run PELT + Chow on the C1-corrected (same-article paired) series.

Compares break locations and Chow test verdicts between:
  - V1 series (HL = all-article mean, FT = body-subset mean) — original
  - C1 series (both restricted to same-article body-having subset)
  - C3 series (pooled estimator on the C1 subset)

Purpose: check whether the JCPOA 2016-02 finding (97% bootstrap stability)
and the six original PELT breaks survive the same-article correction.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_CSV = PROJECT_ROOT / "data" / "structural_break" / "monthly_series.csv"
V2_CSV = PROJECT_ROOT / "data" / "structural_break" / "monthly_series_v2.csv"
OUT_DIR = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR = PROJECT_ROOT / "figures" / "structural_break"

ACCENT, COOL, CHAR = "#B85042", "#2F5F5D", "#363A3E"
BREAK_COLOR = "#E08D17"

HISTORICAL_EVENTS = [
    ("1979-02-01", "Khomeini returns"),
    ("1979-11-04", "Embassy seized"),
    ("1980-09-22", "Iran-Iraq war"),
    ("1988-07-03", "IR655"),
    ("1989-06-03", "Khomeini dies"),
    ("2002-01-29", "Axis of Evil"),
    ("2009-06-13", "Green Movement"),
    ("2013-06-15", "Rouhani elected"),
    ("2015-07-14", "JCPOA"),
    ("2018-05-08", "JCPOA exit"),
    ("2020-01-03", "Soleimani"),
    ("2022-09-16", "Mahsa Amini"),
]
EVENT_DATES = [pd.Timestamp(d) for d, _ in HISTORICAL_EVENTS]
EVENT_LABELS = [lbl for _, lbl in HISTORICAL_EVENTS]

MIN_SEGMENT_MONTHS = 12
BIC_PEN_FACTOR = 2.0
EVENT_MATCH_WINDOW = 12


def detect_breaks(values, min_size=MIN_SEGMENT_MONTHS):
    if len(values) < 2 * min_size:
        return []
    sigma = np.std(values)
    pen = BIC_PEN_FACTOR * sigma**2 * np.log(len(values))
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(values.reshape(-1, 1))
    breaks = algo.predict(pen=pen)
    return [b for b in breaks if b < len(values)]


def nearest_event(break_date, window=EVENT_MATCH_WINDOW):
    deltas = [(lbl, (break_date - d).days / 30.4)
              for d, lbl in zip(EVENT_DATES, EVENT_LABELS)]
    deltas.sort(key=lambda x: abs(x[1]))
    label, delta = deltas[0]
    if abs(delta) <= window:
        return label, round(delta)
    return None, None


def chow_test(values, break_idx, min_segment=6):
    n = len(values)
    pre = values[:break_idx]
    post = values[break_idx:]
    if len(pre) < min_segment or len(post) < min_segment:
        return dict(f_stat=np.nan, p_value=np.nan, shift=np.nan)
    rss_full = np.sum((values - values.mean()) ** 2)
    rss_unrest = np.sum((pre - pre.mean()) ** 2) + np.sum((post - post.mean()) ** 2)
    f = ((rss_full - rss_unrest) / 1) / (rss_unrest / (n - 2))
    p = 1.0 - stats.f.cdf(f, 1, n - 2)
    return dict(f_stat=float(f), p_value=float(p),
                pre_mean=float(pre.mean()), post_mean=float(post.mean()),
                shift=float(post.mean() - pre.mean()))


def analyse_series(name, col, dates, values):
    detected = detect_breaks(values)
    records = []
    if detected:
        segs = [0] + detected + [len(values)]
        for i, idx in enumerate(detected):
            seg_pre = values[segs[i]:idx]
            seg_post = values[idx:segs[i + 2]]
            br_date = dates[idx]
            ev_label, ev_delta = nearest_event(br_date)
            records.append(dict(
                source=name,
                series=col,
                break_date=br_date.date(),
                shift=float(seg_post.mean() - seg_pre.mean()),
                n_pre=len(seg_pre),
                n_post=len(seg_post),
                nearest_event=ev_label,
                event_delta_months=ev_delta,
            ))
    return pd.DataFrame(records)


SERIES_MAP = [
    ("threat_HL_FT",  "Headline−body THREAT gap",    ACCENT),
    ("diplo_HL_FT",   "Headline−body DIPLOMACY gap", COOL),
    ("human_HL_FT",   "Headline−body HUMANIZING gap", CHAR),
    ("threat_news_ed","News−editorial THREAT gap",   ACCENT),
    ("diplo_news_ed", "News−editorial DIPLOMACY gap", COOL),
]


def main():
    v1 = pd.read_csv(V1_CSV, parse_dates=["pub_month"]).set_index("pub_month")
    v2 = pd.read_csv(V2_CSV, parse_dates=["pub_month"]).set_index("pub_month")

    all_breaks = []
    for series_key, label, color in SERIES_MAP:
        v1_col = f"gap_{series_key}"
        c1_col = f"gap_{series_key}_mean_C1"
        c3_col = f"gap_{series_key}_pooled_C1"

        for name, df, col in [("V1", v1, v1_col),
                              ("C1_mean", v2, c1_col),
                              ("C3_pooled", v2, c3_col)]:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if len(s) < 2 * MIN_SEGMENT_MONTHS:
                continue
            recs = analyse_series(name, series_key, s.index, s.values)
            all_breaks.append(recs)

    out = pd.concat(all_breaks, ignore_index=True)
    out.to_csv(OUT_DIR / "baiperron_breaks_c1_compare.csv", index=False)

    print("\n=== PELT break comparison: V1 vs C1 vs C3 ===\n")
    for series_key, label, color in SERIES_MAP:
        sub = out[out["series"] == series_key]
        if sub.empty:
            print(f"[{label}] no breaks detected in any variant.")
            continue
        print(f"\n[{label}]")
        for src in ["V1", "C1_mean", "C3_pooled"]:
            ss = sub[sub["source"] == src]
            if ss.empty:
                print(f"  {src:10s}: (none)")
            else:
                for _, r in ss.iterrows():
                    ev = r["nearest_event"] or "—"
                    print(f"  {src:10s}: {r['break_date']}  Δ={r['shift']:+5.1f}  near={ev}")

    # Make a focused comparison plot — keep dimensions safe
    fig, axes = plt.subplots(5, 1, figsize=(12, 9), dpi=150, sharex=True)
    for ax, (series_key, label, color) in zip(axes, SERIES_MAP):
        v1_col = f"gap_{series_key}"
        c1_col = f"gap_{series_key}_mean_C1"
        if v1_col in v1.columns:
            s = v1[v1_col].dropna()
            ax.plot(s.index, s.rolling(12, min_periods=1).mean(),
                    color=ACCENT, lw=1.0, alpha=0.6, label="V1 (12mo MA)")
        if c1_col in v2.columns:
            s = v2[c1_col].dropna()
            ax.plot(s.index, s.rolling(12, min_periods=1).mean(),
                    color=CHAR, lw=1.4, label="C1 same-article (12mo MA)")
        # mark C1 detected breaks for this series
        c1_breaks = out[(out["series"] == series_key) & (out["source"] == "C1_mean")]
        for _, r in c1_breaks.iterrows():
            ax.axvline(pd.Timestamp(r["break_date"]), color=BREAK_COLOR,
                       lw=1.2, alpha=0.8, ls="--")
        ax.set_title(label, fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=7, frameon=False)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("PELT breaks on C1-corrected series (orange dashed = detected break)",
                 fontsize=11, y=0.995)
    plt.tight_layout()
    fig_path = FIG_DIR / "breakpoints_c1.png"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"\nSaved: {fig_path}")
    print(f"Saved: {OUT_DIR / 'baiperron_breaks_c1_compare.csv'}")


if __name__ == "__main__":
    main()
