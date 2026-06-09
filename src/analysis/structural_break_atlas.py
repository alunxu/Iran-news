#!/usr/bin/env python3
"""
Structural Break Detection — Module 1: The Atlas
=================================================

Build the visual + numerical foundation for everything downstream:

  • Resample the enriched corpus to monthly resolution (1979-01 → 2026-04)
  • Compute six framing tension time series:
       1.  vol_t                   — article volume per month
       2.  gap_threat_HL_FT        — headline-vs-body threat density gap
       3.  gap_diplo_HL_FT         — headline-vs-body diplomacy gap
       4.  gap_human_HL_FT         — headline-vs-body humanizing gap
       5.  gap_threat_news_ed      — news-vs-editorial threat gap (voice tension)
       6.  gap_diplo_news_ed       — news-vs-editorial diplomacy gap
  • Anchor 11 historical events on the time axis
  • Save:
       figures/structural_break/atlas.png   — 6-panel framing atlas
       data/structural_break/monthly_series.csv — clean monthly series
       data/structural_break/monthly_volumes.csv — diagnostic counts per voice

Downstream modules (M2-M5: break detection, layer-synchronization, taxonomy,
hysteresis) consume monthly_series.csv directly — do NOT recompute per-article
densities; that's expensive.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "iran_articles_full.parquet"
LEX_PATH     = PROJECT_ROOT / "data" / "lexicons.json"
OUT_DIR      = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR      = PROJECT_ROOT / "figures" / "structural_break"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Project color palette (matches existing figures) ────────────────────
ACCENT = "#B85042"   # terracotta — threat / shock
COOL   = "#2F5F5D"   # deep teal — diplomacy
CHAR   = "#363A3E"   # charcoal — humanizing / neutral
SAGE   = "#A7BEAE"   # sage — secondary
MUTED  = "#707070"

MIN_ART_HL_FT     = 5   # min monthly articles-with-body for headline–body gap
MIN_ART_PER_VOICE = 2   # min monthly articles per voice for news–editorial gap
SMOOTH_WINDOW     = 12  # months — 12-mo centered rolling mean

# ── Historical anchor events (shortened labels to avoid overlap) ──────
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


# ────────────────────────────────────────────────────────────────────────
# Density helpers
# ────────────────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"\b[a-zA-Z]+\b")

def lexicon_density(text: str | None, lex_set: set[str]) -> float:
    """Hits per 1,000 tokens. Returns 0 for empty text."""
    if not text:
        return 0.0
    toks = _TOKEN_RE.findall(text.lower())
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in lex_set)
    return 1000.0 * hits / len(toks)


# ────────────────────────────────────────────────────────────────────────
# Per-article tension scores
# ────────────────────────────────────────────────────────────────────────
def compute_article_densities(df: pd.DataFrame, lex: dict) -> pd.DataFrame:
    """Adds 6 density columns: {threat,diplo,human} × {HL, FT}."""
    threat  = set(lex["threat"])
    diplo   = set(lex["diplomacy"])
    human   = set(lex["humanizing"])

    print("  computing headline densities ...", flush=True)
    df["d_threat_HL"] = df["headline"].apply(lambda t: lexicon_density(t, threat))
    df["d_diplo_HL"]  = df["headline"].apply(lambda t: lexicon_density(t, diplo))
    df["d_human_HL"]  = df["headline"].apply(lambda t: lexicon_density(t, human))

    print("  computing body densities ...", flush=True)
    df["d_threat_FT"] = df["fulltext"].apply(lambda t: lexicon_density(t, threat))
    df["d_diplo_FT"]  = df["fulltext"].apply(lambda t: lexicon_density(t, diplo))
    df["d_human_FT"]  = df["fulltext"].apply(lambda t: lexicon_density(t, human))

    return df


# ────────────────────────────────────────────────────────────────────────
# Monthly aggregation
# ────────────────────────────────────────────────────────────────────────
def build_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return monthly DataFrame indexed by month-start (Timestamp), with columns:

      vol_total / vol_with_text
      threat_HL / threat_FT  / diplo_HL / diplo_FT / human_HL / human_FT
      threat_news / threat_editorial   diplo_news / diplo_editorial
      gap_*  derived columns
    """
    df = df.dropna(subset=["year", "month"]).copy()
    df["pub_month"] = pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" +
        df["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )

    has_body = df["fulltext"].str.len() > 0

    # 1. Volume series
    vol_total      = df.groupby("pub_month").size().rename("vol_total")
    vol_with_text  = df[has_body].groupby("pub_month").size().rename("vol_with_text")

    # 2. Component-level density (HL is on all articles; FT only where present)
    def _mean_metric(group_df, col):
        return group_df.groupby("pub_month")[col].mean()

    threat_HL = _mean_metric(df, "d_threat_HL").rename("threat_HL")
    diplo_HL  = _mean_metric(df, "d_diplo_HL").rename("diplo_HL")
    human_HL  = _mean_metric(df, "d_human_HL").rename("human_HL")

    df_body = df[has_body]
    threat_FT = _mean_metric(df_body, "d_threat_FT").rename("threat_FT")
    diplo_FT  = _mean_metric(df_body, "d_diplo_FT").rename("diplo_FT")
    human_FT  = _mean_metric(df_body, "d_human_FT").rename("human_FT")

    # 3. Voice-stratified body densities (news vs editorial)
    df_news  = df_body[df_body["voice"] == "news"]
    df_ed    = df_body[df_body["voice"] == "editorial"]
    threat_news      = df_news.groupby("pub_month")["d_threat_FT"].mean().rename("threat_news")
    threat_editorial = df_ed.groupby("pub_month")["d_threat_FT"].mean().rename("threat_editorial")
    diplo_news       = df_news.groupby("pub_month")["d_diplo_FT"].mean().rename("diplo_news")
    diplo_editorial  = df_ed.groupby("pub_month")["d_diplo_FT"].mean().rename("diplo_editorial")
    n_news      = df_news.groupby("pub_month").size().rename("n_news")
    n_editorial = df_ed.groupby("pub_month").size().rename("n_editorial")

    out = pd.concat([
        vol_total, vol_with_text,
        threat_HL, threat_FT, diplo_HL, diplo_FT, human_HL, human_FT,
        threat_news, threat_editorial, diplo_news, diplo_editorial,
        n_news, n_editorial,
    ], axis=1).sort_index()

    # 4. Derived tension gaps
    out["gap_threat_HL_FT"] = out["threat_HL"] - out["threat_FT"]
    out["gap_diplo_HL_FT"]  = out["diplo_HL"]  - out["diplo_FT"]
    out["gap_human_HL_FT"]  = out["human_HL"]  - out["human_FT"]
    out["gap_threat_news_ed"] = out["threat_news"] - out["threat_editorial"]
    out["gap_diplo_news_ed"]  = out["diplo_news"]  - out["diplo_editorial"]

    # 5. Mask tension cells with too few articles
    out.loc[out["vol_with_text"] < MIN_ART_HL_FT,
            ["gap_threat_HL_FT", "gap_diplo_HL_FT", "gap_human_HL_FT"]] = np.nan
    out.loc[(out["n_news"] < MIN_ART_PER_VOICE) | (out["n_editorial"] < MIN_ART_PER_VOICE),
            ["gap_threat_news_ed", "gap_diplo_news_ed"]] = np.nan

    return out


# ────────────────────────────────────────────────────────────────────────
# Atlas figure (6 panels)
# ────────────────────────────────────────────────────────────────────────
def plot_atlas(monthly: pd.DataFrame, out_path: Path) -> None:
    """A 6-panel multi-decade Framing Atlas anchored to historical events.

    Layout:
      [thin ribbon: event timeline with labels]
      [vol_total]
      [HL-body THREAT / DIPLO / HUMAN gaps × 3]
      [news-editorial THREAT / DIPLO gaps × 2]
    """
    panels = [
        ("vol_total",          "Article volume (per month)",                "#888888", False),
        ("gap_threat_HL_FT",   "Headline−body THREAT gap (per 1k tokens)",  ACCENT, True),
        ("gap_diplo_HL_FT",    "Headline−body DIPLOMACY gap",               COOL,   True),
        ("gap_human_HL_FT",    "Headline−body HUMANIZING gap",              CHAR,   True),
        ("gap_threat_news_ed", "News−editorial THREAT gap (body)",          ACCENT, True),
        ("gap_diplo_news_ed",  "News−editorial DIPLOMACY gap (body)",       COOL,   True),
    ]

    # Top ribbon (event labels) + 6 panels.  Total ~13" × 13" @ 150 dpi → 1950 × 1950 px
    height_ratios = [1.0] + [1.6] * 6
    fig = plt.figure(figsize=(13, 13.0), dpi=150)
    gs = fig.add_gridspec(
        nrows=len(panels) + 1, ncols=1,
        height_ratios=height_ratios, hspace=0.42,
        top=0.945, bottom=0.04, left=0.07, right=0.985,
    )
    ribbon_ax = fig.add_subplot(gs[0])
    axes = [fig.add_subplot(gs[i+1], sharex=ribbon_ax) for i in range(len(panels))]

    # Suptitle + subtitle (placed above the gridspec area)
    fig.text(
        0.07, 0.985,
        "Framing Atlas: NYT Iran coverage, 1979 – 2026 (monthly)",
        fontsize=15, fontweight="bold", ha="left", va="top", color=CHAR,
    )
    fig.text(
        0.07, 0.962,
        "Thin line = raw monthly aggregate · thick line = 12-month rolling mean ·"
        " vertical guides = candidate geopolitical breakpoints",
        fontsize=10, color=MUTED, ha="left", va="top",
    )

    event_dates  = [pd.to_datetime(d) for d, _ in HISTORICAL_EVENTS]
    event_labels = [lbl for _, lbl in HISTORICAL_EVENTS]

    # ── Event ribbon (4-position vertical stagger, less label overlap) ─────
    ribbon_ax.set_xlim(monthly.index.min(), monthly.index.max())
    ribbon_ax.set_ylim(0, 1)
    y_positions = [0.82, 0.58, 0.33, 0.10]
    for i, (d, lbl) in enumerate(zip(event_dates, event_labels)):
        y = y_positions[i % 4]
        ribbon_ax.axvline(d, ymin=0, ymax=1, color=MUTED, linewidth=0.5, alpha=0.65)
        ribbon_ax.annotate(
            lbl, xy=(d, y), xytext=(3, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=7.5, color=CHAR, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=MUTED, lw=0.3, alpha=0.92),
        )
    ribbon_ax.set_yticks([])
    ribbon_ax.spines[["left", "right", "top"]].set_visible(False)
    ribbon_ax.spines["bottom"].set_color(MUTED)
    ribbon_ax.tick_params(axis="x", labelbottom=False, bottom=False)
    ribbon_ax.set_title("Historical anchors", fontsize=10, loc="left", color=CHAR, pad=2)

    # ── Data panels ────────────────────────────────────────────────
    for i, (ax, (col, title, color, is_gap)) in enumerate(zip(axes, panels)):
        series = monthly[col].copy()
        smoothed = series.rolling(SMOOTH_WINDOW, center=True, min_periods=4).mean()

        # Raw thin line
        ax.plot(series.index, series.values, color=color, linewidth=0.4, alpha=0.35)
        # Smoothed thick line
        ax.plot(smoothed.index, smoothed.values, color=color, linewidth=1.8)

        # Zero reference for gap panels
        if is_gap:
            ax.axhline(0, color=MUTED, linewidth=0.4, linestyle=":", zorder=0)

        # Vertical event markers
        for d in event_dates:
            ax.axvline(d, color=MUTED, linewidth=0.4, linestyle="-", alpha=0.3, zorder=0)

        ax.set_title(title, fontsize=10.5, loc="left", color=CHAR, pad=4)
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=9, colors=CHAR)
        ax.margins(x=0.005)

        # Hide x-tick labels on all but the bottom panel
        is_bottom = (i == len(panels) - 1)
        if not is_bottom:
            ax.tick_params(axis="x", labelbottom=False)

    # X-axis formatting on bottom panel
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
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  {len(df):,} articles")

    print(f"Loading {LEX_PATH} ...")
    with LEX_PATH.open() as f:
        lex = json.load(f)
    print(f"  threat={len(lex['threat'])} diplo={len(lex['diplomacy'])} human={len(lex['humanizing'])}")

    df = compute_article_densities(df, lex)

    print("Building monthly series ...")
    monthly = build_monthly_series(df)
    print(f"  {len(monthly)} months from {monthly.index.min().date()} to {monthly.index.max().date()}")
    print(f"  non-null tension cells:")
    for col in ["gap_threat_HL_FT", "gap_diplo_HL_FT", "gap_human_HL_FT",
                "gap_threat_news_ed", "gap_diplo_news_ed"]:
        print(f"    {col:25s} {monthly[col].notna().sum():>3} months")

    csv_path = OUT_DIR / "monthly_series.csv"
    monthly.to_csv(csv_path, index_label="pub_month")
    print(f"  saved → {csv_path}")

    # Diagnostic: volume per voice over time (for Module 2)
    df["pub_month"] = pd.to_datetime(
        df.dropna(subset=["year", "month"])["year"].astype(int).astype(str) + "-" +
        df.dropna(subset=["year", "month"])["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    vol_by_voice = (
        df.dropna(subset=["pub_month"])
          .groupby(["pub_month", "voice"]).size().unstack(fill_value=0)
          .sort_index()
    )
    vol_by_voice.to_csv(OUT_DIR / "monthly_volumes.csv", index_label="pub_month")
    print(f"  saved → {OUT_DIR / 'monthly_volumes.csv'}")

    print("\nPlotting atlas ...")
    plot_atlas(monthly, FIG_DIR / "atlas.png")


if __name__ == "__main__":
    main()
