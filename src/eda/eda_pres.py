#!/usr/bin/env python3
"""
Regenerate the 3 key EDA figures for presentation slides.
- Times New Roman font
- All text significantly larger than original
- Saves to figures/eda_pres/
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path

# Import config from eda.py
from eda import load_data, INFLECTION_POINTS, FRAMING_TERMS, COLORS

# ── Output directory ──────────────────────────────────────────────────
OUT_DIR = Path("figures/eda_pres")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Apply style FIRST, then override font ─────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Font must be set AFTER plt.style.use, otherwise the style resets it
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"


def plot_yearly_volume_pres(df):
    """Plot 1: Article count by year — presentation version."""
    fig, ax = plt.subplots(figsize=(18, 7))

    yearly = df.groupby("year").size()
    ax.bar(yearly.index, yearly.values, color=COLORS["secondary"], alpha=0.8, width=0.8)

    # Annotate inflection points — with offset to avoid overlap
    # "1979 Revolution" and "Hostage Crisis" are very close (1979.1 and 1979.8)
    # so we need to offset them
    label_offsets = {
        "1979 Revolution": 0.8,
        "Hostage Crisis": 1.8,
        "Iran-Iraq War": 1.5,
    }

    for label, date_str in INFLECTION_POINTS.items():
        year = pd.Timestamp(date_str).year
        if year in yearly.index:
            x_offset = label_offsets.get(label, 0)
            ax.axvline(x=year, color=COLORS["accent"], alpha=0.4, linestyle="--", linewidth=1.2)
            ax.text(
                year + x_offset, ax.get_ylim()[1] * 0.95, label,
                rotation=90, va="top", ha="right",
                fontsize=14, fontweight="bold",
                color=COLORS["accent"], alpha=0.9,
            )

    ax.set_xlabel("Year", fontsize=20, fontweight="bold")
    ax.set_ylabel("Number of Articles", fontsize=20, fontweight="bold")
    ax.set_title("NYT Iran Coverage Volume Over Time (1979–2026)",
                 fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim(1978, 2027)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_yearly_volume.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_yearly_volume.png (presentation)")


def plot_keyword_trends_pres(df):
    """Plot 3: Framing term frequency — presentation version."""
    import re

    combined_text = (
        df["headline"].fillna("") + " " +
        df["abstract"].fillna("") + " " +
        df["lead_paragraph"].fillna("")
    ).str.lower()

    records = {}
    for term_label, pattern in FRAMING_TERMS.items():
        records[term_label] = combined_text.str.contains(pattern, regex=True, na=False)

    term_df = pd.DataFrame(records, index=df.index)
    term_df["pub_date"] = df["pub_date"]
    term_df = term_df.set_index("pub_date")

    monthly_terms = term_df.resample("ME").sum()
    monthly_total = df.set_index("pub_date").resample("ME").size()
    monthly_pct = monthly_terms.div(monthly_total, axis=0) * 100
    rolling_pct = monthly_pct.rolling(12, min_periods=3).mean()

    top_terms = ["nuclear", "terrorism/terror", "sanctions", "diplomacy/diplomatic",
                 "protest", "hostage", "military", "women"]

    fig, ax = plt.subplots(figsize=(18, 8))
    for term in top_terms:
        if term in rolling_pct.columns:
            ax.plot(rolling_pct.index, rolling_pct[term], linewidth=2.5, label=term)

    for label, date_str in INFLECTION_POINTS.items():
        ts = pd.Timestamp(date_str, tz="UTC")
        ax.axvline(x=ts, color="gray", alpha=0.3, linestyle=":", linewidth=1)

    ax.set_xlabel("Date", fontsize=22, fontweight="bold")
    ax.set_ylabel("% of Iran articles mentioning term\n(12-month rolling avg)",
                  fontsize=18, fontweight="bold")
    ax.set_title("Framing Term Prevalence", fontsize=26, fontweight="bold")
    ax.legend(loc="upper left", fontsize=16, ncol=2)
    ax.tick_params(axis="both", labelsize=16)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_keyword_trends.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 03_keyword_trends.png (presentation)")


def plot_cooccurrence_pres(df):
    """Plot 8: Co-occurrence heatmap — presentation version."""
    event_terms = {
        "1979 Revolution": r"\b(?:revolution|revolutionary|revolt)\b",
        "Hostage Crisis": r"\bhostage\b",
        "Iran-Iraq War": r"\b(?:iran[\s-]?iraq|saddam|war)\b",
        "Nuclear": r"\bnuclear\b",
        "Terrorism": r"\bterror(?:ism|ist)?\b",
        "Sanctions": r"\bsanction",
        "JCPOA/Deal": r"\b(?:jcpoa|nuclear deal|agreement)\b",
        "Protest": r"\bprotest",
        "Women": r"\bwomen\b",
        "Israel": r"\bisrael",
        "Axis of Evil": r"\baxis of evil\b",
    }

    combined_text = (
        df["headline"].fillna("") + " " +
        df["abstract"].fillna("") + " " +
        df["lead_paragraph"].fillna("")
    ).str.lower()

    term_matrix = pd.DataFrame()
    for label, pattern in event_terms.items():
        term_matrix[label] = combined_text.str.contains(pattern, regex=True, na=False).astype(int)

    cooc = term_matrix.T.dot(term_matrix)

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = pd.DataFrame(
        data=[[i >= j for j in range(len(cooc))] for i in range(len(cooc))],
        index=cooc.index, columns=cooc.columns
    )

    sns.heatmap(cooc, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                mask=mask.values, square=True, linewidths=0.5,
                annot_kws={"size": 15, "fontweight": "bold"})
    ax.set_title("Term Co-occurrence", fontsize=26, fontweight="bold")
    ax.tick_params(axis="both", labelsize=15)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_cooccurrence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 08_cooccurrence.png (presentation)")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} articles loaded.\n")

    print("Generating presentation figures (Times New Roman, large text)...")
    plot_yearly_volume_pres(df)
    plot_keyword_trends_pres(df)
    plot_cooccurrence_pres(df)
    print("\nDone! Figures saved to figures/eda_pres/")
