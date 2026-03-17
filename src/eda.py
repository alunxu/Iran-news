#!/usr/bin/env python3
"""
Exploratory Data Analysis on the Iran articles corpus.

Generates visualizations to help settle research directions:
1. Article volume over time (yearly + monthly detail at inflection points)
2. Keyword frequency tracking for key framing terms
3. Section/news desk distribution
4. Material type analysis (News vs. Opinion vs. Editorial)
5. Image availability by period
6. Keyword co-occurrence network (for re-narration signals)
"""

import json
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Patch

# ── Config ────────────────────────────────────────────────────────────
FIGURES_DIR = Path("figures/eda")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Key inflection points for annotation
INFLECTION_POINTS = {
    "1979 Revolution": "1979-02-01",
    "Hostage Crisis": "1979-11-04",
    "Iran-Iraq War": "1980-09-22",
    "Iran-Contra": "1986-11-03",
    "9/11 & Axis of Evil": "2002-01-29",
    "Nuclear Deal (JCPOA)": "2015-07-14",
    "Trump Exits JCPOA": "2018-05-08",
    "Soleimani Killing": "2020-01-03",
    "Women Life Freedom": "2022-09-16",
    "2025 Escalation": "2025-01-01",
}

# Framing terms to track
FRAMING_TERMS = {
    "nuclear": r"\bnuclear\b",
    "terrorism/terror": r"\bterror(?:ism|ist)?\b",
    "sanctions": r"\bsanction",
    "diplomacy/diplomatic": r"\bdiplomat(?:ic|cy)?\b",
    "protest": r"\bprotest",
    "human rights": r"\bhuman rights\b",
    "hostage": r"\bhostage",
    "revolution": r"\brevolution",
    "military": r"\bmilitar",
    "oil/energy": r"\b(?:oil|petroleum|energy)\b",
    "women": r"\bwomen\b",
    "democracy": r"\bdemocrac",
    "missile": r"\bmissile",
    "threat": r"\bthreat",
}

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# ── Color scheme ──────────────────────────────────────────────────────
COLORS = {
    "primary": "#2c3e50",
    "accent": "#e74c3c",
    "secondary": "#3498db",
    "bg": "#fafafa",
    "grid": "#ecf0f1",
}


def load_data():
    """Load the filtered Iran articles corpus."""
    parquet_path = Path("data/iran_articles.parquet")
    if not parquet_path.exists():
        print("No data found. Run filter_iran.py first.")
        return None

    df = pd.read_parquet(parquet_path)
    df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce", utc=True)
    df["year"] = df["pub_date"].dt.year
    df["month"] = df["pub_date"].dt.month
    return df


def plot_yearly_volume(df):
    """Plot 1: Article count by year with inflection point annotations."""
    fig, ax = plt.subplots(figsize=(16, 6))

    yearly = df.groupby("year").size()
    ax.bar(yearly.index, yearly.values, color=COLORS["secondary"], alpha=0.8, width=0.8)

    # Annotate inflection points
    for label, date_str in INFLECTION_POINTS.items():
        year = pd.Timestamp(date_str).year
        if year in yearly.index:
            ax.axvline(x=year, color=COLORS["accent"], alpha=0.4, linestyle="--", linewidth=1)
            ax.text(
                year, ax.get_ylim()[1] * 0.95, label,
                rotation=90, va="top", ha="right",
                fontsize=7, color=COLORS["accent"], alpha=0.8,
            )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.set_title("NYT Iran Coverage Volume Over Time (1979–2026)", fontsize=14, fontweight="bold")
    ax.set_xlim(1978, 2027)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_yearly_volume.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_yearly_volume.png")


def plot_monthly_detail(df):
    """Plot 2: Monthly article counts — detail around inflection points."""
    fig, ax = plt.subplots(figsize=(16, 5))

    monthly = df.set_index("pub_date").resample("ME").size()
    ax.plot(monthly.index, monthly.values, color=COLORS["secondary"], linewidth=0.8, alpha=0.7)
    ax.fill_between(monthly.index, monthly.values, alpha=0.2, color=COLORS["secondary"])

    for label, date_str in INFLECTION_POINTS.items():
        ts = pd.Timestamp(date_str, tz="UTC")
        ax.axvline(x=ts, color=COLORS["accent"], alpha=0.5, linestyle="--", linewidth=1)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Articles per Month", fontsize=12)
    ax.set_title("Monthly Iran Coverage Density", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_monthly_detail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 02_monthly_detail.png")


def plot_keyword_trends(df):
    """Plot 3: Framing term frequency over time (rolling 12-month)."""
    import re

    # Create binary columns for each framing term
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

    # Resample to monthly counts
    monthly_terms = term_df.resample("ME").sum()

    # Normalize by total articles per month
    monthly_total = df.set_index("pub_date").resample("ME").size()
    monthly_pct = monthly_terms.div(monthly_total, axis=0) * 100

    # Rolling 12-month average
    rolling_pct = monthly_pct.rolling(12, min_periods=3).mean()

    # Select top terms for readability
    top_terms = ["nuclear", "terrorism/terror", "sanctions", "diplomacy/diplomatic",
                 "protest", "hostage", "military", "women"]

    fig, ax = plt.subplots(figsize=(16, 7))
    for term in top_terms:
        if term in rolling_pct.columns:
            ax.plot(rolling_pct.index, rolling_pct[term], linewidth=1.5, label=term)

    for label, date_str in INFLECTION_POINTS.items():
        ts = pd.Timestamp(date_str, tz="UTC")
        ax.axvline(x=ts, color="gray", alpha=0.3, linestyle=":", linewidth=1)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("% of Iran articles mentioning term (12-month rolling avg)", fontsize=10)
    ax.set_title("Framing Term Prevalence in Iran Coverage", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_keyword_trends.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 03_keyword_trends.png")


def plot_section_distribution(df):
    """Plot 4: Where does Iran coverage live? Section/desk analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top sections
    section_counts = df["section_name"].value_counts().head(15)
    axes[0].barh(section_counts.index[::-1], section_counts.values[::-1],
                 color=COLORS["secondary"], alpha=0.8)
    axes[0].set_xlabel("Number of Articles")
    axes[0].set_title("Top 15 Sections", fontweight="bold")

    # Top news desks
    desk_counts = df["news_desk"].value_counts().head(15)
    axes[1].barh(desk_counts.index[::-1], desk_counts.values[::-1],
                 color=COLORS["primary"], alpha=0.8)
    axes[1].set_xlabel("Number of Articles")
    axes[1].set_title("Top 15 News Desks", fontweight="bold")

    fig.suptitle("Where Does Iran Coverage Live?", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_section_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 04_section_distribution.png")


def plot_material_type_over_time(df):
    """Plot 5: News vs. Opinion vs. Editorial over time."""
    # Categorize material types
    def categorize_material(mat):
        if pd.isna(mat):
            return "Other"
        mat = str(mat).lower()
        if "news" in mat:
            return "News"
        elif "editorial" in mat:
            return "Editorial"
        elif "op-ed" in mat or "opinion" in mat:
            return "Op-Ed/Opinion"
        elif "letter" in mat:
            return "Letters"
        elif "review" in mat:
            return "Review"
        else:
            return "Other"

    df["material_cat"] = df["type_of_material"].apply(categorize_material)

    # Compute yearly shares
    yearly_type = df.groupby(["year", "material_cat"]).size().unstack(fill_value=0)
    yearly_pct = yearly_type.div(yearly_type.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(16, 6))
    categories = ["News", "Editorial", "Op-Ed/Opinion", "Letters", "Other"]
    colors_cat = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71", "#95a5a6"]

    bottom = None
    for cat, color in zip(categories, colors_cat):
        if cat in yearly_pct.columns:
            values = yearly_pct[cat]
            ax.bar(yearly_pct.index, values, bottom=bottom, label=cat,
                   color=color, alpha=0.85, width=0.8)
            if bottom is None:
                bottom = values.copy()
            else:
                bottom = bottom + values

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("% of Coverage", fontsize=12)
    ax.set_title("Material Type Composition of Iran Coverage", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(1978, 2027)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_material_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 05_material_type.png")


def plot_image_availability(df):
    """Plot 6: Image availability over time."""
    yearly_images = df.groupby("year")["has_image"].agg(["sum", "count"])
    yearly_images["pct"] = yearly_images["sum"] / yearly_images["count"] * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Absolute count
    ax1.bar(yearly_images.index, yearly_images["sum"], color=COLORS["secondary"],
            alpha=0.8, label="With images", width=0.8)
    ax1.bar(yearly_images.index, yearly_images["count"] - yearly_images["sum"],
            bottom=yearly_images["sum"], color=COLORS["grid"],
            alpha=0.8, label="No images", width=0.8)
    ax1.set_ylabel("Number of Articles")
    ax1.set_title("Image Availability in Iran Coverage", fontsize=14, fontweight="bold")
    ax1.legend()

    # Percentage
    ax2.plot(yearly_images.index, yearly_images["pct"],
             color=COLORS["accent"], linewidth=2, marker="o", markersize=3)
    ax2.fill_between(yearly_images.index, yearly_images["pct"],
                     alpha=0.2, color=COLORS["accent"])
    ax2.set_ylabel("% Articles with Images")
    ax2.set_xlabel("Year")
    ax2.set_xlim(1978, 2027)
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_image_availability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 06_image_availability.png")


def plot_top_keywords(df):
    """Plot 7: Top structured keywords (from NYT keyword tags)."""
    all_keywords = []
    for kw_json in df["keywords_json"]:
        try:
            keywords = json.loads(kw_json)
            for kw in keywords:
                name = kw.get("name", "")
                value = kw.get("value", "")
                if value:
                    all_keywords.append((name, value))
        except (json.JSONDecodeError, TypeError):
            continue

    # Split by keyword type
    subject_counts = Counter()
    person_counts = Counter()
    geo_counts = Counter()
    org_counts = Counter()

    for name, value in all_keywords:
        if name == "subject":
            subject_counts[value] += 1
        elif name == "persons":
            person_counts[value] += 1
        elif name == "glocations":
            geo_counts[value] += 1
        elif name == "organizations":
            org_counts[value] += 1

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, (title, counts) in zip(
        axes.flat,
        [
            ("Top Subject Tags", subject_counts),
            ("Top Person Tags", person_counts),
            ("Top Location Tags", geo_counts),
            ("Top Organization Tags", org_counts),
        ],
    ):
        top = counts.most_common(20)
        if top:
            labels, values = zip(*top)
            ax.barh(list(labels)[::-1], list(values)[::-1],
                    color=COLORS["secondary"], alpha=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Count")

    fig.suptitle("Most Frequent Keyword Tags in Iran Coverage",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_top_keywords.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 07_top_keywords.png")


def plot_keyword_cooccurrence(df):
    """Plot 8: Keyword co-occurrence heatmap for re-narration signals."""
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

    # Create binary matrix
    import re
    term_matrix = pd.DataFrame()
    for label, pattern in event_terms.items():
        term_matrix[label] = combined_text.str.contains(pattern, regex=True, na=False).astype(int)

    # Co-occurrence matrix
    cooc = term_matrix.T.dot(term_matrix)

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = pd.DataFrame(
        data=[[i >= j for j in range(len(cooc))] for i in range(len(cooc))],
        index=cooc.index, columns=cooc.columns
    )

    sns.heatmap(cooc, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                mask=mask.values, square=True, linewidths=0.5)
    ax.set_title("Term Co-occurrence in Iran Coverage\n(potential re-narration signals)",
                 fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_cooccurrence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 08_cooccurrence.png")


def print_summary_stats(df):
    """Print key summary statistics."""
    print(f"\n{'='*60}")
    print("CORPUS SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total articles: {len(df):,}")
    print(f"Date range: {df['pub_date'].min().strftime('%Y-%m-%d')} to {df['pub_date'].max().strftime('%Y-%m-%d')}")
    print(f"Articles with images: {df['has_image'].sum():,} ({df['has_image'].mean()*100:.1f}%)")
    print(f"Average word count: {df['word_count'].mean():.0f}")
    print(f"Unique sections: {df['section_name'].nunique()}")

    print(f"\nArticles per period:")
    periods = [
        ("Pre-Revolution", 1979, 1979),
        ("Revolution & Hostage Crisis", 1979, 1981),
        ("Iran-Iraq War", 1980, 1988),
        ("Post-Cold-War/90s", 1989, 2000),
        ("Post-9/11 Era", 2001, 2014),
        ("Nuclear Deal Era", 2015, 2017),
        ("Trump Max Pressure", 2018, 2021),
        ("Women Life Freedom", 2022, 2023),
        ("Direct Confrontation", 2024, 2026),
    ]
    for label, y_start, y_end in periods:
        mask = (df["year"] >= y_start) & (df["year"] <= y_end)
        count = mask.sum()
        years = y_end - y_start + 1
        print(f"  {label} ({y_start}–{y_end}): {count:,} articles ({count/max(years,1):.0f}/year)")


def main():
    print("Loading Iran articles corpus...")
    df = load_data()
    if df is None:
        return

    print(f"Loaded {len(df):,} articles\n")

    print("Generating figures:")
    plot_yearly_volume(df)
    plot_monthly_detail(df)
    plot_keyword_trends(df)
    plot_section_distribution(df)
    plot_material_type_over_time(df)
    plot_image_availability(df)
    plot_top_keywords(df)
    plot_keyword_cooccurrence(df)

    print_summary_stats(df)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Review these to decide on the most promising research directions!")


if __name__ == "__main__":
    main()
