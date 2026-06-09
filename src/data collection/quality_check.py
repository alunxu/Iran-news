#!/usr/bin/env python3
"""
Data quality check for the enriched NYT Iran corpus.

Loads iran_articles_full.parquet and runs a battery of checks:
  - Coverage breakdown (decade, year, section, document type)
  - Word-count distribution + outlier detection
  - Quality anomalies (paywall stubs, advertisement leakage, duplicates)
  - Iran-relevance (mention frequency in body)
  - Metadata completeness
  - Date / URL consistency

Outputs:
  figures/quality/*.png   — 8 charts
  report/data_quality_check.md — structured summary with recommended preprocessing
"""

from __future__ import annotations

import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "iran_articles_full.parquet"
FIG_DIR = PROJECT_ROOT / "figures" / "quality"
REPORT_PATH = PROJECT_ROOT / "report" / "data_quality_check.md"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

PAYWALL_PHRASES = [
    "subscribe to keep reading",
    "subscribe to continue",
    "you've reached your limit",
    "subscribe for unlimited",
    "create your free account to continue",
    "thanks for your interest",
    "view full article in timesmachine",
]
AD_PHRASES = [
    "advertisement\nskip advertisement",
    "skip advertisement",
    "continue reading the main story",
    "supported by",
]
PRINT_ONLY_MARKERS = [
    "from print",
    "the new york times archives",
]


def fmt_int(n) -> str:
    return f"{int(n):,}"


def fmt_pct(n, total) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"


# ─────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        sys.exit(f"FATAL: {DATA_PATH} not found")
    df = pd.read_parquet(DATA_PATH)
    n_missing_year = int(df["year"].isna().sum())
    if n_missing_year:
        print(f"  ⚠️  {n_missing_year} articles have missing year — dropping for analysis")
        df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df["decade"] = (df["year"] // 10) * 10
    df["has_text"] = df["fulltext"].str.len() > 0
    return df


# ─────────────────────────────────────────────────────────────────
# Section 1 · Coverage
# ─────────────────────────────────────────────────────────────────
def section_coverage(df: pd.DataFrame) -> dict:
    out = {}
    out["total"] = len(df)
    out["with_text"] = int(df["has_text"].sum())
    out["pct"] = 100 * out["with_text"] / out["total"]

    out["by_decade"] = df.groupby("decade").agg(
        total=("uri", "count"),
        with_text=("has_text", "sum"),
    )
    out["by_decade"]["pct"] = (out["by_decade"]["with_text"] / out["by_decade"]["total"] * 100).round(1)

    out["by_year"] = df.groupby("year").agg(
        total=("uri", "count"),
        with_text=("has_text", "sum"),
    )
    out["by_year"]["pct"] = (out["by_year"]["with_text"] / out["by_year"]["total"] * 100).round(1)

    # By scrape_status
    out["by_status"] = df["scrape_status"].value_counts()

    # By section_name (top 15)
    out["by_section"] = df.groupby("section_name").agg(
        total=("uri", "count"),
        with_text=("has_text", "sum"),
    ).sort_values("total", ascending=False).head(15)
    out["by_section"]["pct"] = (out["by_section"]["with_text"] / out["by_section"]["total"] * 100).round(1)

    # By document_type
    out["by_doctype"] = df.groupby("document_type").agg(
        total=("uri", "count"),
        with_text=("has_text", "sum"),
    ).sort_values("total", ascending=False).head(8)
    out["by_doctype"]["pct"] = (out["by_doctype"]["with_text"] / out["by_doctype"]["total"] * 100).round(1)

    return out


def fig_coverage_by_year(df: pd.DataFrame, path: Path):
    by_year = df.groupby("year").agg(
        total=("uri", "count"),
        with_text=("has_text", "sum"),
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(by_year.index, by_year["total"], color="#d8d4c9", label="Total articles")
    ax.bar(by_year.index, by_year["with_text"], color="#2f5f5d", label="With full text")
    ax.set_xlabel("Year")
    ax.set_ylabel("Articles")
    ax.set_title("Article volume and full-text coverage by year")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_coverage_by_decade(coverage: dict, path: Path):
    bd = coverage["by_decade"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh([f"{int(d)}s" for d in bd.index], bd["pct"].values, color="#2f5f5d")
    for i, (pct, n) in enumerate(zip(bd["pct"].values, bd["with_text"].values)):
        ax.text(min(pct + 1.5, 99), i, f"{pct:.1f}%  ({n:,})",
                va="center", fontsize=9, color="#363a3e")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Full-text coverage (%)")
    ax.set_title("Coverage by decade")
    ax.invert_yaxis()
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Section 2 · Word-count distribution
# ─────────────────────────────────────────────────────────────────
def section_wordcount(df: pd.DataFrame) -> dict:
    wc = df.loc[df["has_text"], "fulltext_word_count"]
    out = {
        "n": len(wc),
        "min": int(wc.min()),
        "p1": int(wc.quantile(0.01)),
        "p5": int(wc.quantile(0.05)),
        "median": int(wc.median()),
        "mean": int(wc.mean()),
        "p95": int(wc.quantile(0.95)),
        "p99": int(wc.quantile(0.99)),
        "max": int(wc.max()),
        "lt_50": int((wc < 50).sum()),
        "lt_100": int((wc < 100).sum()),
        "gt_5000": int((wc > 5000).sum()),
        "gt_10000": int((wc > 10000).sum()),
    }
    return out


def fig_wordcount_hist(df: pd.DataFrame, path: Path):
    wc = df.loc[df["has_text"], "fulltext_word_count"]
    wc_clipped = wc.clip(upper=3000)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(wc_clipped, bins=60, color="#2f5f5d", edgecolor="white", linewidth=0.4)
    ax.axvline(wc.median(), color="#b85042", linestyle="--", linewidth=1)
    ax.text(wc.median() + 50, ax.get_ylim()[1] * 0.9,
            f"median = {int(wc.median())}", color="#b85042", fontsize=10)
    ax.set_xlabel("Full-text word count (clipped at 3000)")
    ax.set_ylabel("Articles")
    ax.set_title("Distribution of full-text word counts")
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Section 3 · Quality anomalies
# ─────────────────────────────────────────────────────────────────
def detect_phrase_leakage(df: pd.DataFrame, phrases: list[str]) -> tuple[int, list[str]]:
    """Count how many fulltexts contain any of the given phrases (case-insensitive)."""
    if not df["has_text"].any():
        return 0, []
    mask = pd.Series(False, index=df.index)
    sample = []
    for p in phrases:
        hits = df.loc[df["has_text"], "fulltext"].str.contains(re.escape(p), case=False, regex=True, na=False)
        # Re-index back to df
        full = pd.Series(False, index=df.index)
        full.loc[hits.index] = hits
        mask |= full
        if hits.sum() and len(sample) < 4:
            for uri in df.loc[hits[hits].index, "uri"].head(2):
                sample.append(uri)
    return int(mask.sum()), sample


def section_anomalies(df: pd.DataFrame) -> dict:
    out = {}

    paywall_n, paywall_uri = detect_phrase_leakage(df, PAYWALL_PHRASES)
    ad_n, ad_uri = detect_phrase_leakage(df, AD_PHRASES)
    out["paywall_leakage"] = paywall_n
    out["paywall_uri_samples"] = paywall_uri
    out["ad_leakage"] = ad_n
    out["ad_uri_samples"] = ad_uri

    # Duplicates by content hash (first 1000 chars of fulltext)
    text_df = df.loc[df["has_text"], ["uri", "fulltext"]].copy()
    text_df["hash"] = text_df["fulltext"].str[:1000].apply(
        lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()
    )
    dup_groups = text_df["hash"].value_counts()
    out["duplicate_groups"] = int((dup_groups > 1).sum())
    out["duplicate_articles"] = int(dup_groups[dup_groups > 1].sum() - (dup_groups > 1).sum())

    # Suspiciously short texts (potentially paywall stubs that slipped through)
    if "fulltext_word_count" in df.columns:
        out["short_lt100"] = int(((df["fulltext_word_count"] < 100) & df["has_text"]).sum())
        out["short_lt50"]  = int(((df["fulltext_word_count"] < 50)  & df["has_text"]).sum())
    else:
        out["short_lt100"] = 0
        out["short_lt50"] = 0

    return out


def fig_wordcount_by_status(df: pd.DataFrame, path: Path):
    """Word count distribution by recovery source."""
    statuses = ["success", "recovered", "subscription"]
    data = []
    labels = []
    for s in statuses:
        wc = df.loc[(df["scrape_status"] == s) & df["has_text"], "fulltext_word_count"]
        if len(wc) > 0:
            data.append(wc.clip(upper=3000).values)
            labels.append(f"{s}\n(n={len(wc):,})")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, tick_labels=labels, vert=True, patch_artist=True,
                    showfliers=False, widths=0.5)
    colors = ["#2f5f5d", "#69a297", "#b85042"]
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Word count (clipped at 3000)")
    ax.set_title("Word count by recovery source")
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Section 4 · Iran relevance
# ─────────────────────────────────────────────────────────────────
def section_iran_relevance(df: pd.DataFrame) -> dict:
    """Count Iran-related mentions in body (case-insensitive).

    Counts both literal Iran/Iranian/Iranians AND common proxy terms
    (Tehran, Khomeini, Ayatollah, Persian, Shia, Shah) — many articles
    in the corpus discuss Iran without using the country name itself.
    """
    iran_pat  = r"\biran(?:ian)?s?\b"
    proxy_pat = r"\b(?:tehran|teheran|khomeini|ayatollah|persian|shia|shiite|shah)\b"
    counts_iran  = df.loc[df["has_text"], "fulltext"].str.count(iran_pat,  flags=re.IGNORECASE)
    counts_proxy = df.loc[df["has_text"], "fulltext"].str.count(proxy_pat, flags=re.IGNORECASE)
    counts = counts_iran + counts_proxy
    out = {
        "n": int(df["has_text"].sum()),
        "no_mention": int((counts == 0).sum()),
        "lte_1": int((counts <= 1).sum()),
        "lte_3": int((counts <= 3).sum()),
        "median": int(counts.median()),
        "p95": int(counts.quantile(0.95)),
    }
    out["no_mention_pct"] = 100 * out["no_mention"] / out["n"] if out["n"] else 0
    out["lte_3_pct"] = 100 * out["lte_3"] / out["n"] if out["n"] else 0
    return out


def fig_iran_mentions(df: pd.DataFrame, path: Path):
    iran_pat  = r"\biran(?:ian)?s?\b"
    proxy_pat = r"\b(?:tehran|teheran|khomeini|ayatollah|persian|shia|shiite|shah)\b"
    c_iran  = df.loc[df["has_text"], "fulltext"].str.count(iran_pat,  flags=re.IGNORECASE)
    c_proxy = df.loc[df["has_text"], "fulltext"].str.count(proxy_pat, flags=re.IGNORECASE)
    counts_clip = (c_iran + c_proxy).clip(upper=30)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(counts_clip, bins=31, color="#2f5f5d", edgecolor="white", linewidth=0.4)
    ax.axvline(3, color="#b85042", linestyle="--", linewidth=1)
    ax.text(3.5, ax.get_ylim()[1] * 0.85, "≤3 mentions = peripheral?",
            color="#b85042", fontsize=10)
    ax.set_xlabel("Iran + proxy term mentions per article body (clipped at 30)")
    ax.set_ylabel("Articles")
    ax.set_title("Iran-relevance signal in body (literal + Tehran / Khomeini / Shah etc.)")
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Section 5 · Metadata completeness
# ─────────────────────────────────────────────────────────────────
def section_metadata(df: pd.DataFrame) -> dict:
    out = {}
    fields = ["headline", "abstract", "lead_paragraph", "pub_date", "web_url",
              "section_name", "news_desk", "document_type", "type_of_material"]
    out["missing"] = {
        f: int(df[f].isna().sum()) + int((df[f] == "").sum())
        for f in fields if f in df.columns
    }
    out["images"] = {
        "with_image": int(df.get("has_image", pd.Series([False]*len(df))).sum()),
        "n_images_max": int(df.get("n_images", pd.Series([0]*len(df))).max()),
    }
    return out


# ─────────────────────────────────────────────────────────────────
# Section 6 · Consistency checks
# ─────────────────────────────────────────────────────────────────
def section_consistency(df: pd.DataFrame) -> dict:
    out = {}

    # URL year vs pub_date year
    url_year = df["web_url"].str.extract(r"nytimes\.com/(\d{4})/", expand=False)
    url_year_int = pd.to_numeric(url_year, errors="coerce")
    out["url_year_missing"] = int(url_year.isna().sum())
    out["url_pubdate_mismatch"] = int(
        ((url_year_int.notna()) & (url_year_int != df["year"])).sum()
    )

    # Future-dated articles
    out["future_dated"] = int((df["year"] > 2026).sum())

    return out


# ─────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────
def write_report(df, cov, wc, anom, iran, meta, cons):
    lines = []
    lines.append("# Data Quality Check — NYT Iran Corpus")
    lines.append("")
    lines.append("_Auto-generated by `src/data collection/quality_check.py`._")
    lines.append("")

    # 1. Coverage
    lines.append("## 1. Coverage")
    lines.append("")
    lines.append(f"- **Total articles**: {fmt_int(cov['total'])}")
    lines.append(f"- **With full text**: {fmt_int(cov['with_text'])} ({cov['pct']:.1f}%)")
    lines.append(f"- **Without full text**: {fmt_int(cov['total'] - cov['with_text'])} (relies on API metadata only)")
    lines.append("")
    lines.append("### By decade")
    lines.append("| Decade | Total | With full text | Coverage |")
    lines.append("|---|---|---|---|")
    for d, row in cov["by_decade"].iterrows():
        lines.append(f"| {int(d)}s | {fmt_int(row['total'])} | {fmt_int(row['with_text'])} | {row['pct']:.1f}% |")
    lines.append("")
    lines.append("### Scrape status breakdown")
    lines.append("| Status | Count |")
    lines.append("|---|---|")
    for s, n in cov["by_status"].items():
        lines.append(f"| `{s}` | {fmt_int(n)} |")
    lines.append("")
    lines.append("### Top 15 sections by article volume")
    lines.append("| Section | Total | With full text | Coverage |")
    lines.append("|---|---|---|---|")
    for sec, row in cov["by_section"].iterrows():
        lines.append(f"| {sec or '(none)'} | {fmt_int(row['total'])} | {fmt_int(row['with_text'])} | {row['pct']:.1f}% |")
    lines.append("")
    lines.append("![Coverage by year](../figures/quality/coverage_by_year.png)")
    lines.append("")
    lines.append("![Coverage by decade](../figures/quality/coverage_by_decade.png)")
    lines.append("")

    # 2. Word counts
    lines.append("## 2. Word-count distribution")
    lines.append("")
    lines.append(f"Among {fmt_int(wc['n'])} articles with full text:")
    lines.append("")
    lines.append("| Statistic | Words |")
    lines.append("|---|---|")
    lines.append(f"| Min | {wc['min']} |")
    lines.append(f"| 1st percentile | {wc['p1']} |")
    lines.append(f"| 5th percentile | {wc['p5']} |")
    lines.append(f"| **Median** | **{wc['median']}** |")
    lines.append(f"| **Mean** | **{wc['mean']}** |")
    lines.append(f"| 95th percentile | {wc['p95']} |")
    lines.append(f"| 99th percentile | {wc['p99']} |")
    lines.append(f"| Max | {fmt_int(wc['max'])} |")
    lines.append("")
    lines.append(f"- Articles with **<50 words** (likely paywall stubs / extraction failures): **{fmt_int(wc['lt_50'])}**")
    lines.append(f"- Articles with **<100 words**: {fmt_int(wc['lt_100'])}")
    lines.append(f"- Articles with **>5,000 words**: {fmt_int(wc['gt_5000'])} (likely listings, transcripts, year-end roundups)")
    lines.append(f"- Articles with **>10,000 words**: {fmt_int(wc['gt_10000'])}")
    lines.append("")
    lines.append("![Word count distribution](../figures/quality/wordcount_hist.png)")
    lines.append("")
    lines.append("![Word count by recovery source](../figures/quality/wordcount_by_status.png)")
    lines.append("")

    # 3. Anomalies
    lines.append("## 3. Quality anomalies")
    lines.append("")
    lines.append(f"- **Paywall-text leakage** (e.g. \"Subscribe to keep reading\"): {fmt_int(anom['paywall_leakage'])} articles")
    lines.append(f"- **Advertisement / boilerplate leakage**: {fmt_int(anom['ad_leakage'])} articles")
    lines.append(f"- **Suspiciously short** (<50 words while marked has_text): {fmt_int(anom['short_lt50'])}")
    lines.append(f"- **Duplicate content groups** (same first-1000-char hash): {fmt_int(anom['duplicate_groups'])} groups, {fmt_int(anom['duplicate_articles'])} articles to dedupe")
    lines.append("")

    # 4. Iran relevance
    lines.append("## 4. Iran relevance in body")
    lines.append("")
    lines.append("Many NYT articles discuss Iran without using the country's name — they refer to *Tehran*, the *Ayatollah*, the *Shah*, or *Persian* / *Shia* topics instead. We therefore count mentions of either Iran/Iranian/Iranians **or** the proxy terms *Tehran · Teheran · Khomeini · Ayatollah · Persian · Shia · Shiite · Shah* in each article body.")
    lines.append("")
    lines.append(f"Among the {fmt_int(iran['n'])} articles with full text:")
    lines.append("")
    lines.append(f"- **Zero Iran-relevance signal** in body (no Iran term **and** no proxy): {fmt_int(iran['no_mention'])} ({iran['no_mention_pct']:.1f}%) — likely false positives from NYT keyword tagging")
    lines.append(f"- **≤1 mention**: {fmt_int(iran['lte_1'])}")
    lines.append(f"- **≤3 mentions**: {fmt_int(iran['lte_3'])} ({iran['lte_3_pct']:.1f}%) — Iran is peripheral if at all")
    lines.append(f"- **Median mentions per article**: {iran['median']}")
    lines.append(f"- **95th percentile**: {iran['p95']}")
    lines.append("")
    lines.append("![Iran-relevance signal](../figures/quality/iran_mentions.png)")
    lines.append("")

    # 5. Metadata completeness
    lines.append("## 5. Metadata completeness")
    lines.append("")
    lines.append("| Field | Missing |")
    lines.append("|---|---|")
    for f, n in meta["missing"].items():
        lines.append(f"| `{f}` | {fmt_int(n)} |")
    lines.append("")
    lines.append(f"- Articles with at least one image (NYT-supplied metadata): {fmt_int(meta['images']['with_image'])}")
    lines.append(f"- Max images attached: {meta['images']['n_images_max']}")
    lines.append("")

    # 6. Consistency
    lines.append("## 6. Date / URL consistency")
    lines.append("")
    lines.append(f"- URLs without an extractable year: {fmt_int(cons['url_year_missing'])}")
    lines.append(f"- URL year ≠ pub_date year (potential routing/redirect issues): {fmt_int(cons['url_pubdate_mismatch'])}")
    lines.append(f"- Articles dated after 2026 (invalid): {fmt_int(cons['future_dated'])}")
    lines.append("")

    # 7. Recommendations
    lines.append("## 7. Recommended preprocessing actions")
    lines.append("")
    lines.append("Based on the findings above, before downstream analysis:")
    lines.append("")
    lines.append("1. **Filter out paywall stubs**: drop or set `fulltext = \"\"` for articles with `<50 words` AND containing paywall phrases. Affects ~"
                 + fmt_int(min(wc['lt_50'], anom['paywall_leakage'])) + " articles.")
    lines.append("2. **Strip residual ad/boilerplate**: regex-strip `^Advertisement\\s*` prefixes and `Continue reading the main story` suffixes (already partially handled in `merge_fulltext.py` `clean_fulltext`).")
    lines.append("3. **Deduplicate** based on first-1000-character hash: keep the entry with most metadata; drop the other " + fmt_int(anom['duplicate_articles']) + ".")
    lines.append("4. **Iran-relevance threshold**: optionally restrict body-level analyses (LDA, syntactic) to articles with **≥3 Iran/Iranian mentions in body** (drops " + fmt_int(iran['lte_3']) + " peripheral articles).")
    lines.append("5. **Stratify by decade in all body-dependent analyses** — coverage varies from 27.8% (1970s) to 95.2% (2020s); pooling across decades without stratification will conflate coverage drift with framing drift.")
    lines.append("6. **Headline + abstract analyses**: 100% available regardless of full-text recovery — use these as the primary axis for cross-decade comparisons.")
    lines.append("7. **Future work**: 1979-80 `/archives/` URLs require NYT TimesMachine access (authenticated scraping) — currently blocked by NYT DataDome WAF; ProQuest Historical Newspapers is a fallback if available.")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to: {REPORT_PATH}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {DATA_PATH} ...")
    df = load_data()
    print(f"  {len(df):,} articles, {df['has_text'].sum():,} with full text "
          f"({df['has_text'].mean()*100:.1f}%)")

    print("\nSection 1 — coverage ...")
    cov = section_coverage(df)
    fig_coverage_by_year(df, FIG_DIR / "coverage_by_year.png")
    fig_coverage_by_decade(cov, FIG_DIR / "coverage_by_decade.png")

    print("Section 2 — word counts ...")
    wc = section_wordcount(df)
    fig_wordcount_hist(df, FIG_DIR / "wordcount_hist.png")
    fig_wordcount_by_status(df, FIG_DIR / "wordcount_by_status.png")

    print("Section 3 — anomalies ...")
    anom = section_anomalies(df)

    print("Section 4 — Iran relevance ...")
    iran = section_iran_relevance(df)
    fig_iran_mentions(df, FIG_DIR / "iran_mentions.png")

    print("Section 5 — metadata ...")
    meta = section_metadata(df)

    print("Section 6 — consistency ...")
    cons = section_consistency(df)

    print("\nWriting markdown report ...")
    write_report(df, cov, wc, anom, iran, meta, cons)

    print("\nFigures written to:")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
