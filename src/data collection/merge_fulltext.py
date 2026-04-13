#!/usr/bin/env python3
"""
Merge scraped full-text articles back into the main dataset.

Reads the original iran_articles.parquet and the scraped fulltext files,
producing an enriched dataset with the full article body.

Output:
  - data/iran_articles_full.parquet  (main enriched dataset)
  - data/iran_articles_full.csv      (CSV mirror)

Usage:
    python merge_fulltext.py
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas pyarrow")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FULLTEXT_DIR = DATA_DIR / "fulltext"
SCRAPE_RESULTS_PATH = DATA_DIR / "fulltext_articles.csv"


def uri_to_hash(uri: str) -> str:
    """Same hash function as the scraper uses for filenames."""
    return hashlib.md5(uri.encode()).hexdigest()[:12]


import re

def clean_fulltext(text: str) -> str:
    """Clean up known scraping artifacts from extracted text."""
    # Strip "Advertisement" / "SKIP ADVERTISEMENT" prefix lines
    text = re.sub(
        r"^(?:Advertisement\s*(?:SKIP\s+ADVERTISEMENT\s*(?:Supported\s+by\s*)?(?:SKIP\s+ADVERTISEMENT\s*)?)?)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Strip trailing "Continue reading the main story" boilerplate
    text = re.sub(
        r"\s*Continue reading the main story\.?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def main():
    # Load original dataset
    parquet_path = DATA_DIR / "iran_articles.parquet"
    if not parquet_path.exists():
        print("ERROR: iran_articles.parquet not found. Run filter_iran.py first.")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Original dataset: {len(df):,} articles")

    # Load scrape results metadata
    if not SCRAPE_RESULTS_PATH.exists():
        print("ERROR: fulltext_articles.csv not found. Run scrape_fulltext.py first.")
        return

    scrape_df = pd.read_csv(SCRAPE_RESULTS_PATH)
    print(f"Scrape results: {len(scrape_df):,} entries")
    print(f"  Status breakdown:")
    for status, count in scrape_df["status"].value_counts().items():
        print(f"    {status}: {count:,}")

    # Build fulltext lookup: uri -> fulltext content
    fulltext_map = {}
    status_map = {}
    wordcount_map = {}

    for _, row in scrape_df.iterrows():
        uri = str(row["uri"])
        status = row["status"]
        status_map[uri] = status
        wordcount_map[uri] = row.get("word_count", 0)

        if status == "success":
            # Read the text file
            txt_path = FULLTEXT_DIR / f"{uri_to_hash(uri)}.txt"
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8")
                # Clean up known artifacts
                text = clean_fulltext(text)
                fulltext_map[uri] = text
            else:
                fulltext_map[uri] = ""

    print(f"\nFull texts loaded: {len(fulltext_map):,}")

    # Merge into main dataset
    df["fulltext"] = df["uri"].map(fulltext_map).fillna("")
    df["scrape_status"] = df["uri"].map(status_map).fillna("not_scraped")
    df["fulltext_word_count"] = df["uri"].map(wordcount_map).fillna(0).astype(int)

    # Summary
    print(f"\n{'='*60}")
    print("MERGED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total articles: {len(df):,}")

    has_fulltext = (df["fulltext"].str.len() > 0).sum()
    print(f"With full text: {has_fulltext:,} ({has_fulltext/len(df)*100:.1f}%)")

    print(f"\nScrape status breakdown:")
    for status, count in df["scrape_status"].value_counts().items():
        print(f"  {status}: {count:,}")

    # Coverage by decade
    print(f"\nFull-text coverage by decade:")
    df["decade"] = (df["year"] // 10) * 10
    for decade, group in df.groupby("decade"):
        total = len(group)
        with_text = (group["fulltext"].str.len() > 0).sum()
        print(f"  {int(decade)}s: {with_text:,}/{total:,} ({with_text/total*100:.1f}%)")

    # Average text length comparison
    mask = df["fulltext"].str.len() > 0
    if mask.any():
        avg_fulltext = df.loc[mask, "fulltext"].str.split().str.len().mean()
        avg_abstract = df["abstract"].fillna("").str.split().str.len().mean()
        avg_lead = df["lead_paragraph"].fillna("").str.split().str.len().mean()
        print(f"\nAverage text length (words):")
        print(f"  Abstract only: {avg_abstract:.0f}")
        print(f"  Lead paragraph only: {avg_lead:.0f}")
        print(f"  Full text: {avg_fulltext:.0f}")
        print(f"  Enrichment factor: {avg_fulltext / max(avg_abstract + avg_lead, 1):.1f}x more text")

    # Save enriched dataset
    out_parquet = DATA_DIR / "iran_articles_full.parquet"
    out_csv = DATA_DIR / "iran_articles_full.csv"

    df.drop(columns=["decade"], inplace=True)
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"\nSaved enriched dataset:")
    print(f"  {out_parquet}")
    print(f"  {out_csv}")


if __name__ == "__main__":
    main()
