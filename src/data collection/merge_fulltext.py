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
import argparse
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
RECOVERY_RESULTS_PATH = DATA_DIR / "recovery_articles.csv"
SUBSCRIPTION_RESULTS_PATH = DATA_DIR / "nyt_subscription_articles.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Merge scraped full text into an Iran article metadata dataset.")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "iran_articles.parquet",
                        help="Input metadata parquet.")
    parser.add_argument("--output-prefix", default="iran_articles_full",
                        help="Output basename without extension.")
    return parser.parse_args()


def uri_to_hash(uri: str) -> str:
    """Same hash function as the scraper uses for filenames."""
    return hashlib.md5(uri.encode()).hexdigest()[:12]


import re

def clean_fulltext(text: str) -> str:
    """Clean known scraping artifacts (ads, navigation, archive boilerplate)
    from extracted article text. Patterns target trafilatura's known leaks
    in NYT-rendered pages.
    """
    # 1. Strip standalone "Advertisement" / "SKIP ADVERTISEMENT" / "Supported by"
    #    lines wherever they appear (prefix, between paragraphs, suffix)
    text = re.sub(
        r"(?im)^[ \t]*(?:advertisement|skip advertisement|supported by)[ \t]*$\n?",
        "",
        text,
    )
    # 2. Strip "Continue reading the main story" navigation prompts (anywhere)
    text = re.sub(
        r"(?i)\s*continue reading the main story\.?\s*",
        " ",
        text,
    )
    # 3. Strip TimesMachine "View Full Article" stubs that slipped through
    text = re.sub(
        r"(?i)\s*view full article in timesmachine\.?\s*",
        " ",
        text,
    )
    # 4. Strip "From print:" / "Archives" provenance markers
    text = re.sub(
        r"(?im)^[ \t]*(?:from print|the new york times archives)[ \t]*[:.]?[ \t]*$\n?",
        "",
        text,
    )
    # 5. Collapse 3+ consecutive newlines to 2 (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 6. Trim leading/trailing whitespace
    return text.strip()


def main():
    args = parse_args()

    # Load original dataset
    parquet_path = args.input
    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found. Run filter_iran.py first.")
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

    # Apply recovery results (overrides original status for recovered articles)
    if RECOVERY_RESULTS_PATH.exists():
        recovery_df = pd.read_csv(RECOVERY_RESULTS_PATH).drop_duplicates("uri", keep="last")
        n_recovered = (recovery_df["status"] == "recovered").sum()
        print(f"\nRecovery results: {len(recovery_df):,} entries ({n_recovered:,} recovered)")

        for _, row in recovery_df.iterrows():
            uri = str(row["uri"])
            if row["status"] == "recovered":
                txt_path = FULLTEXT_DIR / f"{uri_to_hash(uri)}.txt"
                if txt_path.exists():
                    text = clean_fulltext(txt_path.read_text(encoding="utf-8"))
                    fulltext_map[uri] = text
                    status_map[uri] = "recovered"
                    wordcount_map[uri] = int(row.get("word_count", 0))

    # Apply NYT subscription scraper results (highest priority — direct full text)
    if SUBSCRIPTION_RESULTS_PATH.exists():
        sub_df = pd.read_csv(SUBSCRIPTION_RESULTS_PATH).drop_duplicates("uri", keep="last")
        n_sub = (sub_df["status"] == "success").sum()
        print(f"\nSubscription results: {len(sub_df):,} entries ({n_sub:,} recovered)")

        for _, row in sub_df.iterrows():
            uri = str(row["uri"])
            if row["status"] == "success":
                txt_path = FULLTEXT_DIR / f"{uri_to_hash(uri)}.txt"
                if txt_path.exists():
                    text = clean_fulltext(txt_path.read_text(encoding="utf-8"))
                    fulltext_map[uri] = text
                    status_map[uri] = "subscription"
                    wordcount_map[uri] = int(row.get("word_count", 0))

    print(f"\nFull texts loaded: {len(fulltext_map):,}")

    # Deduplicate by content hash (first 1000 chars). Keep the longest text;
    # blank out the others (preserves their metadata row but prevents counting
    # the same article twice in body-level analyses).
    from collections import defaultdict
    hash_to_uris: dict[str, list[str]] = defaultdict(list)
    for uri, text in fulltext_map.items():
        if text:
            h = hashlib.sha1(text[:1000].encode("utf-8")).hexdigest()
            hash_to_uris[h].append(uri)

    dropped_dupes = 0
    for h, uris in hash_to_uris.items():
        if len(uris) > 1:
            # Keep the URI with the longest text
            keep = max(uris, key=lambda u: len(fulltext_map[u]))
            for u in uris:
                if u != keep:
                    fulltext_map[u] = ""
                    status_map[u] = "duplicate"
                    wordcount_map[u] = 0
                    dropped_dupes += 1
    if dropped_dupes:
        print(f"Deduplicated: {dropped_dupes:,} duplicate articles blanked (kept 1 per content-hash group)")

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
    out_parquet = DATA_DIR / f"{args.output_prefix}.parquet"
    out_csv = DATA_DIR / f"{args.output_prefix}.csv"

    df.drop(columns=["decade"], inplace=True)
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"\nSaved enriched dataset:")
    print(f"  {out_parquet}")
    print(f"  {out_csv}")


if __name__ == "__main__":
    main()
