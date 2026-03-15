#!/usr/bin/env python3
"""
Filter raw archive data for Iran-related articles.

Reads all monthly JSON files from data/raw/, filters for articles
mentioning Iran (via keywords, headline, abstract, lead_paragraph),
extracts relevant fields, and saves to Parquet + CSV.
"""

import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Iran-related search terms (case-insensitive)
IRAN_TERMS = [
    r"\biran\b",
    r"\biranian\b",
    r"\biranians\b",
    r"\btehran\b",
    r"\bpersia\b",
    r"\bpersian\b",
    r"\bayatollah\b",
    r"\bkhamenei\b",
    r"\bkhomeini\b",
    r"\brouhani\b",
    r"\braisi\b",
    r"\bpezeshkian\b",
    r"\birgc\b",
    r"\bislamicrepublic\b",
    r"\bislamic\s+republic\b",
]

IRAN_PATTERN = re.compile("|".join(IRAN_TERMS), re.IGNORECASE)


def check_keywords(keywords_list):
    """Check if any keyword value matches Iran terms."""
    if not keywords_list:
        return False
    for kw in keywords_list:
        value = kw.get("value", "")
        if IRAN_PATTERN.search(value):
            return True
    return False


def check_text(text):
    """Check if text contains Iran terms."""
    if not text:
        return False
    return bool(IRAN_PATTERN.search(text))


def extract_article(doc):
    """Extract relevant fields from a raw article document."""
    headline = doc.get("headline", {})
    headline_main = headline.get("main", "") if isinstance(headline, dict) else ""

    # Get keywords as structured data
    keywords = doc.get("keywords", [])
    keyword_values = [
        {"name": kw.get("name", ""), "value": kw.get("value", "")}
        for kw in (keywords or [])
    ]

    # Get multimedia metadata
    multimedia = doc.get("multimedia", [])
    media_entries = []
    for m in (multimedia or []):
        media_entries.append({
            "url": m.get("url", ""),
            "caption": m.get("caption", ""),
            "credit": m.get("credit", ""),
            "type": m.get("type", ""),
            "subtype": m.get("subtype", ""),
            "height": m.get("height", 0),
            "width": m.get("width", 0),
        })

    return {
        "headline": headline_main,
        "abstract": doc.get("abstract", ""),
        "lead_paragraph": doc.get("lead_paragraph", ""),
        "snippet": doc.get("snippet", ""),
        "pub_date": doc.get("pub_date", ""),
        "section_name": doc.get("section_name", ""),
        "news_desk": doc.get("news_desk", ""),
        "document_type": doc.get("document_type", ""),
        "type_of_material": doc.get("type_of_material", ""),
        "web_url": doc.get("web_url", ""),
        "uri": doc.get("uri", ""),
        "word_count": doc.get("word_count", 0),
        "source": doc.get("source", ""),
        "keywords_json": json.dumps(keyword_values),
        "multimedia_json": json.dumps(media_entries),
        "has_image": len(media_entries) > 0,
        "n_images": len(media_entries),
    }


def is_iran_related(doc):
    """Check if an article is Iran-related."""
    headline = doc.get("headline", {})
    headline_main = headline.get("main", "") if isinstance(headline, dict) else ""

    # Check structured keywords first (most reliable)
    if check_keywords(doc.get("keywords", [])):
        return True

    # Check headline
    if check_text(headline_main):
        return True

    # Check abstract
    if check_text(doc.get("abstract", "")):
        return True

    # Check lead paragraph
    if check_text(doc.get("lead_paragraph", "")):
        return True

    # Check snippet
    if check_text(doc.get("snippet", "")):
        return True

    return False


def main():
    json_files = sorted(RAW_DIR.glob("*.json"))
    print(f"Found {len(json_files)} monthly archive files in {RAW_DIR}/")

    if not json_files:
        print("No raw data found. Run collect_archive.py first.")
        return

    all_articles = []
    total_docs = 0

    for json_file in tqdm(json_files, desc="Filtering archives"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            tqdm.write(f"  Error reading {json_file.name}: {e}")
            continue

        docs = data.get("response", {}).get("docs", [])
        total_docs += len(docs)

        for doc in docs:
            if is_iran_related(doc):
                article = extract_article(doc)
                all_articles.append(article)

    print(f"\nTotal articles scanned: {total_docs:,}")
    print(f"Iran-related articles: {len(all_articles):,}")
    print(f"Hit rate: {len(all_articles)/max(total_docs,1)*100:.2f}%")

    if not all_articles:
        print("No Iran-related articles found.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_articles)

    # Parse dates
    df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce", utc=True)
    df["year"] = df["pub_date"].dt.year
    df["month"] = df["pub_date"].dt.month
    df["year_month"] = df["pub_date"].dt.to_period("M")

    # Remove duplicates by URI
    n_before = len(df)
    df = df.drop_duplicates(subset="uri", keep="first")
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"Removed {n_dupes} duplicates")

    # Sort by date
    df = df.sort_values("pub_date").reset_index(drop=True)

    # Save
    parquet_path = OUTPUT_DIR / "iran_articles.parquet"
    csv_path = OUTPUT_DIR / "iran_articles.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df):,} articles to:")
    print(f"  {parquet_path}")
    print(f"  {csv_path}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("CORPUS SUMMARY")
    print(f"{'='*60}")
    print(f"Date range: {df['pub_date'].min()} to {df['pub_date'].max()}")
    print(f"Total articles: {len(df):,}")
    print(f"With images: {df['has_image'].sum():,} ({df['has_image'].mean()*100:.1f}%)")
    print(f"\nArticles by decade:")
    for decade_start in range(1970, 2030, 10):
        decade_end = decade_start + 10
        mask = (df["year"] >= decade_start) & (df["year"] < decade_end)
        count = mask.sum()
        if count > 0:
            print(f"  {decade_start}s: {count:,}")

    print(f"\nTop 10 sections:")
    print(df["section_name"].value_counts().head(10).to_string())

    print(f"\nTop 10 material types:")
    print(df["type_of_material"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
