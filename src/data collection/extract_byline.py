#!/usr/bin/env python3
"""
Tier 3.8: byline extraction.

The NYT Archive API attaches a `byline` object to every article with the
original string ("By NICHOLAS KRISTOF") and a structured list of persons.
We did not surface this in the initial filter; this script walks the raw
monthly JSONs to extract byline + primary author surname for each URI in
the current corpus and joins them onto the enriched parquet.

Useful for downstream analysis differentiating individual columnists
(e.g., Friedman vs Krugman) or NYT staff vs guest contributors.

Outputs (overwrites):
  data/iran_articles_full.parquet  — adds `byline_original` and `byline_lastname`
  data/iran_articles_full.csv
"""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR  = DATA_DIR / "raw"

def extract_byline(doc) -> tuple[Optional[str], Optional[str]]:
    """Return (original, primary_lastname) for an article."""
    b = doc.get("byline") if isinstance(doc, dict) else None
    if not b or not isinstance(b, dict):
        return None, None
    original = b.get("original")
    persons = b.get("person") or []
    primary_lastname = None
    if persons and isinstance(persons[0], dict):
        primary_lastname = persons[0].get("lastname") or None
    return original, primary_lastname


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add NYT byline fields from raw Archive API JSON files."
    )
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "iran_articles_full.parquet"),
        help="Input parquet path.",
    )
    parser.add_argument(
        "--output-prefix",
        default="iran_articles_full",
        help="Output prefix under data/; writes <prefix>.parquet and <prefix>.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    parquet = Path(args.input)
    if not parquet.exists():
        sys.exit(f"FATAL: {parquet} not found")

    out_parquet = DATA_DIR / f"{args.output_prefix}.parquet"
    out_csv = DATA_DIR / f"{args.output_prefix}.csv"

    print(f"Loading {parquet} ...")
    df = pd.read_parquet(parquet)
    uri_set = set(df["uri"].astype(str))
    print(f"  {len(df):,} articles loaded; {len(uri_set):,} unique URIs")

    json_files = sorted(RAW_DIR.glob("*.json"))
    if not json_files:
        sys.exit(f"FATAL: no raw JSONs in {RAW_DIR}")
    print(f"Scanning {len(json_files)} monthly raw files ...")

    byline_map: dict[str, tuple[Optional[str], Optional[str]]] = {}
    scanned = 0
    found = 0
    for jf in json_files:
        try:
            with jf.open() as f:
                data = json.load(f)
        except Exception:
            continue
        docs = data.get("response", {}).get("docs", []) if isinstance(data, dict) else []
        for doc in docs:
            scanned += 1
            uri = doc.get("uri", "")
            if uri not in uri_set:
                continue
            orig, last = extract_byline(doc)
            if orig or last:
                byline_map[uri] = (orig, last)
                found += 1

    print(f"  scanned {scanned:,} raw docs")
    print(f"  byline found for {found:,} / {len(uri_set):,} corpus articles ({found/len(uri_set)*100:.1f}%)")

    df["byline_original"] = df["uri"].map(lambda u: byline_map.get(u, (None, None))[0])
    df["byline_lastname"] = df["uri"].map(lambda u: byline_map.get(u, (None, None))[1])

    # Quick sanity sample (counts only)
    n_orig = df["byline_original"].notna().sum()
    n_last = df["byline_lastname"].notna().sum()
    print()
    print(f"  byline_original non-null: {n_orig:,}")
    print(f"  byline_lastname non-null: {n_last:,}")
    print()
    top = df["byline_lastname"].value_counts().head(10)
    print("Top 10 by surname (likely NYT staff columnists):")
    for name, n in top.items():
        print(f"  {name:25s} {n:,}")

    print(f"\nWriting enriched dataset ...")
    df.to_parquet(out_parquet, index=False)
    print(f"  {out_parquet}")
    df.to_csv(out_csv, index=False)
    print(f"  {out_csv}")
    print(f"\nFinal column count: {len(df.columns)} (added byline_original, byline_lastname)")


if __name__ == "__main__":
    main()
