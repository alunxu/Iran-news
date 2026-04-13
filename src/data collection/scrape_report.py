#!/usr/bin/env python3
"""
Generate a quality report on the scraping results.

Shows coverage statistics by year, decade, section, and identifies
patterns in paywall/failure rates to help decide scraping strategy.

Usage:
    python scrape_report.py
"""

import csv
import sys
from collections import defaultdict

csv.field_size_limit(sys.maxsize)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FULLTEXT_DIR = DATA_DIR / "fulltext"
SCRAPE_RESULTS_PATH = DATA_DIR / "fulltext_articles.csv"


def load_results_and_articles():
    """Load scrape results and original article metadata."""
    # Load scrape results
    results = {}
    if SCRAPE_RESULTS_PATH.exists():
        with open(SCRAPE_RESULTS_PATH) as f:
            for row in csv.DictReader(f):
                results[row["uri"]] = row

    # Load original articles for year/section info
    articles = {}
    csv_path = DATA_DIR / "iran_articles.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                articles[row["uri"]] = row

    return results, articles


def main():
    results, articles = load_results_and_articles()

    if not results:
        print("No scrape results found. Run scrape_fulltext.py first.")
        return

    total_articles = len(articles)
    total_scraped = len(results)

    print(f"{'='*70}")
    print("SCRAPING QUALITY REPORT")
    print(f"{'='*70}")
    print(f"Total articles in dataset: {total_articles:,}")
    print(f"Total articles scraped:    {total_scraped:,}")
    print(f"Not yet scraped:           {total_articles - total_scraped:,}")

    # Overall status breakdown
    status_counts = defaultdict(int)
    for r in results.values():
        status_counts[r["status"]] += 1

    print(f"\n--- Overall Status ---")
    for status in ["success", "paywall", "failed", "http_error", "timeout", "empty_url"]:
        count = status_counts.get(status, 0)
        if count > 0:
            pct = count / total_scraped * 100
            print(f"  {status:15s}: {count:>6,} ({pct:5.1f}%)")

    # Coverage by year
    print(f"\n--- Coverage by Year ---")
    print(f"{'Year':>6s} {'Total':>7s} {'Scraped':>8s} {'Success':>8s} {'Paywall':>8s} {'Failed':>7s} {'Rate':>6s}")
    print("-" * 60)

    year_stats = defaultdict(lambda: defaultdict(int))
    for uri, article in articles.items():
        try:
            year = int(float(article.get("year", 0)))
        except (ValueError, TypeError):
            year = 0
        year_stats[year]["total"] += 1
        if uri in results:
            year_stats[year]["scraped"] += 1
            status = results[uri]["status"]
            year_stats[year][status] += 1

    for year in sorted(year_stats.keys()):
        if year == 0:
            continue
        s = year_stats[year]
        total = s["total"]
        scraped = s["scraped"]
        success = s.get("success", 0)
        paywall = s.get("paywall", 0)
        failed = s.get("failed", 0) + s.get("http_error", 0) + s.get("timeout", 0)
        rate = success / total * 100 if total > 0 else 0
        print(f"{year:>6d} {total:>7,} {scraped:>8,} {success:>8,} {paywall:>8,} {failed:>7,} {rate:>5.1f}%")

    # Coverage by decade
    print(f"\n--- Coverage by Decade ---")
    decade_stats = defaultdict(lambda: defaultdict(int))
    for year, s in year_stats.items():
        if year == 0:
            continue
        decade = (year // 10) * 10
        for key, val in s.items():
            decade_stats[decade][key] += val

    for decade in sorted(decade_stats.keys()):
        s = decade_stats[decade]
        total = s["total"]
        success = s.get("success", 0)
        paywall = s.get("paywall", 0)
        rate = success / total * 100 if total > 0 else 0
        print(f"  {decade}s: {success:,}/{total:,} success ({rate:.1f}%), {paywall:,} paywall")

    # Word count distribution for successful scrapes
    word_counts = []
    for r in results.values():
        if r["status"] == "success":
            try:
                wc = int(r.get("word_count", 0))
                if wc > 0:
                    word_counts.append(wc)
            except (ValueError, TypeError):
                pass

    if word_counts:
        word_counts.sort()
        n = len(word_counts)
        print(f"\n--- Full-Text Word Count Distribution (successful) ---")
        print(f"  Count:   {n:,}")
        print(f"  Min:     {word_counts[0]:,}")
        print(f"  25th %%:  {word_counts[n//4]:,}")
        print(f"  Median:  {word_counts[n//2]:,}")
        print(f"  75th %%:  {word_counts[3*n//4]:,}")
        print(f"  Max:     {word_counts[-1]:,}")
        print(f"  Mean:    {sum(word_counts)/n:,.0f}")

    # Content hash dedup check
    hashes = defaultdict(list)
    for uri, r in results.items():
        if r["status"] == "success" and r.get("content_hash"):
            hashes[r["content_hash"]].append(uri)

    dupes = {h: uris for h, uris in hashes.items() if len(uris) > 1}
    if dupes:
        print(f"\n--- Content Duplicates Detected ---")
        print(f"  {len(dupes)} duplicate content groups ({sum(len(u)-1 for u in dupes.values())} redundant articles)")
    else:
        print(f"\n  No content duplicates detected.")

    # Disk usage
    if FULLTEXT_DIR.exists():
        txt_files = list(FULLTEXT_DIR.glob("*.txt"))
        total_bytes = sum(f.stat().st_size for f in txt_files)
        print(f"\n--- Disk Usage ---")
        print(f"  Text files: {len(txt_files):,}")
        print(f"  Total size: {total_bytes / 1024 / 1024:.1f} MB")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
