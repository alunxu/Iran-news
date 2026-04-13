#!/usr/bin/env python3
"""
Collect all NYT articles via the Archive API.

Downloads one JSON per month from Jan 1979 to Mar 2026.
Supports resume — skips months that are already downloaded.
Rate-limited to 12-second intervals to stay within API limits.
"""

import os
import json
import time
import sys
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("NYT_API_KEY")
if not API_KEY or API_KEY == "your_api_key_here":
    print("ERROR: Set your NYT_API_KEY in the .env file.")
    print("Get one at https://developer.nytimes.com/")
    sys.exit(1)

ARCHIVE_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Date range
START_YEAR, START_MONTH = 1979, 1
END_YEAR, END_MONTH = 2026, 3  # inclusive

# Rate limiting
REQUEST_INTERVAL = 12  # seconds between requests


def generate_months(start_year, start_month, end_year, end_month):
    """Generate (year, month) tuples for the date range."""
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield year, month
        month += 1
        if month > 12:
            month = 1
            year += 1


def download_month(year, month, retries=3):
    """Download archive data for a given month. Returns True on success."""
    url = ARCHIVE_URL.format(year=year, month=month)
    params = {"api-key": API_KEY}

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                return data
            elif resp.status_code == 429:
                # Rate limited — wait and retry
                wait = 60 * (attempt + 1)
                tqdm.write(f"  Rate limited on {year}-{month:02d}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                tqdm.write(
                    f"  HTTP {resp.status_code} for {year}-{month:02d}: {resp.text[:200]}"
                )
                if attempt < retries - 1:
                    time.sleep(30)

        except requests.exceptions.RequestException as e:
            tqdm.write(f"  Request error for {year}-{month:02d}: {e}")
            if attempt < retries - 1:
                time.sleep(30)

    return None


def main():
    months = list(generate_months(START_YEAR, START_MONTH, END_YEAR, END_MONTH))
    print(f"Archive collection: {len(months)} months ({START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d})")

    # Check which months are already downloaded
    existing = set()
    for f in RAW_DIR.glob("*.json"):
        existing.add(f.stem)  # e.g., "2024-01"

    remaining = [(y, m) for y, m in months if f"{y}-{m:02d}" not in existing]
    print(f"Already downloaded: {len(existing)}, remaining: {len(remaining)}")

    if not remaining:
        print("All months already downloaded!")
        return

    success_count = 0
    fail_count = 0

    for year, month in tqdm(remaining, desc="Downloading archives"):
        filename = RAW_DIR / f"{year}-{month:02d}.json"

        data = download_month(year, month)
        if data is not None:
            with open(filename, "w") as f:
                json.dump(data, f)

            n_docs = len(data.get("response", {}).get("docs", []))
            tqdm.write(f"  ✓ {year}-{month:02d}: {n_docs} articles")
            success_count += 1
        else:
            tqdm.write(f"  ✗ {year}-{month:02d}: FAILED after retries")
            fail_count += 1

        # Rate limit
        time.sleep(REQUEST_INTERVAL)

    print(f"\nDone! Downloaded: {success_count}, Failed: {fail_count}")
    print(f"Total raw files: {len(list(RAW_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
