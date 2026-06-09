#!/usr/bin/env python3
"""
Download images for Iran-related articles.

Reads the filtered Iran articles dataset and downloads associated images.
Supports resume — skips images already downloaded.
"""

import json
import os
import time
import hashlib
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

NYT_IMAGE_BASE = "https://static01.nyt.com/"
REQUEST_INTERVAL = 0.3  # seconds between image downloads


def article_id_from_uri(uri):
    """Create a safe filename from an article URI."""
    # URI looks like "nyt://article/xxxx-xxxx-xxxx"
    if uri:
        return hashlib.md5(uri.encode()).hexdigest()[:12]
    return None


def download_image(url, filepath, timeout=30):
    """Download an image from a URL."""
    try:
        # NYT image URLs are relative; prepend base if needed
        if url.startswith("images/"):
            full_url = NYT_IMAGE_BASE + url
        elif url.startswith("http"):
            full_url = url
        else:
            full_url = NYT_IMAGE_BASE + url

        resp = requests.get(full_url, timeout=timeout, stream=True)
        if resp.status_code == 200:
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            return False
    except Exception:
        return False


def main():
    parquet_path = Path("data/iran_articles.parquet")
    if not parquet_path.exists():
        print("No iran_articles.parquet found. Run filter_iran.py first.")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} Iran-related articles")

    # Parse multimedia JSON
    image_records = []
    for idx, row in df.iterrows():
        try:
            multimedia = json.loads(row["multimedia_json"])
        except (json.JSONDecodeError, TypeError):
            continue

        if not multimedia:
            continue

        article_id = article_id_from_uri(row.get("uri", ""))
        if not article_id:
            continue

        # Take the first image (default crop preferred)
        for i, media in enumerate(multimedia):
            url = media.get("url", "")
            if not url:
                continue

            image_records.append({
                "article_id": article_id,
                "uri": row.get("uri", ""),
                "pub_date": row.get("pub_date", ""),
                "headline": row.get("headline", ""),
                "image_url": url,
                "caption": media.get("caption", ""),
                "credit": media.get("credit", ""),
                "image_idx": i,
            })

    print(f"Found {len(image_records):,} images to download")

    if not image_records:
        print("No images found.")
        return

    # Check which are already downloaded
    existing = set(f.stem for f in IMAGE_DIR.glob("*.*"))

    success = 0
    skipped = 0
    failed = 0

    # Create metadata CSV incrementally
    meta_path = Path("data/image_metadata.csv")
    meta_records = []

    for rec in tqdm(image_records, desc="Downloading images"):
        filename = f"{rec['article_id']}_{rec['image_idx']}"

        if filename in existing:
            skipped += 1
            meta_records.append({**rec, "local_file": f"{filename}.jpg", "status": "exists"})
            continue

        filepath = IMAGE_DIR / f"{filename}.jpg"
        ok = download_image(rec["image_url"], filepath)

        if ok:
            success += 1
            meta_records.append({**rec, "local_file": f"{filename}.jpg", "status": "ok"})
        else:
            failed += 1
            meta_records.append({**rec, "local_file": "", "status": "failed"})

        time.sleep(REQUEST_INTERVAL)

    # Save metadata
    meta_df = pd.DataFrame(meta_records)
    meta_df.to_csv(meta_path, index=False)

    print(f"\nDone! Downloaded: {success}, Skipped: {skipped}, Failed: {failed}")
    print(f"Image metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
