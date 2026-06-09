#!/usr/bin/env python3
"""
Recovery pass for previously-failed Wayback scrapes.

Strategy:
  The original scraper used the Wayback "latest" redirect (web/{url}) which
  often lands on a paywall snapshot for post-2012 NYT articles. Many such
  articles have *earlier* snapshots that pre-date paywall enforcement and
  are fully extractable.

  This script queries the Wayback CDX API for every snapshot of each failed
  URL, sorts chronologically, and tries snapshots earliest-first until
  trafilatura extracts a non-trivial article body.

Conventions match scrape_fulltext.py:
  - Text files saved to data/fulltext/{md5(uri)[:12]}.txt
  - Recovery log: data/recovery_articles.csv (separate from original log)
  - Checkpoint: data/recovery_checkpoint.json
  - Output is merge-compatible with merge_fulltext.py

Usage:
    python recover_failed.py                      # default settings
    python recover_failed.py --concurrency 2      # more polite
    python recover_failed.py --decade 2020        # recover only 2020s
    python recover_failed.py --limit 100          # test run
    python recover_failed.py --min-words 200      # stricter quality threshold
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
import trafilatura

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FULLTEXT_DIR = DATA_DIR / "fulltext"
PARQUET_PATH = DATA_DIR / "iran_articles_full.parquet"
CHECKPOINT_PATH = DATA_DIR / "recovery_checkpoint.json"
OUTPUT_PATH = DATA_DIR / "recovery_articles.csv"

CDX_URL = "https://web.archive.org/cdx/search/cdx"
WAYBACK_FETCH_TEMPLATE = "https://web.archive.org/web/{ts}id_/{url}"

# Trustworthy User-Agent — Wayback throttles default Python UAs harder
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

DEFAULT_CONCURRENCY = 2
DEFAULT_DELAY = 2.0
DEFAULT_TIMEOUT = 25
DEFAULT_MIN_WORDS = 150
MAX_SNAPSHOTS_TO_TRY = 5  # earliest-N + middle + latest


# ──────────────────────────────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, path: Path):
        self.path = path
        self.completed: dict[str, str] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.completed = json.loads(self.path.read_text()).get("completed", {})
            except Exception:
                self.completed = {}

    def save(self):
        self.path.write_text(json.dumps({"completed": self.completed}, indent=0))

    def is_done(self, uri: str) -> bool:
        return uri in self.completed

    def mark(self, uri: str, status: str):
        self.completed[uri] = status

    def stats(self) -> dict:
        from collections import Counter
        return dict(Counter(self.completed.values()))


# ──────────────────────────────────────────────────────────────────────
# Wayback CDX + extraction
# ──────────────────────────────────────────────────────────────────────
async def cdx_snapshots(
    session: aiohttp.ClientSession, url: str, timeout: int
) -> tuple[list[str], str]:
    """Return (timestamps, status) for the URL.

    Status is one of:
      "ok"         - query succeeded (timestamps may be empty if truly absent)
      "cdx_error"  - CDX server failed (timeouts, 503, etc.); should retry later
    """
    params = {
        "url": url,
        "output": "json",
        "limit": "100",
        "filter": "statuscode:200",
        "collapse": "timestamp:8",  # collapse same-day captures
    }
    # Retry up to 3 times with exponential backoff for transient CDX failures
    for attempt in range(3):
        try:
            async with session.get(
                CDX_URL, params=params, headers=HEADERS, timeout=timeout
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if not data or len(data) <= 1:
                        return [], "ok"
                    return [row[1] for row in data[1:] if len(row) > 1], "ok"
                # Transient CDX failure: 429, 5xx
                if resp.status in (429, 502, 503, 504):
                    await asyncio.sleep(2 ** attempt * 3)
                    continue
                return [], "ok"  # other 4xx — assume URL truly absent
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(2 ** attempt * 3)
            continue
        except Exception:
            return [], "cdx_error"
    return [], "cdx_error"


def select_snapshots(timestamps: list[str], n: int = MAX_SNAPSHOTS_TO_TRY) -> list[str]:
    """Pick a chronologically diverse subset: earliest, middle, latest."""
    if len(timestamps) <= n:
        return timestamps
    # Earliest 3 (most likely pre-paywall) + middle + latest
    earliest = timestamps[:3]
    middle = [timestamps[len(timestamps) // 2]]
    latest = [timestamps[-1]]
    return earliest + middle + latest


async def fetch_snapshot(
    session: aiohttp.ClientSession, ts: str, url: str, timeout: int
) -> Optional[str]:
    wb_url = WAYBACK_FETCH_TEMPLATE.format(ts=ts, url=url)
    try:
        async with session.get(wb_url, headers=HEADERS, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return await resp.text(errors="replace")
    except Exception:
        return None


def extract_text(html: str) -> Optional[str]:
    """Try precision then recall extraction, return text if substantial."""
    if not html:
        return None
    for kwargs in (
        {"include_comments": False, "include_tables": False, "favor_precision": True},
        {"include_comments": False, "include_tables": False, "favor_recall": True},
    ):
        try:
            text = trafilatura.extract(html, **kwargs)
            if text and len(text.split()) >= 50:
                return text
        except Exception:
            continue
    return None


def is_paywall_or_short(text: str) -> bool:
    """Heuristic: detect TimesMachine paywall preview pages."""
    if not text:
        return True
    lower = text.lower()
    if "timesmachine" in lower or "view full article" in lower:
        return True
    return len(text.split()) < 50


# ──────────────────────────────────────────────────────────────────────
# Per-article recovery
# ──────────────────────────────────────────────────────────────────────
async def recover_one(
    session: aiohttp.ClientSession,
    uri: str,
    url: str,
    timeout: int,
    min_words: int,
    delay: float,
) -> dict:
    """Try CDX deep-query recovery on a single article."""
    timestamps, cdx_status = await cdx_snapshots(session, url, timeout)
    await asyncio.sleep(delay)

    if cdx_status == "cdx_error":
        # Distinguish from genuine absence — leave out of checkpoint so
        # we can retry on the next run
        return {"uri": uri, "url": url, "status": "cdx_error",
                "snapshots_tried": 0, "ts_used": "", "word_count": 0}

    if not timestamps:
        return {"uri": uri, "url": url, "status": "no_cdx_snapshots",
                "snapshots_tried": 0, "ts_used": "", "word_count": 0}

    candidates = select_snapshots(timestamps)
    best_text = None
    best_ts = None
    best_wc = 0

    for ts in candidates:
        html = await fetch_snapshot(session, ts, url, timeout)
        text = extract_text(html) if html else None
        if text and not is_paywall_or_short(text):
            wc = len(text.split())
            if wc > best_wc:
                best_text = text
                best_ts = ts
                best_wc = wc
                if wc >= min_words:
                    break  # good enough — stop early
        await asyncio.sleep(delay)

    if best_text and best_wc >= min_words:
        # Save text, hashed by URI to match existing convention
        text_path = FULLTEXT_DIR / f"{hashlib.md5(uri.encode()).hexdigest()[:12]}.txt"
        text_path.write_text(best_text, encoding="utf-8")
        return {
            "uri": uri, "url": url, "status": "recovered",
            "snapshots_tried": len(candidates), "ts_used": best_ts,
            "word_count": best_wc,
        }
    elif best_text:
        return {
            "uri": uri, "url": url, "status": "too_short",
            "snapshots_tried": len(candidates), "ts_used": best_ts or "",
            "word_count": best_wc,
        }
    else:
        return {
            "uri": uri, "url": url, "status": "all_failed",
            "snapshots_tried": len(candidates), "ts_used": "",
            "word_count": 0,
        }


# ──────────────────────────────────────────────────────────────────────
# Worker pool
# ──────────────────────────────────────────────────────────────────────
async def worker(
    name: int,
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    checkpoint: CheckpointManager,
    csv_writer,
    csv_file,
    args,
    progress: dict,
):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        uri, url = item
        try:
            result = await recover_one(
                session, uri, url, args.timeout, args.min_words, args.delay
            )
            # cdx_error means CDX itself failed — don't checkpoint, retry later
            if result["status"] != "cdx_error":
                checkpoint.mark(uri, result["status"])
            csv_writer.writerow([
                result["uri"], result["url"], result["status"],
                result["snapshots_tried"], result["ts_used"], result["word_count"],
            ])
            csv_file.flush()
            progress["done"] += 1
            if result["status"] == "recovered":
                progress["recovered"] += 1
            if progress["done"] % 25 == 0:
                stats = checkpoint.stats()
                print(
                    f"  [{progress['done']}/{progress['total']}] "
                    f"recovered={progress['recovered']}  "
                    f"all_status={stats}",
                    flush=True,
                )
            if progress["done"] % 100 == 0:
                checkpoint.save()
        except Exception as e:
            print(f"  worker {name}: {uri[:40]} → unexpected: {e}", flush=True)
        finally:
            queue.task_done()


# ──────────────────────────────────────────────────────────────────────
# Article loading
# ──────────────────────────────────────────────────────────────────────
def load_targets(args) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    df["decade"] = (df["year"] // 10) * 10

    # Recoverable: anything currently not 'success'
    targets = df[df["scrape_status"] != "success"].copy()

    # Skip 1970s by default — those are TimesMachine paywall pages,
    # Wayback cannot serve full text for them
    if not args.include_pre2000:
        targets = targets[targets["decade"] >= 2000]

    if args.decade is not None:
        targets = targets[targets["decade"] == args.decade]

    # Optional: focus on the most recoverable subset
    if args.failed_only:
        targets = targets[targets["scrape_status"] == "failed"]

    return targets[["uri", "web_url", "decade", "year"]].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
async def run(args):
    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    targets = load_targets(args)
    print(f"Recoverable targets: {len(targets)}")
    print(f"  by decade:")
    for dec, n in targets["decade"].value_counts().sort_index().items():
        print(f"    {int(dec)}s: {n}")
    if args.limit:
        targets = targets.head(args.limit)
        print(f"Limited to first {args.limit} for testing")

    checkpoint = CheckpointManager(CHECKPOINT_PATH)
    print(f"Resuming with {len(checkpoint.completed)} already processed")
    todo = targets[~targets["uri"].isin(checkpoint.completed.keys())]
    print(f"Remaining to process: {len(todo)}")
    if len(todo) == 0:
        print("Nothing to do. Done.")
        return

    # CSV log (append mode)
    is_new_file = not OUTPUT_PATH.exists()
    csv_file = OUTPUT_PATH.open("a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if is_new_file:
        csv_writer.writerow([
            "uri", "url", "status", "snapshots_tried", "ts_used", "word_count"
        ])

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    progress = {"done": 0, "total": len(todo), "recovered": 0}

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [
            asyncio.create_task(
                worker(i, queue, session, checkpoint, csv_writer, csv_file, args, progress)
            )
            for i in range(args.concurrency)
        ]

        for _, row in todo.iterrows():
            await queue.put((row["uri"], row["web_url"]))

        for _ in range(args.concurrency):
            await queue.put(None)

        await queue.join()
        for w in workers:
            await w

    checkpoint.save()
    csv_file.close()

    print()
    print("=" * 60)
    print("Recovery complete")
    print(f"  Processed: {progress['done']}")
    print(f"  Recovered: {progress['recovered']}")
    print(f"  Status breakdown: {checkpoint.stats()}")
    print(f"  Text saved to: {FULLTEXT_DIR}/")
    print(f"  Log:           {OUTPUT_PATH}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS,
                   help="Minimum extracted word count to count as recovered")
    p.add_argument("--decade", type=int, default=None,
                   help="Restrict to one decade (e.g., 2020)")
    p.add_argument("--include-pre2000", action="store_true",
                   help="Also try pre-2000 articles (mostly TimesMachine — usually fruitless)")
    p.add_argument("--failed-only", action="store_true",
                   help="Only retry status='failed' (skip not_archived/too_short/extraction_failed)")
    p.add_argument("--limit", type=int, default=None, help="Test on first N targets")
    p.add_argument("--force", action="store_true",
                   help="Clear checkpoint and start over")
    args = p.parse_args()

    if args.force and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Checkpoint cleared")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
