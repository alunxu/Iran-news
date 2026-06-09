#!/usr/bin/env python3
"""
Direct NYT scraper using a logged-in subscription session (cookies).

Why this exists:
  Wayback Machine cannot recover certain article subsets — most notably
  pre-1981 /archives/ URLs (which were never indexed by Wayback) and many
  post-2020 paywalled articles. With a valid NYT subscription session, we
  can request these directly: nytimes.com auto-redirects /archives/ URLs
  to TimesMachine OCR'd full text, and serves recent articles past the
  paywall.

Inputs:
  - data/nyt_cookies.txt      Netscape-format cookie jar exported from a
                              browser logged into nytimes.com
  - data/iran_articles_full.parquet
                              Existing dataset (with scrape_status column)

Outputs (same layout as scrape_fulltext.py / recover_failed.py):
  - data/fulltext/{md5(uri)[:12]}.txt     Recovered full text
  - data/nyt_subscription_articles.csv    Per-article log
  - data/nyt_subscription_checkpoint.json Resume checkpoint

Usage:
  # Test
  python scrape_nyt_subscription.py --limit 30 --decade 2020

  # Full run for 1979-80 archives
  python scrape_nyt_subscription.py --pre1981 --concurrency 2 --delay 4

  # Full run for 2020s paywalled
  python scrape_nyt_subscription.py --decade 2020

⚠️  Be polite. NYT ToS technically prohibits automated access; treat this as
    a research tool used responsibly:
      - Slow pacing (default 4s + jitter)
      - Low concurrency (default 2)
      - Stop if paywall markers appear (session expired)
      - Don't run unattended for hours
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import http.cookiejar
import json
import random
import sys
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
import trafilatura

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FULLTEXT_DIR = DATA_DIR / "fulltext"
COOKIES_PATH = DATA_DIR / "nyt_cookies.txt"
PARQUET_PATH = DATA_DIR / "iran_articles_full.parquet"
CHECKPOINT_PATH = DATA_DIR / "nyt_subscription_checkpoint.json"
OUTPUT_PATH = DATA_DIR / "nyt_subscription_articles.csv"

DEFAULT_CONCURRENCY = 1
DEFAULT_DELAY = 12.0
DEFAULT_TIMEOUT = 30
DEFAULT_MIN_WORDS = 150
PAYWALL_ABORT_THRESHOLD = 5   # consecutive paywall pages → cookies expired
HTTP_403_ABORT_THRESHOLD = 3  # consecutive 403s → DataDome block, stop immediately

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# Strings strongly suggesting we hit a paywall page rather than full content
PAYWALL_MARKERS = [
    "subscribe to keep reading",
    "subscribe to continue",
    "you've reached your limit",
    "subscribe for unlimited",
    "create your free account to continue",
    "thanks for your interest",
]


# ────────────────────────────────────────────────────────────────────
# Cookies → aiohttp
# ────────────────────────────────────────────────────────────────────
def load_cookies(path: Path) -> dict[str, str]:
    """Load Netscape-format cookies into a plain {name: value} dict.

    aiohttp's ClientSession accepts cookies as a dict directly via the
    cookies= kwarg, which is simpler than building a CookieJar.
    """
    if not path.exists():
        sys.exit(f"FATAL: cookie file missing: {path}")
    mc = http.cookiejar.MozillaCookieJar()
    mc.load(str(path), ignore_discard=True, ignore_expires=True)
    return {c.name: c.value for c in mc}


# ────────────────────────────────────────────────────────────────────
# Checkpoint
# ────────────────────────────────────────────────────────────────────
class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.completed: dict[str, str] = {}
        if path.exists():
            try:
                self.completed = json.loads(path.read_text()).get("completed", {})
            except Exception:
                self.completed = {}

    def is_done(self, uri: str) -> bool:
        return uri in self.completed

    def mark(self, uri: str, status: str):
        self.completed[uri] = status

    def save(self):
        self.path.write_text(json.dumps({"completed": self.completed}, indent=0))

    def stats(self) -> dict:
        from collections import Counter
        return dict(Counter(self.completed.values()))


# ────────────────────────────────────────────────────────────────────
# Extraction
# ────────────────────────────────────────────────────────────────────
def extract_text(html: str) -> Optional[str]:
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


def looks_paywalled(html: str, text: Optional[str]) -> bool:
    """Best-effort paywall detection."""
    if not html:
        return False
    lower = html.lower()
    for m in PAYWALL_MARKERS:
        if m in lower:
            return True
    # If extraction yielded almost nothing, suspect paywall
    if text is not None and len(text.split()) < 30:
        return True
    return False


# ────────────────────────────────────────────────────────────────────
# Per-article fetch
# ────────────────────────────────────────────────────────────────────
async def fetch_one(
    session: aiohttp.ClientSession, url: str, timeout: int
) -> tuple[Optional[str], int, str]:
    """Returns (html, http_status, error_msg)."""
    try:
        async with session.get(url, timeout=timeout) as resp:
            html = await resp.text(errors="replace")
            return html, resp.status, ""
    except aiohttp.ClientError as e:
        return None, -1, f"{type(e).__name__}: {str(e)[:60]}"
    except asyncio.TimeoutError:
        return None, -1, "timeout"
    except Exception as e:
        return None, -2, f"{type(e).__name__}: {str(e)[:60]}"


# ────────────────────────────────────────────────────────────────────
# Worker
# ────────────────────────────────────────────────────────────────────
async def worker(
    name: int,
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    checkpoint: Checkpoint,
    csv_writer,
    csv_file,
    args,
    progress: dict,
    abort_flag: asyncio.Event,
):
    while True:
        if abort_flag.is_set():
            queue.task_done()
            break
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        uri, url = item

        try:
            html, http_status, err = await fetch_one(session, url, args.timeout)
            text = extract_text(html) if html else None
            wc = len(text.split()) if text else 0
            paywalled = looks_paywalled(html or "", text) if html else False

            if text and wc >= args.min_words and not paywalled:
                status = "success"
                txt_path = FULLTEXT_DIR / f"{hashlib.md5(uri.encode()).hexdigest()[:12]}.txt"
                txt_path.write_text(text, encoding="utf-8")
                progress["recovered"] += 1
                progress["paywall_streak"] = 0
                progress["http_403_streak"] = 0
            elif paywalled:
                status = "paywall"
                progress["paywall_streak"] += 1
                progress["http_403_streak"] = 0
            elif http_status == 200 and wc < args.min_words:
                status = "too_short"
                progress["paywall_streak"] = 0
                progress["http_403_streak"] = 0
            elif http_status == 403:
                status = "error_403"
                progress["http_403_streak"] += 1
                progress["paywall_streak"] = 0
            else:
                status = f"error_{http_status}"
                progress["paywall_streak"] = 0
                progress["http_403_streak"] = 0

            checkpoint.mark(uri, status)
            csv_writer.writerow([uri, url, status, http_status, wc, err[:80]])
            csv_file.flush()
            progress["done"] += 1

            if progress["done"] % 25 == 0:
                stats = checkpoint.stats()
                print(
                    f"  [{progress['done']}/{progress['total']}] "
                    f"recovered={progress['recovered']} stats={stats}",
                    flush=True,
                )
            if progress["done"] % 100 == 0:
                checkpoint.save()

            # Cookies expired? Stop before racking up failures.
            if progress["paywall_streak"] >= PAYWALL_ABORT_THRESHOLD:
                print(
                    f"\n⚠️  {PAYWALL_ABORT_THRESHOLD} consecutive paywall responses — "
                    "session likely expired. Aborting. Re-export cookies and resume.",
                    flush=True,
                )
                abort_flag.set()
                queue.task_done()
                break

            # DataDome bot-block? Stop immediately to avoid worsening the block.
            if progress["http_403_streak"] >= HTTP_403_ABORT_THRESHOLD:
                print(
                    f"\n⚠️  {HTTP_403_ABORT_THRESHOLD} consecutive 403s — "
                    "DataDome WAF has flagged us. Aborting to let block expire.\n"
                    "  Wait 1-2 hours, re-export cookies, and resume.",
                    flush=True,
                )
                abort_flag.set()
                queue.task_done()
                break

            # Polite pacing with jitter
            await asyncio.sleep(args.delay + random.uniform(0, args.delay * 0.5))

        except Exception as e:
            print(f"  worker {name}: unexpected {type(e).__name__}: {e}", flush=True)
        finally:
            queue.task_done()


# ────────────────────────────────────────────────────────────────────
# Targets
# ────────────────────────────────────────────────────────────────────
def load_targets(args) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    df["decade"] = (df["year"] // 10) * 10

    # Anything not currently a success is a candidate
    targets = df[df["scrape_status"] != "success"].copy()
    # Also exclude already-recovered (via Wayback CDX recovery pass)
    targets = targets[targets["scrape_status"] != "recovered"]

    if args.pre1981:
        targets = targets[targets["year"] < 1981]
    elif args.decade is not None:
        targets = targets[targets["decade"] == args.decade]

    return targets[["uri", "web_url", "year", "decade"]].reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
async def run(args):
    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    targets = load_targets(args)
    print(f"Targets: {len(targets)}")
    print("  by decade:")
    for dec, n in targets["decade"].value_counts().sort_index().items():
        print(f"    {int(dec)}s: {n}")
    if args.limit:
        targets = targets.head(args.limit)
        print(f"  Limited to first {args.limit} for testing")

    checkpoint = Checkpoint(CHECKPOINT_PATH)
    print(f"Resuming with {len(checkpoint.completed)} already done")
    todo = targets[~targets["uri"].isin(checkpoint.completed.keys())]
    print(f"Remaining: {len(todo)}")
    if len(todo) == 0:
        print("Nothing to do.")
        return

    # CSV log
    is_new = not OUTPUT_PATH.exists()
    csv_file = OUTPUT_PATH.open("a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if is_new:
        csv_writer.writerow(["uri", "url", "status", "http_status", "word_count", "error"])

    # aiohttp session with subscription cookies
    cookies = load_cookies(COOKIES_PATH)
    print(f"Loaded {len(cookies)} cookies")

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    progress = {
        "done": 0,
        "total": len(todo),
        "recovered": 0,
        "paywall_streak": 0,
        "http_403_streak": 0,
    }
    abort_flag = asyncio.Event()

    async with aiohttp.ClientSession(
        connector=connector, cookies=cookies, headers=BROWSER_HEADERS, timeout=timeout
    ) as session:
        workers = [
            asyncio.create_task(
                worker(i, queue, session, checkpoint, csv_writer, csv_file, args, progress, abort_flag)
            )
            for i in range(args.concurrency)
        ]
        for _, row in todo.iterrows():
            if abort_flag.is_set():
                break
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
    print(f"Done. Processed: {progress['done']}, Recovered: {progress['recovered']}")
    print(f"Status breakdown: {checkpoint.stats()}")
    print(f"Texts saved to: {FULLTEXT_DIR}/")
    print(f"Log:            {OUTPUT_PATH}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                   help="Base delay (s) between requests; jitter is added")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    p.add_argument("--decade", type=int, default=None,
                   help="Restrict to one decade (e.g., 2020)")
    p.add_argument("--pre1981", action="store_true",
                   help="Restrict to pre-1981 articles (TimesMachine archives)")
    p.add_argument("--limit", type=int, default=None, help="Test on first N targets")
    p.add_argument("--force", action="store_true", help="Reset checkpoint")
    args = p.parse_args()

    if args.force and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Checkpoint cleared.")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
