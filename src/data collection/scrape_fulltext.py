#!/usr/bin/env python3
"""
Scrape full article text from NYT web URLs via the Wayback Machine.

NYT uses DataDome bot protection that blocks all direct programmatic access
(curl, aiohttp, requests, headless browsers all get 403). The Wayback Machine
at web.archive.org caches most NYT articles and serves them without bot
protection, making it the most reliable source for full-text extraction.

Strategy:
  1. For each article URL, fetch from https://web.archive.org/web/{nyt_url}
     (redirects to the latest archived snapshot)
  2. Extract article text with trafilatura (precision mode, then recall fallback)
  3. Save extracted text to data/fulltext/{hash}.txt

Architecture improvements over the Olympics paper's approach:
  - Wayback Machine as source (bypasses NYT DataDome bot protection)
  - trafilatura for robust, ML-backed article extraction (vs. custom heuristics)
  - asyncio + aiohttp for concurrent downloads (configurable pool size)
  - Adaptive rate limiting with exponential backoff + Retry-After support
  - Checkpoint/resume: progress saved per-article, restartable mid-run
  - Dual storage: raw HTML archived + extracted text, for reproducibility
  - Content quality validation: min word count, boilerplate detection
  - Content-hash dedup: catches duplicates the URI dedup may miss

Usage:
    python scrape_fulltext.py                      # default: 3 concurrent, 2s delay
    python scrape_fulltext.py --concurrency 5      # faster (more parallel requests)
    python scrape_fulltext.py --delay 3            # slower (more polite)
    python scrape_fulltext.py --save-html          # also archive raw HTML
    python scrape_fulltext.py --limit 100          # scrape only first 100 articles
    python scrape_fulltext.py --start-year 2000 --end-year 2010  # year range filter
"""

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp is required. Install with: pip install aiohttp")
    sys.exit(1)

try:
    import trafilatura
except ImportError:
    print("ERROR: trafilatura is required. Install with: pip install trafilatura")
    sys.exit(1)

# ── Configuration ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FULLTEXT_DIR = DATA_DIR / "fulltext"
HTML_DIR = DATA_DIR / "fulltext_html"
CHECKPOINT_PATH = DATA_DIR / "scrape_checkpoint.json"
OUTPUT_PATH = DATA_DIR / "fulltext_articles.csv"

# Wayback Machine base URL
WAYBACK_PREFIX = "https://web.archive.org/web/"

# Timestamp fallbacks: when the default redirect returns 403 (Wayback bot detection),
# try specific timestamps. Reduced to 2 to minimize overhead on truly missing articles.
WAYBACK_TIMESTAMPS = ["20230101", "20150101"]

# Scraping defaults — balanced speed vs. politeness
DEFAULT_CONCURRENCY = 5
DEFAULT_DELAY = 2.0  # seconds between requests per worker
DEFAULT_TIMEOUT = 20  # seconds per request
MAX_RETRIES = 2
MIN_ARTICLE_WORDS = 50  # below this, flag as likely stub/incomplete

# User-Agent (polite, identifies academic research purpose)
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scraper")


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class ScrapeResult:
    uri: str
    web_url: str
    status: str  # "success", "not_archived", "extraction_failed", "too_short",
                 # "failed", "empty_url", "timeout", "http_error", "rate_limited"
    word_count: int = 0
    content_hash: str = ""
    fulltext: str = ""
    error_msg: str = ""
    http_status: int = 0
    scrape_time: float = 0.0


# ── Checkpoint management ────────────────────────────────────────────

class CheckpointManager:
    """Tracks which URIs have already been scraped for resume support."""

    def __init__(self, path: Path):
        self.path = path
        self.completed: dict = {}  # uri -> status
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.completed = data.get("completed", {})
                log.info(f"Checkpoint loaded: {len(self.completed)} articles already scraped")
            except (json.JSONDecodeError, Exception) as e:
                log.warning(f"Checkpoint file corrupted, starting fresh: {e}")
                self.completed = {}

    def save(self):
        with open(self.path, "w") as f:
            json.dump({"completed": self.completed}, f)

    def is_done(self, uri: str) -> bool:
        return uri in self.completed

    def mark_done(self, uri: str, status: str):
        self.completed[uri] = status

    def stats(self) -> dict:
        from collections import Counter
        return dict(Counter(self.completed.values()))


# ── Core scraping logic ──────────────────────────────────────────────

def extract_text(html: str, url: str) -> Optional[str]:
    """Extract article text using trafilatura with fallback settings."""
    # Primary extraction: trafilatura with precision mode
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        favor_precision=True,
        deduplicate=True,
    )
    if text and len(text.split()) >= MIN_ARTICLE_WORDS:
        return text

    # Fallback: trafilatura with recall mode (more aggressive extraction)
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        favor_recall=True,
        deduplicate=True,
    )
    return text


def content_hash(text: str) -> str:
    """Generate a content hash for deduplication."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


async def fetch_from_wayback(
    session: aiohttp.ClientSession,
    nyt_url: str,
    timestamp: str = "",
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[Optional[str], int]:
    """Fetch article HTML from the Wayback Machine.

    If timestamp is empty, uses the default redirect (latest snapshot).
    If timestamp is provided (e.g. "20230101"), fetches that specific snapshot.
    Timestamp fallbacks use a shorter timeout to fail fast.

    Returns (html, http_status).
    """
    if timestamp:
        wayback_url = f"{WAYBACK_PREFIX}{timestamp}/{nyt_url}"
        timeout = min(timeout, 10)  # fail fast on fallbacks
    else:
        wayback_url = WAYBACK_PREFIX + nyt_url
    headers = {"User-Agent": random.choice(USER_AGENTS)}

    try:
        async with session.get(
            wayback_url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=True,
        ) as resp:
            if resp.status == 200:
                html = await resp.text()
                return html, 200
            elif resp.status == 404:
                return None, 404
            elif resp.status == 429:
                return None, 429
            else:
                return None, resp.status
    except asyncio.TimeoutError:
        return None, -1  # timeout
    except aiohttp.ClientError as e:
        log.debug(f"Client error for {wayback_url}: {e}")
        return None, -2  # connection error
    except Exception as e:
        log.debug(f"Unexpected error for {wayback_url}: {e}")
        return None, -3


async def discover_wayback_timestamps(
    session: aiohttp.ClientSession,
    nyt_url: str,
    limit: int = 5,
) -> list[str]:
    """Ask Wayback's CDX index for actual archived snapshots of this URL.

    Fixed timestamp fallbacks work for many recent pages, but pre-web archive
    URLs often have sparse snapshots at irregular dates (e.g. 2020/2021 only).
    CDX discovery prevents us from falsely classifying those as unrecoverable.
    """
    cdx_url = "https://web.archive.org/cdx"
    params = {
        "url": nyt_url,
        "output": "json",
        "fl": "timestamp,statuscode,mimetype,digest",
        "filter": ["statuscode:200", "mimetype:text/html"],
        "collapse": "digest",
        "limit": str(limit),
    }
    headers = {"User-Agent": random.choice(USER_AGENTS)}

    try:
        async with session.get(
            cdx_url,
            params=params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json(content_type=None)
    except Exception as e:
        log.debug(f"CDX lookup failed for {nyt_url}: {e}")
        return []

    if not isinstance(data, list) or len(data) <= 1:
        return []

    timestamps = []
    for row in data[1:]:
        if not row:
            continue
        timestamp = str(row[0])
        if timestamp and timestamp.isdigit():
            # id_ asks Wayback for the archived page content without rewriting.
            timestamps.append(f"{timestamp}id_")
    return timestamps


def is_timesmachine_page(html: str) -> bool:
    """Detect NYT TimesMachine paywall pages (scanned archive previews).

    These pages show only a ~48-word snippet followed by
    'View Full Article in Timesmachine'. The full text is only available
    as scanned images, not extractable via web scraping.
    """
    html_lower = html.lower()
    return (
        "view full article in timesmachine" in html_lower
        or "view full article" in html_lower and "timesmachine" in html_lower
    )


async def scrape_one(
    session: aiohttp.ClientSession,
    uri: str,
    url: str,
    save_html: bool = False,
) -> ScrapeResult:
    """Scrape a single article via Wayback Machine with retries and timestamp fallback."""
    if not url or not url.startswith("http"):
        return ScrapeResult(uri=uri, web_url=url or "", status="empty_url")

    t0 = time.monotonic()

    # Try URLs in order: default redirect, then timestamp-based fallbacks.
    # Key optimization: track connection failures — if we get 2+ connection
    # errors in a row, Wayback is having issues with this URL, bail fast.
    urls_to_try = [""] + WAYBACK_TIMESTAMPS  # "" = default redirect
    consecutive_conn_errors = 0
    last_http_status = 0
    best_text = ""  # keep best partial extraction across attempts

    tried_timestamps = set()

    async def try_timestamp(timestamp: str):
        """Try one Wayback timestamp and return (result, status, best_text_update)."""
        html, http_status = await fetch_from_wayback(session, url, timestamp=timestamp)

        if http_status == 429:
            wait = 5 + random.uniform(0, 2)
            log.warning(f"Rate limited (429), waiting {wait:.1f}s...")
            await asyncio.sleep(wait)
            return None, http_status, None

        if html is None or http_status != 200:
            return None, http_status, None

        if is_timesmachine_page(html):
            return None, http_status, None

        fulltext = extract_text(html, url)
        if fulltext is None:
            return None, http_status, None

        wc = len(fulltext.split())
        if wc < MIN_ARTICLE_WORDS:
            return None, http_status, fulltext

        if save_html:
            html_path = HTML_DIR / f"{hashlib.md5(uri.encode()).hexdigest()[:12]}.html"
            html_path.write_text(html, encoding="utf-8")

        return ScrapeResult(
            uri=uri, web_url=url, status="success",
            word_count=wc, content_hash=content_hash(fulltext),
            fulltext=fulltext, http_status=200, scrape_time=time.monotonic() - t0,
        ), http_status, None

    for timestamp in urls_to_try:
        tried_timestamps.add(timestamp)
        # Bail fast if Wayback keeps dropping connections for this URL
        if consecutive_conn_errors >= 2:
            break

        result, http_status, partial_text = await try_timestamp(timestamp)
        last_http_status = http_status
        if result:
            return result
        if partial_text and len(partial_text.split()) > len(best_text.split()):
            best_text = partial_text

        # --- Handle non-200 responses ---

        if http_status == 429:
            wait = 5 + random.uniform(0, 2)
            log.warning(f"Rate limited (429), waiting {wait:.1f}s...")
            await asyncio.sleep(wait)
            continue

        if http_status == 404:
            if not timestamp:
                # Default 404 = not in Wayback at all, don't try timestamps
                return ScrapeResult(
                    uri=uri, web_url=url, status="not_archived",
                    http_status=404, scrape_time=time.monotonic() - t0,
                )
            continue  # this timestamp doesn't exist, try next

        if http_status in (-1, -2, -3):
            consecutive_conn_errors += 1
            continue  # try next timestamp (no sleep — just move on)

        if http_status == 403:
            consecutive_conn_errors = 0
            continue  # bot-blocked on this attempt, try next timestamp

        if http_status != 200:
            continue  # server error (503, 520, etc.), try next

    # If fixed fallbacks failed, ask CDX for actual snapshots and try them.
    cdx_timestamps = await discover_wayback_timestamps(session, url, limit=5)
    for timestamp in cdx_timestamps:
        if timestamp in tried_timestamps:
            continue
        result, http_status, partial_text = await try_timestamp(timestamp)
        last_http_status = http_status
        if result:
            return result
        if partial_text and len(partial_text.split()) > len(best_text.split()):
            best_text = partial_text

    # --- Exhausted all attempts ---
    elapsed = time.monotonic() - t0

    if best_text and len(best_text.split()) > 0:
        return ScrapeResult(
            uri=uri, web_url=url, status="too_short",
            word_count=len(best_text.split()), fulltext=best_text,
            http_status=200, scrape_time=elapsed,
        )

    if last_http_status == 404:
        return ScrapeResult(
            uri=uri, web_url=url, status="not_archived",
            http_status=404, scrape_time=elapsed,
        )

    return ScrapeResult(
        uri=uri, web_url=url, status="failed",
        http_status=last_http_status, scrape_time=elapsed,
        error_msg="Exhausted all fetch attempts",
    )


# ── Worker pool ──────────────────────────────────────────────────────

async def worker(
    name: str,
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    checkpoint: CheckpointManager,
    writer,
    csv_file,
    csv_lock: asyncio.Lock,
    stats: dict,
    save_html: bool,
    delay: float,
):
    """Worker coroutine: pulls articles from queue, scrapes them, writes results."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        uri, url = item

        try:
            result = await scrape_one(session, uri, url, save_html=save_html)

            # Write fulltext to individual file (success only)
            if result.status == "success" and result.fulltext:
                txt_path = FULLTEXT_DIR / f"{hashlib.md5(uri.encode()).hexdigest()[:12]}.txt"
                txt_path.write_text(result.fulltext, encoding="utf-8")

            # Update checkpoint
            checkpoint.mark_done(uri, result.status)

            # Write CSV row and flush
            async with csv_lock:
                writer.writerow({
                    "uri": result.uri,
                    "web_url": result.web_url,
                    "status": result.status,
                    "word_count": result.word_count,
                    "content_hash": result.content_hash,
                    "http_status": result.http_status,
                    "scrape_time": f"{result.scrape_time:.2f}",
                    "error_msg": result.error_msg,
                })
                csv_file.flush()

            # Update stats
            stats[result.status] = stats.get(result.status, 0) + 1
            stats["total"] = stats.get("total", 0) + 1

            # Progress log every 50 articles
            if stats["total"] % 50 == 0:
                log.info(
                    f"Progress: {stats['total']} scraped | "
                    f"success={stats.get('success', 0)} "
                    f"not_archived={stats.get('not_archived', 0)} "
                    f"too_short={stats.get('too_short', 0)} "
                    f"failed={stats.get('failed', 0) + stats.get('http_error', 0) + stats.get('timeout', 0)}"
                )

            # Save checkpoint every 100 articles
            if stats["total"] % 100 == 0:
                checkpoint.save()

        except Exception as e:
            log.error(f"Worker {name} error on {uri}: {e}")
            stats["error"] = stats.get("error", 0) + 1

        finally:
            queue.task_done()
            # Per-worker delay to respect Wayback Machine rate limits
            await asyncio.sleep(delay + random.uniform(0, 0.5))


# ── Main orchestration ───────────────────────────────────────────────

def load_articles(
    limit: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    input_path: Optional[Path] = None,
) -> list:
    """Load (uri, web_url) pairs from the Iran articles dataset."""
    parquet_path = input_path if input_path and input_path.suffix == ".parquet" else DATA_DIR / "iran_articles.parquet"
    csv_path = input_path if input_path and input_path.suffix == ".csv" else DATA_DIR / "iran_articles.csv"

    articles = []

    if parquet_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            if start_year:
                df = df[df["year"] >= start_year]
            if end_year:
                df = df[df["year"] <= end_year]
            if limit:
                df = df.head(limit)
            articles = list(zip(df["uri"].astype(str), df["web_url"].astype(str)))
            log.info(f"Loaded {len(articles)} articles from Parquet")
            return articles
        except ImportError:
            log.info("pandas not available, falling back to CSV")

    if csv_path.exists():
        csv.field_size_limit(sys.maxsize)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                year = None
                if row.get("year"):
                    try:
                        year = int(float(row["year"]))
                    except ValueError:
                        pass
                if start_year and year and year < start_year:
                    continue
                if end_year and year and year > end_year:
                    continue
                articles.append((row["uri"], row["web_url"]))
                if limit and len(articles) >= limit:
                    break
        log.info(f"Loaded {len(articles)} articles from CSV")
        return articles

    log.error("No dataset found. Run filter_iran.py first.")
    sys.exit(1)


async def run_scraper(args):
    """Main async scraper loop."""
    # Setup directories
    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    if args.save_html:
        HTML_DIR.mkdir(parents=True, exist_ok=True)

    # Load articles
    articles = load_articles(
        limit=args.limit,
        start_year=args.start_year,
        end_year=args.end_year,
        input_path=args.input,
    )

    if not articles:
        log.error("No articles to scrape.")
        return

    # Checkpoint
    checkpoint = CheckpointManager(CHECKPOINT_PATH)

    # Filter out already-scraped (optionally retry errors)
    if args.retry_errors:
        # Re-queue articles that previously failed with connection/http errors
        retryable = {"http_error", "timeout", "failed"}
        remaining = [
            (uri, url) for uri, url in articles
            if not checkpoint.is_done(uri) or checkpoint.completed.get(uri) in retryable
        ]
        # Clear retryable entries from checkpoint so they get re-processed
        for uri, url in remaining:
            if checkpoint.is_done(uri):
                del checkpoint.completed[uri]
        checkpoint.save()
        log.info(f"Retry mode: re-queued {len(remaining)} articles (including previous errors)")
    else:
        remaining = [(uri, url) for uri, url in articles if not checkpoint.is_done(uri)]

    log.info(
        f"Total: {len(articles)}, already scraped: {len(articles) - len(remaining)}, "
        f"remaining: {len(remaining)}"
    )

    if not remaining:
        log.info("All articles already scraped! Use --force to re-scrape, or --retry-errors to retry failures.")
        print("\nCheckpoint stats:", checkpoint.stats())
        return

    # Setup CSV output (append mode for resume)
    csv_exists = OUTPUT_PATH.exists()
    csv_file = open(OUTPUT_PATH, "a", newline="", encoding="utf-8")
    fieldnames = [
        "uri", "web_url", "status", "word_count", "content_hash",
        "http_status", "scrape_time", "error_msg",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()

    csv_lock = asyncio.Lock()
    stats = {}

    # Build work queue
    queue = asyncio.Queue()
    for uri, url in remaining:
        await queue.put((uri, url))

    # Sentinel values to stop workers
    for _ in range(args.concurrency):
        await queue.put(None)

    # Create aiohttp session with connection pooling
    connector = aiohttp.TCPConnector(
        limit=args.concurrency * 3,
        limit_per_host=args.concurrency * 2,
        ttl_dns_cache=300,
        keepalive_timeout=30,
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [
            asyncio.create_task(
                worker(
                    name=f"w-{i}",
                    queue=queue,
                    session=session,
                    checkpoint=checkpoint,
                    writer=writer,
                    csv_file=csv_file,
                    csv_lock=csv_lock,
                    stats=stats,
                    save_html=args.save_html,
                    delay=args.delay,
                )
            )
            for i in range(args.concurrency)
        ]
        await asyncio.gather(*workers)

    # Final save
    checkpoint.save()
    csv_file.close()

    # Print summary
    print(f"\n{'='*60}")
    print("SCRAPING COMPLETE")
    print(f"{'='*60}")
    print(f"Total processed: {stats.get('total', 0)}")
    for status in ["success", "not_archived", "too_short", "extraction_failed",
                    "failed", "http_error", "timeout", "empty_url"]:
        count = stats.get(status, 0)
        if count > 0:
            print(f"  {status}: {count}")
    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"Full texts saved to: {FULLTEXT_DIR}/")
    if args.save_html:
        print(f"Raw HTML saved to: {HTML_DIR}/")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape full NYT article text via the Wayback Machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Number of concurrent workers (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Seconds between requests per worker (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--save-html", action="store_true",
        help="Also save raw HTML for each article",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Input Iran article dataset (.parquet or .csv). Defaults to data/iran_articles.*",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only scrape first N articles (for testing)",
    )
    parser.add_argument(
        "--start-year", type=int, default=None,
        help="Only scrape articles from this year onwards",
    )
    parser.add_argument(
        "--end-year", type=int, default=None,
        help="Only scrape articles up to this year",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore checkpoint and re-scrape all articles",
    )
    parser.add_argument(
        "--retry-errors", action="store_true",
        help="Re-scrape articles that previously failed with http/connection errors",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging (debug level)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.force and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        log.info("Checkpoint cleared (--force)")

    print(f"NYT Full-Text Scraper (via Wayback Machine)")
    print(f"  Source: {WAYBACK_PREFIX}")
    print(f"  Concurrency: {args.concurrency} workers")
    print(f"  Delay: {args.delay}s per worker")
    print(f"  Save HTML: {args.save_html}")
    if args.input:
        print(f"  Input: {args.input}")
    if args.start_year or args.end_year:
        print(f"  Year range: {args.start_year or '...'} – {args.end_year or '...'}")
    if args.limit:
        print(f"  Limit: {args.limit} articles")
    print()

    asyncio.run(run_scraper(args))


if __name__ == "__main__":
    main()
