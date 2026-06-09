#!/usr/bin/env python3
"""
Playwright-based NYT scraper — bypasses DataDome WAF.

Why this exists:
  Direct HTTP requests with cookies (scrape_nyt_subscription.py) get
  blocked by NYT's DataDome WAF after ~30 articles. Playwright runs a
  real Chromium browser, executes DataDome's JS challenges, and presents
  a believable browser fingerprint — significantly less likely to trip
  bot detection.

Cost:
  - Slower (15-25s per article including page load + jitter)
  - Heavier (~200MB Chromium download)
  - Single-threaded (one browser context per process)

Inputs:
  - data/nyt_cookies.txt    Netscape-format NYT subscription cookies
  - data/iran_articles_full.parquet

Outputs (same as scrape_nyt_subscription.py — both share the merge path):
  - data/fulltext/{md5(uri)[:12]}.txt
  - data/nyt_subscription_articles.csv  (appended)
  - data/nyt_subscription_checkpoint.json

Usage:
  # Test
  python scrape_nyt_playwright.py --pre1981 --limit 5 --headed

  # Overnight slow run (1979 only — finishes in ~7h)
  python scrape_nyt_playwright.py --year 1979 --delay 15

  # Both 1979-80 (~14h, two nights or weekend)
  python scrape_nyt_playwright.py --pre1981 --delay 15
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

import pandas as pd
import trafilatura

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FULLTEXT_DIR = DATA_DIR / "fulltext"
COOKIES_PATH = DATA_DIR / "nyt_cookies.txt"
PARQUET_PATH = DATA_DIR / "iran_articles_full.parquet"
CHECKPOINT_PATH = DATA_DIR / "nyt_subscription_checkpoint.json"
OUTPUT_PATH = DATA_DIR / "nyt_subscription_articles.csv"

DEFAULT_DELAY = 15.0
DEFAULT_TIMEOUT = 60  # page load timeout (s)
DEFAULT_MIN_WORDS = 150
PAYWALL_ABORT_THRESHOLD = 5
HTTP_403_ABORT_THRESHOLD = 3

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Stealth tweaks: hide common headless markers.
STEALTH_INIT_JS = """
() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    Object.defineProperty(navigator, 'plugins',   { get: () => [1, 2, 3, 4, 5] });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    window.chrome = { runtime: {} };
    const originalQuery = window.navigator.permissions?.query;
    if (originalQuery) {
        window.navigator.permissions.query = (parameters) =>
            parameters.name === 'notifications'
                ? Promise.resolve({ state: Notification.permission })
                : originalQuery(parameters);
    }
}
"""

PAYWALL_MARKERS = [
    "subscribe to keep reading",
    "subscribe to continue",
    "you've reached your limit",
    "subscribe for unlimited",
    "create your free account to continue",
]


# ───────────────────────────────────────────────────────────────────
# Cookies → Playwright storage_state
# ───────────────────────────────────────────────────────────────────
def cookies_to_storage_state(cookie_path: Path) -> dict:
    """Convert Netscape cookie file to Playwright storage_state format."""
    if not cookie_path.exists():
        sys.exit(f"FATAL: cookie file missing: {cookie_path}")
    mc = http.cookiejar.MozillaCookieJar()
    mc.load(str(cookie_path), ignore_discard=True, ignore_expires=True)

    cookies = []
    for c in mc:
        cookies.append({
            "name": c.name,
            "value": c.value,
            "domain": c.domain,
            "path": c.path,
            "expires": c.expires if c.expires else -1,
            "httpOnly": False,
            "secure": c.secure,
            "sameSite": "Lax",
        })
    return {"cookies": cookies, "origins": []}


# ───────────────────────────────────────────────────────────────────
# Checkpoint
# ───────────────────────────────────────────────────────────────────
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


# ───────────────────────────────────────────────────────────────────
# Extraction & paywall detection
# ───────────────────────────────────────────────────────────────────
def extract_text(html: str) -> Optional[str]:
    if not html:
        return None
    for kw in (
        {"include_comments": False, "include_tables": False, "favor_precision": True},
        {"include_comments": False, "include_tables": False, "favor_recall": True},
    ):
        try:
            t = trafilatura.extract(html, **kw)
            if t and len(t.split()) >= 50:
                return t
        except Exception:
            continue
    return None


def looks_paywalled(html: str, text: Optional[str]) -> bool:
    if not html:
        return False
    lower = html.lower()
    for m in PAYWALL_MARKERS:
        if m in lower:
            return True
    if text is not None and len(text.split()) < 30:
        return True
    return False


# ───────────────────────────────────────────────────────────────────
# Targets
# ───────────────────────────────────────────────────────────────────
def load_targets(args) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    df["decade"] = (df["year"] // 10) * 10
    targets = df[~df["scrape_status"].isin(["success", "recovered", "subscription"])].copy()

    if args.year is not None:
        targets = targets[targets["year"] == args.year]
    elif args.pre1981:
        targets = targets[targets["year"] < 1981]
    elif args.decade is not None:
        targets = targets[targets["decade"] == args.decade]

    return targets[["uri", "web_url", "year", "decade"]].reset_index(drop=True)


# ───────────────────────────────────────────────────────────────────
# Main scrape loop
# ───────────────────────────────────────────────────────────────────
async def run(args):
    from playwright.async_api import async_playwright

    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    targets = load_targets(args)
    print(f"Targets: {len(targets)}")
    print("  by decade:")
    for dec, n in targets["decade"].value_counts().sort_index().items():
        print(f"    {int(dec)}s: {n}")
    if args.limit:
        targets = targets.head(args.limit)
        print(f"  Limited to first {args.limit}")

    checkpoint = Checkpoint(CHECKPOINT_PATH)
    print(f"Checkpoint: {len(checkpoint.completed)} already processed")
    todo = targets[~targets["uri"].isin(checkpoint.completed.keys())]
    print(f"Remaining: {len(todo)}")
    if len(todo) == 0:
        print("Nothing to do.")
        return

    storage_state = cookies_to_storage_state(COOKIES_PATH)
    print(f"Loaded {len(storage_state['cookies'])} cookies into storage_state")

    is_new = not OUTPUT_PATH.exists()
    csv_file = OUTPUT_PATH.open("a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if is_new:
        csv_writer.writerow(["uri", "url", "status", "http_status", "word_count", "error"])

    progress = {"done": 0, "total": len(todo), "recovered": 0,
                "paywall_streak": 0, "http_403_streak": 0}

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=not args.headed,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
                "--no-sandbox",
            ],
        )
        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
            storage_state=storage_state,
        )
        await context.add_init_script(STEALTH_INIT_JS)
        page = await context.new_page()

        # Warm up — visit homepage so DataDome sees a normal landing
        try:
            print("Warming up at nytimes.com homepage...")
            await page.goto("https://www.nytimes.com/", timeout=args.timeout * 1000)
            await asyncio.sleep(3)
        except Exception as e:
            print(f"  warmup warning: {e}")

        try:
            for _, row in todo.iterrows():
                uri, url = row["uri"], row["web_url"]
                http_status, html, err = 0, "", ""

                try:
                    resp = await page.goto(url, timeout=args.timeout * 1000,
                                           wait_until="domcontentloaded")
                    http_status = resp.status if resp else 0
                    # Small wait for client-rendered content + DataDome challenge
                    try:
                        await page.wait_for_load_state("networkidle", timeout=10_000)
                    except Exception:
                        pass
                    # Light human-like behavior
                    await page.evaluate("window.scrollBy(0, 400)")
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                    html = await page.content()
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:80]}"

                text = extract_text(html) if html else None
                wc = len(text.split()) if text else 0
                paywalled = looks_paywalled(html or "", text) if html else False

                if text and wc >= args.min_words and not paywalled and http_status == 200:
                    status = "success"
                    txt_path = FULLTEXT_DIR / f"{hashlib.md5(uri.encode()).hexdigest()[:12]}.txt"
                    txt_path.write_text(text, encoding="utf-8")
                    progress["recovered"] += 1
                    progress["paywall_streak"] = 0
                    progress["http_403_streak"] = 0
                elif http_status == 403:
                    status = "error_403"
                    progress["http_403_streak"] += 1
                    progress["paywall_streak"] = 0
                elif paywalled:
                    status = "paywall"
                    progress["paywall_streak"] += 1
                    progress["http_403_streak"] = 0
                elif http_status == 200 and wc < args.min_words:
                    status = "too_short"
                    progress["paywall_streak"] = 0
                    progress["http_403_streak"] = 0
                else:
                    status = f"error_{http_status}" if http_status else "fetch_failed"
                    progress["paywall_streak"] = 0
                    progress["http_403_streak"] = 0

                checkpoint.mark(uri, status)
                csv_writer.writerow([uri, url, status, http_status, wc, err[:80]])
                csv_file.flush()
                progress["done"] += 1

                if progress["done"] % 10 == 0:
                    stats = checkpoint.stats()
                    print(f"  [{progress['done']}/{progress['total']}] "
                          f"recovered={progress['recovered']} stats={stats}", flush=True)
                if progress["done"] % 50 == 0:
                    checkpoint.save()

                # Abort triggers
                if progress["paywall_streak"] >= PAYWALL_ABORT_THRESHOLD:
                    print(f"\n⚠️  {PAYWALL_ABORT_THRESHOLD} consecutive paywalls — "
                          "session likely expired. Aborting.", flush=True)
                    break
                if progress["http_403_streak"] >= HTTP_403_ABORT_THRESHOLD:
                    print(f"\n⚠️  {HTTP_403_ABORT_THRESHOLD} consecutive 403s — "
                          "DataDome flagged us. Aborting; wait + retry later.", flush=True)
                    break

                # Polite delay with jitter
                await asyncio.sleep(args.delay + random.uniform(0, args.delay * 0.4))

        finally:
            await context.close()
            await browser.close()

    checkpoint.save()
    csv_file.close()
    print()
    print("=" * 60)
    print(f"Done. Processed: {progress['done']}, Recovered: {progress['recovered']}")
    print(f"Status breakdown: {checkpoint.stats()}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                   help="Base delay (s) between articles; jitter is added")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="Per-page load timeout (s)")
    p.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    p.add_argument("--year", type=int, default=None,
                   help="Restrict to one year (e.g., 1979)")
    p.add_argument("--pre1981", action="store_true")
    p.add_argument("--decade", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--headed", action="store_true",
                   help="Run with visible browser window (better DataDome bypass)")
    p.add_argument("--force", action="store_true", help="Reset checkpoint")
    args = p.parse_args()

    if args.force and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Checkpoint cleared.")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
