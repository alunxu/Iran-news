#!/usr/bin/env python3
"""
Tier 1 + 2 enrichment: derive analysis-ready columns and save a versioned
lexicons.json alongside the dataset.

Adds to the parquet/CSV:
  • `voice` — news / editorial / column / review / letter / blog / other
            (from type_of_material; news_desk as fallback for missing types)
  • `decade` — integer (1970, 1980, ...) for stratification
  • `iran_mention_count` — Iran/Iranian + proxy terms in fulltext
  • `headline_word_count`, `lead_word_count`, `abstract_word_count`
            (component-level word counts for tension scoring)

Saves:
  • data/lexicons.json — versioned threat/diplomacy/humanizing word lists
                          (and verb-asymmetry sets), so analyses can load
                          them deterministically.

Idempotent: safe to re-run; overwrites existing columns.
"""

from __future__ import annotations

import json
import re
import sys
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LEX_JSON = DATA_DIR / "lexicons.json"


# ───────────────────────────────────────────────────────────────────
# Lexicons (single source of truth for the project)
# Version-bump when categories change; the version string travels with
# the JSON so downstream analyses can pin to a specific lexicon set.
# ───────────────────────────────────────────────────────────────────
LEXICONS = {
    "version": "1.0.0",
    "description": "NYT Iran framing lexicons, calibrated against Lee & Maslog (2005) war/peace journalism indicators.",
    "threat": sorted({
        "threat","threaten","threatens","threatening","threatened",
        "attack","attacks","strike","strikes","bomb","bombs","bombing",
        "missile","missiles","weapon","weapons","nuclear","enrichment",
        "uranium","centrifuge","warhead",
        "terror","terrorism","terrorist","terrorists",
        "war","warfare","military","militia","armed",
        "danger","dangerous","hostile","hostility",
        "aggression","aggressive","provocative","provocation",
        "destabilize","destabilizing","rogue",
        "sanctions","sanction","embargo","punish","punishment",
        "crisis","conflict","confrontation","escalation","escalate",
        "kill","killed","killing","death","deaths","deadly",
        "execute","execution","executed",
        "seize","seized","capture","captured",
        "proxy","proxies","retaliation","retaliate",
    }),
    "diplomacy": sorted({
        "diplomacy","diplomatic","diplomat","diplomats",
        "negotiate","negotiation","negotiations","negotiating","negotiator",
        "agreement","accord","deal","treaty","pact",
        "talk","talks","dialogue","discussion",
        "peace","peaceful","peacemaker",
        "cooperate","cooperation","cooperative",
        "compromise","concession","concessions",
        "resolve","resolution",
        "engage","engagement","outreach",
        "moderate","moderation","moderates","reform","reformist","reformers",
        "rapprochement","detente","thaw",
    }),
    "humanizing": sorted({
        "people","family","families","children","woman","women","girl","girls",
        "student","students","young","youth",
        "artist","writer","poet","filmmaker","musician","director",
        "ordinary","everyday","daily","life","lives",
        "hope","dream","love","joy","happy","happiness",
        "culture","cultural","art","arts","film","cinema","music","poetry",
        "beauty","beautiful","tradition","traditions",
        "freedom","liberty","rights","dignity",
        "protest","protesters","protester","demonstrators","movement",
        "community","neighborhood","street","home",
    }),
    "credible_verbs": sorted({
        "said","says","stated","announced","confirmed",
        "acknowledged","noted","explained","argued","described",
    }),
    "dubious_verbs": sorted({
        "claimed","claims","insisted","insists","alleged",
        "denied","denies","vowed","warned","warns",
        "threatened","boasted","bragged","declared",
    }),
    "iran_proxy_terms": sorted({
        # Used for relevance scoring — articles can be about Iran
        # without using "Iran"; metonymy via capital, leaders, etc.
        "tehran","teheran","khomeini","khamenei","ayatollah",
        "ahmadinejad","khatami","rafsanjani","rouhani","raisi","pezeshkian",
        "soleimani","mahsa","amini","mossadegh","mossadeq","pahlavi",
        "persian","farsi","shia","shiite","shah",
        "irgc","pasdaran","quds","basij","majlis",
        "natanz","fordow","bushehr","jcpoa",
    }),
}


# ───────────────────────────────────────────────────────────────────
# Voice derivation
# ───────────────────────────────────────────────────────────────────
def derive_voice(tom: str | None, nd: str | None) -> str:
    """Return one of: news / editorial / column / review / letter / blog / other.

    Primary signal: type_of_material (genre tag, NYT-supplied).
    Fallback: news_desk for missing/ambiguous type_of_material.
    """
    tom = (tom or "").strip().lower()
    nd = (nd or "").strip().lower()

    # Primary
    if tom == "editorial":     return "editorial"
    if tom == "op-ed":         return "column"
    if tom == "letter":        return "letter"
    if tom == "review":        return "review"
    if tom == "web log":       return "blog"
    if tom in {"correction", "video", "obituary", "schedule",
               "biography", "addendum", "interview", "list",
               "recipe", "premium"}:
        return "other"
    if tom in {"news", "archives", "summary", "text", "briefing",
               "front page", "news analysis", "an analysis",
               "an appraisal"}:
        return "news"

    # Fallback to news_desk for missing/unknown type_of_material
    if "editorial" in nd:      return "editorial"
    if "oped" in nd or "op-ed" in nd:
        return "column"
    if "letter" in nd:         return "letter"

    return "news"  # safe default


# ───────────────────────────────────────────────────────────────────
# Iran mention count (literal + proxy terms)
# ───────────────────────────────────────────────────────────────────
IRAN_RELEVANCE_PAT = (
    r"\b(?:iran(?:ian)?s?|tehran|teheran|khomeini|khamenei|ayatollah|"
    r"ahmadinejad|khatami|rafsanjani|rouhani|raisi|pezeshkian|soleimani|"
    r"mahsa|mossade[gq]h?|pahlavi|persian|farsi|shia|shiite|shah|"
    r"irgc|pasdaran|quds\s+force|basij|majlis|natanz|fordow|bushehr|jcpoa)\b"
)


def count_iran_mentions(text: str | None) -> int:
    if not text:
        return 0
    return len(re.findall(IRAN_RELEVANCE_PAT, text, flags=re.IGNORECASE))


# ───────────────────────────────────────────────────────────────────
# Word counter (whitespace-split)
# ───────────────────────────────────────────────────────────────────
def wc(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Add analysis-ready derived columns to a full-text Iran corpus."
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
        sys.exit(f"FATAL: {parquet} not found — run merge_fulltext.py first")

    out_parquet = DATA_DIR / f"{args.output_prefix}.parquet"
    out_csv = DATA_DIR / f"{args.output_prefix}.csv"

    print(f"Loading {parquet} ...")
    df = pd.read_parquet(parquet)
    print(f"  {len(df):,} articles loaded")

    # 1. voice
    print("Deriving voice ...")
    df["voice"] = [
        derive_voice(tom, nd)
        for tom, nd in zip(df["type_of_material"], df["news_desk"])
    ]
    voice_counts = df["voice"].value_counts()
    print(f"  voice distribution:")
    for v, n in voice_counts.items():
        print(f"    {v:10s}  {n:>6,}")

    # 2. decade
    print("Deriving decade ...")
    df["decade"] = (df["year"].fillna(-1).astype(int) // 10) * 10
    df.loc[df["decade"] < 0, "decade"] = pd.NA
    print(f"  decade distribution: {dict(df['decade'].value_counts().sort_index())}")

    # 3. Iran mention count (in fulltext)
    print("Counting Iran + proxy mentions in fulltext ...")
    df["iran_mention_count"] = df["fulltext"].fillna("").apply(count_iran_mentions)
    nonzero = (df["iran_mention_count"] > 0).sum()
    has_text = (df["fulltext"].str.len() > 0).sum()
    print(f"  {nonzero:,} / {has_text:,} ({nonzero/has_text*100:.1f}%) fulltext articles have ≥1 mention")
    print(f"  median count (fulltext articles): {df.loc[df['fulltext'].str.len()>0, 'iran_mention_count'].median():.0f}")

    # 4. Component word counts
    print("Computing component word counts ...")
    df["headline_word_count"] = df["headline"].fillna("").apply(wc)
    df["lead_word_count"]     = df["lead_paragraph"].fillna("").apply(wc)
    df["abstract_word_count"] = df["abstract"].fillna("").apply(wc)
    print(f"  median headline wc: {df['headline_word_count'].median():.0f}")
    print(f"  median lead wc:     {df['lead_word_count'].median():.0f}")
    print(f"  median abstract wc: {df['abstract_word_count'].median():.0f}")

    # 5. Save lexicons.json
    LEX_JSON.write_text(json.dumps(LEXICONS, indent=2, ensure_ascii=False))
    print(f"\nSaved lexicons → {LEX_JSON}")
    for cat in ("threat", "diplomacy", "humanizing"):
        print(f"  {cat:10s}: {len(LEXICONS[cat])} terms")
    print(f"  credible_verbs: {len(LEXICONS['credible_verbs'])} terms")
    print(f"  dubious_verbs:  {len(LEXICONS['dubious_verbs'])} terms")
    print(f"  iran_proxy:     {len(LEXICONS['iran_proxy_terms'])} terms")

    # 6. Save enriched parquet + csv
    print(f"\nWriting enriched dataset ...")
    df.to_parquet(out_parquet, index=False)
    print(f"  {out_parquet}")
    df.to_csv(out_csv, index=False)
    print(f"  {out_csv}")

    print(f"\nFinal column count: {len(df.columns)}")
    print(f"New columns: voice, decade, iran_mention_count, "
          f"headline_word_count, lead_word_count, abstract_word_count")


if __name__ == "__main__":
    main()
