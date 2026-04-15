#!/usr/bin/env python3
"""
Framing tension analysis: measure divergence between different structural
components of NYT Iran articles (headline vs body, news vs opinion, etc.)
"""
import pandas as pd
import numpy as np
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ── Framing lexicons ──────────────────────────────────────────────────
THREAT_WORDS = {
    'threat', 'threaten', 'threatens', 'threatening', 'threatened',
    'attack', 'attacks', 'strike', 'strikes', 'bomb', 'bombs', 'bombing',
    'missile', 'missiles', 'weapon', 'weapons', 'nuclear', 'enrichment',
    'uranium', 'centrifuge', 'warhead',
    'terror', 'terrorism', 'terrorist', 'terrorists',
    'war', 'warfare', 'military', 'militia', 'armed',
    'danger', 'dangerous', 'hostile', 'hostility',
    'aggression', 'aggressive', 'provocative', 'provocation',
    'destabilize', 'destabilizing', 'rogue',
    'sanctions', 'sanction', 'embargo', 'punish', 'punishment',
    'crisis', 'conflict', 'confrontation', 'escalation', 'escalate',
    'kill', 'killed', 'killing', 'death', 'deaths', 'deadly',
    'execute', 'execution', 'executed',
    'seize', 'seized', 'capture', 'captured',
    'proxy', 'proxies', 'retaliation', 'retaliate',
}

DIPLOMACY_WORDS = {
    'diplomacy', 'diplomatic', 'diplomat', 'diplomats',
    'negotiate', 'negotiation', 'negotiations', 'negotiating', 'negotiator',
    'agreement', 'accord', 'deal', 'treaty', 'pact',
    'talk', 'talks', 'dialogue', 'discussion',
    'peace', 'peaceful', 'peacemaker',
    'cooperate', 'cooperation', 'cooperative',
    'compromise', 'concession', 'concessions',
    'resolve', 'resolution',
    'engage', 'engagement', 'outreach',
    'moderate', 'moderation', 'moderates', 'reform', 'reformist', 'reformers',
    'rapprochement', 'detente', 'thaw',
}

HUMANIZING_WORDS = {
    'people', 'family', 'families', 'children', 'woman', 'women', 'girl', 'girls',
    'student', 'students', 'young', 'youth',
    'artist', 'writer', 'poet', 'filmmaker', 'musician', 'director',
    'ordinary', 'everyday', 'daily', 'life', 'lives',
    'hope', 'dream', 'love', 'joy', 'happy', 'happiness',
    'culture', 'cultural', 'art', 'arts', 'film', 'cinema', 'music', 'poetry',
    'beauty', 'beautiful', 'tradition', 'traditions',
    'freedom', 'liberty', 'rights', 'dignity',
    'protest', 'protesters', 'protester', 'demonstrators', 'movement',
    'community', 'neighborhood', 'street', 'home',
}

CREDIBLE_VERBS = {'said', 'says', 'stated', 'announced', 'confirmed',
                  'acknowledged', 'noted', 'explained', 'argued', 'described'}
DUBIOUS_VERBS = {'claimed', 'claims', 'insisted', 'insists', 'alleged',
                 'denied', 'denies', 'vowed', 'warned', 'warns',
                 'threatened', 'boasted', 'bragged', 'declared'}


def word_tokenize(text):
    if not isinstance(text, str):
        return []
    return re.findall(r'\b[a-z]+\b', text.lower())


def lexicon_density(tokens, lexicon):
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in lexicon)
    return hits / len(tokens) * 1000


def extract_speech_verbs(text, actor_pattern):
    if not isinstance(text, str):
        return []
    verbs = []
    for m in re.finditer(actor_pattern + r'\s+(\w+)', text, re.IGNORECASE):
        verb = m.group(1).lower()
        if verb in CREDIBLE_VERBS | DUBIOUS_VERBS:
            verbs.append(verb)
    return verbs


def main():
    print("Loading dataset...")
    df = pd.read_parquet(DATA_DIR / "iran_articles_full.parquet")
    print(f"Total articles: {len(df):,}")
    has_ft = df['fulltext'].notna() & (df['fulltext'] != '')
    df_ft = df[has_ft].copy()
    print(f"With fulltext: {len(df_ft):,}")

    # ── 1. Lexicon densities for each component ──
    print("\n" + "=" * 65)
    print("TENSION 1: Headline vs. Abstract vs. Lead vs. Fulltext")
    print("=" * 65)
    components = {
        'headline': df_ft['headline'],
        'abstract': df_ft['abstract'],
        'lead_paragraph': df_ft['lead_paragraph'],
        'fulltext': df_ft['fulltext'],
    }

    for name, series in components.items():
        tokens_list = series.apply(word_tokenize)
        df_ft[f'{name}_threat'] = tokens_list.apply(lambda t: lexicon_density(t, THREAT_WORDS))
        df_ft[f'{name}_diplo'] = tokens_list.apply(lambda t: lexicon_density(t, DIPLOMACY_WORDS))
        df_ft[f'{name}_human'] = tokens_list.apply(lambda t: lexicon_density(t, HUMANIZING_WORDS))

    print(f"\n{'Component':<18} {'Threat':>8} {'Diplomacy':>10} {'Humanizing':>11} {'T/D':>6} {'H/T':>6}")
    print("-" * 63)
    for name in components:
        t = df_ft[f'{name}_threat'].mean()
        d = df_ft[f'{name}_diplo'].mean()
        h = df_ft[f'{name}_human'].mean()
        td = t / d if d > 0 else 0
        ht = h / t if t > 0 else 0
        print(f"{name:<18} {t:>8.2f} {d:>10.2f} {h:>11.2f} {td:>6.2f} {ht:>6.2f}")

    # Headline-body divergence
    df_ft['hl_body_threat_div'] = df_ft['headline_threat'] - df_ft['fulltext_threat']
    pct_hl = (df_ft['hl_body_threat_div'] > 0).mean() * 100
    print(f"\nHeadline threat > body threat in {pct_hl:.1f}% of articles")
    print(f"Mean divergence: {df_ft['hl_body_threat_div'].mean():+.2f} per 1000 words")

    # Temporal trend
    df_ft['decade'] = (df_ft['year'] // 10) * 10
    print("\nBy decade:")
    for decade, group in df_ft.groupby('decade'):
        gap = group['hl_body_threat_div'].mean()
        pct = (group['hl_body_threat_div'] > 0).mean() * 100
        n = len(group)
        print(f"  {int(decade)}s (n={n:,}): gap={gap:+.2f}, {pct:.1f}% HL more threatening")

    # ── 2. News vs Opinion ──
    print("\n" + "=" * 65)
    print("TENSION 2: News vs. Opinion/Editorial")
    print("=" * 65)
    genre_map = {
        'News': 'News', 'An Analysis': 'Analysis',
        'Editorial': 'Editorial', 'Op-Ed': 'Op-Ed',
        'Letter': 'Letter', 'Review': 'Review',
    }
    df_ft['genre'] = df_ft['type_of_material'].map(genre_map)
    df_genre = df_ft[df_ft['genre'].notna()]

    print(f"\n{'Genre':<12} {'N':>6} {'Threat':>8} {'Diplo':>7} {'Human':>7} {'T/D':>6} {'H/T':>6}")
    print("-" * 55)
    for genre in ['News', 'Analysis', 'Editorial', 'Op-Ed', 'Letter', 'Review']:
        group = df_genre[df_genre['genre'] == genre]
        if len(group) < 10:
            continue
        t = group['fulltext_threat'].mean()
        d = group['fulltext_diplo'].mean()
        h = group['fulltext_human'].mean()
        td = t / d if d > 0 else 0
        ht = h / t if t > 0 else 0
        print(f"{genre:<12} {len(group):>6} {t:>8.2f} {d:>7.2f} {h:>7.2f} {td:>6.2f} {ht:>6.2f}")

    # Temporal: News vs Op-Ed gap
    print("\nNews vs Op-Ed threat gap by decade:")
    for decade in sorted(df_ft['decade'].unique()):
        news = df_genre[(df_genre['decade'] == decade) & (df_genre['genre'] == 'News')]
        oped = df_genre[(df_genre['decade'] == decade) & (df_genre['genre'] == 'Op-Ed')]
        if len(news) > 20 and len(oped) > 5:
            gap = oped['fulltext_threat'].mean() - news['fulltext_threat'].mean()
            print(f"  {int(decade)}s: Op-Ed - News = {gap:+.2f} ({'Op-Ed more hawkish' if gap > 0 else 'News more hawkish'})")

    # ── 3. Section framing ──
    print("\n" + "=" * 65)
    print("TENSION 3: Section Framing")
    print("=" * 65)
    top_sections = df_ft['section_name'].value_counts().head(12).index
    print(f"\n{'Section':<22} {'N':>6} {'Threat':>8} {'Human':>7} {'H/T':>6}")
    print("-" * 52)
    for sec in top_sections:
        group = df_ft[df_ft['section_name'] == sec]
        t = group['fulltext_threat'].mean()
        h = group['fulltext_human'].mean()
        ht = h / t if t > 0 else 0
        print(f"{str(sec)[:22]:<22} {len(group):>6} {t:>8.2f} {h:>7.2f} {ht:>6.2f}")

    # ── 4. Speech verb asymmetry ──
    print("\n" + "=" * 65)
    print("TENSION 4: Speech Verb Asymmetry")
    print("=" * 65)
    iran_verbs = Counter()
    us_verbs = Counter()
    for text in df_ft['fulltext'].dropna():
        iran_verbs.update(extract_speech_verbs(text, r'Iran'))
        us_verbs.update(extract_speech_verbs(text, r'(?:United States|America|Washington|U\.S\.)'))

    print(f"\n{'Verb':<15} {'Iran':>6} {'US':>6} {'Ratio':>8}")
    print("-" * 38)
    for verb in sorted(set(iran_verbs) | set(us_verbs),
                       key=lambda v: iran_verbs.get(v, 0) + us_verbs.get(v, 0), reverse=True)[:15]:
        i, u = iran_verbs.get(verb, 0), us_verbs.get(verb, 0)
        ratio = f"{i/u:.1f}:1" if u > 0 else "inf"
        print(f"{verb:<15} {i:>6} {u:>6} {ratio:>8}")

    iran_cred = sum(iran_verbs.get(v, 0) for v in CREDIBLE_VERBS)
    iran_dub = sum(iran_verbs.get(v, 0) for v in DUBIOUS_VERBS)
    us_cred = sum(us_verbs.get(v, 0) for v in CREDIBLE_VERBS)
    us_dub = sum(us_verbs.get(v, 0) for v in DUBIOUS_VERBS)
    iran_pct = iran_dub / (iran_cred + iran_dub) * 100 if (iran_cred + iran_dub) > 0 else 0
    us_pct = us_dub / (us_cred + us_dub) * 100 if (us_cred + us_dub) > 0 else 0
    print(f"\nIran: {iran_dub}/{iran_cred+iran_dub} dubious ({iran_pct:.1f}%)")
    print(f"US:   {us_dub}/{us_cred+us_dub} dubious ({us_pct:.1f}%)")

    # ── 5. Abstract vs Fulltext ──
    print("\n" + "=" * 65)
    print("TENSION 5: Abstract vs. Fulltext Securitization")
    print("=" * 65)
    valid = df_ft['fulltext_threat'] > 0
    ratio = df_ft.loc[valid, 'abstract_threat'] / df_ft.loc[valid, 'fulltext_threat']
    print(f"Abstract/Fulltext threat ratio: mean={ratio.mean():.2f}, median={ratio.median():.2f}")
    h_ratio = df_ft.loc[df_ft['fulltext_human'] > 0, 'abstract_human'] / df_ft.loc[df_ft['fulltext_human'] > 0, 'fulltext_human']
    print(f"Abstract/Fulltext humanizing ratio: mean={h_ratio.mean():.2f}, median={h_ratio.median():.2f}")

    print("\nAbstract-Fulltext threat gap by decade:")
    for decade, group in df_ft.groupby('decade'):
        gap = (group['abstract_threat'] - group['fulltext_threat']).mean()
        print(f"  {int(decade)}s: {gap:+.2f}")

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE")
    print("=" * 65)


if __name__ == '__main__':
    main()
