#!/usr/bin/env python3
"""
Exploratory analysis of two linguistic framing dimensions beyond lexicon density:

  1. Agency / blame     — who is the grammatical subject of verbs (Iran vs US),
                          and what is the rate of agentless-passive constructions?
  2. Certainty / hedging — epistemic modality markers (clearly, confirmed) vs
                          hedges (may, alleged, reportedly).

We run this on a stratified sample of the fulltext corpus to see whether these
dimensions produce publishable, theory-grounded patterns that would complement
the threat/diplomacy/humanizing lexicon analysis.
"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy

DATA = Path("data/iran_articles_full.parquet")
SAMPLE_PER_DECADE = 200  # speed vs. coverage tradeoff

# Minimal disable for speed: keep tagger, parser, attribute_ruler; drop NER.
NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

IRAN_TOKENS = {
    "iran", "iranian", "iranians", "tehran",
    "ayatollah", "khamenei", "rouhani", "ahmadinejad", "raisi",
    "khomeini", "mullahs", "regime",
}
US_TOKENS = {
    "america", "american", "americans", "u.s.", "us", "usa", "united states",
    "washington", "biden", "trump", "obama", "bush", "clinton",
    "reagan", "carter", "pentagon", "cia",
}

HEDGE_WORDS = {
    "may", "might", "could", "possibly", "perhaps", "apparently",
    "seemingly", "suggests", "suggested", "suggesting", "appears", "appeared",
    "alleged", "allegedly", "reportedly", "reported", "purportedly",
    "supposedly", "presumably", "likely", "unlikely", "maybe",
}
CERTAIN_WORDS = {
    "clearly", "obviously", "evidently", "undoubtedly", "undeniably",
    "certainly", "definitely", "conclusively", "incontrovertibly",
    "confirmed", "proved", "proven", "established", "demonstrated",
    "shown", "beyond doubt", "without question",
}


def load_sample():
    print("Loading corpus...")
    df = pd.read_parquet(DATA)
    df = df[df["fulltext"].notna() & (df["fulltext_word_count"] > 100)].copy()
    df["decade"] = (df["year"] // 10) * 10
    # Stratified sample
    samples = []
    for decade, group in df.groupby("decade"):
        n = min(SAMPLE_PER_DECADE, len(group))
        samples.append(group.sample(n=n, random_state=42))
    sample = pd.concat(samples).reset_index(drop=True)
    print(f"  sampled {len(sample):,} articles across decades "
          f"{sorted(sample['decade'].unique().tolist())}")
    return sample


def actor_of_token(token):
    """Classify a token's text into iran/us/other."""
    t = token.text.lower().rstrip(".")
    if t in IRAN_TOKENS:
        return "iran"
    if t in US_TOKENS:
        return "us"
    return None


def analyze_doc(doc):
    """Return per-doc counts: passives, iran/us subject, iran/us object,
    hedge words, certainty words."""
    n_verbs = 0
    n_passives = 0  # verbs with nsubjpass or auxpass
    n_agentless_pass = 0  # passives with no "by" agent
    iran_subj = iran_obj = 0
    us_subj = us_obj = 0
    for tok in doc:
        if tok.pos_ == "VERB":
            n_verbs += 1
            # Passive detection: any child is nsubjpass or auxpass
            deps = [c.dep_ for c in tok.children]
            if "nsubjpass" in deps or "auxpass" in deps:
                n_passives += 1
                has_by_agent = any(
                    c.dep_ == "agent" for c in tok.children
                )
                if not has_by_agent:
                    n_agentless_pass += 1

            # Subject / object actor detection
            for c in tok.children:
                if c.dep_ in ("nsubj", "nsubjpass"):
                    # Walk the noun phrase to find actor tokens
                    for sub in c.subtree:
                        a = actor_of_token(sub)
                        if a == "iran":
                            iran_subj += 1
                            break
                        elif a == "us":
                            us_subj += 1
                            break
                if c.dep_ in ("dobj", "pobj", "iobj"):
                    for sub in c.subtree:
                        a = actor_of_token(sub)
                        if a == "iran":
                            iran_obj += 1
                            break
                        elif a == "us":
                            us_obj += 1
                            break

    # Certainty / hedging counts
    n_hedge = sum(1 for t in doc if t.text.lower() in HEDGE_WORDS)
    n_certain = sum(1 for t in doc if t.text.lower() in CERTAIN_WORDS)

    return dict(
        n_tokens=len(doc),
        n_verbs=n_verbs,
        n_passives=n_passives,
        n_agentless_passives=n_agentless_pass,
        iran_subj=iran_subj, iran_obj=iran_obj,
        us_subj=us_subj, us_obj=us_obj,
        n_hedge=n_hedge, n_certain=n_certain,
    )


def run(sample):
    print(f"Running spaCy on {len(sample)} docs (tagger+parser only)...")
    results = []
    texts = sample["fulltext"].tolist()
    # nlp.pipe is batched
    for i, doc in enumerate(NLP.pipe(texts, batch_size=50)):
        r = analyze_doc(doc)
        r["decade"] = sample.iloc[i]["decade"]
        r["section"] = sample.iloc[i].get("section_name") or "?"
        r["genre"] = sample.iloc[i].get("type_of_material") or "?"
        results.append(r)
        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(sample)}")
    return pd.DataFrame(results)


def summarize(df):
    total_verbs = df["n_verbs"].sum()
    total_pass = df["n_passives"].sum()
    total_agentless = df["n_agentless_passives"].sum()
    print("\n=== AGENCY / BLAME ===")
    print(f"Total verbs analyzed:            {total_verbs:,}")
    print(f"Passive constructions:           {total_pass:,} "
          f"({100*total_pass/total_verbs:.1f}% of verbs)")
    print(f"  of which agentless:            {total_agentless:,} "
          f"({100*total_agentless/total_pass:.1f}% of passives)")

    print("\n--- Passive rate by decade ---")
    by_dec = df.groupby("decade").agg(
        n_verbs=("n_verbs", "sum"),
        n_pass=("n_passives", "sum"),
        n_agless=("n_agentless_passives", "sum"),
    )
    by_dec["pass_pct"] = 100 * by_dec["n_pass"] / by_dec["n_verbs"]
    by_dec["agless_pct_of_verbs"] = 100 * by_dec["n_agless"] / by_dec["n_verbs"]
    print(by_dec.to_string())

    print("\n--- Iran vs US: subject / object asymmetry ---")
    iran_subj = df["iran_subj"].sum()
    iran_obj = df["iran_obj"].sum()
    us_subj = df["us_subj"].sum()
    us_obj = df["us_obj"].sum()
    print(f"Iran — subject: {iran_subj:,}  |  object: {iran_obj:,}  "
          f"→ subj/obj ratio = {iran_subj/max(iran_obj,1):.2f}")
    print(f"US   — subject: {us_subj:,}  |  object: {us_obj:,}  "
          f"→ subj/obj ratio = {us_subj/max(us_obj,1):.2f}")
    print(f"Interpretation: ratio > 1 means the actor appears more as doer "
          f"than done-to.")

    print("\n=== CERTAINTY / HEDGING ===")
    total_tokens = df["n_tokens"].sum()
    total_hedge = df["n_hedge"].sum()
    total_cert = df["n_certain"].sum()
    print(f"Hedge words:   {total_hedge:,} "
          f"({1000*total_hedge/total_tokens:.2f} per 1k tokens)")
    print(f"Certain words: {total_cert:,} "
          f"({1000*total_cert/total_tokens:.2f} per 1k tokens)")
    print(f"Hedge/Certainty ratio: {total_hedge/max(total_cert,1):.2f}")

    print("\n--- Hedging by decade (per 1k tokens) ---")
    by_dec = df.groupby("decade").agg(
        n_tok=("n_tokens", "sum"),
        n_hedge=("n_hedge", "sum"),
        n_cert=("n_certain", "sum"),
    )
    by_dec["hedge_per_1k"] = 1000 * by_dec["n_hedge"] / by_dec["n_tok"]
    by_dec["cert_per_1k"] = 1000 * by_dec["n_cert"] / by_dec["n_tok"]
    by_dec["hedge_over_cert"] = by_dec["n_hedge"] / by_dec["n_cert"].replace(0, 1)
    print(by_dec.to_string())

    # Genre breakdown
    print("\n--- Hedging by genre (per 1k tokens, top 8) ---")
    by_genre = df.groupby("genre").agg(
        n_tok=("n_tokens", "sum"),
        n_hedge=("n_hedge", "sum"),
        n_cert=("n_certain", "sum"),
        n=("n_tokens", "count"),
    )
    by_genre = by_genre[by_genre["n"] >= 50]
    by_genre["hedge_per_1k"] = 1000 * by_genre["n_hedge"] / by_genre["n_tok"]
    by_genre["cert_per_1k"] = 1000 * by_genre["n_cert"] / by_genre["n_tok"]
    print(by_genre.sort_values("hedge_per_1k", ascending=False).head(8).to_string())


def main():
    sample = load_sample()
    results = run(sample)
    summarize(results)
    out = Path("data/agency_hedging_results.csv")
    results.to_csv(out, index=False)
    print(f"\nSaved raw per-article results: {out}")


if __name__ == "__main__":
    main()
