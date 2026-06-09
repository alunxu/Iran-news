#!/usr/bin/env python3
"""
E_M1_4 — Per-decade lexicon contribution to the HL−body THREAT gap.

Quantifies which lexicon words contribute most to the gap in each decade, so
we can see WHETHER the same words drive the gap across time, or different
generations of vocabulary take over as the securitization apparatus shifts.

For each decade × dimension (we focus on THREAT), compute per-word:
  • mean HL hits per article
  • mean body hits per article (per 1000 words)
  • per-word contribution to HL−body density gap

Then rank top contributors per decade.

Outputs:
  data/structural_break/lexicon_decade_contrib_threat.csv
  data/structural_break/lexicon_decade_contrib_diplo.csv
  figures/structural_break/lexicon_decade_topwords.png
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "iran_articles_full.parquet"
LEX_PATH = PROJECT_ROOT / "data" / "lexicons.json"
OUT_DIR = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR = PROJECT_ROOT / "figures" / "structural_break"

ACCENT, COOL, CHAR = "#B85042", "#2F5F5D", "#363A3E"

LEX = json.load(open(LEX_PATH))


def per_word_density(text: str, word: str) -> tuple[int, int]:
    """Return (hits, n_tokens) for `word` in `text`."""
    if not isinstance(text, str) or not text.strip():
        return 0, 0
    n_tokens = len(text.split())
    pat = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    return len(pat.findall(text)), n_tokens


def compute_word_decade_contrib(df: pd.DataFrame, words: list[str]) -> pd.DataFrame:
    """For each (word, decade), compute POOLED HL density, FT density, gap.

    Pooled: sum(hits across all articles) / sum(tokens across all articles) × 1000.
    This is more interpretable per word — "per 1000 words in this decade, the
    word appeared X times in headlines vs Y times in bodies".
    """
    same = df[df["fulltext_word_count"] > 0].copy()
    same["decade"] = (pd.to_datetime(same["pub_date"]).dt.year // 10) * 10
    same["hl_tokens"] = same["headline"].fillna("").str.split().str.len()

    rows = []
    for word in words:
        pat = re.compile(r"\b" + re.escape(word.lower()) + r"\b", re.IGNORECASE)
        same["hl_hits"] = same["headline"].fillna("").apply(lambda t: len(pat.findall(t)))
        same["ft_hits"] = same["fulltext"].fillna("").apply(lambda t: len(pat.findall(t)))
        for dec, g in same.groupby("decade"):
            hl_tot_tokens = g["hl_tokens"].sum()
            ft_tot_tokens = g["fulltext_word_count"].sum()
            hl_dens = (g["hl_hits"].sum() / max(hl_tot_tokens, 1)) * 1000
            ft_dens = (g["ft_hits"].sum() / max(ft_tot_tokens, 1)) * 1000
            rows.append(dict(
                word=word,
                decade=int(dec),
                hl_dens=hl_dens,
                ft_dens=ft_dens,
                gap=hl_dens - ft_dens,
                hl_total_hits=int(g["hl_hits"].sum()),
                ft_total_hits=int(g["ft_hits"].sum()),
                n=len(g),
            ))
    return pd.DataFrame(rows)


def plot_top_words_heatmap(contrib: pd.DataFrame, lex_name: str, color: str,
                          out_path: Path) -> None:
    """Heatmap of top-N words × decade with HL−body gap as color."""
    # Pick top words by total absolute contribution across all decades
    top_words = (contrib.groupby("word")["gap"].apply(lambda x: x.abs().sum())
                 .sort_values(ascending=False).head(15).index.tolist())
    sub = contrib[contrib["word"].isin(top_words)]
    pivot = sub.pivot(index="word", columns="decade", values="gap").reindex(top_words)

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
    vmax = float(pivot.abs().max().max())
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(d)}s" for d in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(f"Top {lex_name} words by HL−body density gap (per 1000 words)",
                 fontsize=11, loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("HL density − body density", fontsize=9)
    # Annotate cells
    for i, w in enumerate(pivot.index):
        for j, d in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(val) < vmax * 0.6 else "white")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    df = pd.read_parquet(DATA)
    print(f"Loaded {len(df):,} articles")

    for lex_name, color in [("threat", ACCENT), ("diplomacy", COOL)]:
        words = LEX[lex_name]
        print(f"\n=== {lex_name.upper()} ({len(words)} words) ===")
        contrib = compute_word_decade_contrib(df, words)
        contrib.to_csv(OUT_DIR / f"lexicon_decade_contrib_{lex_name}.csv", index=False)

        # Print top 10 contributors per decade
        for dec in sorted(contrib["decade"].unique()):
            sub = contrib[contrib["decade"] == dec].nlargest(8, "gap")
            top_words = ", ".join(f"{r.word}({r.gap:+.1f})" for r in sub.itertuples())
            print(f"  {int(dec)}s top HL-body gap: {top_words}")

        plot_top_words_heatmap(contrib, lex_name, color,
                              FIG_DIR / f"lexicon_decade_topwords_{lex_name}.png")


if __name__ == "__main__":
    main()
