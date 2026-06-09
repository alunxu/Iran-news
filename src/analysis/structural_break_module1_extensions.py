#!/usr/bin/env python3
"""
Module 1 Extensions — E_M1_2 (per-voice raw trajectories) + E_M1_1 (4-layer decomposition)
==========================================================================================

Two extensions to the framing atlas:

  E_M1_2  Plot RAW density (not gap) for each voice — news / editorial / column —
          on each framing dimension (threat / diplomacy / humanizing). The reader
          can see absolute levels per voice instead of only the news−editorial gap.

  E_M1_1  Expand the syntactical decomposition from {HL, body} → {headline, abstract,
          lead, body}. Per Pan & Kosicki, NYT articles have 4 distinct syntactical
          slots, each with its own framing pressure. Show three tension pairs:
            • HL − Abstract  (NYT desk framing vs editor summary)
            • HL − Lead       (reader-facing tension — first words after the headline)
            • Lead − Body     (article opener vs main text)

Both extensions operate on the SAME C1 same-article subset (articles with body)
to avoid the V1 selection bias.

Outputs:
  data/structural_break/voice_raw_series.csv
  data/structural_break/layer_series.csv
  figures/structural_break/voice_raw_trajectories.png
  figures/structural_break/layer_decomposition.png
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PARQUET = PROJECT_ROOT / "data" / "iran_articles_full.parquet"
LEX_PATH = PROJECT_ROOT / "data" / "lexicons.json"
OUT_DIR = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR = PROJECT_ROOT / "figures" / "structural_break"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ACCENT, COOL, CHAR, MUTED = "#B85042", "#2F5F5D", "#363A3E", "#707070"

SMOOTH_WINDOW = 12
MIN_ART_PER_VOICE = 3
MIN_ART_LAYER = 5

LEX = json.load(open(LEX_PATH))
THREAT = LEX["threat"]
DIPLO = LEX["diplomacy"]
HUMAN = LEX["humanizing"]

PATTERNS = {
    "threat": re.compile(r"\b(" + "|".join(re.escape(w) for w in THREAT) + r")\b", re.IGNORECASE),
    "diplo":  re.compile(r"\b(" + "|".join(re.escape(w) for w in DIPLO) + r")\b", re.IGNORECASE),
    "human":  re.compile(r"\b(" + "|".join(re.escape(w) for w in HUMAN) + r")\b", re.IGNORECASE),
}


# ────────────────────────────────────────────────────────────────────────
# Per-article density computation for a text field
# ────────────────────────────────────────────────────────────────────────
def density(text: str, pattern: re.Pattern) -> float | None:
    """Returns hits per 1000 tokens, or None if text is empty."""
    if not isinstance(text, str) or not text.strip():
        return None
    n_tokens = len(text.split())
    if n_tokens == 0:
        return None
    hits = len(pattern.findall(text))
    return hits / n_tokens * 1000.0


def add_layer_densities(df: pd.DataFrame, field: str, prefix: str) -> pd.DataFrame:
    """Add per-row {prefix}_threat / _diplo / _human columns from text field."""
    text = df[field].fillna("").astype(str)
    for dim, pat in PATTERNS.items():
        df[f"{prefix}_{dim}"] = text.apply(lambda t: density(t, pat))
    return df


# ────────────────────────────────────────────────────────────────────────
# E_M1_2: per-voice raw trajectories
# ────────────────────────────────────────────────────────────────────────
def build_voice_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly mean fulltext-density per (voice, dim).

    Restricted to same-article body-having subset (the C1 fix).
    """
    same_article = df[df["fulltext_word_count"] > 0].copy()
    same_article = add_layer_densities(same_article, "fulltext", "ft")
    same_article["pub_month"] = pd.to_datetime(same_article["pub_date"]).dt.to_period("M").dt.to_timestamp()

    rows = []
    voices = ["news", "editorial", "column"]
    for (m, v), g in same_article.groupby(["pub_month", "voice"]):
        if v not in voices:
            continue
        if len(g) < MIN_ART_PER_VOICE:
            continue
        rows.append(dict(
            pub_month=m,
            voice=v,
            n=len(g),
            threat=g["ft_threat"].mean(),
            diplo=g["ft_diplo"].mean(),
            human=g["ft_human"].mean(),
        ))
    return pd.DataFrame(rows)


def plot_voice_raw(voice_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=150, sharex=True)
    dims = [("threat", "THREAT density (per 1000 body words)", ACCENT),
            ("diplo",  "DIPLOMACY density",                    COOL),
            ("human",  "HUMANIZING density",                   CHAR)]
    voice_styles = {
        "news":      dict(color=CHAR,   lw=1.4, label="News"),
        "editorial": dict(color=ACCENT, lw=1.2, label="Editorial"),
        "column":    dict(color=COOL,   lw=1.2, label="Column / Op-Ed"),
    }
    for ax, (dim, ylabel, _) in zip(axes, dims):
        for voice, style in voice_styles.items():
            s = voice_df[voice_df["voice"] == voice].set_index("pub_month")[dim]
            if s.empty:
                continue
            s = s.rolling(SMOOTH_WINDOW, min_periods=1).mean()
            ax.plot(s.index, s.values, alpha=0.85, **style)
        ax.set_title(ylabel, fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        if dim == "threat":
            ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Year", fontsize=9)
    fig.suptitle("Raw framing density by voice (12-month moving average, body fulltext only)",
                 fontsize=11, y=0.995)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved: {out_path}")


# ────────────────────────────────────────────────────────────────────────
# E_M1_1: 4-layer decomposition
# ────────────────────────────────────────────────────────────────────────
LAYER_SPECS = [
    ("headline",       "hl"),
    ("abstract",       "ab"),
    ("lead_paragraph", "ld"),
    ("fulltext",       "ft"),
]


def build_layer_series(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly mean density at each of 4 layers, on the same-article subset."""
    same_article = df[df["fulltext_word_count"] > 0].copy()
    for field, prefix in LAYER_SPECS:
        same_article = add_layer_densities(same_article, field, prefix)

    same_article["pub_month"] = pd.to_datetime(same_article["pub_date"]).dt.to_period("M").dt.to_timestamp()

    agg_cols = []
    for _, prefix in LAYER_SPECS:
        for dim in ["threat", "diplo", "human"]:
            agg_cols.append(f"{prefix}_{dim}")

    monthly = (same_article
               .groupby("pub_month")[agg_cols]
               .mean()
               .reset_index())
    monthly["n"] = same_article.groupby("pub_month").size().values
    monthly = monthly[monthly["n"] >= MIN_ART_LAYER]

    # Compute the three flagship tension pairs for THREAT
    for dim in ["threat", "diplo", "human"]:
        monthly[f"gap_{dim}_HL_Abs"]  = monthly[f"hl_{dim}"] - monthly[f"ab_{dim}"]
        monthly[f"gap_{dim}_HL_Lead"] = monthly[f"hl_{dim}"] - monthly[f"ld_{dim}"]
        monthly[f"gap_{dim}_Lead_FT"] = monthly[f"ld_{dim}"] - monthly[f"ft_{dim}"]
    return monthly


def plot_layer_decomp(layer_df: pd.DataFrame, out_path: Path) -> None:
    """3 rows (framing dim) × 2 columns (left: raw layer levels, right: tension pairs)."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), dpi=150, sharex=True)

    dims = [("threat", "THREAT",     ACCENT),
            ("diplo",  "DIPLOMACY",  COOL),
            ("human",  "HUMANIZING", CHAR)]

    layer_styles = {
        "hl": dict(color=ACCENT, lw=1.3, label="Headline"),
        "ab": dict(color=COOL,   lw=1.0, label="Abstract"),
        "ld": dict(color=CHAR,   lw=1.0, label="Lead para"),
        "ft": dict(color=MUTED,  lw=1.0, label="Body"),
    }
    gap_styles = {
        "HL_Abs":  dict(color=ACCENT, lw=1.2, label="HL − Abstract"),
        "HL_Lead": dict(color=COOL,   lw=1.2, label="HL − Lead"),
        "Lead_FT": dict(color=CHAR,   lw=1.0, label="Lead − Body"),
    }

    idx = layer_df.set_index("pub_month")
    for row, (dim, dim_label, _color) in enumerate(dims):
        # Left column — raw layers
        ax = axes[row, 0]
        for layer_key, style in layer_styles.items():
            s = idx[f"{layer_key}_{dim}"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
            ax.plot(s.index, s.values, alpha=0.85, **style)
        ax.set_title(f"{dim_label} — raw density per layer", fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        if row == 0:
            ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)

        # Right column — gap pairs
        ax = axes[row, 1]
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        for gap_key, style in gap_styles.items():
            col = f"gap_{dim}_{gap_key}"
            if col not in idx.columns:
                continue
            s = idx[col].rolling(SMOOTH_WINDOW, min_periods=1).mean()
            ax.plot(s.index, s.values, alpha=0.85, **style)
        ax.set_title(f"{dim_label} — layer tensions (gap)", fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        if row == 0:
            ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)

    for ax in axes[-1, :]:
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlabel("Year", fontsize=9)
    fig.suptitle(
        "4-layer syntactical decomposition (Headline / Abstract / Lead / Body)  —  same-article subset, 12mo MA",
        fontsize=11, y=0.995)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved: {out_path}")


# ────────────────────────────────────────────────────────────────────────
# Decade-summary for a sanity check
# ────────────────────────────────────────────────────────────────────────
def decade_summary(layer_df: pd.DataFrame) -> pd.DataFrame:
    df = layer_df.copy()
    df["decade"] = (df["pub_month"].dt.year // 10) * 10
    out_rows = []
    for dec, g in df.groupby("decade"):
        row = {"decade": int(dec), "n_months": len(g)}
        for dim in ["threat", "diplo", "human"]:
            for layer in ["hl", "ab", "ld", "ft"]:
                row[f"{layer}_{dim}"] = g[f"{layer}_{dim}"].mean()
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def main():
    print("Loading enriched parquet…")
    df = pd.read_parquet(DATA_PARQUET)
    print(f"  {len(df):,} articles")

    print("\nE_M1_2 — per-voice raw trajectories")
    voice_df = build_voice_series(df)
    voice_df.to_csv(OUT_DIR / "voice_raw_series.csv", index=False)
    plot_voice_raw(voice_df, FIG_DIR / "voice_raw_trajectories.png")
    print(f"  rows: {len(voice_df)}  voices: {voice_df['voice'].unique().tolist()}")
    print("  per-voice grand-mean density (whole period):")
    print(voice_df.groupby("voice")[["threat","diplo","human"]].mean().round(2))

    print("\nE_M1_1 — 4-layer decomposition")
    layer_df = build_layer_series(df)
    layer_df.to_csv(OUT_DIR / "layer_series.csv", index=False)
    plot_layer_decomp(layer_df, FIG_DIR / "layer_decomposition.png")
    print(f"  rows: {len(layer_df)}")
    print("\n  decade-mean by layer (THREAT density):")
    dec = decade_summary(layer_df)
    print(dec[["decade","hl_threat","ab_threat","ld_threat","ft_threat"]].to_string(index=False))
    print("\n  decade-mean by layer (DIPLO density):")
    print(dec[["decade","hl_diplo","ab_diplo","ld_diplo","ft_diplo"]].to_string(index=False))
    print("\n  decade-mean by layer (HUMAN density):")
    print(dec[["decade","hl_human","ab_human","ld_human","ft_human"]].to_string(index=False))


if __name__ == "__main__":
    main()
