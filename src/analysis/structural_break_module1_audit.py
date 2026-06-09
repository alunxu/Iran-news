#!/usr/bin/env python3
"""
Structural Break — Module 1 self-audit (C1, C2, C3 fixes)
==========================================================

Three corrections to the original Module 1 monthly aggregation:

  C1  Same-article HL-FT comparison.  The original computed `threat_HL` over
      ALL articles in a month and `threat_FT` over the subset with bodies.
      Those two populations differ (Wayback coverage is decade-stratified),
      so the gap conflates a selection effect with a real framing tension.
      → Restrict both sides to articles with `fulltext_word_count > 0`.

  C2  Article-length diagnostic.  The non-stationary upward trend in the
      HL-body threat gap (V3) could partly be a consequence of growing
      article length (more body tokens dilute or concentrate density
      differently).  We report median fulltext_word_count per decade.

  C3  Pooled vs mean-of-ratios.  Headlines are short (~7 words) so 1 hit
      becomes ~140 per 1000.  Mean-of-ratios over-weights short-headline
      noise.  Length-weighted pooled aggregation:
                 sum(hits) / sum(tokens) * 1000
      is the alternative.  We compute both side by side.

Outputs
-------
  data/structural_break/monthly_series_v2.csv         (corrected series)
  data/structural_break/article_length_diagnostic.csv (C2 table)
  figures/structural_break/atlas_comparison.png       (3-way comparison)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "iran_articles_full.parquet"
LEX_PATH     = PROJECT_ROOT / "data" / "lexicons.json"
OLD_SERIES   = PROJECT_ROOT / "data" / "structural_break" / "monthly_series.csv"
SB_DIR       = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR      = PROJECT_ROOT / "figures" / "structural_break"

ACCENT = "#B85042"; COOL = "#2F5F5D"; CHAR = "#363A3E"; MUTED = "#707070"
SAGE   = "#A7BEAE"

MIN_ART_HL_FT     = 5
MIN_ART_PER_VOICE = 2
SMOOTH_WINDOW     = 12

_TOKEN_RE = re.compile(r"\b[a-zA-Z]+\b")


# ─────────────────────────────────────────────────────────────────
# Per-article hits + token counts (the building block for both
# mean-of-ratios and pooled aggregation)
# ─────────────────────────────────────────────────────────────────
def tokenize_and_count(text: str | None, lex_set: set[str]
                      ) -> tuple[int, int]:
    """Return (hits, n_tokens)."""
    if not text:
        return 0, 0
    toks = _TOKEN_RE.findall(text.lower())
    if not toks:
        return 0, 0
    hits = sum(1 for t in toks if t in lex_set)
    return hits, len(toks)


def compute_article_features(df: pd.DataFrame, lex: dict) -> pd.DataFrame:
    threat = set(lex["threat"])
    diplo  = set(lex["diplomacy"])
    human  = set(lex["humanizing"])

    print("  tokenizing headlines + bodies ...", flush=True)
    for src_col, dest_prefix in [("headline", "HL"), ("fulltext", "FT")]:
        hits_t, hits_d, hits_h, nt = [], [], [], []
        for text in df[src_col]:
            n_toks_local = None
            ht, n = tokenize_and_count(text, threat); hits_t.append(ht)
            n_toks_local = n
            hd, _ = tokenize_and_count(text, diplo);  hits_d.append(hd)
            hh, _ = tokenize_and_count(text, human);  hits_h.append(hh)
            nt.append(n_toks_local)
        df[f"hits_threat_{dest_prefix}"] = hits_t
        df[f"hits_diplo_{dest_prefix}"]  = hits_d
        df[f"hits_human_{dest_prefix}"]  = hits_h
        df[f"n_tokens_{dest_prefix}"]    = nt

    return df


# ─────────────────────────────────────────────────────────────────
# Aggregation: C1 (same-article subset) + both mean & pooled (C3)
# ─────────────────────────────────────────────────────────────────
def aggregate_monthly_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["year", "month"]).copy()
    df["pub_month"] = pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" +
        df["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    df["decade"] = (df["year"] // 10).astype(int) * 10

    has_body = df["fulltext_word_count"].fillna(0) > 0
    df_body  = df[has_body]
    print(f"  articles with body: {len(df_body):,} / {len(df):,} "
          f"({len(df_body)/len(df)*100:.1f}%)")

    out_rows = {}

    # Volumes
    out_rows["vol_total"]     = df.groupby("pub_month").size()
    out_rows["vol_with_text"] = df_body.groupby("pub_month").size()

    # ── HL-FT gap (C1: both on body-having articles) ──
    for cat, lex_name in [("threat", "threat"), ("diplo", "diplo"), ("human", "human")]:
        # Mean-of-ratios (legacy approach)
        # density per article = hits / max(1, n_tokens) * 1000
        for comp in ["HL", "FT"]:
            hits  = df_body[f"hits_{lex_name}_{comp}"]
            ntoks = df_body[f"n_tokens_{comp}"].replace(0, np.nan)
            df_body[f"d_{lex_name}_{comp}"] = (1000.0 * hits / ntoks).fillna(0)

        mean_HL = df_body.groupby("pub_month")[f"d_{lex_name}_HL"].mean()
        mean_FT = df_body.groupby("pub_month")[f"d_{lex_name}_FT"].mean()
        out_rows[f"gap_{lex_name}_HL_FT_mean_C1"] = mean_HL - mean_FT

        # Pooled: sum(hits) / sum(tokens) * 1000
        def pooled(grp, cat=lex_name, comp="HL"):
            tot_n = grp[f"n_tokens_{comp}"].sum()
            return 1000.0 * grp[f"hits_{cat}_{comp}"].sum() / max(1, tot_n)

        pooled_HL = df_body.groupby("pub_month").apply(lambda g: pooled(g, comp="HL"))
        pooled_FT = df_body.groupby("pub_month").apply(lambda g: pooled(g, comp="FT"))
        out_rows[f"gap_{lex_name}_HL_FT_pooled_C1"] = pooled_HL - pooled_FT

    # ── News-editorial gap (within body-having articles, by voice) ──
    df_news = df_body[df_body["voice"] == "news"]
    df_ed   = df_body[df_body["voice"] == "editorial"]
    n_news  = df_news.groupby("pub_month").size().rename("n_news")
    n_ed    = df_ed.groupby("pub_month").size().rename("n_editorial")
    out_rows["n_news"]      = n_news
    out_rows["n_editorial"] = n_ed

    for cat in ["threat", "diplo"]:
        # Mean-of-ratios
        mean_news = df_news.groupby("pub_month")[f"d_{cat}_FT"].mean()
        mean_ed   = df_ed.groupby("pub_month")[f"d_{cat}_FT"].mean()
        out_rows[f"gap_{cat}_news_ed_mean_C1"] = mean_news - mean_ed

        # Pooled
        pooled_news = df_news.groupby("pub_month").apply(
            lambda g: 1000.0 * g[f"hits_{cat}_FT"].sum() / max(1, g["n_tokens_FT"].sum())
        )
        pooled_ed = df_ed.groupby("pub_month").apply(
            lambda g: 1000.0 * g[f"hits_{cat}_FT"].sum() / max(1, g["n_tokens_FT"].sum())
        )
        out_rows[f"gap_{cat}_news_ed_pooled_C1"] = pooled_news - pooled_ed

    monthly = pd.concat(out_rows.values(), axis=1, keys=out_rows.keys()).sort_index()
    monthly.index.name = "pub_month"

    # Apply min-N filters
    hl_ft_cols = [c for c in monthly.columns if "HL_FT" in c]
    monthly.loc[monthly["vol_with_text"].fillna(0) < MIN_ART_HL_FT, hl_ft_cols] = np.nan

    voice_cols = [c for c in monthly.columns if "news_ed" in c]
    voice_mask = ((monthly["n_news"].fillna(0) < MIN_ART_PER_VOICE) |
                  (monthly["n_editorial"].fillna(0) < MIN_ART_PER_VOICE))
    monthly.loc[voice_mask, voice_cols] = np.nan

    return monthly


# ─────────────────────────────────────────────────────────────────
# C2: Article length diagnostic per decade
# ─────────────────────────────────────────────────────────────────
def article_length_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["year"]).copy()
    df["decade"] = (df["year"] // 10).astype(int) * 10
    has_body = df["fulltext_word_count"].fillna(0) > 0
    df_body = df[has_body]

    diag = df_body.groupby("decade")["fulltext_word_count"].agg(
        n="count",
        mean="mean",
        median="median",
        p10=lambda x: x.quantile(0.10),
        p90=lambda x: x.quantile(0.90),
    ).round(0).astype(int)

    # Headline lengths too
    hl_diag = df.groupby("decade")["headline_word_count"].median().round(1).rename("headline_median")
    diag = diag.join(hl_diag)
    return diag


# ─────────────────────────────────────────────────────────────────
# Comparison plot: old (v1) vs new (C1-mean) vs new (C1-pooled)
# ─────────────────────────────────────────────────────────────────
def plot_comparison(monthly_v1: pd.DataFrame, monthly_v2: pd.DataFrame,
                    out_path: Path) -> None:
    """Three series per panel: original vs C1+mean vs C1+pooled."""

    # 5 tension series mapping (v1 col → v2 cols)
    panels = [
        ("gap_threat_HL_FT",   "Headline−body THREAT gap",     ACCENT,
         "gap_threat_HL_FT_mean_C1",   "gap_threat_HL_FT_pooled_C1"),
        ("gap_diplo_HL_FT",    "Headline−body DIPLOMACY gap",  COOL,
         "gap_diplo_HL_FT_mean_C1",    "gap_diplo_HL_FT_pooled_C1"),
        ("gap_human_HL_FT",    "Headline−body HUMANIZING gap", CHAR,
         "gap_human_HL_FT_mean_C1",    "gap_human_HL_FT_pooled_C1"),
        ("gap_threat_news_ed", "News−editorial THREAT gap",    ACCENT,
         "gap_threat_news_ed_mean_C1", "gap_threat_news_ed_pooled_C1"),
        ("gap_diplo_news_ed",  "News−editorial DIPLOMACY gap", COOL,
         "gap_diplo_news_ed_mean_C1",  "gap_diplo_news_ed_pooled_C1"),
    ]

    fig, axes = plt.subplots(
        nrows=5, ncols=1, sharex=True, figsize=(13, 11.5), dpi=150,
        gridspec_kw={"hspace": 0.38, "top": 0.945, "bottom": 0.05,
                     "left": 0.07, "right": 0.985},
    )

    fig.text(0.07, 0.985,
        "Module 1 self-audit: original (v1) vs same-article fix (C1) vs C1+pooled (C1+C3)",
        fontsize=14.5, fontweight="bold", ha="left", va="top", color=CHAR)
    fig.text(0.07, 0.964,
        "Thick = 12-month rolling mean of each variant · "
        "v1 = original buggy mix · C1 = same-article subset · "
        "C1+C3 = length-weighted pooled",
        fontsize=9.5, color=MUTED, ha="left", va="top")

    def smooth(s):
        return s.rolling(SMOOTH_WINDOW, center=True, min_periods=4).mean()

    for i, (ax, (v1_col, title, color, mean_col, pooled_col)) in enumerate(zip(axes, panels)):
        v1 = monthly_v1[v1_col]
        v2_mean = monthly_v2[mean_col]
        v2_pooled = monthly_v2[pooled_col]

        ax.plot(smooth(v1).index, smooth(v1).values,
                color=MUTED, linewidth=1.4, linestyle=":", label="v1 (original)")
        ax.plot(smooth(v2_mean).index, smooth(v2_mean).values,
                color=color, linewidth=1.8, label="C1 (same-article, mean)")
        ax.plot(smooth(v2_pooled).index, smooth(v2_pooled).values,
                color=color, linewidth=1.4, linestyle="--", alpha=0.85,
                label="C1+C3 (same-article, pooled)")
        ax.axhline(0, color=MUTED, linewidth=0.4, linestyle=":", zorder=0)
        ax.set_title(title, fontsize=10.5, loc="left", color=CHAR, pad=4)
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=9, colors=CHAR)
        ax.margins(x=0.005)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
        if i != 4:
            ax.tick_params(axis="x", labelbottom=False)

    bottom = axes[-1]
    bottom.xaxis.set_major_locator(mdates.YearLocator(5))
    bottom.xaxis.set_minor_locator(mdates.YearLocator(1))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    bottom.set_xlabel("Year", fontsize=10, color=CHAR)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    SB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  {len(df):,} articles")

    print(f"Loading {LEX_PATH} ...")
    with LEX_PATH.open() as f:
        lex = json.load(f)

    df = compute_article_features(df, lex)

    print("\n=== C2: Article length per decade ===")
    diag = article_length_diagnostic(df)
    print(diag.to_string())
    diag.to_csv(SB_DIR / "article_length_diagnostic.csv")
    print(f"  saved → {SB_DIR / 'article_length_diagnostic.csv'}")

    print("\nBuilding monthly series with C1 + both mean & pooled ...")
    monthly_v2 = aggregate_monthly_v2(df)
    print(f"  {len(monthly_v2)} months")
    monthly_v2.to_csv(SB_DIR / "monthly_series_v2.csv", index_label="pub_month")
    print(f"  saved → {SB_DIR / 'monthly_series_v2.csv'}")

    print("\nNon-null cell counts per series:")
    for col in [c for c in monthly_v2.columns if c.startswith("gap_")]:
        print(f"  {col:42s}  {monthly_v2[col].notna().sum():>4} months")

    # Coverage comparison
    print("\n=== Coverage diff vs original v1 ===")
    monthly_v1 = pd.read_csv(OLD_SERIES, parse_dates=["pub_month"]).set_index("pub_month")
    for v1_col, _, _, mean_col, _ in [
        ("gap_threat_HL_FT",   "", "", "gap_threat_HL_FT_mean_C1",   ""),
        ("gap_diplo_HL_FT",    "", "", "gap_diplo_HL_FT_mean_C1",    ""),
        ("gap_human_HL_FT",    "", "", "gap_human_HL_FT_mean_C1",    ""),
        ("gap_threat_news_ed", "", "", "gap_threat_news_ed_mean_C1", ""),
        ("gap_diplo_news_ed",  "", "", "gap_diplo_news_ed_mean_C1",  ""),
    ]:
        n_v1 = monthly_v1[v1_col].notna().sum()
        n_v2 = monthly_v2[mean_col].notna().sum()
        diff = n_v2 - n_v1
        print(f"  {v1_col:25s}  v1={n_v1:>4}  C1={n_v2:>4}  Δ={diff:+d}")

    print("\nPlotting comparison ...")
    plot_comparison(monthly_v1, monthly_v2, FIG_DIR / "atlas_comparison.png")


if __name__ == "__main__":
    main()
