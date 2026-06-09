#!/usr/bin/env python3
"""
V_M1_6 — News-desk confound check.

Tests whether the HL-body THREAT gap trajectory is driven by a CHANGE in WHICH
desk covers Iran (e.g. Foreign Desk in 1980s → mixed Foreign+Politics+Washington
in 2010s) rather than a change in framing behavior within any single desk.

Pipeline:
  1. Normalize desk strings (Foreign Desk / Foreign → "foreign", etc.).
  2. Report decade × desk volume matrix.
  3. Compute decade-level HL-body THREAT gap restricted to Foreign-only — the
     desk with longest continuous coverage (1980s-2020s).
  4. Compare to all-desks gap (same-article subset).

Outputs:
  data/structural_break/desk_decade_volume.csv
  data/structural_break/desk_confound_summary.csv
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "iran_articles_full.parquet"
LEX_PATH = PROJECT_ROOT / "data" / "lexicons.json"
OUT_DIR = PROJECT_ROOT / "data" / "structural_break"

LEX = json.load(open(LEX_PATH))
THREAT_PAT = re.compile(r"\b(" + "|".join(re.escape(w) for w in LEX["threat"]) + r")\b",
                        re.IGNORECASE)


def normalize_desk(s):
    if pd.isna(s) or not isinstance(s, str):
        return "unknown"
    s = s.strip().lower()
    if s in ("", "none", "nan"):
        return "unknown"
    if "foreign" in s:
        return "foreign"
    if "editorial" in s or "op-ed" in s or "opinion" in s or "oped" in s:
        return "opinion"
    if "washington" in s or "national" in s or "politics" in s:
        return "national"
    if "business" in s or "financial" in s:
        return "business"
    if "metropol" in s or "metro" in s:
        return "metro"
    if "weekend" in s or "arts" in s or "culture" in s or "magazine" in s:
        return "culture"
    return "other"


def density(text: str, pat) -> float | None:
    if not isinstance(text, str) or not text.strip():
        return None
    n = len(text.split())
    if n == 0:
        return None
    return len(pat.findall(text)) / n * 1000.0


def main():
    df = pd.read_parquet(DATA)
    df["desk_norm"] = df["news_desk"].apply(normalize_desk)
    df["decade"] = (pd.to_datetime(df["pub_date"]).dt.year // 10) * 10

    # 1. Decade × desk volume
    vol = (df.groupby(["decade", "desk_norm"]).size().unstack(fill_value=0)
             .sort_index(axis=1))
    vol.to_csv(OUT_DIR / "desk_decade_volume.csv")
    print("=== Desk volume by decade ===")
    print(vol)
    print()
    print("=== Desk share (%) by decade ===")
    share = vol.div(vol.sum(axis=1), axis=0) * 100
    print(share.round(1))
    print()

    # 2. Same-article subset for gap computation
    same = df[df["fulltext_word_count"] > 0].copy()
    same["hl_threat"] = same["headline"].apply(lambda t: density(t, THREAT_PAT))
    same["ft_threat"] = same["fulltext"].apply(lambda t: density(t, THREAT_PAT))
    same["gap_threat"] = same["hl_threat"] - same["ft_threat"]

    # 3. Foreign-only gap by decade vs all-desks gap by decade
    decade_summary = []
    for dec, g in same.groupby("decade"):
        all_mean = g["gap_threat"].mean()
        all_n = len(g)
        foreign = g[g["desk_norm"] == "foreign"]
        foreign_mean = foreign["gap_threat"].mean() if len(foreign) >= 30 else np.nan
        foreign_n = len(foreign)
        opinion = g[g["desk_norm"] == "opinion"]
        opinion_mean = opinion["gap_threat"].mean() if len(opinion) >= 30 else np.nan
        opinion_n = len(opinion)
        decade_summary.append(dict(
            decade=int(dec),
            n_all=all_n,
            gap_all=all_mean,
            n_foreign=foreign_n,
            gap_foreign=foreign_mean,
            n_opinion=opinion_n,
            gap_opinion=opinion_mean,
        ))
    summary = pd.DataFrame(decade_summary)
    summary.to_csv(OUT_DIR / "desk_confound_summary.csv", index=False)
    print("=== HL-body THREAT gap by decade × desk ===")
    print(summary.round(2).to_string(index=False))
    print()

    # 4. Quick stability test: if "all-desks" and "foreign-only" trajectories track each other,
    # the gap growth isn't a desk-mix artifact.
    valid = summary.dropna(subset=["gap_all", "gap_foreign"])
    if len(valid) >= 3:
        corr_all_foreign = valid["gap_all"].corr(valid["gap_foreign"])
        diff_all_foreign = (valid["gap_all"] - valid["gap_foreign"]).abs().mean()
        print(f"Decade-level corr(all-desks, foreign-only): r = {corr_all_foreign:.3f}")
        print(f"Mean |all − foreign| per decade: {diff_all_foreign:.2f}")
        print()
        if corr_all_foreign > 0.85:
            print("VERDICT: Desk mix is NOT a primary driver — foreign-only and all-desks")
            print("         gap trajectories are highly correlated across decades.")
        else:
            print("VERDICT: Desk mix may be confounding — divergence between foreign-only")
            print("         and all-desks gap trajectories.")

    print()
    print(f"Saved: {OUT_DIR / 'desk_decade_volume.csv'}")
    print(f"Saved: {OUT_DIR / 'desk_confound_summary.csv'}")


if __name__ == "__main__":
    main()
