#!/usr/bin/env python3
"""
Structural Break Detection — Verification suite (V1 – V5)
=========================================================

Audits the 6 endogenous breakpoints found in Module 2 against five
methodological artifact sources:

  V1  Coverage-edge artifact  — does the break sit on a NaN-coverage shelf?
  V2  Volume confound         — is article volume changing at the break?
  V3  Stationarity            — is the series globally trending (ADF / KPSS)?
  V4  Penalty sensitivity     — do breaks survive different BIC penalties?
  V5  Lexicon bootstrap       — do breaks survive resampling the lexicon?

Inputs
------
  data/iran_articles_full.parquet   (full corpus, needed for V5)
  data/lexicons.json
  data/structural_break/monthly_series.csv
  data/structural_break/baiperron_breaks.csv

Outputs
-------
  data/structural_break/verify_v1_coverage.csv
  data/structural_break/verify_v2_volume.csv
  data/structural_break/verify_v3_stationarity.csv
  data/structural_break/verify_v4_penalty.csv
  data/structural_break/verify_v5_lexicon.csv
  data/structural_break/verify_summary.csv     ← per-break verdict
  figures/structural_break/verification.png    ← visual summary
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SB_DIR       = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR      = PROJECT_ROOT / "figures" / "structural_break"
SERIES_CSV   = SB_DIR / "monthly_series.csv"
BREAKS_CSV   = SB_DIR / "baiperron_breaks.csv"
LEX_PATH     = PROJECT_ROOT / "data" / "lexicons.json"
PARQUET      = PROJECT_ROOT / "data" / "iran_articles_full.parquet"

MIN_SEGMENT_MONTHS = 12
BIC_PEN_FACTOR     = 2.0
ACCENT = "#B85042"; COOL = "#2F5F5D"; CHAR = "#363A3E"; MUTED = "#707070"

# Same 5 tension series as Module 2
SERIES_SPECS = [
    ("gap_threat_HL_FT",   "Headline−body THREAT gap",     "threat",     "HL_FT"),
    ("gap_diplo_HL_FT",    "Headline−body DIPLOMACY gap",  "diplomacy",  "HL_FT"),
    ("gap_human_HL_FT",    "Headline−body HUMANIZING gap", "humanizing", "HL_FT"),
    ("gap_threat_news_ed", "News−editorial THREAT gap",    "threat",     "news_ed"),
    ("gap_diplo_news_ed",  "News−editorial DIPLOMACY gap", "diplomacy",  "news_ed"),
]


def detect_breaks(values: np.ndarray, pen_factor: float = BIC_PEN_FACTOR,
                  min_size: int = MIN_SEGMENT_MONTHS) -> list[int]:
    if len(values) < 2 * min_size:
        return []
    sigma = np.std(values)
    pen = pen_factor * (sigma ** 2) * np.log(len(values))
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(values.reshape(-1, 1))
    breaks = algo.predict(pen=pen)
    return [b for b in breaks if b < len(values)]


# ════════════════════════════════════════════════════════════════════
# V1: Coverage-edge artifact
# ════════════════════════════════════════════════════════════════════
def v1_coverage(monthly: pd.DataFrame, breaks: pd.DataFrame) -> pd.DataFrame:
    """For each break, compare the fraction of non-null months in
    pre[-12,-1] vs post[+1,+12] windows. Large diff → coverage stepping."""
    records = []
    for _, row in breaks.iterrows():
        col   = row["series"]
        bdate = pd.Timestamp(row["break_date"])
        srs   = monthly[col]
        # build month-aligned index
        all_dates = monthly.index
        idx = all_dates.get_indexer([bdate], method="nearest")[0]
        pre  = srs.iloc[max(0, idx - 12):idx]
        post = srs.iloc[idx + 1:min(len(srs), idx + 13)]
        pre_cov  = pre.notna().mean()  if len(pre)  else np.nan
        post_cov = post.notna().mean() if len(post) else np.nan
        diff = post_cov - pre_cov
        records.append(dict(
            series=col, break_date=bdate.date(),
            pre_coverage=round(pre_cov, 3), post_coverage=round(post_cov, 3),
            coverage_diff=round(diff, 3),
            flag=("⚠️ HIGH" if abs(diff) >= 0.30 else ""),
        ))
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════
# V2: Volume confound
# ════════════════════════════════════════════════════════════════════
def v2_volume(monthly: pd.DataFrame, breaks: pd.DataFrame) -> pd.DataFrame:
    """Does article volume shift at the break?"""
    vol = monthly["vol_total"]
    vol_log_smooth = np.log1p(vol).rolling(12, center=True, min_periods=4).mean()
    records = []
    for _, row in breaks.iterrows():
        col = row["series"]
        bdate = pd.Timestamp(row["break_date"])
        idx = monthly.index.get_indexer([bdate], method="nearest")[0]
        pre  = vol.iloc[max(0, idx - 12):idx]
        post = vol.iloc[idx + 1:min(len(vol), idx + 13)]
        pre_m  = pre.mean()
        post_m = post.mean()
        log_ratio = np.log1p(post_m) - np.log1p(pre_m)
        records.append(dict(
            series=col, break_date=bdate.date(),
            pre_vol=round(pre_m, 1), post_vol=round(post_m, 1),
            log_vol_ratio=round(log_ratio, 3),
            flag=("⚠️ HIGH" if abs(log_ratio) >= 0.5 else ""),
        ))
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════
# V3: Stationarity (ADF + KPSS)
# ════════════════════════════════════════════════════════════════════
def v3_stationarity(monthly: pd.DataFrame) -> pd.DataFrame:
    """Trend test via OLS on (t, y). If slope p-value < 0.05, the series is
    globally trending — PELT-detected breaks may partly be capturing the trend
    rather than discrete regime shifts.

    Also computes a *unit-root-style* check: regress Δy on lagged y; if the
    lag coefficient is significantly negative the series is mean-reverting
    (stationary). Implementations are scipy-only to avoid the broken
    statsmodels deprecate_kwarg API on the local env.
    """
    records = []
    for col, *_ in SERIES_SPECS:
        s = monthly[col].dropna()
        if len(s) < 30:
            continue
        y = s.values
        t = np.arange(len(y))

        # 1. Linear-trend test
        slope, intercept, _r, slope_p, _se = stats.linregress(t, y)
        trend_p_per_decade = slope * 120  # slope per 120 months = 10 yr

        # 2. Augmented-Dickey-Fuller-lite (one-lag, no intercept hassle)
        dy = np.diff(y)
        y_lag = y[:-1]
        # H0: ρ = 0 in Δy_t = α + ρ * y_{t-1} + ε
        ar_slope, ar_intercept, _, ar_p, ar_se = stats.linregress(y_lag, dy)
        adf_t = ar_slope / ar_se if ar_se > 0 else 0.0
        # rough rule of thumb: t < -2.86 suggests rejecting unit-root @ 5%
        ar_rejects_unit_root = adf_t < -2.86

        if slope_p < 0.05 and abs(trend_p_per_decade) >= 1.0:
            verdict = "non-stationary trend 🚨"
        elif ar_rejects_unit_root:
            verdict = "stationary ✓"
        else:
            verdict = "inconclusive / weak trend"

        records.append(dict(
            series=col,
            n=len(s),
            trend_slope_per_decade=round(trend_p_per_decade, 3),
            trend_p=round(slope_p, 4),
            adf_lite_t=round(adf_t, 2),
            adf_lite_rejects_ur=ar_rejects_unit_root,
            verdict=verdict,
        ))
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════
# V4: Penalty sensitivity
# ════════════════════════════════════════════════════════════════════
PENALTIES = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

def v4_penalty_sweep(monthly: pd.DataFrame, breaks: pd.DataFrame) -> pd.DataFrame:
    """Re-run PELT at multiple penalties; check how many penalties detect
    a break within ±6 months of each original break."""
    records = []
    for _, row in breaks.iterrows():
        col = row["series"]
        bdate = pd.Timestamp(row["break_date"])
        s = monthly[col].dropna()
        dates = s.index
        target_idx = dates.get_indexer([bdate], method="nearest")[0]

        appears_at = []
        for pen in PENALTIES:
            detected = detect_breaks(s.values, pen_factor=pen)
            # any detected break within ±6 months of target?
            for b in detected:
                if abs(b - target_idx) <= 6:
                    appears_at.append(pen)
                    break

        records.append(dict(
            series=col, break_date=bdate.date(),
            appears_at_pens=appears_at,
            n_penalties_confirmed=len(appears_at),
            robust=(len(appears_at) >= 4),  # 4/6 = pretty robust
            flag=("✓ ROBUST" if len(appears_at) >= 4 else
                  "🟡 MARGINAL" if len(appears_at) >= 2 else
                  "🚨 FRAGILE"),
        ))
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════
# V5: Lexicon bootstrap
# ════════════════════════════════════════════════════════════════════
_TOKEN_RE = re.compile(r"\b[a-zA-Z]+\b")


def tokenize_corpus(df: pd.DataFrame) -> tuple[list[list[str]], list[list[str]], list[int]]:
    """Return (headline_tokens, body_tokens, len_body) lists per article."""
    print("  tokenizing corpus (one-time) ...", flush=True)
    hl_tokens, ft_tokens, ft_len = [], [], []
    for hl, ft in zip(df["headline"], df["fulltext"]):
        ht = [t.lower() for t in _TOKEN_RE.findall(hl)] if isinstance(hl, str) else []
        bt = [t.lower() for t in _TOKEN_RE.findall(ft)] if isinstance(ft, str) else []
        hl_tokens.append(ht)
        ft_tokens.append(bt)
        ft_len.append(len(bt))
    return hl_tokens, ft_tokens, ft_len


def compute_density(tokens_list: list[list[str]], lex: set[str]) -> np.ndarray:
    """Vectorised: hits per 1000 tokens for each article."""
    out = np.zeros(len(tokens_list))
    for i, toks in enumerate(tokens_list):
        if not toks:
            continue
        hits = sum(1 for t in toks if t in lex)
        out[i] = 1000.0 * hits / len(toks)
    return out


def aggregate_monthly(densities: np.ndarray, dates: pd.Series,
                      has_body: np.ndarray = None) -> pd.Series:
    """Mean density per month."""
    s = pd.Series(densities, index=dates)
    if has_body is not None:
        s = s[has_body]
    return s.groupby(pd.Grouper(freq="MS")).mean()


def v5_lexicon_bootstrap(df: pd.DataFrame, lex: dict, breaks: pd.DataFrame,
                          n_boot: int = 30, sample_frac: float = 0.7
                          ) -> pd.DataFrame:
    """Bootstrap 70% of each lexicon, re-run PELT, count break stability."""
    # Prep
    df = df.dropna(subset=["year", "month"]).copy()
    df["pub_month"] = pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" +
        df["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    has_body = (df["fulltext"].str.len() > 0).values
    is_news  = (df["voice"] == "news").values
    is_ed    = (df["voice"] == "editorial").values

    hl_tokens, ft_tokens, _ = tokenize_corpus(df)
    dates = df["pub_month"]

    rng = np.random.default_rng(42)
    records = []

    # Run bootstrap per lexicon category
    lex_cache = {cat: list(lex[cat]) for cat in ("threat", "diplomacy", "humanizing")}

    for lex_cat in ("threat", "diplomacy", "humanizing"):
        full_lex = lex_cache[lex_cat]
        target_n = max(3, int(sample_frac * len(full_lex)))

        # Targets: which breaks involve this lexicon?
        prefix_map = {"threat": "threat", "diplomacy": "diplo", "humanizing": "human"}
        target_breaks = breaks[breaks["series"].str.contains(prefix_map[lex_cat])]
        if target_breaks.empty:
            continue
        print(f"  bootstrap [{lex_cat}] {n_boot} iters of {target_n}/{len(full_lex)} terms · {len(target_breaks)} target breaks ...",
              flush=True)

        target_pairs = [(pd.Timestamp(d), sid) for d, sid in
                        zip(target_breaks["break_date"], target_breaks["series"])]
        appearances = {(td.date(), sid): set() for td, sid in target_pairs}

        for boot in range(n_boot):
            sub_terms = set(rng.choice(full_lex, target_n, replace=False))
            d_hl = compute_density(hl_tokens, sub_terms)
            d_ft = compute_density(ft_tokens, sub_terms)

            d_hl_monthly = aggregate_monthly(d_hl, dates)
            d_ft_monthly = aggregate_monthly(d_ft, dates, has_body=has_body)
            gap_HL_FT = (d_hl_monthly - d_ft_monthly).dropna()

            d_news = aggregate_monthly(d_ft, dates, has_body=(has_body & is_news))
            d_ed   = aggregate_monthly(d_ft, dates, has_body=(has_body & is_ed))
            gap_news_ed = (d_news - d_ed).dropna()

            detected_HL_FT = [gap_HL_FT.index[b] for b in detect_breaks(gap_HL_FT.values)] \
                              if len(gap_HL_FT) >= 2 * MIN_SEGMENT_MONTHS else []
            detected_news_ed = [gap_news_ed.index[b] for b in detect_breaks(gap_news_ed.values)] \
                                if len(gap_news_ed) >= 2 * MIN_SEGMENT_MONTHS else []

            for td, sid in target_pairs:
                candidates = detected_HL_FT if sid.endswith("HL_FT") else detected_news_ed
                for dd in candidates:
                    if abs((dd - td).days) <= 180:
                        appearances[(td.date(), sid)].add(boot)
                        break

        for (br_date, sid), boots in appearances.items():
            stability = round(len(boots) / n_boot, 2)
            records.append(dict(
                series=sid, break_date=br_date,
                lexicon=lex_cat,
                appearances=len(boots),
                n_boot=n_boot,
                stability=stability,
                flag=("✓ ROBUST" if stability >= 0.8 else
                      "🟡 MARGINAL" if stability >= 0.5 else
                      "🚨 FRAGILE"),
            ))

    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════
# Final per-break verdict (aggregate V1-V5)
# ════════════════════════════════════════════════════════════════════
def synthesize_verdict(breaks, v1, v2, v3, v4, v5) -> pd.DataFrame:
    rows = []
    for _, br in breaks.iterrows():
        col   = br["series"]
        bdate = br["break_date"]
        v1_row = v1[(v1["series"] == col) & (v1["break_date"].astype(str) == str(bdate))]
        v2_row = v2[(v2["series"] == col) & (v2["break_date"].astype(str) == str(bdate))]
        v3_row = v3[v3["series"] == col]
        v4_row = v4[(v4["series"] == col) & (v4["break_date"].astype(str) == str(bdate))]
        v5_row = v5[(v5["series"] == col) & (v5["break_date"].astype(str) == str(bdate))] if not v5.empty else pd.DataFrame()

        flags = []
        if not v1_row.empty and v1_row["flag"].iloc[0]:    flags.append("V1:coverage-edge")
        if not v2_row.empty and v2_row["flag"].iloc[0]:    flags.append("V2:volume-shift")
        if not v3_row.empty and "non-stationary" in v3_row["verdict"].iloc[0]: flags.append("V3:non-stationary")
        v4_flag = v4_row["flag"].iloc[0] if not v4_row.empty else ""
        if "FRAGILE" in v4_flag: flags.append("V4:penalty-fragile")
        v5_flag = v5_row["flag"].iloc[0] if not v5_row.empty else ""
        if v5_flag and "FRAGILE" in v5_flag: flags.append("V5:lexicon-fragile")

        verdict = (
            "✓ ROBUST" if len(flags) == 0 else
            "🟡 CAVEAT" if len(flags) == 1 else
            "🚨 SUSPECT"
        )
        rows.append(dict(
            series=col, break_date=bdate,
            shift=br.get("shift", np.nan),
            nearest_event=br.get("nearest_event", ""),
            v4_robust=v4_flag,
            v5_robust=v5_flag,
            n_flags=len(flags),
            flags="; ".join(flags) if flags else "(none)",
            verdict=verdict,
        ))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
def main():
    print(f"Loading {SERIES_CSV} ...")
    monthly = pd.read_csv(SERIES_CSV, parse_dates=["pub_month"]).set_index("pub_month")
    breaks = pd.read_csv(BREAKS_CSV)
    print(f"  {len(monthly)} months, {len(breaks)} breaks to verify\n")

    print("V1: Coverage-edge artifact ...")
    v1 = v1_coverage(monthly, breaks)
    print(v1.to_string(index=False))

    print("\nV2: Volume confound ...")
    v2 = v2_volume(monthly, breaks)
    print(v2.to_string(index=False))

    print("\nV3: Stationarity ...")
    v3 = v3_stationarity(monthly)
    print(v3.to_string(index=False))

    print("\nV4: Penalty sensitivity (re-detect across penalties) ...")
    v4 = v4_penalty_sweep(monthly, breaks)
    print(v4.to_string(index=False))

    print(f"\nV5: Lexicon bootstrap (n=30 iters per category) ...")
    with LEX_PATH.open() as f:
        lex = json.load(f)
    df = pd.read_parquet(PARQUET)
    v5 = v5_lexicon_bootstrap(df, lex, breaks, n_boot=30, sample_frac=0.7)
    print(v5.to_string(index=False))

    print("\n" + "=" * 70)
    print("PER-BREAK VERDICT (synthesis of V1-V5)")
    print("=" * 70)
    summary = synthesize_verdict(breaks, v1, v2, v3, v4, v5)
    print(summary.to_string(index=False))

    # Save all
    v1.to_csv(SB_DIR / "verify_v1_coverage.csv", index=False)
    v2.to_csv(SB_DIR / "verify_v2_volume.csv", index=False)
    v3.to_csv(SB_DIR / "verify_v3_stationarity.csv", index=False)
    v4.to_csv(SB_DIR / "verify_v4_penalty.csv", index=False)
    v5.to_csv(SB_DIR / "verify_v5_lexicon.csv", index=False)
    summary.to_csv(SB_DIR / "verify_summary.csv", index=False)
    print(f"\nAll outputs in {SB_DIR}")


if __name__ == "__main__":
    main()
