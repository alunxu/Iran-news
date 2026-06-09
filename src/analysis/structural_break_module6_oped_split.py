#!/usr/bin/env python3
r"""
Module 6 -- Op-Ed staff vs. guest split.

\S 1 of the notes found that editorials run hotter than news on \scl{threat}
and \scl{diplo} (grand means 19.6 vs.\ 14.2 for \scl{threat}).  The Op-Ed /
column bucket aggregates two very different things:

  (a) NYT staff columnists -- Friedman, Lewis, Safire, Kristof, Stephens etc.
      who carry an institutional named voice and write regularly.
  (b) Guest contributors -- one-off pieces by outside experts, politicians,
      academics, refugees, with no shared institutional commitment.

This module separates them and asks: is the editorial-runs-hot signal
carried by a small set of named staff columnists, or is it institutional?

Buckets:
  - STAFF       byline appears >= 10 times in column / editorial
  - GUEST       byline appears 1..9 times (named but irregular)
  - UNSIGNED    no byline (Editorial Board, the institutional unsigned voice)

Outputs:
  data/structural_break/oped_split_means.csv
  data/structural_break/oped_split_monthly.csv
  figures/structural_break/oped_split.png
  figures/structural_break/voice_divergence.png
  figures/structural_break/voice_event_heatmap.png
  figures/structural_break/voice_jcpoa_window.png
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json
import re
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "iran_articles_full.parquet"
LEX_PATH = PROJECT_ROOT / "data" / "lexicons.json"
OUT_DIR = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR = PROJECT_ROOT / "figures" / "structural_break"

ACCENT, COOL, CHAR, MUTED = "#B85042", "#2F5F5D", "#363A3E", "#707070"
STAFF_COLOR = "#B85042"
GUEST_COLOR = "#9B6A4F"
UNSIGNED_COLOR = COOL
NEWS_COLOR = CHAR

STAFF_THRESHOLD = 10  # bylines with >= 10 pieces are staff
SMOOTH_WINDOW = 12

LABEL_EVENTS = [
    ("2002-01-29", "Axis"),
    ("2013-06-15", "Rouhani"),
    ("2016-01-16", "JCPOA"),
    ("2018-05-08", "Exit"),
    ("2020-01-03", "Soleimani"),
    ("2022-09-16", "Mahsa"),
    ("2023-10-07", "Gaza"),
]

EVENT_TESTS = [
    ("2002-01-29", "Axis", "policy"),
    ("2009-06-13", "Green", "domestic"),
    ("2013-06-15", "Rouhani", "domestic"),
    ("2015-07-14", "JCPOA signed", "diplomacy"),
    ("2016-01-16", "JCPOA impl.", "diplomacy"),
    ("2018-05-08", "JCPOA exit", "diplomacy"),
    ("2020-01-03", "Soleimani", "conflict"),
    ("2022-09-16", "Mahsa", "domestic"),
    ("2023-10-07", "Gaza", "conflict"),
]
WINDOW_MONTHS = 12
EXCLUDE_CENTRAL_MONTHS = 2

LEX = json.load(open(LEX_PATH))
PATTERNS = {
    "threat":  re.compile(r"\b(" + "|".join(re.escape(w) for w in LEX["threat"])     + r")\b", re.IGNORECASE),
    "diplo":   re.compile(r"\b(" + "|".join(re.escape(w) for w in LEX["diplomacy"])  + r")\b", re.IGNORECASE),
    "human":   re.compile(r"\b(" + "|".join(re.escape(w) for w in LEX["humanizing"]) + r")\b", re.IGNORECASE),
}


def density(text: str, pat) -> float | None:
    if not isinstance(text, str) or not text.strip():
        return None
    n = len(text.split())
    if n == 0:
        return None
    return len(pat.findall(text)) / n * 1000.0


def classify_oped_bucket(row, staff_set):
    """Classify each article into news / staff / guest / unsigned / other."""
    voice = row.get("voice", "")
    if voice == "news":
        return "news"
    if voice not in ("column", "editorial"):
        return "other"
    byline = row.get("byline_lastname")
    if pd.isna(byline) or not isinstance(byline, str) or byline.strip() == "":
        return "unsigned"
    if byline in staff_set:
        return "staff"
    return "guest"


def main():
    df = pd.read_parquet(DATA)
    # Restrict to body-having articles (consistent with §1 C1 fix)
    df = df[df["fulltext_word_count"] > 0].copy()

    # Identify staff bylines (>= STAFF_THRESHOLD pieces across column+editorial)
    oped_mask = df["voice"].isin(["column", "editorial"])
    byline_counts = df.loc[oped_mask, "byline_lastname"].dropna().value_counts()
    staff_set = set(byline_counts[byline_counts >= STAFF_THRESHOLD].index)
    print(f"Staff columnists (>= {STAFF_THRESHOLD} pieces): {len(staff_set)}")
    print(f"  Names: {sorted(staff_set)}")

    # Bucket every article
    df["oped_bucket"] = df.apply(lambda r: classify_oped_bucket(r, staff_set), axis=1)
    print(f"  Bucket counts: {df['oped_bucket'].value_counts().to_dict()}")

    # Compute per-article densities on the fulltext field
    for dim, pat in PATTERNS.items():
        df[f"ft_{dim}"] = df["fulltext"].apply(lambda t: density(t, pat))

    # Aggregate by bucket
    buckets = ["staff", "guest", "unsigned", "news"]
    rows = []
    for bkt in buckets:
        sub = df[df["oped_bucket"] == bkt]
        if sub.empty:
            continue
        row = dict(bucket=bkt, n_articles=len(sub),
                   n_unique_bylines=int(sub["byline_lastname"].dropna().nunique()))
        for dim in ["threat", "diplo", "human"]:
            row[f"mean_{dim}"] = float(sub[f"ft_{dim}"].mean())
        rows.append(row)
    means_df = pd.DataFrame(rows)
    means_df.to_csv(OUT_DIR / "oped_split_means.csv", index=False)

    print("\n=== Grand-mean body density by bucket (per 1000 words) ===")
    print(means_df.to_string(index=False))

    # Monthly time series per bucket
    df["pub_month"] = pd.to_datetime(df["pub_date"]).dt.to_period("M").dt.to_timestamp()
    monthly_rows = []
    for (m, bkt), g in df.groupby(["pub_month", "oped_bucket"]):
        if bkt is None or len(g) < 2:
            continue
        monthly_rows.append(dict(
            pub_month=m, bucket=bkt, n=len(g),
            threat=float(g["ft_threat"].mean()) if g["ft_threat"].notna().any() else np.nan,
            diplo=float(g["ft_diplo"].mean())  if g["ft_diplo"].notna().any()  else np.nan,
            human=float(g["ft_human"].mean())  if g["ft_human"].notna().any()  else np.nan,
        ))
    monthly_df = pd.DataFrame(monthly_rows)
    monthly_df.to_csv(OUT_DIR / "oped_split_monthly.csv", index=False)

    # ────────────────────────────────────────────────────────────────────
    # Figure: 3 rows (axis) × 1 col, each shows 4 trajectories
    # ────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), dpi=150, sharex=True)
    bucket_style = {
        "news":     dict(color=NEWS_COLOR,     label="News",       lw=1.2, alpha=0.8),
        "staff":    dict(color=STAFF_COLOR,    label="Staff cols", lw=1.4, alpha=0.95),
        "guest":    dict(color=GUEST_COLOR,    label="Guest ctrb", lw=1.0, alpha=0.8),
        "unsigned": dict(color=UNSIGNED_COLOR, label="Unsigned (Ed Board)", lw=1.2, alpha=0.9),
    }
    axis_specs = [("threat", "THREAT density (per 1000 body words)", ACCENT),
                  ("diplo",  "DIPLOMACY density",                    COOL),
                  ("human",  "HUMANIZING density",                   CHAR)]
    for ax, (dim, ylabel, _) in zip(axes, axis_specs):
        for bkt, style in bucket_style.items():
            sub = monthly_df[monthly_df["bucket"] == bkt].set_index("pub_month")
            if sub.empty or sub[dim].dropna().empty:
                continue
            s = sub[dim].rolling(SMOOTH_WINDOW, min_periods=1).mean()
            ax.plot(s.index, s.values, **style)
        for ed, _ in LABEL_EVENTS:
            ax.axvline(pd.Timestamp(ed), color=MUTED, lw=0.60, alpha=0.36, ls="-", zorder=0)
        ax.set_title(ylabel, fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        if dim == "threat":
            ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=4)
    for idx, (ed, label) in enumerate(LABEL_EVENTS):
        d = mdates.date2num(pd.Timestamp(ed).to_pydatetime())
        axes[0].annotate(
            label,
            xy=(d, 1.01),
            xycoords=axes[0].get_xaxis_transform(),
            xytext=(d, 1.14 + (idx % 2) * 0.09),
            textcoords=axes[0].get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=6.8,
            color=MUTED,
            rotation=42,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.65, alpha=0.75),
            annotation_clip=False,
        )
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Year (12-month moving average)", fontsize=9)
    fig.suptitle(f"Op-Ed split by byline bucket (staff = {len(staff_set)} columnists, "
                 f">= {STAFF_THRESHOLD} pieces; guest = irregular named; unsigned = Editorial Board)",
                 fontsize=10, y=0.995)
    plt.tight_layout()
    fig_path = FIG_DIR / "oped_split.png"
    fig.savefig(fig_path, bbox_inches="tight")

    # ────────────────────────────────────────────────────────────────────
    # Figure: cross-voice divergence index.
    # For each month/frame, smooth each bucket first, then measure the
    # cross-bucket spread. This keeps the slide-level claim temporal:
    # when do NYT voices converge or diverge?
    # ────────────────────────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
    dim_style = {
        "threat": dict(label="threat", color=ACCENT, lw=1.8),
        "diplo": dict(label="diplomacy", color=COOL, lw=1.8),
        "human": dict(label="humanizing", color=CHAR, lw=1.8),
    }
    divergence_rows = []
    for dim, style in dim_style.items():
        pivot = monthly_df.pivot(index="pub_month", columns="bucket", values=dim).sort_index()
        smoothed = pivot.rolling(SMOOTH_WINDOW, min_periods=1).mean()
        available = smoothed[["news", "staff", "guest", "unsigned"]].dropna(thresh=3)
        spread = available.max(axis=1) - available.min(axis=1)
        for m, value in spread.items():
            divergence_rows.append(dict(pub_month=m, frame=dim, voice_spread=float(value)))
        ax.plot(spread.index, spread.values, **style)
    pd.DataFrame(divergence_rows).to_csv(OUT_DIR / "voice_divergence_monthly.csv", index=False)

    for ed, _ in LABEL_EVENTS:
        ax.axvline(pd.Timestamp(ed), color=MUTED, lw=0.75, alpha=0.34, ls="-", zorder=0)
    for idx, (ed, label) in enumerate(LABEL_EVENTS):
        d = mdates.date2num(pd.Timestamp(ed).to_pydatetime())
        ax.annotate(
            label,
            xy=(d, 1.01),
            xycoords=ax.get_xaxis_transform(),
            xytext=(d, 1.13 + (idx % 2) * 0.10),
            textcoords=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7.2,
            color=MUTED,
            rotation=42,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.70, alpha=0.78),
            annotation_clip=False,
        )
    ax.set_title("Cross-voice divergence index", loc="left", fontsize=14, color=CHAR, pad=8)
    ax.set_ylabel("voice spread\n(per 1,000 words; 12-month MA)", fontsize=9)
    ax.set_xlabel("Year", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=3)
    plt.tight_layout()
    div_path = FIG_DIR / "voice_divergence.png"
    fig2.savefig(div_path, bbox_inches="tight")

    # ────────────────────────────────────────────────────────────────────
    # Figure: voice-specific event test.
    # Rows are known historical events; columns are voice × frame cells.
    # Values are post-event mean minus pre-event baseline, so this asks:
    # when an event happens, which institutional voice actually moves?
    # ────────────────────────────────────────────────────────────────────
    def event_shift(series: pd.Series, event_date: pd.Timestamp) -> float:
        pre_start = event_date - pd.DateOffset(months=WINDOW_MONTHS)
        pre_end = event_date - pd.DateOffset(months=EXCLUDE_CENTRAL_MONTHS)
        post_start = event_date + pd.DateOffset(months=EXCLUDE_CENTRAL_MONTHS)
        post_end = event_date + pd.DateOffset(months=WINDOW_MONTHS)
        pre = series.loc[pre_start:pre_end].dropna()
        post = series.loc[post_start:post_end].dropna()
        if len(pre) < 3 or len(post) < 3:
            return np.nan
        return float(post.mean() - pre.mean())

    bucket_names = {
        "news": "News",
        "staff": "Staff cols",
        "guest": "Guest op-eds",
        "unsigned": "Ed board",
    }
    event_rows = []
    for event_date_str, event_label, subtype in EVENT_TESTS:
        event_date = pd.Timestamp(event_date_str)
        for bucket in ["news", "staff", "guest", "unsigned"]:
            sub = monthly_df[monthly_df["bucket"] == bucket].set_index("pub_month").sort_index()
            for dim in ["threat", "diplo"]:
                smoothed = sub[dim].rolling(SMOOTH_WINDOW, min_periods=1).mean()
                event_rows.append(
                    dict(
                        event=event_label,
                        event_date=event_date.date(),
                        subtype=subtype,
                        bucket=bucket,
                        frame=dim,
                        shift=event_shift(smoothed, event_date),
                    )
                )
    event_voice_df = pd.DataFrame(event_rows)
    event_voice_df.to_csv(OUT_DIR / "voice_event_shifts.csv", index=False)

    event_voice_df["cell"] = (
        event_voice_df["bucket"].map(bucket_names)
        + "·"
        + event_voice_df["frame"].map({"threat": "threat", "diplo": "diplomacy"})
    )
    cell_order = [
        f"{bucket_names[b]}·{frame}"
        for b in ["news", "staff", "guest", "unsigned"]
        for frame in ["threat", "diplomacy"]
    ]
    event_order = [label for _, label, _ in EVENT_TESTS]
    heat = (
        event_voice_df.pivot(index="event", columns="cell", values="shift")
        .reindex(event_order)[cell_order]
    )
    fig3, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    vmax = np.nanmax(np.abs(heat.values))
    im = ax.imshow(heat.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cell_order)))
    ax.set_xticklabels(cell_order, rotation=42, ha="right", fontsize=7.4)
    ax.set_yticks(range(len(event_order)))
    ax.set_yticklabels(event_order, fontsize=8.4)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black" if abs(v) < vmax * 0.52 else "white",
                )
    for x in [1.5, 3.5, 5.5]:
        ax.axvline(x, color="black", lw=1.0, alpha=0.65)
    cb = plt.colorbar(im, ax=ax, fraction=0.030, pad=0.012)
    cb.set_label("post − pre shift", fontsize=8)
    ax.set_title(
        "Voice-specific event shifts",
        loc="left",
        fontsize=14,
        color=CHAR,
        pad=8,
    )
    ax.text(
        0,
        -0.95,
        "Chow-style event test: 12-month post-event mean minus 12-month pre-event baseline; central ±2 months excluded.",
        ha="left",
        va="top",
        fontsize=8.2,
        color=MUTED,
        transform=ax.transData,
    )
    plt.tight_layout()
    voice_event_path = FIG_DIR / "voice_event_heatmap.png"
    fig3.savefig(voice_event_path, bbox_inches="tight")

    # ────────────────────────────────────────────────────────────────────
    # Figure: focused JCPOA/nuclear window.
    # This is more presentation-friendly than the full divergence index:
    # it keeps the voice comparison, but anchors it to the strongest
    # regime-shift evidence found elsewhere in the temporal analysis.
    # ────────────────────────────────────────────────────────────────────
    focus_start, focus_end = pd.Timestamp("2012-01-01"), pd.Timestamp("2019-12-31")
    fig4, axes = plt.subplots(2, 1, figsize=(11, 5.2), dpi=150, sharex=True)
    focus_specs = [
        ("diplo", "diplomacy", COOL),
        ("threat", "threat", ACCENT),
    ]
    for ax, (dim, label, title_color) in zip(axes, focus_specs):
        for bkt, style in bucket_style.items():
            sub = monthly_df[monthly_df["bucket"] == bkt].set_index("pub_month").sort_index()
            if sub.empty or sub[dim].dropna().empty:
                continue
            s = sub[dim].rolling(SMOOTH_WINDOW, min_periods=1).mean()
            s = s.loc[focus_start:focus_end]
            if s.dropna().empty:
                continue
            ax.plot(s.index, s.values, **style)
        for ed, lab in [
            ("2013-06-15", "Rouhani"),
            ("2015-07-14", "JCPOA signed"),
            ("2016-01-16", "Implementation"),
            ("2018-05-08", "US exit"),
        ]:
            ax.axvline(pd.Timestamp(ed), color=MUTED, lw=0.9, alpha=0.45, ls="-", zorder=0)
            ax.annotate(
                lab,
                xy=(mdates.date2num(pd.Timestamp(ed).to_pydatetime()), 1.01),
                xycoords=ax.get_xaxis_transform(),
                xytext=(mdates.date2num(pd.Timestamp(ed).to_pydatetime()), 1.13),
                textcoords=ax.get_xaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=7.2,
                color=MUTED,
                rotation=35,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7),
                annotation_clip=False,
            )
        ax.set_title(f"{label} density by voice", loc="left", fontsize=11, color=title_color)
        ax.grid(True, alpha=0.22)
        ax.tick_params(labelsize=8)
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=False, ncol=4)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Year (12-month moving average)", fontsize=9)
    plt.tight_layout()
    focus_path = FIG_DIR / "voice_jcpoa_window.png"
    fig4.savefig(focus_path, bbox_inches="tight")

    print(f"\nSaved: {fig_path}")
    print(f"Saved: {div_path}")
    print(f"Saved: {voice_event_path}")
    print(f"Saved: {focus_path}")
    print(f"Saved: {OUT_DIR / 'oped_split_means.csv'}")
    print(f"Saved: {OUT_DIR / 'oped_split_monthly.csv'}")
    print(f"Saved: {OUT_DIR / 'voice_divergence_monthly.csv'}")
    print(f"Saved: {OUT_DIR / 'voice_event_shifts.csv'}")


if __name__ == "__main__":
    main()
