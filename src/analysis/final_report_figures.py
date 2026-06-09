#!/usr/bin/env python3
"""Generate report-safe figures for the DH-412 final report.

All outputs are capped below 2000 px on each side. The figures are designed
for the final written report rather than the presentation deck: fewer labels,
more direct evidence, and no slide-specific typography.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
STRUCT = DATA / "structural_break"
OUT = ROOT / "figures" / "final_report"

INK = "#363A3E"
MUTED = "#707070"
THREAT = "#B85042"
DIPLO = "#2F5F5D"
HUMAN = "#8C6D31"
GRID = "#E8E5E0"

LAYERS = [("hl", "HL"), ("ab", "AB"), ("ld", "LD"), ("ft", "Body")]
LAYER_FULL = {"hl": "Headline", "ab": "Abstract", "ld": "Lead", "ft": "Body"}
FRAMES = [("threat", "Threat", THREAT), ("diplo", "Diplomacy", DIPLO), ("human", "Humanizing", HUMAN)]
DECADES = ["1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]
VOICE_ORDER = ["news", "editorial", "column", "review"]
VOICE_LABELS = {"news": "News", "editorial": "Editorial", "column": "Column", "review": "Review"}
TOKEN_RE = re.compile(r"\b[a-zA-Z]+\b")
MIN_VOICE_CELL_N = 10


def weighted_mean(df: pd.DataFrame, value: str, weight: str = "n") -> float:
    valid = df[[value, weight]].dropna()
    if len(valid) == 0 or valid[weight].sum() == 0:
        return np.nan
    return float((valid[value] * valid[weight]).sum() / valid[weight].sum())


def add_decade(df: pd.DataFrame, date_col: str = "pub_month") -> pd.DataFrame:
    out = df.copy()
    year = pd.to_datetime(out[date_col]).dt.year
    out["decade"] = (year // 10 * 10).astype(int).astype(str) + "s"
    return out[out["decade"].isin(DECADES)]


def frame_cmap(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("", ["#FBFAF7", color])


def load_lexicons() -> dict[str, set[str]]:
    with open(DATA / "lexicons.json", "r", encoding="utf-8") as f:
        lex = json.load(f)
    return {
        "threat": set(lex["threat"]),
        "diplo": set(lex["diplomacy"]),
        "human": set(lex["humanizing"]),
    }


def token_and_hits(text: str | None, lex_set: set[str]) -> tuple[int, int]:
    if not isinstance(text, str) or not text:
        return 0, 0
    toks = TOKEN_RE.findall(text.lower())
    return len(toks), sum(1 for t in toks if t in lex_set)


def decade_from_year(year: float | int | str | None) -> str | None:
    if pd.isna(year):
        return None
    y = int(float(year))
    d = f"{(y // 10) * 10}s"
    return d if d in DECADES else None


def draw_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    color: str,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".1f",
) -> None:
    vals = matrix.to_numpy(dtype=float)
    if vmin is None:
        vmin = float(np.nanmin(vals))
    if vmax is None:
        vmax = float(np.nanmax(vals))
    im = ax.imshow(vals, cmap=frame_cmap(color), vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=10.5, color=INK, pad=6)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, fontsize=8.0, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8.0)
    ax.tick_params(length=0)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            val = vals[y, x]
            if np.isnan(val):
                ax.text(x, y, "—", ha="center", va="center", fontsize=7.4, color="#9A9A9A")
                continue
            text_color = "white" if val > vmin + 0.66 * (vmax - vmin) else INK
            ax.text(x, y, format(val, fmt), ha="center", va="center", fontsize=7.4, color=text_color)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_layer_density() -> Path:
    layer = add_decade(pd.read_csv(STRUCT / "layer_series.csv"))
    rows = []
    for decade in DECADES:
        sub = layer[layer["decade"] == decade]
        for layer_id, layer_label in LAYERS:
            for frame_id, frame_label, _ in FRAMES:
                rows.append(
                    {
                        "decade": decade,
                        "layer": layer_label,
                        "frame": frame_label,
                        "density": weighted_mean(sub, f"{layer_id}_{frame_id}") if len(sub) else np.nan,
                    }
                )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(DATA / "final_report_layer_decade_density.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 5.4), dpi=150)
    for ax, (_, frame_label, color) in zip(axes, FRAMES):
        matrix = (
            out_df[out_df["frame"] == frame_label]
            .pivot(index="decade", columns="layer", values="density")
            .reindex(index=DECADES, columns=[label for _, label in LAYERS])
        )
        draw_heatmap(ax, matrix, frame_label, color, vmin=0.0)
        ax.set_xlabel("")
    axes[0].set_ylabel("Decade", fontsize=8.5, color=MUTED)
    fig.suptitle("Layer framing density by decade", fontsize=14, color=INK, y=0.965)
    fig.text(0.055, 0.055, "Density per 1,000 tokens; same-article full-text subset.", fontsize=8.2, color=MUTED)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.18, wspace=0.26)
    out = OUT / "fig1_layer_density.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def make_voice_grammar() -> Path:
    lex = load_lexicons()
    corpus = pd.read_parquet(DATA / "iran_articles_full.parquet")
    corpus = corpus[
        (corpus["fulltext_word_count"] > 0)
        & corpus["voice"].isin(VOICE_ORDER)
        & corpus["year"].notna()
    ].copy()
    corpus["decade_label"] = corpus["year"].apply(decade_from_year)
    corpus = corpus[corpus["decade_label"].isin(DECADES)]

    rows = []
    for decade in DECADES:
        for v in VOICE_ORDER:
            sub = corpus[(corpus["decade_label"] == decade) & (corpus["voice"] == v)]
            token_counts = None
            for frame_id, frame_label, _ in FRAMES:
                counts = [token_and_hits(text, lex[frame_id]) for text in sub["fulltext"]]
                if token_counts is None:
                    token_counts = [tok for tok, _ in counts]
                total_tokens = sum(token_counts)
                total_hits = sum(hit for _, hit in counts)
                density = 1000.0 * total_hits / total_tokens if total_tokens else np.nan
                if len(sub) < MIN_VOICE_CELL_N:
                    density = np.nan
                rows.append(
                    {
                        "decade": decade,
                        "voice": VOICE_LABELS[v],
                        "frame": frame_label,
                        "density": density,
                        "n_articles": int(len(sub)),
                    }
                )
    voice_decade = pd.DataFrame(rows)
    voice_decade.to_csv(DATA / "final_report_voice_decade_density.csv", index=False)

    oped = pd.read_csv(STRUCT / "oped_split_means.csv")
    oped["bucket"] = oped["bucket"].replace({"staff": "Staff", "guest": "Guest", "unsigned": "Unsigned", "news": "News"})
    oped_matrix = oped.set_index("bucket")[["mean_threat", "mean_diplo", "mean_human"]]
    oped_matrix.columns = ["Threat", "Diplomacy", "Humanizing"]
    oped_matrix = oped_matrix.reindex(["News", "Staff", "Guest", "Unsigned"])
    oped_matrix.to_csv(DATA / "final_report_oped_split_density.csv")

    # Grammatical results are taken from the team's spaCy agency analysis.
    # Units are intentionally printed in labels/caption: ratios for agency,
    # per-1k-token rates for nominalization.
    agency_layer = pd.DataFrame(
        [
            {"series": "Headline", "decade": "1970s", "ratio": 0.51},
            {"series": "Headline", "decade": "1990s", "ratio": 0.87},
            {"series": "Headline", "decade": "2000s", "ratio": 0.62},
            {"series": "Headline", "decade": "2010s", "ratio": 0.52},
            {"series": "Body", "decade": "1970s", "ratio": 0.48},
            {"series": "Body", "decade": "2020s", "ratio": 0.74},
        ]
    )
    nominal = pd.DataFrame(
        {
            "Component": ["Headline", "Body", "News", "Editorial"],
            "Rate": [13.40, 16.39, 16.71, 17.55],
        }
    )
    agency_layer.to_csv(DATA / "final_report_agency_layer_summary.csv", index=False)
    nominal.to_csv(DATA / "final_report_nominalization_summary.csv", index=False)

    fig = plt.figure(figsize=(12.2, 7.2), dpi=150)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0], hspace=0.48, wspace=0.62)

    for idx, (_, frame_label, color) in enumerate(FRAMES):
        ax = fig.add_subplot(gs[0, idx * 2 : idx * 2 + 2])
        matrix = (
            voice_decade[voice_decade["frame"] == frame_label]
            .pivot(index="decade", columns="voice", values="density")
            .reindex(index=DECADES, columns=["News", "Editorial", "Column", "Review"])
        )
        draw_heatmap(ax, matrix, frame_label, color, vmin=0.0)
        if idx == 0:
            ax.set_ylabel("Decade", fontsize=8.5, color=MUTED)

    ax = fig.add_subplot(gs[1, 0:2])
    draw_heatmap(ax, oped_matrix, "Op-Ed split", THREAT, vmin=0.0)

    ax = fig.add_subplot(gs[1, 2:4])
    x_positions = {d: i for i, d in enumerate(DECADES)}
    for label, color in [("Headline", THREAT), ("Body", DIPLO)]:
        sub = agency_layer[agency_layer["series"] == label]
        xs = [x_positions[d] for d in sub["decade"]]
        ys = sub["ratio"].tolist()
        ax.plot(xs, ys, marker="o", lw=1.6, color=color, label=label)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.025, f"{y:.2f}", ha="center", va="bottom", fontsize=7.0, color=INK)
    ax.set_xticks(range(len(DECADES)))
    ax.set_xticklabels(DECADES, rotation=35, ha="right", fontsize=8.0)
    ax.set_ylim(0.35, 0.95)
    ax.set_title("Agency ratio", fontsize=10.5, color=INK, pad=6)
    ax.set_ylabel("subject / object", fontsize=8.5, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    ax = fig.add_subplot(gs[1, 4:6])
    colors = [MUTED, THREAT, MUTED, THREAT]
    ax.barh(np.arange(len(nominal)), nominal["Rate"], color=colors, alpha=0.86)
    ax.set_yticks(np.arange(len(nominal)))
    ax.set_yticklabels(nominal["Component"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("/ 1,000 tokens", fontsize=8.5, color=MUTED)
    ax.set_title("Nominalization", fontsize=10.5, color=INK, pad=6)
    for yv, val in enumerate(nominal["Rate"]):
        ax.text(val + 0.22, yv, f"{val:.1f}", va="center", ha="left", fontsize=8.2, color=INK)
    ax.set_xlim(0, 20)
    ax.grid(axis="x", color=GRID, linewidth=0.8)

    for ax in fig.axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=INK)
        ax.set_axisbelow(True)

    fig.suptitle("Voice framing and grammar", fontsize=14, y=0.975, color=INK)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.89, bottom=0.11)
    out = OUT / "fig2_voice_grammar.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(make_layer_density())
    print(make_voice_grammar())


if __name__ == "__main__":
    main()
