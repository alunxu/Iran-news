#!/usr/bin/env python3
"""
E_M1_4 v2 — Information-rich visualizations of per-decade lexicon contribution.

Original heatmap had ~80% empty cells because most lexicon words are rare. This
revision shows three more compact, informative views:

  Plot A  Top-8 contributors per decade as small-multiples bar charts
          (each decade's actual top contributors, not a fixed word list).

  Plot B  Stacked composition: each decade's HL-body gap decomposed into
          named contributors + "rest", showing share-of-total drift.

  Plot C  Generational turnover: Jaccard similarity of top-10 words between
          adjacent decades. Quantifies "apparatus replacement" pace.

Inputs:  data/structural_break/lexicon_decade_contrib_{threat,diplomacy}.csv
         (produced by robustness_e_m1_4_lexicon_contrib.py)
"""

from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR = PROJECT_ROOT / "figures" / "structural_break"

ACCENT, COOL, CHAR, MUTED = "#B85042", "#2F5F5D", "#363A3E", "#707070"
HIGHLIGHT = "#E08D17"

TOP_N_PER_DECADE = 8
TOP_N_FOR_STACK = 10


def load_contrib(lex_name: str) -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / f"lexicon_decade_contrib_{lex_name}.csv")


# ────────────────────────────────────────────────────────────────────────
# Plot A — small-multiples top contributors per decade
# ────────────────────────────────────────────────────────────────────────
def plot_top_per_decade(contrib: pd.DataFrame, lex_name: str, color: str,
                       out_path: Path) -> None:
    decades = sorted(contrib["decade"].unique())
    n_dec = len(decades)
    ncols = 3
    nrows = (n_dec + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 7), dpi=150)
    axes = axes.flatten()

    # Find shared x-range so panels are comparable
    sub = (contrib.groupby("decade", group_keys=False)
                  .apply(lambda g: g.nlargest(TOP_N_PER_DECADE, "gap")))
    xmax = sub["gap"].max() * 1.1

    for ax, dec in zip(axes, decades):
        sub_d = contrib[contrib["decade"] == dec].nlargest(TOP_N_PER_DECADE, "gap").iloc[::-1]
        bars = ax.barh(sub_d["word"], sub_d["gap"], color=color, alpha=0.85, edgecolor="white", lw=0.5)
        # Highlight the top contributor
        if len(bars) > 0:
            bars[-1].set_color(HIGHLIGHT)
            bars[-1].set_alpha(0.95)
        ax.set_xlim(0, xmax)
        ax.set_title(f"{int(dec)}s", fontsize=11, loc="left", weight="bold")
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for sp, gap in zip(sub_d["word"], sub_d["gap"]):
            ax.text(gap + xmax * 0.01, list(sub_d["word"]).index(sp),
                    f"{gap:+.1f}", va="center", fontsize=7, color=MUTED)

    # Hide unused axes
    for ax in axes[n_dec:]:
        ax.axis("off")

    fig.suptitle(f"Top-{TOP_N_PER_DECADE} {lex_name.upper()} words by HL−body density gap, by decade  "
                 f"(highlighted bar = leader per decade)",
                 fontsize=11, y=1.0)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ────────────────────────────────────────────────────────────────────────
# Plot B — stacked composition of total HL-body gap
# ────────────────────────────────────────────────────────────────────────
def plot_stacked_composition(contrib: pd.DataFrame, lex_name: str,
                             out_path: Path) -> None:
    """Stacked bar (one per decade) showing per-word share of total HL−body gap.

    Words appearing in any decade's top-N are tracked individually; others
    pooled into 'rest of lexicon'.
    """
    # Identify top-N words per decade, take union → tracked vocab
    top_words = set()
    for dec in contrib["decade"].unique():
        top = contrib[contrib["decade"] == dec].nlargest(TOP_N_FOR_STACK, "gap")["word"].tolist()
        top_words.update(top)
    tracked = sorted(top_words,
                     key=lambda w: contrib[contrib["word"] == w]["gap"].sum(),
                     reverse=True)
    # Cap at 12 words for legibility
    tracked = tracked[:12]

    decades = sorted(contrib["decade"].unique())
    # Build matrix: rows = words, cols = decades, values = gap
    mat = np.zeros((len(tracked) + 1, len(decades)))  # +1 for 'rest'
    for j, dec in enumerate(decades):
        sub = contrib[contrib["decade"] == dec]
        # Only positive gaps contribute to the 'total HL>body' bar
        sub_pos = sub[sub["gap"] > 0]
        for i, w in enumerate(tracked):
            mat[i, j] = float(sub_pos[sub_pos["word"] == w]["gap"].sum())
        # 'rest' = sum of all positive contributions not in tracked
        mat[-1, j] = float(sub_pos[~sub_pos["word"].isin(tracked)]["gap"].sum())

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    cmap = plt.cm.get_cmap("Spectral", len(tracked) + 1)
    colors = [cmap(i) for i in range(len(tracked))]
    colors.append("#CCCCCC")  # 'rest' is gray
    labels = tracked + ["(rest of lexicon)"]

    bottom = np.zeros(len(decades))
    x = np.arange(len(decades))
    width = 0.7
    for i, lbl in enumerate(labels):
        ax.bar(x, mat[i], width, bottom=bottom, color=colors[i],
               edgecolor="white", lw=0.5, label=lbl)
        # In-bar label if band thick enough
        for j in range(len(decades)):
            if mat[i, j] > 1.5:
                ax.text(x[j], bottom[j] + mat[i, j] / 2, lbl,
                        ha="center", va="center", fontsize=7,
                        color="black" if i >= len(tracked) else
                        ("white" if mat[i, j] > 4 else "black"))
        bottom += mat[i]

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)}s" for d in decades], fontsize=9)
    ax.set_ylabel(f"HL−body {lex_name.upper()} density gap (per 1000 words, summed across positive contributors)",
                  fontsize=9)
    ax.set_title(f"Composition of the HL−body {lex_name.upper()} gap by decade  "
                 f"(stacked = sum of per-word HL−body density)",
                 fontsize=11, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
              frameon=False, title="Word")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ────────────────────────────────────────────────────────────────────────
# Plot C — generational turnover (Jaccard similarity between adjacent decades)
# ────────────────────────────────────────────────────────────────────────
def plot_turnover(contrib: pd.DataFrame, lex_name: str, color: str,
                  out_path: Path) -> None:
    decades = sorted(contrib["decade"].unique())
    top_sets = {
        dec: set(contrib[contrib["decade"] == dec].nlargest(TOP_N_FOR_STACK, "gap")["word"])
        for dec in decades
    }

    rows = []
    for d_a, d_b in zip(decades[:-1], decades[1:]):
        a, b = top_sets[d_a], top_sets[d_b]
        if not (a | b):
            continue
        jacc = len(a & b) / len(a | b)
        kept = a & b
        dropped = a - b
        added = b - a
        rows.append(dict(
            from_decade=int(d_a), to_decade=int(d_b),
            jaccard=jacc, kept=", ".join(sorted(kept)),
            dropped=", ".join(sorted(dropped)), added=", ".join(sorted(added)),
        ))
    turn = pd.DataFrame(rows)
    turn.to_csv(OUT_DIR / f"lexicon_turnover_{lex_name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    x = np.arange(len(turn))
    bars = ax.bar(x, turn["jaccard"], color=color, alpha=0.85, edgecolor="white", lw=0.5)
    for i, r in enumerate(turn.itertuples()):
        ax.text(i, r.jaccard + 0.015, f"{r.jaccard:.2f}", ha="center",
                fontsize=8, color=CHAR)
        # Brief annotation: top added word
        added_words = r.added.split(", ") if r.added else []
        dropped_words = r.dropped.split(", ") if r.dropped else []
        annot = ""
        if added_words:
            annot += f"+{added_words[0]}"
        if dropped_words:
            annot += f"\n−{dropped_words[0]}"
        ax.text(i, -0.07, annot, ha="center", va="top", fontsize=7, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.from_decade}s→{r.to_decade}s" for r in turn.itertuples()],
                       fontsize=9)
    ax.set_ylim(-0.15, 1.05)
    ax.set_ylabel("Jaccard similarity (top-10 words)", fontsize=9)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"Generational turnover of the {lex_name.upper()} apparatus  "
                 f"(low Jaccard = vocabulary replaced)",
                 fontsize=11, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print()
    print(f"  {lex_name.upper()} turnover details:")
    for r in turn.itertuples():
        print(f"    {r.from_decade}s → {r.to_decade}s  J={r.jaccard:.2f}  "
              f"kept={{{r.kept}}}  +{{{r.added}}}  −{{{r.dropped}}}")


def main():
    for lex_name, color in [("threat", ACCENT), ("diplomacy", COOL)]:
        print(f"=== {lex_name.upper()} ===")
        contrib = load_contrib(lex_name)
        plot_top_per_decade(contrib, lex_name, color,
                           FIG_DIR / f"lexicon_top_per_decade_{lex_name}.png")
        plot_stacked_composition(contrib, lex_name,
                                FIG_DIR / f"lexicon_stack_{lex_name}.png")
        plot_turnover(contrib, lex_name, color,
                     FIG_DIR / f"lexicon_turnover_{lex_name}.png")
        print()


if __name__ == "__main__":
    main()
