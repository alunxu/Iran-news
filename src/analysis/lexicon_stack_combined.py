#!/usr/bin/env python3
"""Combine the THREAT and DIPLO stacked-composition figures into one image
with (a) and (b) panels, sharing the x-axis."""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR = PROJECT_ROOT / "figures" / "structural_break"

TOP_N_FOR_STACK = 10


def load_contrib(lex_name: str) -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / f"lexicon_decade_contrib_{lex_name}.csv")


def stack_for(contrib: pd.DataFrame, ax, panel_label: str, lex_name: str):
    top_words = set()
    for dec in contrib["decade"].unique():
        top = contrib[contrib["decade"] == dec].nlargest(TOP_N_FOR_STACK, "gap")["word"].tolist()
        top_words.update(top)
    tracked = sorted(top_words,
                     key=lambda w: contrib[contrib["word"] == w]["gap"].sum(),
                     reverse=True)[:12]
    decades = sorted(contrib["decade"].unique())
    mat = np.zeros((len(tracked) + 1, len(decades)))
    for j, dec in enumerate(decades):
        sub = contrib[(contrib["decade"] == dec) & (contrib["gap"] > 0)]
        for i, w in enumerate(tracked):
            mat[i, j] = float(sub[sub["word"] == w]["gap"].sum())
        mat[-1, j] = float(sub[~sub["word"].isin(tracked)]["gap"].sum())

    cmap = plt.cm.get_cmap("Spectral", len(tracked) + 1)
    colors = [cmap(i) for i in range(len(tracked))] + ["#CCCCCC"]
    labels = tracked + ["(rest of lexicon)"]

    bottom = np.zeros(len(decades))
    x = np.arange(len(decades))
    for i, lbl in enumerate(labels):
        ax.bar(x, mat[i], 0.7, bottom=bottom, color=colors[i],
               edgecolor="white", lw=0.4, label=lbl)
        for j in range(len(decades)):
            # Only label thick bands; keep figure readable
            if mat[i, j] > 1.5:
                ax.text(x[j], bottom[j] + mat[i, j] / 2, lbl,
                        ha="center", va="center", fontsize=6.5,
                        color=("white" if mat[i, j] > 4 and i < len(tracked)
                               else "black"))
        bottom += mat[i]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)}s" for d in decades], fontsize=9)
    ax.set_ylabel(f"HL$-$body density gap\n(per 1000 words)", fontsize=8)
    ax.set_title(f"({panel_label})  {lex_name.upper()}", fontsize=10, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.5,
              frameon=False, title="Word")


def main():
    threat = load_contrib("threat")
    diplo = load_contrib("diplomacy")

    fig, (ax_t, ax_d) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=130)
    stack_for(threat, ax_t, "a", "threat")
    stack_for(diplo, ax_d, "b", "diplo")
    plt.tight_layout()
    out_path = FIG_DIR / "lexicon_stack_combined.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
