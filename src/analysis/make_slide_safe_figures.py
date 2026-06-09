#!/usr/bin/env python3
"""Create slide-safe figures for the final presentation.

These figures are intentionally separate from the formal notes figures:
they keep the same underlying results, but add enough margins for direct
PowerPoint/Google Slides placement without cropping labels.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "structural_break"
FIG_DIR = PROJECT_ROOT / "figures" / "structural_break"

CHAR = "#363A3E"
MUTED = "#707070"
ORIG_BREAK = "#E08D17"
DETR_BREAK = "#1f77b4"
FRAME_COLORS = {
    "threat": "#B85042",
    "diplo": "#2F5F5D",
    "human": "#363A3E",
}
FRAME_LABELS = {
    "threat": "threat",
    "diplo": "diplomacy",
    "human": "humanizing",
}
EVENT_ORDER_FULL = [
    "Embassy seized",
    "Iran-Iraq war",
    "IR655",
    "Khomeini dies",
    "Axis of Evil",
    "Green Movement",
    "Rouhani elected",
    "JCPOA signed",
    "JCPOA Imp Day",
    "JCPOA exit",
    "Soleimani",
    "Mahsa Amini",
]
EVENT_DISPLAY_FULL = {
    "Embassy seized": "Embassy seized",
    "Iran-Iraq war": "Iran-Iraq war",
    "IR655": "IR655",
    "Khomeini dies": "Khomeini dies",
    "Axis of Evil": "Axis of Evil",
    "Green Movement": "Green Movement",
    "Rouhani elected": "Rouhani elected",
    "JCPOA signed": "JCPOA signed",
    "JCPOA Imp Day": "JCPOA implementation",
    "JCPOA exit": "US exits JCPOA",
    "Soleimani": "Soleimani",
    "Mahsa Amini": "Mahsa Amini",
}
LAYER_LABELS = {"hl": "Headline", "ab": "Abstract", "ld": "Lead", "ft": "Body"}

PANEL_SPECS = [
    ("gap_threat_HL_FT_mean_C1", "Headline-body threat gap", "threat"),
    ("gap_diplo_HL_FT_mean_C1", "Headline-body diplomacy gap", "diplomacy"),
]

EVENTS = [
    ("2002-01-29", "Axis"),
    ("2013-06-15", "Rouhani"),
    ("2016-01-16", "JCPOA impl."),
    ("2018-05-08", "US exit"),
    ("2020-01-03", "Soleimani"),
    ("2022-09-16", "Mahsa"),
]


def linear_residuals(series: pd.Series) -> np.ndarray:
    dates = series.index
    y = series.values.astype(float)
    t_years = np.array([(d - dates[0]).days / 365.25 for d in dates])
    slope, intercept = np.polyfit(t_years, y, 1)
    return y - (slope * t_years + intercept)


def detrend_zscore(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if len(s) < 24:
        return s
    residuals = linear_residuals(s)
    residuals = pd.Series(residuals, index=s.index)
    std = residuals.std()
    return (residuals - residuals.mean()) / std if std > 1e-6 else residuals * 0


def persistence_verdict(d6: float, d36: float) -> tuple[str, str]:
    if abs(d6) < 1.0:
        return "weak", "#D9D4CE"
    if np.sign(d6) != np.sign(d36):
        return "reverses", "#B85042"
    if abs(d36) >= 1.2 * abs(d6):
        return "intensifies", "#2F5F5D"
    if abs(d36) >= 0.7 * abs(d6):
        return "persists", "#2F5F5D"
    return "fades", "#707070"


def main() -> None:
    monthly = pd.read_csv(DATA_DIR / "monthly_series_v2.csv", parse_dates=["pub_month"])
    monthly = monthly.set_index("pub_month").sort_index()
    breaks = pd.read_csv(DATA_DIR / "detrended_breaks.csv", parse_dates=["break_date"])
    trends = pd.read_csv(DATA_DIR / "trend_estimates.csv")

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 5.4), dpi=150, sharex=True)
    for ax, (col, title, frame_name) in zip(axes, PANEL_SPECS):
        s = monthly[col].dropna()
        residuals = pd.Series(linear_residuals(s), index=s.index)
        smoothed = residuals.rolling(12, min_periods=1).mean()

        ax.axhline(0, color="black", lw=0.55, alpha=0.35)
        ax.plot(smoothed.index, smoothed.values, color=CHAR, lw=1.25, label="Detrended residual (12mo MA)")

        bsub = breaks[breaks["series"] == col]
        original_dates = list(bsub.loc[bsub["kind"] == "original_C1", "break_date"])
        detrended_dates = list(bsub.loc[bsub["kind"] == "detrended", "break_date"])
        overlap_dates = {d.date() for d in original_dates}.intersection({d.date() for d in detrended_dates})

        # If original and detrended PELT find the same date, draw the lines
        # slightly apart so both remain visible on a slide. The event itself
        # is still the labelled date; the small offset is visual only.
        for d in original_dates:
            plot_date = d - pd.DateOffset(days=45) if d.date() in overlap_dates else d
            ax.axvline(plot_date, color=ORIG_BREAK, lw=1.45, ls="--", alpha=0.92)
        for d in detrended_dates:
            plot_date = d + pd.DateOffset(days=45) if d.date() in overlap_dates else d
            ax.axvline(plot_date, color=DETR_BREAK, lw=1.65, alpha=0.96)

        for event_date, _ in EVENTS:
            ax.axvline(pd.Timestamp(event_date), color=MUTED, lw=0.70, alpha=0.30, zorder=0)

        trend = trends.loc[trends["series"] == col, "slope_per_decade"].iloc[0]
        ax.set_title(f"{title}  (trend = {trend:+.2f}/decade)", loc="left", fontsize=11, color=CHAR, pad=9)
        ax.grid(True, alpha=0.18)
        ax.tick_params(labelsize=8.5)
        ax.set_ylabel(frame_name, fontsize=9, color=MUTED)

    axes[0].plot([], [], color=ORIG_BREAK, lw=1.35, ls="--", label="Original PELT break")
    axes[0].plot([], [], color=DETR_BREAK, lw=1.55, label="Detrended PELT break")
    axes[0].legend(loc="upper left", fontsize=8.2, frameon=False, ncol=3)

    for idx, (event_date, label) in enumerate(EVENTS):
        x = mdates.date2num(pd.Timestamp(event_date).to_pydatetime())
        axes[0].annotate(
            label,
            xy=(x, 1.01),
            xycoords=axes[0].get_xaxis_transform(),
            xytext=(x, 1.17 + (idx % 2) * 0.11),
            textcoords=axes[0].get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7.6,
            color=MUTED,
            rotation=40,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.75, alpha=0.75),
            annotation_clip=False,
        )

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Year", fontsize=9.5)
    fig.suptitle("Detrended PELT: drift disappears, baseline-shift candidates survive", fontsize=13, y=0.965)
    fig.text(
        0.08,
        0.035,
        "Note: when original and detrended PELT breaks fall on the same month, lines are slightly offset for visibility.",
        fontsize=8.2,
        color=MUTED,
        ha="left",
    )

    # Reserve explicit margins; avoid bbox_inches='tight' so PowerPoint does
    # not crop edge text when the image is inserted.
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.78, hspace=0.42)
    out = FIG_DIR / "detrended_pelt_slide_safe.png"
    fig.savefig(out, facecolor="white")
    print(out)

    # ------------------------------------------------------------------
    # Drift-vs-baseline-shift explainer.
    # This version makes the contrast visual rather than textual:
    # - the original panel shows the slow slope and the original PELT cut;
    # - the residual panel shows whether a step-like break remains after
    #   the slope has been removed.
    # ------------------------------------------------------------------
    fig_exp, axes_exp = plt.subplots(2, 2, figsize=(12.0, 6.8), dpi=150, sharex="col")
    exp_specs = [
        ("gap_threat_HL_FT_mean_C1", "Threat gap", "#B85042", "drift-like"),
        ("gap_diplo_HL_FT_mean_C1", "Diplomacy gap", "#2F5F5D", "baseline-shift candidate"),
    ]
    for row_idx, (col, label, color, verdict) in enumerate(exp_specs):
        s = monthly[col].dropna()
        dates = s.index
        y = s.values.astype(float)
        t_years = np.array([(d - dates[0]).days / 365.25 for d in dates])
        slope, intercept = np.polyfit(t_years, y, 1)
        trend = pd.Series(slope * t_years + intercept, index=dates)
        residuals = pd.Series(y - trend.values, index=dates)
        original_smoothed = s.rolling(12, min_periods=1).mean()
        residual_smoothed = residuals.rolling(12, min_periods=1).mean()
        bsub = breaks[breaks["series"] == col]
        original_dates = list(bsub.loc[bsub["kind"] == "original_C1", "break_date"])
        detrended_dates = list(bsub.loc[bsub["kind"] == "detrended", "break_date"])

        ax_l = axes_exp[row_idx, 0]
        ax_r = axes_exp[row_idx, 1]

        # Original: foreground the slow trend, not only the break line.
        ax_l.plot(original_smoothed.index, original_smoothed.values, color=CHAR, lw=1.0, alpha=0.72)
        ax_l.plot(trend.index, trend.values, color=color, lw=3.1, alpha=0.92)
        for d in original_dates:
            ax_l.axvline(d, color=ORIG_BREAK, lw=1.65, ls="--", alpha=0.96)
        ax_l.annotate(
            "",
            xy=(trend.index[-1], trend.iloc[-1]),
            xytext=(trend.index[0], trend.iloc[0]),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.2, alpha=0.75),
        )
        ax_l.set_title(f"{label}: before detrending", loc="left", fontsize=11.2, color=color, pad=7)

        # Residual: foreground step survival/disappearance.
        ax_r.axhline(0, color="black", lw=0.7, alpha=0.42)
        ax_r.plot(residual_smoothed.index, residual_smoothed.values, color=CHAR, lw=1.05, alpha=0.82)
        if detrended_dates:
            for d in detrended_dates:
                d = pd.Timestamp(d)
                ax_r.axvline(d, color=DETR_BREAK, lw=2.0, alpha=0.98)
            # Show the detected structural break as segment means. This makes the
            # diplomacy row read as a plateau, not just as two vertical lines.
            segment_edges = [residual_smoothed.index.min()] + [pd.Timestamp(d) for d in detrended_dates] + [residual_smoothed.index.max()]
            for start, end in zip(segment_edges[:-1], segment_edges[1:]):
                segment = residual_smoothed.loc[start:end].dropna()
                if len(segment):
                    ax_r.hlines(segment.mean(), start, end, color=DETR_BREAK, lw=4.0, alpha=0.40)
                    if row_idx == 1 and start.year <= 2014 <= end.year:
                        ax_r.axvspan(start, end, color=DETR_BREAK, alpha=0.125, lw=0)
                        ax_r.text(
                            start + (end - start) / 2,
                            segment.mean() + 1.1,
                            "surviving plateau",
                            ha="center",
                            va="bottom",
                            fontsize=8.8,
                            color=DETR_BREAK,
                            fontweight="bold",
                        )
        else:
            # No residual break: visually mark the whole panel as one noisy
            # fluctuation around zero.
            ax_r.axhspan(-2.0, 2.0, color=MUTED, alpha=0.055, lw=0)
            ax_r.text(
                0.51,
                0.54,
                "no surviving step",
                transform=ax_r.transAxes,
                ha="center",
                va="center",
                fontsize=12.0,
                color=MUTED,
                fontweight="bold",
                alpha=0.92,
            )
        ax_r.set_title(f"{label}: after detrending", loc="left", fontsize=11.2, color=color, pad=7)

        # Row verdict badge.
        badge_color = MUTED if row_idx == 0 else DETR_BREAK
        ax_r.text(
            0.98,
            0.90,
            verdict,
            transform=ax_r.transAxes,
            ha="right",
            va="center",
            fontsize=9.3,
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=badge_color, edgecolor="none", alpha=0.90),
        )

        for ax in [ax_l, ax_r]:
            ax.grid(True, alpha=0.16)
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(mdates.YearLocator(10))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            for event_date, _ in EVENTS:
                ax.axvline(pd.Timestamp(event_date), color=MUTED, lw=0.55, alpha=0.14, zorder=0)

    axes_exp[0, 0].plot([], [], color=CHAR, lw=1.0, alpha=0.72, label="12-month moving average")
    axes_exp[0, 0].plot([], [], color="#B85042", lw=3.1, label="linear drift")
    axes_exp[0, 0].plot([], [], color=ORIG_BREAK, lw=1.65, ls="--", label="original PELT break")
    axes_exp[0, 0].legend(loc="lower left", fontsize=7.6, frameon=False, ncol=1)
    axes_exp[1, 1].plot([], [], color=DETR_BREAK, lw=2.0, label="detrended PELT break")
    axes_exp[1, 1].plot([], [], color=DETR_BREAK, lw=4.0, alpha=0.40, label="residual segment mean")
    axes_exp[1, 1].legend(loc="lower left", fontsize=7.6, frameon=False)
    for ax in axes_exp[-1, :]:
        ax.set_xlabel("Year", fontsize=9)
    for ax in axes_exp[:, 0]:
        ax.set_ylabel("original gap", fontsize=8.8, color=MUTED)
    for ax in axes_exp[:, 1]:
        ax.set_ylabel("residual gap", fontsize=8.8, color=MUTED)
    fig_exp.suptitle("Detrending separates slow drift from step-like baseline shifts", fontsize=14.5, y=0.972)
    fig_exp.text(
        0.07,
        0.917,
        "Left: PELT on the original series can cut a long slope. Right: after subtracting that slope, only step-like changes survive.",
        fontsize=9.1,
        color=MUTED,
        ha="left",
    )
    fig_exp.subplots_adjust(left=0.075, right=0.985, top=0.835, bottom=0.105, hspace=0.39, wspace=0.17)
    out_exp = FIG_DIR / "detrended_pelt_drift_vs_baseline_shift_slide_safe.png"
    fig_exp.savefig(out_exp, facecolor="white")
    print(out_exp)

    # ------------------------------------------------------------------
    # Persistence summary for slides.
    # Focus on diplomacy because the strongest temporal finding is the
    # JCPOA diplomacy baseline-shift candidate. This turns the dense
    # hysteresis grid into a direct Δ6 -> Δ36 reading.
    # ------------------------------------------------------------------
    persist = pd.read_csv(DATA_DIR / "hysteresis_persistence.csv")
    selected_events = ["JCPOA signed", "JCPOA Imp Day", "JCPOA exit", "Soleimani"]
    labels = {
        "JCPOA signed": "JCPOA signed",
        "JCPOA Imp Day": "Implementation",
        "JCPOA exit": "US exit",
        "Soleimani": "Soleimani",
    }
    rows = []
    for event in selected_events:
        sub = persist[(persist["event"] == event) & (persist["axis"] == "diplo")]
        sub = sub.dropna(subset=["shift_6mo", "shift_36mo"])
        d6 = float(sub["shift_6mo"].mean())
        d36 = float(sub["shift_36mo"].mean())
        verdict, color = persistence_verdict(d6, d36)
        rows.append(dict(event=event, label=labels[event], d6=d6, d36=d36, verdict=verdict, color=color))
    summary = pd.DataFrame(rows)
    summary.to_csv(DATA_DIR / "persistence_diplo_summary.csv", index=False)

    fig2, ax = plt.subplots(figsize=(11.8, 4.8), dpi=150)
    y_positions = np.arange(len(summary))[::-1]
    ax.axvline(0, color="black", lw=0.8, alpha=0.45)
    for y, row in zip(y_positions, summary.itertuples(index=False)):
        ax.plot([row.d6, row.d36], [y, y], color=row.color, lw=2.2, alpha=0.80)
        ax.scatter(row.d6, y, s=110, facecolor="white", edgecolor=row.color, lw=1.9, zorder=3)
        ax.scatter(row.d36, y, s=72, facecolor=row.color, edgecolor=row.color, lw=1.2, zorder=4)
        ax.text(9.7, y, row.verdict, ha="left", va="center", fontsize=11, color=row.color, fontweight="bold")

    ax.scatter([], [], s=90, facecolor="white", edgecolor=CHAR, lw=1.8, label="Δ6: short-run shift")
    ax.scatter([], [], s=90, facecolor=CHAR, edgecolor=CHAR, lw=1.2, label="Δ36: 36-month shift")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=2)
    ax.set_xlim(-17.8, 11.8)
    ax.set_ylim(-0.8, len(summary) - 0.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary["label"], fontsize=11, color=CHAR)
    ax.set_xlabel("Diplomacy framing shift vs. pre-event baseline  (per 1,000 tokens)", fontsize=10)
    ax.set_title("Persistence check: does the diplomacy shift remain after the news cycle?", loc="left", fontsize=14, pad=12)
    ax.text(
        0.0,
        1.01,
        "Open dot = average shift after 6 months; filled dot = average shift after 36 months. Values average headline, abstract, lead, and body.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )
    ax.grid(True, axis="x", alpha=0.18)
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    fig2.subplots_adjust(left=0.16, right=0.92, top=0.78, bottom=0.20)
    out2 = FIG_DIR / "persistence_diplo_slide_safe.png"
    fig2.savefig(out2, facecolor="white")
    print(out2)

    # ------------------------------------------------------------------
    # Three-frame persistence summary.
    # Same reading as above, but shown across threat, diplomacy, and
    # humanizing. This is slightly denser but still slide-safe.
    # ------------------------------------------------------------------
    all_rows = []
    for axis in ["threat", "diplo", "human"]:
        for event in selected_events:
            sub = persist[(persist["event"] == event) & (persist["axis"] == axis)]
            sub = sub.dropna(subset=["shift_6mo", "shift_36mo"])
            d6 = float(sub["shift_6mo"].mean())
            d36 = float(sub["shift_36mo"].mean())
            verdict, verdict_color = persistence_verdict(d6, d36)
            all_rows.append(
                dict(
                    axis=axis,
                    event=event,
                    label=labels[event],
                    d6=d6,
                    d36=d36,
                    verdict=verdict,
                    verdict_color=verdict_color,
                )
            )
    all_summary = pd.DataFrame(all_rows)
    all_summary.to_csv(DATA_DIR / "persistence_all_frames_summary.csv", index=False)

    fig3, axes3 = plt.subplots(1, 3, figsize=(12.0, 5.9), dpi=150, sharey=True)
    y_positions = np.arange(len(selected_events))[::-1]
    for ax, axis in zip(axes3, ["threat", "diplo", "human"]):
        sub = all_summary[all_summary["axis"] == axis].set_index("event").loc[selected_events].reset_index()
        frame_color = FRAME_COLORS[axis]
        ax.axvline(0, color="black", lw=0.8, alpha=0.42)
        for y, row in zip(y_positions, sub.itertuples(index=False)):
            ax.plot([row.d6, row.d36], [y, y], color=row.verdict_color, lw=2.0, alpha=0.78)
            ax.scatter(row.d6, y, s=95, facecolor="white", edgecolor=row.verdict_color, lw=1.75, zorder=3)
            ax.scatter(row.d36, y, s=60, facecolor=row.verdict_color, edgecolor=row.verdict_color, lw=1.1, zorder=4)
            ax.text(11.0, y, row.verdict, ha="left", va="center", fontsize=8.3, color=row.verdict_color, fontweight="bold")
        ax.set_xlim(-18, 15)
        ax.set_title(FRAME_LABELS[axis], fontsize=12, color=frame_color, pad=9)
        ax.grid(True, axis="x", alpha=0.18)
        ax.tick_params(axis="x", labelsize=8)
        for spine in ["top", "left", "right"]:
            ax.spines[spine].set_visible(False)
        if ax is axes3[0]:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([labels[e] for e in selected_events], fontsize=10.2, color=CHAR)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    fig3.suptitle("Persistence check across framing categories", fontsize=15, y=0.965)
    fig3.text(
        0.07,
        0.905,
        "Open dot = 6-month shift; filled dot = 36-month shift. Values average headline, abstract, lead, and body.",
        fontsize=9,
        color=MUTED,
        ha="left",
    )
    fig3.text(
        0.50,
        0.08,
        "Framing shift vs. pre-event baseline  (per 1,000 tokens)",
        fontsize=10,
        color=CHAR,
        ha="center",
    )
    fig3.subplots_adjust(left=0.15, right=0.95, top=0.82, bottom=0.17, wspace=0.28)
    out3 = FIG_DIR / "persistence_all_frames_slide_safe.png"
    fig3.savefig(out3, facecolor="white")
    print(out3)

    # ------------------------------------------------------------------
    # All-event verdict matrix.
    # This answers the "why these events?" question: the selected events
    # are not the only ones tested; they are the cleanest examples from a
    # broader event grid.
    # ------------------------------------------------------------------
    matrix_rows = []
    for axis in ["threat", "diplo", "human"]:
        for event in EVENT_ORDER_FULL:
            sub = persist[(persist["event"] == event) & (persist["axis"] == axis)]
            sub = sub.dropna(subset=["shift_6mo", "shift_36mo"])
            if sub.empty:
                verdict, color, d6, d36 = "no baseline", "#EFEDE9", np.nan, np.nan
            else:
                d6 = float(sub["shift_6mo"].mean())
                d36 = float(sub["shift_36mo"].mean())
                verdict, color = persistence_verdict(d6, d36)
            matrix_rows.append(dict(event=event, axis=axis, verdict=verdict, color=color, d6=d6, d36=d36))
    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(DATA_DIR / "persistence_all_events_verdicts.csv", index=False)

    color_map = {
        "persists": "#2F5F5D",
        "intensifies": "#5A8F7B",
        "fades": "#8B8B8B",
        "reverses": "#B85042",
        "weak": "#D9D4CE",
        "no baseline": "#EFEDE9",
    }
    label_map = {
        "persists": "persists",
        "intensifies": "intensifies",
        "fades": "fades",
        "reverses": "reverses",
        "weak": "weak",
        "no baseline": "no baseline",
    }
    fig4, ax = plt.subplots(figsize=(10.8, 7.0), dpi=150)
    for y, event in enumerate(EVENT_ORDER_FULL):
        for x, axis in enumerate(["threat", "diplo", "human"]):
            row = matrix[(matrix["event"] == event) & (matrix["axis"] == axis)].iloc[0]
            verdict = row["verdict"]
            color = color_map[verdict]
            rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=1.4)
            ax.add_patch(rect)
            text_color = "white" if verdict in {"persists", "intensifies", "reverses"} else CHAR
            ax.text(x + 0.5, y + 0.5, label_map[verdict], ha="center", va="center",
                    fontsize=8.2, color=text_color, fontweight="bold" if verdict != "weak" else "normal")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, len(EVENT_ORDER_FULL))
    ax.invert_yaxis()
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["threat", "diplomacy", "humanizing"], fontsize=12)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(EVENT_ORDER_FULL)) + 0.5)
    ax.set_yticklabels([EVENT_DISPLAY_FULL[e] for e in EVENT_ORDER_FULL], fontsize=9.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig4.suptitle("Persistence verdicts across all tested events", fontsize=15, y=0.965)
    fig4.text(
        0.12,
        0.915,
        "Each cell compares average Δ6 and Δ36 across headline, abstract, lead, and body. Weak = short-run shift |Δ6| < 1 per 1,000 tokens.",
        fontsize=8.7,
        color=MUTED,
        ha="left",
    )
    fig4.subplots_adjust(left=0.27, right=0.96, top=0.84, bottom=0.08)
    out4 = FIG_DIR / "persistence_all_events_verdict_matrix.png"
    fig4.savefig(out4, facecolor="white")
    print(out4)

    # ------------------------------------------------------------------
    # All-event dumbbell persistence summary.
    # Same information as the verdict matrix, but it preserves magnitude:
    # how far Δ6 and Δ36 sit from the pre-event baseline.
    # ------------------------------------------------------------------
    fig4b, axes4b = plt.subplots(1, 3, figsize=(12.0, 8.8), dpi=150, sharey=True)
    y_positions = np.arange(len(EVENT_ORDER_FULL))[::-1]
    for ax, axis in zip(axes4b, ["threat", "diplo", "human"]):
        sub = matrix[matrix["axis"] == axis].set_index("event").loc[EVENT_ORDER_FULL].reset_index()
        ax.axvline(0, color="black", lw=0.8, alpha=0.42)
        for y, row in zip(y_positions, sub.itertuples(index=False)):
            verdict = row.verdict
            color = color_map[verdict]
            if pd.isna(row.d6) or pd.isna(row.d36):
                continue
            alpha = 0.38 if verdict in {"weak", "no baseline"} else 0.82
            ax.plot([row.d6, row.d36], [y, y], color=color, lw=1.9, alpha=alpha)
            ax.scatter(row.d6, y, s=82, facecolor="white", edgecolor=color, lw=1.65, zorder=3, alpha=alpha)
            ax.scatter(row.d36, y, s=54, facecolor=color, edgecolor=color, lw=1.0, zorder=4, alpha=alpha)
            ax.text(18.2, y, verdict, ha="left", va="center", fontsize=7.3, color=color, fontweight="bold" if verdict not in {"weak", "no baseline"} else "normal")
        ax.set_xlim(-18.5, 27.0)
        ax.set_title(FRAME_LABELS[axis], fontsize=12, color=FRAME_COLORS[axis], pad=9)
        ax.grid(True, axis="x", alpha=0.16)
        ax.tick_params(axis="x", labelsize=8)
        for spine in ["top", "left", "right"]:
            ax.spines[spine].set_visible(False)
        if ax is axes4b[0]:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([EVENT_DISPLAY_FULL[e] for e in EVENT_ORDER_FULL], fontsize=8.8, color=CHAR)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
    fig4b.suptitle("Persistence magnitude across all tested events", fontsize=15, y=0.975)
    fig4b.text(
        0.07,
        0.935,
        "Open dot = 6-month shift; filled dot = 36-month shift. Larger distance from 0 means stronger afterlife; values average headline, abstract, lead, and body.",
        fontsize=8.7,
        color=MUTED,
        ha="left",
    )
    fig4b.text(
        0.50,
        0.060,
        "Framing shift vs. pre-event baseline  (per 1,000 tokens)",
        fontsize=10,
        color=CHAR,
        ha="center",
    )
    fig4b.subplots_adjust(left=0.18, right=0.93, top=0.87, bottom=0.12, wspace=0.25)
    out4b = FIG_DIR / "persistence_all_events_dumbbell_slide_safe.png"
    fig4b.savefig(out4b, facecolor="white")
    print(out4b)

    # ------------------------------------------------------------------
    # Joint-break summary matrix.
    # Replaces the 12-panel line plot with the actual contribution pattern:
    # which frame × layer cells move at each multivariate PELT break?
    # ------------------------------------------------------------------
    layer_df = pd.read_csv(DATA_DIR / "layer_series.csv", parse_dates=["pub_month"]).set_index("pub_month").sort_index()
    joint_breaks = pd.read_csv(DATA_DIR / "joint_breaks.csv", parse_dates=["break_date"])
    axes_order = ["threat", "diplo", "human"]
    layers_order = ["hl", "ab", "ld", "ft"]
    cells = [f"{l}_{a}" for a in axes_order for l in layers_order]
    panel = pd.DataFrame(index=layer_df.index)
    for c in cells:
        panel[c] = detrend_zscore(layer_df[c])
    panel = panel.dropna()

    rows = []
    row_labels = []
    for _, br in joint_breaks.iterrows():
        bdate = pd.Timestamp(br["break_date"])
        if bdate not in panel.index:
            b = panel.index.get_indexer([bdate], method="nearest")[0]
        else:
            b = panel.index.get_loc(bdate)
        pre = panel.iloc[max(0, b - 12):b]
        post = panel.iloc[b:min(len(panel), b + 12)]
        shift = (post.mean() - pre.mean()).reindex(cells)
        rows.append(shift.values)
        if bdate.year == 2013:
            label = "2013 Rouhani\njoint break"
        elif bdate.year == 2016:
            label = "2016 JCPOA implementation\njoint break"
        elif bdate.year == 2023:
            label = "2023 Gaza-war period\nprovisional"
        else:
            label = f"{bdate:%Y-%m}\njoint break"
        row_labels.append(label)
    matrix = np.vstack(rows)
    out_matrix = pd.DataFrame(matrix, index=row_labels, columns=cells)
    out_matrix.to_csv(DATA_DIR / "joint_break_contribution_matrix.csv")

    fig5, ax = plt.subplots(figsize=(12.0, 4.9), dpi=150)
    vmax = max(0.5, np.nanmax(np.abs(matrix)))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    col_labels = [LAYER_LABELS[l].replace("Headline", "HL").replace("Abstract", "Abs").replace("Lead", "Lead").replace("Body", "Body")
                  for a in axes_order for l in layers_order]
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels(col_labels, fontsize=8.5, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    for x in [3.5, 7.5]:
        ax.axvline(x, color="white", lw=2.2)
    for x_center, title, color in [(1.5, "threat", FRAME_COLORS["threat"]),
                                   (5.5, "diplomacy", FRAME_COLORS["diplo"]),
                                   (9.5, "humanizing", FRAME_COLORS["human"])]:
        ax.text(x_center, -0.95, title, ha="center", va="center", fontsize=12, color=color)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(
                j,
                i,
                f"{v:+.1f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if abs(v) > vmax * 0.55 else CHAR,
                fontweight="bold" if abs(v) > 1.0 else "normal",
            )
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cb.set_label("post − pre z-shift", fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig5.suptitle("What actually moves at each joint break?", fontsize=15, y=0.965)
    fig5.text(
        0.12,
        0.895,
        "Cells show 12-month post-minus-pre shifts on detrended, z-scored layer series. Strong same-frame movement across layers = cross-layer synchronization.",
        fontsize=8.8,
        color=MUTED,
        ha="left",
    )
    fig5.subplots_adjust(left=0.24, right=0.92, top=0.78, bottom=0.20)
    out5 = FIG_DIR / "joint_break_contribution_matrix_slide_safe.png"
    fig5.savefig(out5, facecolor="white")
    print(out5)


if __name__ == "__main__":
    main()
