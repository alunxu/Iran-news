#!/usr/bin/env python3
"""Generate report-style temporal figures without external Python packages.

The execution environment used by Codex may not have matplotlib/numpy.  This
script writes SVG directly and converts it to PNG with ImageMagick.  All PNGs
are capped below 2000 px on each side.
"""

from __future__ import annotations

import csv
import math
import subprocess
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "structural_break"
FIG = ROOT / "figures" / "final_report"
FIG.mkdir(parents=True, exist_ok=True)

CHAR = "#34383b"
MUTED = "#6d6d6d"
GRID = "#eeeeee"
ZERO = "#bdbdbd"
THREAT = "#c6533e"
DIPLO = "#2e6f6d"
HUMAN = "#4d5459"
ORANGE = "#e28a1d"
BLUE = "#1f78b4"
ORANGE_LIGHT = "#f7deb3"
BLUE_LIGHT = "#cfe3f2"


def esc(s: object) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def parse_month(s: str) -> date:
    y, m, _ = s.split("-")
    return date(int(y), int(m), 1)


def year_float(d: date) -> float:
    return d.year + (d.month - 1) / 12


def safe_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if math.isnan(x):
        return None
    return x


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rolling_mean(vals: list[float | None], window: int = 12) -> list[float | None]:
    out: list[float | None] = []
    half = window // 2
    for i in range(len(vals)):
        lo = max(0, i - half)
        hi = min(len(vals), i + half + 1)
        xs = [v for v in vals[lo:hi] if v is not None]
        out.append(sum(xs) / len(xs) if len(xs) >= 4 else None)
    return out


def linreg(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    intercept = my - slope * mx
    return slope, intercept


def colormap(v: float | None, vmax: float) -> str:
    if v is None:
        return "#f4f4f4"
    x = max(-vmax, min(vmax, v)) / vmax
    if x >= 0:
        # white -> terracotta red
        t = x
        a = (248, 248, 248)
        b = (181, 58, 48)
    else:
        t = -x
        a = (248, 248, 248)
        b = (48, 114, 154)
    r = round(a[0] + (b[0] - a[0]) * t)
    g = round(a[1] + (b[1] - a[1]) * t)
    b2 = round(a[2] + (b[2] - a[2]) * t)
    return f"#{r:02x}{g:02x}{b2:02x}"


def text(
    x: float,
    y: float,
    s: object,
    size: int = 16,
    color: str = CHAR,
    anchor: str = "start",
    weight: str = "400",
    rotate: float | None = None,
) -> str:
    tr = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial" '
        f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
        f'font-weight="{weight}"{tr}>{esc(s)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", sw: float = 1) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def polyline(points: list[tuple[float, float]], color: str, width: float = 2, dash: str | None = None) -> str:
    if not points:
        return ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    parts: list[str] = []
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}" stroke-linecap="round"{dash_attr}/>'
        )
    return "\n".join(parts)


def svg_wrap(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def convert(svg_path: Path, png_path: Path) -> None:
    font = "/System/Library/Fonts/HelveticaNeue.ttc"
    subprocess.run(
        [
            "magick",
            "-font",
            font,
            str(svg_path),
            "-background",
            "white",
            "-alpha",
            "remove",
            "-alpha",
            "off",
            "-depth",
            "8",
            str(png_path),
        ],
        check=True,
    )


def draw_temporal_drift() -> None:
    rows = read_csv(DATA / "monthly_series_v2.csv")
    dates = [parse_month(r["pub_month"]) for r in rows]
    xvals = [year_float(d) for d in dates]

    series = [
        ("Threat gap", "gap_threat_HL_FT_mean_C1", THREAT),
        ("Diplomacy gap", "gap_diplo_HL_FT_mean_C1", DIPLO),
    ]
    breaks = read_csv(DATA / "detrended_breaks.csv")
    break_map: dict[tuple[str, str], list[float]] = {}
    for r in breaks:
        break_map.setdefault((r["series"], r["kind"]), []).append(year_float(parse_month(r["break_date"])))

    W, H = 1680, 880
    body: list[str] = [rect(0, 0, W, H, "white")]
    body.append(text(56, 42, "Temporal drift and baseline shifts", 28, CHAR, weight="600"))

    plot_w, plot_h = 705, 270
    lefts = [72, 878]
    tops = [102, 462]
    xmin, xmax = 1979, 2026
    xticks = [1980, 1990, 2000, 2010, 2020]

    for row_idx, (label, col, color) in enumerate(series):
        raw = [safe_float(r[col]) for r in rows]
        sm = rolling_mean(raw, 12)
        fit_x = [x for x, y in zip(xvals, sm) if y is not None]
        fit_y = [y for y in sm if y is not None]
        slope, intercept = linreg(fit_x, fit_y)
        residual = [(y - (slope * x + intercept)) if y is not None else None for x, y in zip(xvals, sm)]
        panels = [("original", sm), ("detrended residual", residual)]
        y_ranges = [(min(fit_y) - 3, max(fit_y) + 3), (-13, 13) if row_idx == 0 else (-7, 18)]

        for col_idx, (kind_label, vals) in enumerate(panels):
            x0, y0 = lefts[col_idx], tops[row_idx]
            y_min, y_max = y_ranges[col_idx]
            body.append(text(x0, y0 - 23, f"{label}: {kind_label}", 22, color, weight="600"))
            body.append(rect(x0, y0, plot_w, plot_h, "#fbfbfb", "#222", 1.5))

            def sx(x: float) -> float:
                return x0 + (x - xmin) / (xmax - xmin) * plot_w

            def sy(y: float) -> float:
                return y0 + plot_h - (y - y_min) / (y_max - y_min) * plot_h

            # break bands sit behind the data, so they read clearly after scaling
            if col_idx == 0:
                for bx in break_map.get((col, "original_C1"), []):
                    body.append(rect(sx(bx) - 6, y0, 12, plot_h, ORANGE_LIGHT))
            else:
                for bx in break_map.get((col, "detrended"), []):
                    body.append(rect(sx(bx) - 6, y0, 12, plot_h, BLUE_LIGHT))

            # grid and axes
            for xt in xticks:
                xx = sx(xt)
                body.append(line(xx, y0, xx, y0 + plot_h, GRID, 0.8))
                if row_idx == 1:
                    body.append(text(xx, y0 + plot_h + 30, str(xt), 15, CHAR, "middle"))
            for t in range(math.ceil(y_min / 5) * 5, math.floor(y_max / 5) * 5 + 1, 5):
                yy = sy(t)
                body.append(line(x0, yy, x0 + plot_w, yy, GRID if t != 0 else ZERO, 1.1 if t == 0 else 0.8))
                body.append(text(x0 - 12, yy + 5, str(t), 13, MUTED, "end"))

            pts: list[tuple[float, float]] = []
            for x, y in zip(xvals, vals):
                if y is not None:
                    pts.append((sx(x), sy(y)))
            body.append(polyline(pts, "#5c6063", 2))

            if col_idx == 0:
                trend_pts = [(sx(xmin), sy(slope * xmin + intercept)), (sx(xmax), sy(slope * xmax + intercept))]
                body.append(polyline(trend_pts, color, 5))
                for bx in break_map.get((col, "original_C1"), []):
                    body.append(line(sx(bx), y0, sx(bx), y0 + plot_h, ORANGE, 7))
            else:
                for bx in break_map.get((col, "detrended"), []):
                    body.append(line(sx(bx), y0, sx(bx), y0 + plot_h, BLUE, 7))

    # legend
    yleg = 835
    body.append(polyline([(90, yleg), (130, yleg)], "#5c6063", 2))
    body.append(text(140, yleg + 5, "12-month moving average", 15, MUTED))
    body.append(polyline([(420, yleg), (460, yleg)], THREAT, 5))
    body.append(text(470, yleg + 5, "linear drift", 15, MUTED))
    body.append(rect(682, yleg - 16, 16, 28, ORANGE_LIGHT))
    body.append(line(690, yleg - 16, 690, yleg + 12, ORANGE, 7))
    body.append(text(710, yleg + 5, "PELT break before detrending", 15, MUTED))
    body.append(rect(1032, yleg - 16, 16, 28, BLUE_LIGHT))
    body.append(line(1040, yleg - 16, 1040, yleg + 12, BLUE, 7))
    body.append(text(1060, yleg + 5, "break after detrending", 15, MUTED))

    svg = svg_wrap(W, H, body)
    svg_path = FIG / "fig3_temporal_drift_baseline.svg"
    png_path = FIG / "fig3_temporal_drift_baseline.png"
    svg_path.write_text(svg)
    convert(svg_path, png_path)


def draw_heatmap(
    matrix: list[list[float | None]],
    row_labels: list[str],
    col_labels: list[str],
    group_spans: list[tuple[int, int, str]],
    title: str,
    subtitle: str,
    out_name: str,
    vmax: float,
    width: int,
    height: int,
    left: int = 270,
) -> None:
    n_rows = len(row_labels)
    n_cols = len(col_labels)
    top = 102
    cell_w = (width - left - 92) / n_cols
    cell_h = (height - top - 86) / n_rows
    body: list[str] = [rect(0, 0, width, height, "white")]
    body.append(text(width / 2, 36, title, 27, CHAR, anchor="middle", weight="600"))

    # group labels
    for start, end, label in group_spans:
        x1 = left + start * cell_w
        x2 = left + end * cell_w
        body.append(text((x1 + x2) / 2, top - 36, label, 18, CHAR, "middle", "600"))
        body.append(line(x1, top - 25, x2, top - 25, CHAR, 1.2))
    for j, lab in enumerate(col_labels):
        x = left + j * cell_w + cell_w / 2
        body.append(text(x, top - 10, lab, 13, MUTED, "middle"))

    for i, rlab in enumerate(row_labels):
        y = top + i * cell_h
        body.append(text(left - 12, y + cell_h * 0.63, rlab, 14, CHAR, "end"))
        for j, val in enumerate(matrix[i]):
            x = left + j * cell_w
            fill = colormap(val, vmax)
            body.append(rect(x, y, cell_w, cell_h, fill, "white", 1))
            if val is None:
                # light diagonal hatch for insufficient data
                body.append(line(x + 5, y + cell_h - 5, x + cell_w - 5, y + 5, "#d9d9d9", 1))
            else:
                txt_col = "white" if abs(val) > vmax * 0.55 else "#1f1f1f"
                body.append(text(x + cell_w / 2, y + cell_h * 0.62, f"{val:+.1f}", 11, txt_col, "middle"))

    # separators
    for start, _, _ in group_spans[1:]:
        x = left + start * cell_w
        body.append(line(x, top - 34, x, top + n_rows * cell_h, "#222", 2))

    # color legend
    lx = width - 62
    ly = top + 10
    lh = 260
    steps = 80
    for k in range(steps):
        v = vmax - 2 * vmax * k / (steps - 1)
        body.append(rect(lx, ly + k * lh / steps, 20, lh / steps + 1, colormap(v, vmax)))
    body.append(rect(lx, ly, 20, lh, "none", "#333", 1))
    body.append(text(lx + 32, ly + 8, f"+{vmax:g}", 12, MUTED))
    body.append(text(lx + 32, ly + lh / 2 + 5, "0", 12, MUTED))
    body.append(text(lx + 32, ly + lh + 4, f"-{vmax:g}", 12, MUTED))
    body.append(text(lx + 10, ly + lh + 32, "post-pre", 12, MUTED, "middle"))

    svg_path = FIG / f"{out_name}.svg"
    png_path = FIG / f"{out_name}.png"
    svg_path.write_text(svg_wrap(width, height, body))
    convert(svg_path, png_path)


def draw_layer_event_heatmap() -> None:
    rows = read_csv(DATA / "event_shifts.csv")
    event_order = [
        "Embassy seized",
        "Iran-Iraq war",
        "IR655",
        "Soleimani",
        "JCPOA signed",
        "JCPOA Imp Day",
        "JCPOA exit",
        "Khomeini returns",
        "Khomeini dies",
        "Green Movement",
        "Rouhani elected",
        "Mahsa Amini",
        "Axis of Evil",
    ]
    event_short = {
        "Embassy seized": "Embassy seized",
        "Iran-Iraq war": "Iran-Iraq War",
        "IR655": "IR655",
        "Soleimani": "Soleimani",
        "JCPOA signed": "JCPOA signed",
        "JCPOA Imp Day": "JCPOA implementation",
        "JCPOA exit": "JCPOA exit",
        "Khomeini returns": "Khomeini returns",
        "Khomeini dies": "Khomeini dies",
        "Green Movement": "Green Movement",
        "Rouhani elected": "Rouhani elected",
        "Mahsa Amini": "Mahsa Amini",
        "Axis of Evil": "Axis of Evil speech",
    }
    layers = [("hl", "Headline"), ("ab", "Abstract"), ("ld", "Lead"), ("ft", "Body")]
    axes = [("threat", "Threat"), ("diplo", "Diplomacy"), ("human", "Humanizing")]
    val = {(r["event"], r["axis"], r["layer"]): safe_float(r["shift"]) for r in rows}
    matrix = []
    for ev in event_order:
        row = []
        for ax, _ in axes:
            for layer, _ in layers:
                row.append(val.get((ev, ax, layer)))
        matrix.append(row)
    col_labels = [lab for _ax, _glab in axes for _layer, lab in layers]
    group_spans = [(0, 4, "Threat"), (4, 8, "Diplomacy"), (8, 12, "Humanizing")]
    draw_heatmap(
        matrix,
        [event_short[e] for e in event_order],
        col_labels,
        group_spans,
        "Event-driven shifts across article layers",
        "",
        "fig4_event_layer_heatmap",
        vmax=30,
        width=1680,
        height=960,
        left=270,
    )


def draw_voice_event_heatmap() -> None:
    rows = read_csv(DATA / "voice_event_shifts.csv")
    event_order = [
        "Axis",
        "Green",
        "Rouhani",
        "JCPOA signed",
        "JCPOA impl.",
        "JCPOA exit",
        "Soleimani",
        "Mahsa",
    ]
    event_labels = {
        "Axis": "Axis of Evil speech",
        "Green": "Green Movement",
        "Rouhani": "Rouhani elected",
        "JCPOA signed": "JCPOA signed",
        "JCPOA impl.": "JCPOA implementation",
        "JCPOA exit": "JCPOA exit",
        "Soleimani": "Soleimani killing",
        "Mahsa": "Mahsa Amini",
    }
    buckets = [("news", "News"), ("staff", "Staff columns"), ("guest", "Guest opinion"), ("unsigned", "Editorial")]
    frames = [("threat", "Threat"), ("diplo", "Diplomacy")]
    val = {(r["event"], r["bucket"], r["frame"]): safe_float(r["shift"]) for r in rows}
    matrix = []
    for ev in event_order:
        row = []
        for b, _ in buckets:
            for fr, _ in frames:
                row.append(val.get((ev, b, fr)))
        matrix.append(row)
    col_labels = [lab for _b, _blab in buckets for _f, lab in frames]
    group_spans = [(0, 2, "News"), (2, 4, "Staff columns"), (4, 6, "Guest opinion"), (6, 8, "Editorial")]
    draw_heatmap(
        matrix,
        [event_labels[e] for e in event_order],
        col_labels,
        group_spans,
        "Event-driven shifts across institutional voices",
        "",
        "fig5_event_voice_heatmap",
        vmax=7.5,
        width=1680,
        height=700,
        left=255,
    )


def main() -> None:
    draw_temporal_drift()
    draw_layer_event_heatmap()
    draw_voice_event_heatmap()
    for name in [
        "fig3_temporal_drift_baseline.png",
        "fig4_event_layer_heatmap.png",
        "fig5_event_voice_heatmap.png",
    ]:
        print(FIG / name)


if __name__ == "__main__":
    main()
