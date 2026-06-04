#!/usr/bin/env python3
"""Create a summary figure for exp14 manual open-ended evaluation."""

from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path

import pandas as pd


DECODER_ORDER = (
    "exp14_update1",
    "exp14_update2",
    "exp14_update4",
    "exp14_update8",
    "exp14_frozen",
)

DISPLAY_LABELS = {
    "exp14_update1": "update1",
    "exp14_update2": "update2",
    "exp14_update4": "update4",
    "exp14_update8": "update8",
    "exp14_frozen": "frozen",
}

COLORS = {
    "exp14_update1": "#c8553d",
    "exp14_update2": "#2a9d8f",
    "exp14_update4": "#457b9d",
    "exp14_update8": "#b08968",
    "exp14_frozen": "#6d597a",
}

SCORE_COLORS = {
    0: "#d1495b",
    1: "#e9c46a",
    2: "#2a9d8f",
}

BACKGROUND = "#ffffff"
PANEL_BG = "#ffffff"
GRID = "#d9d9d9"
AXIS = "#222222"
TEXT = "#222222"
TEXT_MUTED = "#444444"
FONT_FAMILY = "Times New Roman, Times, serif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a presentation-friendly manual-eval summary for exp14."
    )
    parser.add_argument(
        "--manual-csv",
        type=Path,
        default=Path(
            "results/experiments/exp14_openended_factuality/runs/run_01_default/"
            "run_01_default_manual_eval.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/exp14_manual_eval_summary.svg"),
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional JSON path. Defaults to the image path with a .json suffix.",
    )
    return parser.parse_args()


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    fill: str = TEXT,
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
        f'font-family="{FONT_FAMILY}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{escape(text)}</text>'
    )


def svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 1.0,
    rx: float = 0.0,
) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" rx="{rx:.2f}" />'
    )


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = AXIS,
    stroke_width: float = 1.0,
) -> str:
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" />'
    )


def wrap_svg(body: list[str], width: int, height: int) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            *body,
            "</svg>",
        ]
    )


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def load_manual_eval(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df = df[df["decoder"].isin(DECODER_ORDER)].copy()
    df["manual_score_0_2"] = pd.to_numeric(df["manual_score_0_2"])
    return df


def build_stats(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for decoder in DECODER_ORDER:
        part = df[df["decoder"] == decoder]["manual_score_0_2"]
        counts = part.value_counts().to_dict()
        mean = float(part.mean())
        std = float(part.std(ddof=1))
        count = int(part.count())
        sem = std / math.sqrt(count)
        stats[decoder] = {
            "count": count,
            "mean_manual_score": mean,
            "sem_manual_score": sem,
            "score0_count": int(counts.get(0, 0)),
            "score1_count": int(counts.get(1, 0)),
            "score2_count": int(counts.get(2, 0)),
            "score0_rate": float(counts.get(0, 0) / count),
            "score1_rate": float(counts.get(1, 0) / count),
            "score2_rate": float(counts.get(2, 0) / count),
        }
    return stats


def draw_mean_panel(
    svg: list[str],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    stats: dict[str, dict[str, float | int]],
) -> None:
    svg.append(svg_rect(left, top, width, height, fill=PANEL_BG, stroke="#dddddd", stroke_width=1.0, rx=14))

    axis_left = left + 76
    axis_right = left + width - 24
    axis_top = top + 58
    axis_bottom = top + height - 72
    axis_width = axis_right - axis_left

    svg.append(svg_text(left + width / 2, top + 24, "Mean manual score", size=17, anchor="middle", weight="bold"))

    ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
    for tick in ticks:
        y = scale(tick, 0.0, 2.0, axis_bottom, axis_top)
        svg.append(svg_line(axis_left, y, axis_right, y, stroke=GRID, stroke_width=1.0))
        svg.append(svg_text(axis_left - 10, y + 4, f"{tick:.1f}", size=10, fill=TEXT_MUTED, anchor="end"))

    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_line(axis_left, axis_bottom, axis_right, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_text(left + width / 2, top + height - 28, "Higher is better", size=11, fill=TEXT_MUTED, anchor="middle"))

    band = axis_width / len(DECODER_ORDER)
    bar_width = min(62.0, band * 0.6)
    leader = max(DECODER_ORDER, key=lambda key: float(stats[key]["mean_manual_score"]))
    for idx, decoder in enumerate(DECODER_ORDER):
        x_center = axis_left + band * (idx + 0.5)
        mean = float(stats[decoder]["mean_manual_score"])
        sem = float(stats[decoder]["sem_manual_score"])
        y = scale(mean, 0.0, 2.0, axis_bottom, axis_top)
        bar_top = y
        bar_height = max(1.0, axis_bottom - y)
        stroke = "#1f1f1f" if decoder == leader else "none"
        stroke_width = 1.4 if decoder == leader else 1.0
        svg.append(
            svg_rect(
                x_center - bar_width / 2,
                bar_top,
                bar_width,
                bar_height,
                fill=COLORS[decoder],
                stroke=stroke,
                stroke_width=stroke_width,
                rx=8,
            )
        )

        y_hi = scale(min(2.0, mean + sem), 0.0, 2.0, axis_bottom, axis_top)
        y_lo = scale(max(0.0, mean - sem), 0.0, 2.0, axis_bottom, axis_top)
        svg.append(svg_line(x_center, y_hi, x_center, y_lo, stroke="#1f1f1f", stroke_width=1.2))
        svg.append(svg_line(x_center - 6, y_hi, x_center + 6, y_hi, stroke="#1f1f1f", stroke_width=1.2))
        svg.append(svg_line(x_center - 6, y_lo, x_center + 6, y_lo, stroke="#1f1f1f", stroke_width=1.2))

        svg.append(svg_text(x_center, bar_top - 10, f"{mean:.2f}", size=12, weight="bold", anchor="middle"))
        svg.append(svg_text(x_center, axis_bottom + 18, DISPLAY_LABELS[decoder], size=11, anchor="middle"))


def draw_distribution_panel(
    svg: list[str],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    stats: dict[str, dict[str, float | int]],
) -> None:
    svg.append(svg_rect(left, top, width, height, fill=PANEL_BG, stroke="#dddddd", stroke_width=1.0, rx=14))

    axis_left = left + 76
    axis_right = left + width - 24
    axis_top = top + 58
    axis_bottom = top + height - 92
    axis_width = axis_right - axis_left

    svg.append(svg_text(left + width / 2, top + 24, "Score distribution", size=17, anchor="middle", weight="bold"))

    ticks = [0.0, 0.25, 0.50, 0.75, 1.0]
    for tick in ticks:
        y = scale(tick, 0.0, 1.0, axis_bottom, axis_top)
        svg.append(svg_line(axis_left, y, axis_right, y, stroke=GRID, stroke_width=1.0))
        svg.append(svg_text(axis_left - 10, y + 4, f"{int(tick * 100)}%", size=10, fill=TEXT_MUTED, anchor="end"))

    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_line(axis_left, axis_bottom, axis_right, axis_bottom, stroke=AXIS, stroke_width=1.1))

    band = axis_width / len(DECODER_ORDER)
    bar_width = min(58.0, band * 0.52)
    for idx, decoder in enumerate(DECODER_ORDER):
        x_left = axis_left + band * idx + (band - bar_width) / 2
        running_top = axis_bottom
        for score in [0, 1, 2]:
            frac = float(stats[decoder][f"score{score}_rate"])
            segment_top = scale(frac, 0.0, 1.0, 0.0, axis_bottom - axis_top)
            seg_height = segment_top
            y_top = running_top - seg_height
            svg.append(
                svg_rect(
                    x_left,
                    y_top,
                    bar_width,
                    seg_height,
                    fill=SCORE_COLORS[score],
                    stroke="#ffffff",
                    stroke_width=0.8,
                    rx=0,
                )
            )
            count = int(stats[decoder][f"score{score}_count"])
            if frac >= 0.16:
                svg.append(
                    svg_text(
                        x_left + bar_width / 2,
                        y_top + seg_height / 2 + 4,
                        f"{count}",
                        size=11,
                        fill="#ffffff" if score in (0, 2) else TEXT,
                        anchor="middle",
                        weight="bold",
                    )
                )
            running_top = y_top

        svg.append(svg_text(x_left + bar_width / 2, axis_bottom + 18, DISPLAY_LABELS[decoder], size=11, anchor="middle"))

    legend_y = top + height - 44
    legend_x = left + 64
    labels = {
        0: "0 = wrong / hallucinated",
        1: "1 = mixed / noisy",
        2: "2 = broadly correct",
    }
    cursor = legend_x
    for score in [0, 1, 2]:
        svg.append(svg_rect(cursor, legend_y - 10, 14, 14, fill=SCORE_COLORS[score], rx=3))
        svg.append(svg_text(cursor + 20, legend_y + 2, labels[score], size=11, fill=TEXT_MUTED))
        cursor += 170


def build_svg(stats: dict[str, dict[str, float | int]]) -> str:
    width = 1180
    height = 650
    svg: list[str] = []
    svg.append(svg_rect(0, 0, width, height, fill=BACKGROUND))

    draw_mean_panel(svg, left=34, top=28, width=520, height=580, stats=stats)
    draw_distribution_panel(svg, left=594, top=28, width=552, height=580, stats=stats)

    return wrap_svg(svg, width, height)


def main() -> None:
    args = parse_args()
    stats_output = args.stats_output or args.output.with_suffix(".json")

    df = load_manual_eval(args.manual_csv)
    stats = build_stats(df)

    payload = {
        "manual_csv": str(args.manual_csv),
        "total_rows": int(len(df)),
        "stats": stats,
        "ranking_by_mean_manual_score": [
            {
                "decoder": decoder,
                "label": DISPLAY_LABELS[decoder],
                "mean_manual_score": float(stats[decoder]["mean_manual_score"]),
            }
            for decoder in sorted(
                DECODER_ORDER,
                key=lambda key: float(stats[key]["mean_manual_score"]),
                reverse=True,
            )
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(stats), encoding="utf-8")
    stats_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote figure to {args.output}")
    print(f"Wrote stats to {stats_output}")


if __name__ == "__main__":
    main()
