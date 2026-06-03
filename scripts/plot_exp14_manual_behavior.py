#!/usr/bin/env python3
"""Plot behavior diagnostics from exp14 manual evaluation."""

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

SCORE_LABELS = {
    0: "0 = wrong / hallucinated",
    1: "1 = mixed / noisy",
    2: "2 = broadly correct",
}

BACKGROUND = "#f7f6f3"
PANEL_BG = "#ffffff"
GRID = "#e4e4e4"
AXIS = "#555555"
TEXT = "#222222"
TEXT_MUTED = "#666666"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create manual-score vs behavior diagnostics for exp14."
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
        default=Path("results/figures/exp14_manual_behavior_diagnostics.svg"),
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
        f'font-family="Arial, Helvetica, sans-serif" text-anchor="{anchor}" '
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
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"{dash_attr} />'
    )


def svg_circle(
    cx: float,
    cy: float,
    r: float,
    *,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 1.0,
    opacity: float = 1.0,
) -> str:
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}" />'
    )


def svg_diamond(
    cx: float,
    cy: float,
    size: float,
    *,
    fill: str,
    stroke: str,
    stroke_width: float = 1.0,
) -> str:
    points = [
        (cx, cy - size),
        (cx + size, cy),
        (cx, cy + size),
        (cx - size, cy),
    ]
    joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polygon points="{joined}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.2f}" />'
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
    df["proxy_answer_token_count"] = pd.to_numeric(df["proxy_answer_token_count"], errors="coerce")
    df["switch_rate"] = pd.to_numeric(df["switch_rate"], errors="coerce")
    return df


def jitter(idx: int, decoder: str) -> float:
    decoder_offset = {
        "exp14_update1": -0.16,
        "exp14_update2": -0.08,
        "exp14_update4": 0.00,
        "exp14_update8": 0.08,
        "exp14_frozen": 0.16,
    }[decoder]
    phase = ((idx * 37) % 11) - 5
    return decoder_offset + phase * 0.013


def correlation(xs: pd.Series, ys: pd.Series) -> float:
    return float(xs.corr(ys))


def draw_scatter_panel(
    svg: list[str],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    subtitle: str,
    x_min: float,
    x_max: float,
    x_ticks: list[float],
) -> None:
    svg.append(svg_rect(left, top, width, height, fill=PANEL_BG, stroke="#dddddd", stroke_width=1.0, rx=14))

    axis_left = left + 78
    axis_right = left + width - 30
    axis_top = top + 64
    axis_bottom = top + height - 84

    svg.append(svg_text(left + width / 2, top + 26, title, size=18, anchor="middle", weight="bold"))
    svg.append(svg_text(left + width / 2, top + 46, subtitle, size=11, fill=TEXT_MUTED, anchor="middle"))

    for score in [0, 1, 2]:
        y = scale(float(score), -0.35, 2.35, axis_bottom, axis_top)
        svg.append(svg_line(axis_left, y, axis_right, y, stroke=GRID, stroke_width=1.0))
        svg.append(svg_text(axis_left - 10, y + 4, SCORE_LABELS[score], size=10, fill=TEXT_MUTED, anchor="end"))

    for tick in x_ticks:
        x = scale(tick, x_min, x_max, axis_left, axis_right)
        svg.append(svg_line(x, axis_top, x, axis_bottom, stroke=GRID, stroke_width=1.0))
        label = f"{int(tick)}" if abs(tick - round(tick)) < 1e-9 else f"{tick:.2f}"
        svg.append(svg_text(x, axis_bottom + 24, label, size=10, fill=TEXT_MUTED, anchor="middle"))

    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_line(axis_left, axis_bottom, axis_right, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_text((axis_left + axis_right) / 2, axis_bottom + 48, x_label, size=11, fill=TEXT_MUTED, anchor="middle"))

    for idx, row in enumerate(df.itertuples(index=False)):
        x = scale(float(getattr(row, x_col)), x_min, x_max, axis_left, axis_right)
        y = scale(float(row.manual_score_0_2) + jitter(idx, row.decoder), -0.35, 2.35, axis_bottom, axis_top)
        svg.append(
            svg_circle(
                x,
                y,
                4.2,
                fill=COLORS[row.decoder],
                stroke="#ffffff",
                stroke_width=0.8,
                opacity=0.82,
            )
        )

    means = df.groupby("manual_score_0_2")[x_col].mean().to_dict()
    for score, mean_value in means.items():
        x = scale(float(mean_value), x_min, x_max, axis_left, axis_right)
        y = scale(float(score), -0.35, 2.35, axis_bottom, axis_top)
        svg.append(svg_diamond(x, y, 8.5, fill="#111111", stroke="#ffffff", stroke_width=1.2))
        svg.append(svg_text(x + 12, y - 8, f"mean {mean_value:.1f}" if x_col == "proxy_answer_token_count" else f"mean {mean_value:.3f}", size=10, fill=TEXT_MUTED))

    corr = correlation(df[x_col], df["manual_score_0_2"])
    svg.append(svg_text(left + width - 24, top + 24, f"corr = {corr:+.2f}", size=11, fill=TEXT_MUTED, anchor="end"))


def draw_decoder_legend(svg: list[str], *, left: float, top: float) -> None:
    cursor_x = left
    for decoder in DECODER_ORDER:
        svg.append(svg_circle(cursor_x + 6, top, 5.5, fill=COLORS[decoder]))
        svg.append(svg_text(cursor_x + 18, top + 4, DISPLAY_LABELS[decoder], size=11, fill=TEXT_MUTED))
        cursor_x += 110


def build_stats_payload(df: pd.DataFrame) -> dict[str, object]:
    by_score = {}
    grouped = df.groupby("manual_score_0_2")
    for score, part in grouped:
        by_score[int(score)] = {
            "count": int(len(part)),
            "mean_answer_token_count": float(part["proxy_answer_token_count"].mean()),
            "median_answer_token_count": float(part["proxy_answer_token_count"].median()),
            "mean_switch_rate": float(part["switch_rate"].mean()),
            "median_switch_rate": float(part["switch_rate"].median()),
        }

    return {
        "total_rows": int(len(df)),
        "correlation_manual_score_vs_answer_length": correlation(df["proxy_answer_token_count"], df["manual_score_0_2"]),
        "correlation_manual_score_vs_switch_rate": correlation(df["switch_rate"], df["manual_score_0_2"]),
        "by_score": by_score,
    }


def build_svg(df: pd.DataFrame, payload: dict[str, object]) -> str:
    width = 1380
    height = 760
    svg: list[str] = []
    svg.append(svg_rect(0, 0, width, height, fill=BACKGROUND))

    svg.append(svg_text(54, 54, "exp14 Manual Behavior Diagnostics", size=28, weight="bold"))
    svg.append(
        svg_text(
            54,
            82,
            "Each dot is one manually judged answer. Diamonds mark the mean x-value within each human score bucket.",
            size=14,
            fill=TEXT_MUTED,
        )
    )

    draw_scatter_panel(
        svg,
        left=46,
        top=124,
        width=620,
        height=560,
        df=df,
        x_col="proxy_answer_token_count",
        x_label="Generated answer length (tokens)",
        title="Manual Score vs Answer Length",
        subtitle="Correct answers are usually shorter; long answers more often drift into extra unsupported detail.",
        x_min=8.0,
        x_max=66.0,
        x_ticks=[10, 20, 30, 40, 50, 60],
    )

    draw_scatter_panel(
        svg,
        left=712,
        top=124,
        width=620,
        height=560,
        df=df,
        x_col="switch_rate",
        x_label="Layer switch rate",
        title="Manual Score vs Switch Rate",
        subtitle="In this open-ended run, higher switch_rate is not obviously harmful; the best decoder was update1.",
        x_min=-0.02,
        x_max=0.98,
        x_ticks=[0.0, 0.2, 0.4, 0.6, 0.8],
    )

    draw_decoder_legend(svg, left=72, top=714)
    svg.append(svg_text(732, 708, f"Length corr: {payload['correlation_manual_score_vs_answer_length']:+.2f}", size=12, fill=TEXT_MUTED))
    svg.append(svg_text(732, 730, f"Switch corr: {payload['correlation_manual_score_vs_switch_rate']:+.2f}", size=12, fill=TEXT_MUTED))

    return wrap_svg(svg, width, height)


def main() -> None:
    args = parse_args()
    stats_output = args.stats_output or args.output.with_suffix(".json")

    df = load_manual_eval(args.manual_csv)
    payload = build_stats_payload(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(df, payload), encoding="utf-8")
    stats_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote figure to {args.output}")
    print(f"Wrote stats to {stats_output}")


if __name__ == "__main__":
    main()
