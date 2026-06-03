#!/usr/bin/env python3
"""Create a simple summary figure for exp14 open-ended factuality results."""

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

BACKGROUND = "#f7f6f3"
PANEL_BG = "#ffffff"
GRID = "#e4e4e4"
AXIS = "#555555"
TEXT = "#222222"
TEXT_MUTED = "#666666"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a presentation-friendly exp14 summary figure."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(
            "results/experiments/exp14_openended_factuality/runs/run_01_default/"
            "run_01_default_summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/exp14_openended_factuality_summary.svg"),
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


def round_up(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def round_down(value: float, step: float) -> float:
    return math.floor(value / step) * step


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["decoder"].isin(DECODER_ORDER)].copy()
    return df


def metric_lookup(df: pd.DataFrame, metric_name: str) -> dict[str, dict[str, float]]:
    rows = df[df["metric_name"] == metric_name]
    return {
        str(row.decoder): {
            "score_mean": float(row.score_mean),
            "score_sem": float(row.score_sem),
            "switch_rate": float(row.switch_rate),
            "selected_layer_match_rate": float(row.selected_layer_match_rate),
            "avg_oracle_jsd_gap": float(row.avg_oracle_jsd_gap),
        }
        for row in rows.itertuples()
    }


def bar_top_and_bottom(value: float, zero_value: float, y_min: float, y_max: float, axis_top: float, axis_bottom: float) -> tuple[float, float]:
    zero_y = scale(zero_value, y_min, y_max, axis_bottom, axis_top)
    value_y = scale(value, y_min, y_max, axis_bottom, axis_top)
    return min(zero_y, value_y), max(zero_y, value_y)


def draw_bar_panel(
    svg: list[str],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    values: dict[str, float],
    sems: dict[str, float] | None,
    y_min: float,
    y_max: float,
    ticks: list[float],
    better: str,
) -> None:
    svg.append(svg_rect(left, top, width, height, fill=PANEL_BG, stroke="#dddddd", stroke_width=1.0, rx=14))

    axis_left = left + 72
    axis_right = left + width - 24
    axis_top = top + 58
    axis_bottom = top + height - 72
    axis_width = axis_right - axis_left

    svg.append(svg_text(left + width / 2, top + 24, title, size=18, anchor="middle", weight="bold"))
    svg.append(svg_text(left + width / 2, top + 44, subtitle, size=11, fill=TEXT_MUTED, anchor="middle"))

    for tick in ticks:
        y = scale(tick, y_min, y_max, axis_bottom, axis_top)
        svg.append(svg_line(axis_left, y, axis_right, y, stroke=GRID, stroke_width=1.0))
        label = f"{tick:.3f}" if abs(tick) < 0.1 else f"{tick:.2f}"
        svg.append(svg_text(axis_left - 10, y + 4, label, size=10, fill=TEXT_MUTED, anchor="end"))

    zero_tick = 0.0 if y_min <= 0.0 <= y_max else y_min
    zero_y = scale(zero_tick, y_min, y_max, axis_bottom, axis_top)
    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_line(axis_left, zero_y, axis_right, zero_y, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_text(left + width / 2, top + height - 28, better, size=11, fill=TEXT_MUTED, anchor="middle"))

    band = axis_width / len(DECODER_ORDER)
    bar_width = min(62.0, band * 0.6)
    for idx, decoder in enumerate(DECODER_ORDER):
        x_center = axis_left + band * (idx + 0.5)
        value = values[decoder]
        bar_top, bar_bottom = bar_top_and_bottom(value, zero_tick, y_min, y_max, axis_top, axis_bottom)
        svg.append(
            svg_rect(
                x_center - bar_width / 2,
                bar_top,
                bar_width,
                max(1.0, bar_bottom - bar_top),
                fill=COLORS[decoder],
                rx=8,
            )
        )

        if sems is not None:
            sem = sems[decoder]
            y_hi = scale(value + sem, y_min, y_max, axis_bottom, axis_top)
            y_lo = scale(value - sem, y_min, y_max, axis_bottom, axis_top)
            svg.append(svg_line(x_center, y_hi, x_center, y_lo, stroke="#1f1f1f", stroke_width=1.2))
            svg.append(svg_line(x_center - 6, y_hi, x_center + 6, y_hi, stroke="#1f1f1f", stroke_width=1.2))
            svg.append(svg_line(x_center - 6, y_lo, x_center + 6, y_lo, stroke="#1f1f1f", stroke_width=1.2))

        label_y = bar_top - 10 if value >= zero_tick else bar_bottom + 16
        svg.append(
            svg_text(
                x_center,
                label_y,
                f"{value:.3f}",
                size=11,
                fill=COLORS[decoder],
                anchor="middle",
                weight="bold",
            )
        )
        svg.append(svg_text(x_center, axis_bottom + 22, DISPLAY_LABELS[decoder], size=11, anchor="middle"))


def build_stats_payload(summary_df: pd.DataFrame) -> dict[str, object]:
    margin = metric_lookup(summary_df, "oref_margin")
    true_f1 = metric_lookup(summary_df, "oref_true_f1")
    win = metric_lookup(summary_df, "oref_win")

    leaders = {
        "best_oref_margin": max(DECODER_ORDER, key=lambda name: margin[name]["score_mean"]),
        "best_oref_true_f1": max(DECODER_ORDER, key=lambda name: true_f1[name]["score_mean"]),
        "best_oref_win": max(DECODER_ORDER, key=lambda name: win[name]["score_mean"]),
        "lowest_switch_rate": min(DECODER_ORDER, key=lambda name: margin[name]["switch_rate"]),
        "highest_selected_layer_match_rate": max(
            DECODER_ORDER,
            key=lambda name: margin[name]["selected_layer_match_rate"],
        ),
    }

    return {
        "figure_claim": (
            "Exp14 tests whether longer open-ended generations expose a clearer refresh-schedule "
            "tradeoff than the earlier short-answer multiple-choice setup."
        ),
        "leaders": leaders,
        "metrics": {
            metric_name: {
                decoder: metric_lookup(summary_df, metric_name)[decoder]
                for decoder in DECODER_ORDER
            }
            for metric_name in ("oref_margin", "oref_true_f1", "oref_win")
        },
        "mechanism": {
            decoder: {
                "switch_rate": margin[decoder]["switch_rate"],
                "selected_layer_match_rate": margin[decoder]["selected_layer_match_rate"],
                "avg_oracle_jsd_gap": margin[decoder]["avg_oracle_jsd_gap"],
            }
            for decoder in DECODER_ORDER
        },
    }


def main() -> None:
    args = parse_args()
    summary_df = load_summary(args.summary_csv)

    margin = metric_lookup(summary_df, "oref_margin")
    true_f1 = metric_lookup(summary_df, "oref_true_f1")
    win = metric_lookup(summary_df, "oref_win")

    stats_payload = build_stats_payload(summary_df)

    width = 1540
    height = 920
    body = [svg_rect(0, 0, width, height, fill=BACKGROUND)]
    body.append(
        svg_text(
            width / 2,
            34,
            "Exp14: Open-Ended Factuality vs Selected-Layer Refresh Schedule",
            size=24,
            anchor="middle",
            weight="bold",
        )
    )
    body.append(
        svg_text(
            width / 2,
            58,
            "Longer generated answers on TruthfulQA questions; quality panels use the saved reference-bank proxy metrics.",
            size=12,
            fill=TEXT_MUTED,
            anchor="middle",
        )
    )

    top_y = 84
    top_h = 330
    panel_w = 470
    gap = 24
    lefts_top = [28, 28 + panel_w + gap, 28 + 2 * (panel_w + gap)]

    margin_abs = max(abs(margin[name]["score_mean"]) + abs(margin[name]["score_sem"]) for name in DECODER_ORDER)
    margin_limit = round_up(max(0.01, margin_abs * 1.4), 0.005)
    margin_ticks = [
        -margin_limit,
        -margin_limit / 2.0,
        0.0,
        margin_limit / 2.0,
        margin_limit,
    ]

    draw_bar_panel(
        body,
        left=lefts_top[0],
        top=top_y,
        width=panel_w,
        height=top_h,
        title="Primary Quality: oref_margin",
        subtitle="Higher means the answer matches true references more than false references.",
        values={name: margin[name]["score_mean"] for name in DECODER_ORDER},
        sems={name: margin[name]["score_sem"] for name in DECODER_ORDER},
        y_min=-margin_limit,
        y_max=margin_limit,
        ticks=margin_ticks,
        better="Higher is better",
    )
    draw_bar_panel(
        body,
        left=lefts_top[1],
        top=top_y,
        width=panel_w,
        height=top_h,
        title="Truth Match: oref_true_f1",
        subtitle="Best token-overlap F1 against any true reference answer.",
        values={name: true_f1[name]["score_mean"] for name in DECODER_ORDER},
        sems={name: true_f1[name]["score_sem"] for name in DECODER_ORDER},
        y_min=0.0,
        y_max=0.26,
        ticks=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
        better="Higher is better",
    )
    draw_bar_panel(
        body,
        left=lefts_top[2],
        top=top_y,
        width=panel_w,
        height=top_h,
        title="Open-Ended Win Rate: oref_win",
        subtitle="Fraction of questions where true-reference overlap beats false-reference overlap.",
        values={name: win[name]["score_mean"] for name in DECODER_ORDER},
        sems={name: win[name]["score_sem"] for name in DECODER_ORDER},
        y_min=0.0,
        y_max=0.60,
        ticks=[0.0, 0.15, 0.30, 0.45, 0.60],
        better="Higher is better",
    )

    bottom_y = 454
    bottom_h = 340
    bottom_panel_w = 730
    bottom_lefts = [28, 782]

    draw_bar_panel(
        body,
        left=bottom_lefts[0],
        top=bottom_y,
        width=bottom_panel_w,
        height=bottom_h,
        title="Mechanism: switch_rate",
        subtitle="How often the decoder changes its selected correction layer during the answer.",
        values={name: margin[name]["switch_rate"] for name in DECODER_ORDER},
        sems=None,
        y_min=0.0,
        y_max=0.75,
        ticks=[0.0, 0.15, 0.30, 0.45, 0.60, 0.75],
        better="Lower is better",
    )
    draw_bar_panel(
        body,
        left=bottom_lefts[1],
        top=bottom_y,
        width=bottom_panel_w,
        height=bottom_h,
        title="Mechanism: selected_layer_match_rate",
        subtitle="How often the carried layer still matches the step-local best layer.",
        values={name: margin[name]["selected_layer_match_rate"] for name in DECODER_ORDER},
        sems=None,
        y_min=0.0,
        y_max=1.00,
        ticks=[0.0, 0.25, 0.50, 0.75, 1.0],
        better="Higher is better",
    )

    svg = wrap_svg(body, width, height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")

    stats_path = args.stats_output or args.output.with_suffix(".json")
    stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_csv": str(args.summary_csv),
                "output": str(args.output),
                "stats_output": str(stats_path),
                "leaders": stats_payload["leaders"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
