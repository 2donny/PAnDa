#!/usr/bin/env python3
"""Create simpler presentation-friendly exp12 mechanism figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_exp12_state_persistence_hypothesis import (
    COLORS,
    DECODER_ORDER,
    DISPLAY_LABELS,
    QUALITY_ORDER,
    build_stats,
    load_mechanism_table,
    scale,
    svg_line,
    svg_rect,
    svg_text,
)


BACKGROUND = "#ffffff"
GRID = "#d9d9d9"
AXIS = "#222222"
TEXT_MUTED = "#444444"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create three simpler exp12 figures for presentation use."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(
            "results/experiments/exp12_state_persistence_diagnostics/runs/run_01_default/"
            "run_01_default_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="exp12_state_persistence",
    )
    return parser.parse_args()


def wrap_svg(body: list[str], width: int, height: int) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            *body,
            "</svg>",
        ]
    )


def mechanism_map(mechanism_df):
    return {
        str(row.decoder): {
            "switch_rate": float(row.switch_rate),
            "selected_layer_match_rate": float(row.selected_layer_match_rate),
            "avg_oracle_jsd_gap": float(row.avg_oracle_jsd_gap),
            "refresh_rate": float(row.refresh_rate),
            "mc2": float(row.score_mean),
        }
        for row in mechanism_df.itertuples()
    }


def quality_map(quality_df):
    return {
        (str(row.decoder), str(row.metric_name)): float(row.score_mean)
        for row in quality_df.itertuples()
    }


def draw_bar_chart(
    svg: list[str],
    values: dict[str, float],
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    y_min: float,
    y_max: float,
    ticks: list[float],
    higher_is_better: bool,
) -> None:
    axis_left = left + 72
    axis_right = left + width - 24
    axis_top = top + 54
    axis_bottom = top + height - 72
    axis_w = axis_right - axis_left
    axis_h = axis_bottom - axis_top

    svg.append(svg_rect(left, top, width, height, fill="#ffffff", stroke="#dddddd", rx=14))
    svg.append(svg_text(left + width / 2, top + 24, title, size=17, anchor="middle", weight="bold"))

    for tick in ticks:
        y = scale(tick, y_min, y_max, axis_bottom, axis_top)
        svg.append(svg_line(axis_left, y, axis_right, y, stroke=GRID, stroke_width=1.0))
        svg.append(svg_text(axis_left - 10, y + 4, f"{tick:.2f}", size=10, fill="#666666", anchor="end"))

    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke=AXIS, stroke_width=1.2))
    svg.append(svg_line(axis_left, axis_bottom, axis_right, axis_bottom, stroke=AXIS, stroke_width=1.2))

    band = axis_w / len(DECODER_ORDER)
    bar_w = min(72.0, band * 0.58)
    for idx, decoder in enumerate(DECODER_ORDER):
        x_center = axis_left + band * (idx + 0.5)
        value = values[decoder]
        y = scale(value, y_min, y_max, axis_bottom, axis_top)
        svg.append(
            svg_rect(
                x_center - bar_w / 2,
                y,
                bar_w,
                max(1.0, axis_bottom - y),
                fill=COLORS[decoder],
                rx=7,
            )
        )
        svg.append(svg_text(x_center, max(axis_top + 14, y - 9), f"{value:.3f}", size=11, fill=COLORS[decoder], anchor="middle", weight="bold"))
        svg.append(svg_text(x_center, axis_bottom + 22, DISPLAY_LABELS[decoder], size=11, anchor="middle"))


def build_switch_rate_svg(mechanism_df, stats: dict[str, object]) -> str:
    width = 920
    height = 390
    values = {decoder: mechanism_df.loc[mechanism_df["decoder"] == decoder, "switch_rate"].iloc[0] for decoder in DECODER_ORDER}
    body = [svg_rect(0, 0, width, height, fill=BACKGROUND)]
    draw_bar_chart(
        body,
        values,
        left=28,
        top=18,
        width=864,
        height=344,
        title="Layer switch rate",
        subtitle="",
        y_min=0.0,
        y_max=0.8,
        ticks=[0.0, 0.2, 0.4, 0.6, 0.8],
        higher_is_better=False,
    )
    return wrap_svg(body, width, height)


def build_staleness_svg(mechanism_df, stats: dict[str, object]) -> str:
    width = 1180
    height = 486
    mechanism = mechanism_map(mechanism_df)
    match_values = {decoder: mechanism[decoder]["selected_layer_match_rate"] for decoder in DECODER_ORDER}
    gap_values = {decoder: mechanism[decoder]["avg_oracle_jsd_gap"] for decoder in DECODER_ORDER}
    body = [svg_rect(0, 0, width, height, fill=BACKGROUND)]

    draw_bar_chart(
        body,
        match_values,
        left=26,
        top=26,
        width=554,
        height=412,
        title="Selected-layer match rate",
        subtitle="",
        y_min=0.0,
        y_max=1.05,
        ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        higher_is_better=True,
    )
    draw_bar_chart(
        body,
        gap_values,
        left=600,
        top=26,
        width=554,
        height=412,
        title="Selected-layer gap",
        subtitle="",
        y_min=0.0,
        y_max=0.0038,
        ticks=[0.0, 0.001, 0.002, 0.003],
        higher_is_better=False,
    )
    return wrap_svg(body, width, height)


def build_quality_svg(quality_df, stats: dict[str, object]) -> str:
    width = 980
    height = 420
    scores = quality_map(quality_df)
    body = [svg_rect(0, 0, width, height, fill=BACKGROUND)]
    left = 28
    top = 18
    chart_w = 924
    chart_h = 364
    axis_left = left + 74
    axis_right = left + chart_w - 26
    axis_top = top + 40
    axis_bottom = top + chart_h - 58
    axis_w = axis_right - axis_left
    axis_h = axis_bottom - axis_top

    body.append(svg_rect(left, top, chart_w, chart_h, fill="#ffffff", stroke="#dddddd", rx=14))
    body.append(svg_text(left + chart_w / 2, top + 24, "TruthfulQA MC quality", size=17, anchor="middle", weight="bold"))

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = scale(tick, 0.0, 1.0, axis_bottom, axis_top)
        body.append(svg_line(axis_left, y, axis_right, y, stroke=GRID, stroke_width=1.0))
        body.append(svg_text(axis_left - 10, y + 4, f"{tick:.2f}", size=10, fill="#666666", anchor="end"))
    body.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke=AXIS, stroke_width=1.2))
    body.append(svg_line(axis_left, axis_bottom, axis_right, axis_bottom, stroke=AXIS, stroke_width=1.2))

    group_w = axis_w / len(QUALITY_ORDER)
    bar_w = 28
    spacing = 8
    center_offset = (len(DECODER_ORDER) - 1) / 2.0
    for metric_idx, metric in enumerate(QUALITY_ORDER):
        x_center = axis_left + group_w * (metric_idx + 0.5)
        for decoder_idx, decoder in enumerate(DECODER_ORDER):
            value = scores[(decoder, metric)]
            x = x_center + (decoder_idx - center_offset) * (bar_w + spacing)
            y = scale(value, 0.0, 1.0, axis_bottom, axis_top)
            body.append(svg_rect(x - bar_w / 2, y, bar_w, max(1.0, axis_bottom - y), fill=COLORS[decoder], rx=6))
            body.append(svg_text(x, max(axis_top + 14, y - 8), f"{value:.3f}", size=10, fill=COLORS[decoder], anchor="middle"))
        body.append(svg_text(x_center, axis_bottom + 24, metric.upper(), size=11, anchor="middle"))

    legend_x = left + 22
    legend_y = top + 72
    for idx, decoder in enumerate(DECODER_ORDER):
        y = legend_y + idx * 18
        body.append(svg_rect(legend_x, y - 9, 12, 12, fill=COLORS[decoder], rx=3))
        body.append(svg_text(legend_x + 18, y + 1, DISPLAY_LABELS[decoder], size=10))

    return wrap_svg(body, width, height)


def main() -> None:
    args = parse_args()
    mechanism_df, quality_df = load_mechanism_table(args.summary_csv)
    stats = build_stats(mechanism_df, quality_df)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    switch_path = output_dir / f"{args.prefix}_switch_rate.svg"
    stale_path = output_dir / f"{args.prefix}_staleness.svg"
    quality_path = output_dir / f"{args.prefix}_quality.svg"
    manifest_path = output_dir / f"{args.prefix}_panels.json"

    switch_path.write_text(build_switch_rate_svg(mechanism_df, stats), encoding="utf-8")
    stale_path.write_text(build_staleness_svg(mechanism_df, stats), encoding="utf-8")
    quality_path.write_text(build_quality_svg(quality_df, stats), encoding="utf-8")

    manifest = {
        "switch_rate_figure": str(switch_path),
        "staleness_figure": str(stale_path),
        "quality_figure": str(quality_path),
        "source_summary_csv": str(args.summary_csv),
        "key_claims": stats["key_claims"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
