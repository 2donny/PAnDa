#!/usr/bin/env python3
"""Plot prefix-probe summaries for exp15."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import pandas as pd


DECODER_ORDER = ("exp14_frozen", "exp14_update1")
DISPLAY_LABELS = {
    "exp14_frozen": "frozen",
    "exp14_update1": "update1",
}
COLORS = {
    "exp14_frozen": "#6d597a",
    "exp14_update1": "#c8553d",
}

BACKGROUND = "#f7f6f3"
PANEL_BG = "#ffffff"
GRID = "#e4e4e4"
AXIS = "#555555"
TEXT = "#222222"
TEXT_MUTED = "#666666"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a presentation-friendly exp15 prefix-probe summary figure."
    )
    parser.add_argument(
        "--prefix-csv",
        type=Path,
        default=Path(
            "results/experiments/exp15_prefix_probe/runs/run_01_default/"
            "run_01_default_prefix_manual_eval.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/exp15_prefix_probe_summary.svg"),
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
) -> str:
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
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


def load_prefix_eval(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, keep_default_na=False)
    df = df[df["decoder"].isin(DECODER_ORDER)].copy()
    df["proxy_oref_margin"] = pd.to_numeric(df["proxy_oref_margin"], errors="coerce")
    df["manual_score_num"] = pd.to_numeric(df["manual_score_0_2"], errors="coerce")

    budgets = []
    for value in df["prefix_variant"].astype(str):
        if value == "full":
            continue
        if value.startswith("word_"):
            try:
                budgets.append(int(value.split("_", 1)[1]))
            except ValueError:
                continue
    prefix_order = [f"word_{budget}" for budget in sorted(set(budgets))]
    if "full" in set(df["prefix_variant"].astype(str)):
        prefix_order.append("full")
    return df, prefix_order


def prefix_label(prefix_variant: str) -> str:
    if prefix_variant == "full":
        return "full"
    return prefix_variant.replace("word_", "")


def build_stats(df: pd.DataFrame, prefix_order: list[str]) -> dict[str, object]:
    proxy_means = {}
    manual_means = {}
    for decoder in DECODER_ORDER:
        proxy_means[decoder] = {}
        manual_means[decoder] = {}
        part = df[df["decoder"] == decoder]
        for prefix in prefix_order:
            block = part[part["prefix_variant"] == prefix]
            proxy_means[decoder][prefix] = (
                None if block.empty else float(block["proxy_oref_margin"].mean())
            )
            reviewed = block["manual_score_num"].dropna()
            manual_means[decoder][prefix] = (
                None if reviewed.empty else float(reviewed.mean())
            )

    return {
        "total_rows": int(len(df)),
        "question_count": int(df["example_idx"].nunique()),
        "prefix_order": prefix_order,
        "reviewed_prefix_rows": int(df["manual_score_num"].notna().sum()),
        "proxy_margin_means": proxy_means,
        "manual_score_means": manual_means,
    }


def draw_legend(svg: list[str], *, left: float, top: float) -> None:
    cursor_x = left
    for decoder in DECODER_ORDER:
        svg.append(svg_circle(cursor_x + 6, top, 5.0, fill=COLORS[decoder]))
        svg.append(svg_text(cursor_x + 18, top + 4, DISPLAY_LABELS[decoder], size=11, fill=TEXT_MUTED))
        cursor_x += 110


def draw_line_panel(
    svg: list[str],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    prefix_order: list[str],
    values_by_decoder: dict[str, dict[str, float | None]],
    title: str,
    subtitle: str,
    y_label: str,
    y_min: float,
    y_max: float,
    ticks: list[float],
    empty_message: str | None = None,
) -> None:
    svg.append(svg_rect(left, top, width, height, fill=PANEL_BG, stroke="#dddddd", stroke_width=1.0, rx=14))
    svg.append(svg_text(left + width / 2, top + 24, title, size=18, anchor="middle", weight="bold"))
    svg.append(svg_text(left + width / 2, top + 44, subtitle, size=11, fill=TEXT_MUTED, anchor="middle"))

    axis_left = left + 72
    axis_right = left + width - 28
    axis_top = top + 62
    axis_bottom = top + height - 72

    for tick in ticks:
        y = scale(tick, y_min, y_max, axis_bottom, axis_top)
        svg.append(svg_line(axis_left, y, axis_right, y, stroke=GRID, stroke_width=1.0))
        tick_label = f"{tick:.2f}" if abs(tick) < 1 else f"{tick:.1f}"
        svg.append(svg_text(axis_left - 10, y + 4, tick_label, size=10, fill=TEXT_MUTED, anchor="end"))

    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_line(axis_left, axis_bottom, axis_right, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_text(left + 20, (axis_top + axis_bottom) / 2, y_label, size=11, fill=TEXT_MUTED))

    if len(prefix_order) == 1:
        x_positions = {prefix_order[0]: (axis_left + axis_right) / 2}
    else:
        step = (axis_right - axis_left) / (len(prefix_order) - 1)
        x_positions = {prefix: axis_left + idx * step for idx, prefix in enumerate(prefix_order)}

    for prefix in prefix_order:
        x = x_positions[prefix]
        svg.append(svg_line(x, axis_top, x, axis_bottom, stroke=GRID, stroke_width=1.0, dash="3 4"))
        svg.append(svg_text(x, axis_bottom + 24, prefix_label(prefix), size=10, fill=TEXT_MUTED, anchor="middle"))

    any_points = False
    for decoder in DECODER_ORDER:
        points = []
        for prefix in prefix_order:
            value = values_by_decoder.get(decoder, {}).get(prefix)
            if value is None:
                continue
            points.append((x_positions[prefix], scale(float(value), y_min, y_max, axis_bottom, axis_top), float(value)))
        if not points:
            continue
        any_points = True
        for idx in range(1, len(points)):
            x1, y1, _ = points[idx - 1]
            x2, y2, _ = points[idx]
            svg.append(svg_line(x1, y1, x2, y2, stroke=COLORS[decoder], stroke_width=2.2))
        for x, y, value in points:
            svg.append(svg_circle(x, y, 5.0, fill=COLORS[decoder], stroke="#ffffff", stroke_width=0.8))
            svg.append(svg_text(x, y - 10, f"{value:.2f}", size=10, fill=TEXT_MUTED, anchor="middle"))

    if not any_points and empty_message:
        svg.append(svg_text((axis_left + axis_right) / 2, (axis_top + axis_bottom) / 2, empty_message, size=14, fill=TEXT_MUTED, anchor="middle"))


def build_svg(stats: dict[str, object]) -> str:
    width = 1280
    height = 760
    prefix_order = list(stats["prefix_order"])
    reviewed_rows = int(stats["reviewed_prefix_rows"])

    svg: list[str] = []
    svg.append(svg_rect(0, 0, width, height, fill=BACKGROUND))
    svg.append(svg_text(54, 54, "exp15 Prefix Probe Summary", size=28, weight="bold"))
    svg.append(
        svg_text(
            54,
            82,
            "Same long generations, sliced into prefixes. Top panel works immediately from proxy scores; bottom panel activates once manual prefix review is filled.",
            size=14,
            fill=TEXT_MUTED,
        )
    )
    svg.append(
        svg_text(
            54,
            108,
            f"Questions: {stats['question_count']} | Prefixes: {', '.join(prefix_label(p) for p in prefix_order)} | Reviewed prefix rows: {reviewed_rows}",
            size=12,
            fill=TEXT_MUTED,
        )
    )

    draw_line_panel(
        svg,
        left=46,
        top=138,
        width=1188,
        height=260,
        prefix_order=prefix_order,
        values_by_decoder=stats["proxy_margin_means"],
        title="Proxy Factuality by Prefix",
        subtitle="Mean oref_margin at each prefix checkpoint",
        y_label="oref_margin",
        y_min=-0.20,
        y_max=0.20,
        ticks=[-0.20, -0.10, 0.0, 0.10, 0.20],
    )

    draw_line_panel(
        svg,
        left=46,
        top=430,
        width=1188,
        height=260,
        prefix_order=prefix_order,
        values_by_decoder=stats["manual_score_means"],
        title="Manual Prefix Score by Prefix",
        subtitle="Mean human prefix score after reviewing the prefix_manual_eval sheet",
        y_label="manual score",
        y_min=0.0,
        y_max=2.0,
        ticks=[0.0, 0.5, 1.0, 1.5, 2.0],
        empty_message="Fill the manual prefix scores, then rerun this script.",
    )

    draw_legend(svg, left=76, top=720)
    return wrap_svg(svg, width, height)


def main() -> None:
    args = parse_args()
    stats_output = args.stats_output or args.output.with_suffix(".json")
    df, prefix_order = load_prefix_eval(args.prefix_csv)
    stats = build_stats(df, prefix_order)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(stats), encoding="utf-8")
    stats_payload = {
        "prefix_csv": str(args.prefix_csv),
        **stats,
    }
    stats_output.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")
    print(f"Wrote figure to {args.output}")
    print(f"Wrote stats to {stats_output}")


if __name__ == "__main__":
    main()
