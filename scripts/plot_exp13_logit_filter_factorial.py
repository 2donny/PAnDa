#!/usr/bin/env python3
"""Plot a simpler exp13 matched 2x2 logit/filter factorial figure."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import pandas as pd


BACKGROUND = "#f7f6f3"
PANEL_BG = "#ffffff"
GRID = "#e4e4e4"
AXIS = "#555555"
TEXT = "#222222"
TEXT_MUTED = "#666666"

METRIC_ORDER = ("mc1", "mc2", "mc3")
CELL_ORDER = (
    "exp13_logprob_top",
    "exp13_logprob_no_top",
    "exp13_logit_top",
    "exp13_logit_no_top",
)
DISPLAY_LABELS = {
    "exp13_logprob_top": "logprob\n+ filter",
    "exp13_logprob_no_top": "logprob\nno filter",
    "exp13_logit_top": "raw-logit\n+ filter",
    "exp13_logit_no_top": "raw-logit\nno filter",
}
COLORS = {
    "exp13_logprob_top": "#b8b8b8",
    "exp13_logprob_no_top": "#2a9d8f",
    "exp13_logit_top": "#d4a373",
    "exp13_logit_no_top": "#c8553d",
}
FACTOR_MAP = {
    "exp13_logprob_top": ("logprob", "on"),
    "exp13_logprob_no_top": ("logprob", "off"),
    "exp13_logit_top": ("raw-logit", "on"),
    "exp13_logit_no_top": ("raw-logit", "off"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a simpler presentation-friendly exp13 factorial summary figure."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(
            "results/experiments/exp13_logit_filter_factorial/runs/run_01_default/"
            "run_01_default_summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/exp13_logit_filter_factorial_summary.svg"),
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("results/figures/exp13_logit_filter_factorial_summary.json"),
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


def metric_label(metric: str) -> str:
    return metric.upper()


def load_factorial(summary_csv: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(summary_csv)
    df = df[df["metric_name"].isin(METRIC_ORDER)].copy()
    df["score_space"] = df["decoder"].map(lambda name: FACTOR_MAP[name][0])
    df["relative_top"] = df["decoder"].map(lambda name: FACTOR_MAP[name][1])

    stats: dict[str, object] = {
        "bars": {},
        "main_effects": {},
        "ranking": {},
    }

    for metric in METRIC_ORDER:
        sub = df[df["metric_name"] == metric].copy()
        stats["bars"][metric] = {
            decoder: float(sub[sub["decoder"] == decoder]["score_mean"].iloc[0])
            for decoder in CELL_ORDER
        }
        pivot = sub.pivot(index="score_space", columns="relative_top", values="score_mean")
        no_top = float(
            sub[sub["relative_top"] == "off"]["score_mean"].mean()
            - sub[sub["relative_top"] == "on"]["score_mean"].mean()
        )
        raw_logit = float(
            sub[sub["score_space"] == "raw-logit"]["score_mean"].mean()
            - sub[sub["score_space"] == "logprob"]["score_mean"].mean()
        )
        interaction = float(
            (pivot.loc["raw-logit", "off"] - pivot.loc["raw-logit", "on"])
            - (pivot.loc["logprob", "off"] - pivot.loc["logprob", "on"])
        )
        stats["main_effects"][metric] = {
            "no_filter": no_top,
            "raw_logit": raw_logit,
            "interaction": interaction,
        }
        stats["ranking"][metric] = (
            sub.sort_values("score_mean", ascending=False)[["decoder", "score_mean"]]
            .to_dict(orient="records")
        )

    stats["avg_no_filter_effect"] = float(
        sum(stats["main_effects"][m]["no_filter"] for m in METRIC_ORDER) / len(METRIC_ORDER)
    )
    stats["avg_raw_logit_effect"] = float(
        sum(stats["main_effects"][m]["raw_logit"] for m in METRIC_ORDER) / len(METRIC_ORDER)
    )
    stats["takeaway"] = (
        "Removing the relative-top filter is the dominant effect. "
        "Changing log-prob contrast to raw-logit is much smaller and mixed."
    )
    return df, stats


def draw_legend(svg: list[str], *, left: float, top: float) -> None:
    cursor_x = left
    for decoder in CELL_ORDER:
        svg.append(svg_rect(cursor_x, top - 10, 18, 12, fill=COLORS[decoder], rx=3))
        label = DISPLAY_LABELS[decoder].replace("\n", " ")
        svg.append(svg_text(cursor_x + 26, top, label, size=11, fill=TEXT_MUTED))
        cursor_x += 210


def draw_bar_panel(
    svg: list[str],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    stats: dict[str, object],
) -> None:
    svg.append(svg_rect(left, top, width, height, fill=PANEL_BG, stroke="#dddddd", stroke_width=1.0, rx=14))
    svg.append(svg_text(left + width / 2, top + 24, "2x2 Cells by Metric", size=18, anchor="middle", weight="bold"))
    svg.append(svg_text(left + width / 2, top + 44, "Same data as before, but flattened into simple bars", size=11, fill=TEXT_MUTED, anchor="middle"))

    axis_left = left + 64
    axis_right = left + width - 28
    axis_top = top + 72
    axis_bottom = top + height - 96
    y_min = 0.0
    y_max = 0.6

    for tick in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        y = scale(tick, y_min, y_max, axis_bottom, axis_top)
        svg.append(svg_line(axis_left, y, axis_right, y, stroke=GRID, stroke_width=1.0))
        svg.append(svg_text(axis_left - 10, y + 4, f"{tick:.1f}", size=10, fill=TEXT_MUTED, anchor="end"))

    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_line(axis_left, axis_bottom, axis_right, axis_bottom, stroke=AXIS, stroke_width=1.1))

    group_w = (axis_right - axis_left) / len(METRIC_ORDER)
    inner_gap = 12
    bar_w = (group_w - inner_gap * 5) / 4.0

    for group_idx, metric in enumerate(METRIC_ORDER):
        group_left = axis_left + group_idx * group_w
        svg.append(svg_text(group_left + group_w / 2, axis_bottom + 24, metric_label(metric), size=12, fill=TEXT_MUTED, anchor="middle", weight="bold"))
        for bar_idx, decoder in enumerate(CELL_ORDER):
            value = float(stats["bars"][metric][decoder])
            x = group_left + inner_gap + bar_idx * (bar_w + inner_gap)
            y = scale(value, y_min, y_max, axis_bottom, axis_top)
            svg.append(svg_rect(x, y, bar_w, axis_bottom - y, fill=COLORS[decoder], rx=6))
            svg.append(svg_text(x + bar_w / 2, y - 8, f"{value:.3f}", size=10, anchor="middle"))

            label_lines = DISPLAY_LABELS[decoder].split("\n")
            base_y = axis_bottom + 42
            for line_idx, line in enumerate(label_lines):
                svg.append(svg_text(x + bar_w / 2, base_y + line_idx * 13, line, size=9, fill=TEXT_MUTED, anchor="middle"))


def draw_effect_panel(
    svg: list[str],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    main_effects: dict[str, dict[str, float]],
) -> None:
    svg.append(svg_rect(left, top, width, height, fill=PANEL_BG, stroke="#dddddd", stroke_width=1.0, rx=14))
    svg.append(svg_text(left + width / 2, top + 24, "Main Effects", size=18, anchor="middle", weight="bold"))
    svg.append(svg_text(left + width / 2, top + 44, "Positive means the factor helps", size=11, fill=TEXT_MUTED, anchor="middle"))

    axis_left = left + 112
    axis_right = left + width - 28
    axis_top = top + 76
    axis_bottom = top + height - 70
    x_min = -0.02
    x_max = 0.10
    zero_x = scale(0.0, x_min, x_max, axis_left, axis_right)

    for tick in (-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
        x = scale(tick, x_min, x_max, axis_left, axis_right)
        svg.append(svg_line(x, axis_top, x, axis_bottom, stroke=GRID, stroke_width=1.0))
        svg.append(svg_text(x, axis_bottom + 22, f"{tick:+.02f}", size=10, fill=TEXT_MUTED, anchor="middle"))
    svg.append(svg_line(zero_x, axis_top, zero_x, axis_bottom, stroke=AXIS, stroke_width=1.2))

    row_gap = (axis_bottom - axis_top) / len(METRIC_ORDER)
    bar_h = 14
    for idx, metric in enumerate(METRIC_ORDER):
        y_mid = axis_top + row_gap * idx + row_gap / 2
        svg.append(svg_text(left + 20, y_mid + 4, metric_label(metric), size=12, fill=TEXT_MUTED))

        nf = float(main_effects[metric]["no_filter"])
        rl = float(main_effects[metric]["raw_logit"])

        nf_x = scale(nf, x_min, x_max, axis_left, axis_right)
        rl_x = scale(rl, x_min, x_max, axis_left, axis_right)

        svg.append(svg_rect(min(zero_x, nf_x), y_mid - 18, abs(nf_x - zero_x), bar_h, fill="#2a9d8f", rx=6))
        svg.append(svg_text(nf_x + (8 if nf >= 0 else -8), y_mid - 6, f"{nf:+.003f}", size=10, anchor="start" if nf >= 0 else "end"))

        svg.append(svg_rect(min(zero_x, rl_x), y_mid + 4, abs(rl_x - zero_x), bar_h, fill="#c8553d", rx=6))
        svg.append(svg_text(rl_x + (8 if rl >= 0 else -8), y_mid + 16, f"{rl:+.003f}", size=10, anchor="start" if rl >= 0 else "end"))

    legend_y = top + height - 30
    svg.append(svg_rect(left + 26, legend_y - 9, 16, 10, fill="#2a9d8f", rx=4))
    svg.append(svg_text(left + 50, legend_y, "remove filter (off - on)", size=11, fill=TEXT_MUTED))
    svg.append(svg_rect(left + 240, legend_y - 9, 16, 10, fill="#c8553d", rx=4))
    svg.append(svg_text(left + 264, legend_y, "raw-logit - logprob", size=11, fill=TEXT_MUTED))


def draw_takeaway_panel(
    svg: list[str],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    stats: dict[str, object],
) -> None:
    svg.append(svg_rect(left, top, width, height, fill=PANEL_BG, stroke="#dddddd", stroke_width=1.0, rx=14))
    svg.append(svg_text(left + 22, top + 28, "Reading", size=17, weight="bold"))
    lines = [
        "The no-filter bars are the important pattern.",
        "On mc1 and mc3, turning the filter off helps a lot.",
        "Once the filter is off, logprob and raw-logit become identical on mc1 and mc3.",
        f"Average no-filter effect: {stats['avg_no_filter_effect']:+.03f}",
        f"Average raw-logit effect: {stats['avg_raw_logit_effect']:+.03f}",
        "So exp13 says the simpler story is the right one:",
        "most of the gain comes from removing the relative-top filter,",
        "not from switching score space by itself.",
    ]
    y = top + 58
    for line in lines:
        svg.append(svg_text(left + 22, y, line, size=12, fill=TEXT_MUTED))
        y += 22


def main() -> None:
    args = parse_args()
    _, stats = load_factorial(args.summary_csv)

    width = 1280
    height = 760
    body: list[str] = [svg_rect(0, 0, width, height, fill=BACKGROUND)]
    body.append(svg_text(34, 36, "exp13 Logit / Filter Factorial", size=28, weight="bold"))
    body.append(
        svg_text(
            34,
            60,
            "Matched update1 2x2 ablation: easier view of the four cells and the two main effects.",
            size=13,
            fill=TEXT_MUTED,
        )
    )
    draw_legend(body, left=34, top=88)
    draw_bar_panel(body, left=30, top=112, width=760, height=620, stats=stats)
    draw_effect_panel(body, left=812, top=112, width=438, height=300, main_effects=stats["main_effects"])
    draw_takeaway_panel(body, left=812, top=434, width=438, height=298, stats=stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(wrap_svg(body, width, height), encoding="utf-8")
    args.stats_output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print({"figure": str(args.output), "stats": str(args.stats_output)})


if __name__ == "__main__":
    main()
