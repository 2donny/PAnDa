#!/usr/bin/env python3
"""Plot selected-layer traces for strong and weak exp14_update1 open-ended answers."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from html import escape
from pathlib import Path


RUN_ID = "run_01_default"
SELECTED_CASES_CSV_DEFAULT = Path(
    f"results/experiments/exp17_openended_dola_trace/runs/{RUN_ID}/{RUN_ID}_selected_cases.csv"
)
TRACE_CSV_DEFAULT = Path(
    f"results/experiments/exp17_openended_dola_trace/runs/{RUN_ID}/{RUN_ID}_token_trace.csv"
)
OUTPUT_DEFAULT = Path("results/figures/exp17_openended_update1_selected_layers.svg")

FONT_FAMILY = "Times New Roman, Times, serif"
BACKGROUND = "#ffffff"
PANEL_BG = "#ffffff"
PANEL_STROKE = "#d8d8d8"
GRID = "#e6e6e6"
AXIS = "#222222"
TEXT = "#222222"
TEXT_MUTED = "#555555"
GOOD_COLORS = ("#1b9e77", "#377eb8", "#6a51a3")
BAD_COLORS = ("#d95f02", "#e7298a", "#b22222")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot exp17 selected-layer traces from the selected-case trace dump."
    )
    parser.add_argument("--selected-cases-csv", type=Path, default=SELECTED_CASES_CSV_DEFAULT)
    parser.add_argument("--trace-csv", type=Path, default=TRACE_CSV_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument(
        "--title",
        type=str,
        default="Open-ended update1 selected layer over generation time",
    )
    return parser.parse_args()


def short_question(question: str, max_len: int = 58) -> str:
    text = " ".join(str(question).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def load_selected_cases(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "example_idx": int(row["example_idx"]),
                    "decoder": row["decoder"],
                    "decoder_label": row["decoder_label"],
                    "group": row["group"],
                    "group_rank": int(row["group_rank"]),
                    "question": row["question"],
                    "short_question": short_question(row["question"]),
                    "manual_score_0_2": int(row["manual_score_0_2"]),
                    "manual_label": row["manual_label"],
                    "issue_tags": row["issue_tags"],
                    "manual_notes": row["manual_notes"],
                    "review_status": row["review_status"],
                    "reviewer": row["reviewer"],
                    "proxy_oref_margin": float(row["proxy_oref_margin"]),
                    "trace_length": int(row["trace_length"]),
                    "source_prediction": row["source_prediction"],
                    "regenerated_prediction": row["regenerated_prediction"],
                    "normalized_prediction_match": int(row["normalized_prediction_match"]),
                    "unique_selected_layers": json.loads(row["unique_selected_layers"]),
                    "switch_rate": float(row["switch_rate"]) if row.get("switch_rate") not in {"", None} else None,
                }
            )
    color_buckets = {
        "manual_score_0_2=2": GOOD_COLORS,
        "manual_score_0_2=0": BAD_COLORS,
    }
    for row in rows:
        palette = color_buckets.get(row["group"], GOOD_COLORS)
        rank = max(1, int(row["group_rank"]))
        row["color"] = palette[min(rank - 1, len(palette) - 1)]
    return rows


def load_trace_rows(path: Path, selected_cases: list[dict[str, object]]) -> list[dict[str, object]]:
    selected_idx = {int(row["example_idx"]) for row in selected_cases}
    case_by_idx = {int(row["example_idx"]): row for row in selected_cases}
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            example_idx = int(row["example_idx"])
            if example_idx not in selected_idx:
                continue
            case = case_by_idx[example_idx]
            rows.append(
                {
                    "example_idx": example_idx,
                    "group": case["group"],
                    "group_rank": int(case["group_rank"]),
                    "question": row["question"],
                    "token_step": int(row["token_step"]),
                    "selected_layer": int(row["selected_layer"]),
                    "token_id": int(row["token_id"]),
                    "token_text": row["token_text"],
                }
            )
    return rows


def svg_text(x, y, text, *, size=12, fill=TEXT, anchor="start", weight="normal"):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
        f'font-family="{FONT_FAMILY}" text-anchor="{anchor}" font-weight="{weight}">{escape(text)}</text>'
    )


def svg_rect(x, y, width, height, *, fill, stroke="none", stroke_width=1.0, rx=0.0):
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" rx="{rx:.2f}" />'
    )


def svg_line(x1, y1, x2, y2, *, stroke=AXIS, stroke_width=1.0, dash=None, stroke_opacity=1.0):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" stroke-opacity="{stroke_opacity:.3f}"{dash_attr} />'
    )


def svg_circle(cx, cy, r, *, fill, stroke="#ffffff", stroke_width=0.8, fill_opacity=0.92):
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
        f'fill-opacity="{fill_opacity:.3f}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" />'
    )


def wrap_svg(body: list[str], width: int, height: int) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            *body,
            "</svg>",
        ]
    )


def scale(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def nice_x_ticks(max_step: int) -> list[int]:
    if max_step <= 16:
        step = 4
    elif max_step <= 32:
        step = 8
    elif max_step <= 64:
        step = 16
    else:
        step = 24
    ticks = [1]
    current = step
    while current < max_step:
        ticks.append(current)
        current += step
    if ticks[-1] != max_step:
        ticks.append(max_step)
    return sorted(set(ticks))


def draw_panel(svg, *, panel_left, panel_top, panel_width, panel_height, title, group_rows, cases, x_max, y_ticks):
    svg.append(svg_rect(panel_left, panel_top, panel_width, panel_height, fill=PANEL_BG, stroke=PANEL_STROKE, stroke_width=1.0, rx=14))
    svg.append(svg_text(panel_left + panel_width / 2, panel_top + 26, title, size=16, anchor="middle", weight="bold"))

    axis_left = panel_left + 66
    axis_right = panel_left + panel_width - 22
    axis_top = panel_top + 52
    axis_bottom = panel_top + panel_height - 96

    for tick in y_ticks:
        y = scale(float(tick), float(y_ticks[0]), float(y_ticks[-1]), axis_bottom, axis_top)
        svg.append(svg_line(axis_left, y, axis_right, y, stroke=GRID, stroke_width=1.0))
        svg.append(svg_text(axis_left - 10, y + 4, str(tick), size=10, fill=TEXT_MUTED, anchor="end"))

    for tick in nice_x_ticks(x_max):
        x = scale(float(tick), 1.0, float(x_max), axis_left, axis_right)
        svg.append(svg_line(x, axis_top, x, axis_bottom, stroke=GRID, stroke_width=1.0, dash="3 4"))
        svg.append(svg_text(x, axis_bottom + 22, str(tick), size=10, fill=TEXT_MUTED, anchor="middle"))

    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_line(axis_left, axis_bottom, axis_right, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_text(panel_left + panel_width / 2, panel_top + panel_height - 50, "Token timestep", size=12, anchor="middle"))
    svg.append(svg_text(panel_left + 16, panel_top + panel_height / 2, "Layer index", size=12))

    case_by_idx = {int(case["example_idx"]): case for case in cases}
    rows_by_idx = {}
    for row in group_rows:
        rows_by_idx.setdefault(int(row["example_idx"]), []).append(row)

    for example_idx, rows in rows_by_idx.items():
        rows = sorted(rows, key=lambda row: int(row["token_step"]))
        case = case_by_idx[example_idx]
        points = []
        for row in rows:
            x = scale(float(row["token_step"]), 1.0, float(x_max), axis_left, axis_right)
            y = scale(float(row["selected_layer"]), float(y_ticks[0]), float(y_ticks[-1]), axis_bottom, axis_top)
            points.append((x, y))
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            svg.append(svg_line(x1, y1, x2, y2, stroke=str(case["color"]), stroke_width=1.0, stroke_opacity=0.42))
        for x, y in points:
            svg.append(svg_circle(x, y, 3.2, fill=str(case["color"])))

    legend_y = panel_top + panel_height - 66
    step = (panel_width - 48) / max(1, len(cases))
    for idx, case in enumerate(cases):
        x = panel_left + 24 + idx * step
        svg.append(svg_circle(x, legend_y, 4.6, fill=str(case["color"]), stroke="none"))
        label = f"Q{int(case['example_idx'])}: {str(case['short_question'])}"
        svg.append(svg_text(x + 10, legend_y + 4, label, size=10, fill=TEXT_MUTED))


def build_figure(trace_rows, selected_cases, output_path: Path, title: str):
    width = 1500
    height = 860
    body = [svg_rect(0, 0, width, height, fill=BACKGROUND)]
    body.append(svg_text(width / 2, 34, title, size=22, anchor="middle", weight="bold"))
    body.append(
        svg_text(
            width / 2,
            58,
            "Cases chosen from exp14 codex manual eval for exp14_update1: score 2 vs score 0",
            size=12,
            fill=TEXT_MUTED,
            anchor="middle",
        )
    )

    trace_rows = sorted(trace_rows, key=lambda row: (row["group"], row["example_idx"], row["token_step"]))
    selected_layers = sorted({int(row["selected_layer"]) for row in trace_rows})
    x_max = max(int(row["token_step"]) for row in trace_rows)

    good_cases = [row for row in selected_cases if row["group"] == "manual_score_0_2=2"]
    bad_cases = [row for row in selected_cases if row["group"] == "manual_score_0_2=0"]
    good_rows = [row for row in trace_rows if row["group"] == "manual_score_0_2=2"]
    bad_rows = [row for row in trace_rows if row["group"] == "manual_score_0_2=0"]

    draw_panel(
        body,
        panel_left=44,
        panel_top=90,
        panel_width=690,
        panel_height=720,
        title="Strong cases (manual score 2)",
        group_rows=good_rows,
        cases=good_cases,
        x_max=x_max,
        y_ticks=selected_layers,
    )
    draw_panel(
        body,
        panel_left=764,
        panel_top=90,
        panel_width=690,
        panel_height=720,
        title="Weak cases (manual score 0)",
        group_rows=bad_rows,
        cases=bad_cases,
        x_max=x_max,
        y_ticks=selected_layers,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(wrap_svg(body, width, height), encoding="utf-8")


def build_stats(selected_cases, trace_rows):
    case_counts = Counter(case["group"] for case in selected_cases)
    reviewer_counts = Counter(case["reviewer"] for case in selected_cases)
    review_status_counts = Counter(case["review_status"] for case in selected_cases)
    return {
        "num_cases": len(selected_cases),
        "case_counts": dict(case_counts),
        "reviewer_counts": dict(reviewer_counts),
        "review_status_counts": dict(review_status_counts),
        "unique_selected_layers": sorted({int(row["selected_layer"]) for row in trace_rows}),
        "max_token_step": max(int(row["token_step"]) for row in trace_rows),
        "selected_cases": selected_cases,
    }


def main():
    args = parse_args()
    selected_cases = load_selected_cases(args.selected_cases_csv)
    trace_rows = load_trace_rows(args.trace_csv, selected_cases)
    if not selected_cases:
        raise SystemExit(f"No selected cases were found in {args.selected_cases_csv}.")
    if not trace_rows:
        raise SystemExit(f"No trace rows were found in {args.trace_csv}.")

    build_figure(trace_rows, selected_cases, args.output, args.title)

    stats_output = args.stats_output
    if stats_output is None:
        stats_output = args.output.with_suffix(".json")
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.write_text(json.dumps(build_stats(selected_cases, trace_rows), indent=2), encoding="utf-8")

    print(
        {
            "output": str(args.output),
            "stats_output": str(stats_output),
            "selected_cases_csv": str(args.selected_cases_csv),
            "trace_csv": str(args.trace_csv),
        }
    )


if __name__ == "__main__":
    main()
