#!/usr/bin/env python3
"""Plot real DoLa selected-layer traces from experiment token-trace artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path


TRACE_RUN_ID = "trace_dola_mc1_extremes_gpu"
TRACE_CSV_DEFAULT = Path(
    f"results/experiments/exp11_core_decoder_comparison/runs/{TRACE_RUN_ID}/{TRACE_RUN_ID}_token_trace.csv"
)
SELECTED_CASES_CSV_DEFAULT = Path(
    f"results/experiments/exp11_core_decoder_comparison/runs/{TRACE_RUN_ID}/{TRACE_RUN_ID}_selected_cases.csv"
)
OUTPUT_DEFAULT = Path("results/figures/dola_selected_layers_mc1_cases.svg")

FONT_FAMILY = "Times New Roman, Times, serif"
BACKGROUND = "#ffffff"
PANEL_BG = "#ffffff"
PANEL_STROKE = "#d8d8d8"
GRID = "#e6e6e6"
AXIS = "#222222"
TEXT = "#222222"
TEXT_MUTED = "#555555"
WIN_COLORS = ("#1b9e77", "#377eb8", "#6a51a3")
LOSS_COLORS = ("#d95f02", "#e7298a", "#b22222")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected-layer traces from the exp11 DoLa trace-dump artifact."
    )
    parser.add_argument("--trace-csv", type=Path, default=TRACE_CSV_DEFAULT)
    parser.add_argument("--selected-cases-csv", type=Path, default=SELECTED_CASES_CSV_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional JSON output. Defaults to output stem + .json.",
    )
    parser.add_argument("--title", type=str, default="DoLa selected layer over answer time")
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
                    "source_run_id": row["source_run_id"],
                    "trace_run_id": row["trace_run_id"],
                    "decoder": row["decoder"],
                    "metric_name": row["metric_name"],
                    "group": row["group"],
                    "group_rank": int(row["group_rank"]),
                    "example_idx": int(row["example_idx"]),
                    "question": row["question"],
                    "short_question": short_question(row["question"]),
                    "prediction": row["prediction"],
                    "score": float(row["score"]),
                    "decision_margin": float(row["decision_margin"]),
                    "score_detail": row["score_detail"],
                    "sequence_logprob": float(row["sequence_logprob"]),
                    "trace_length": int(row["trace_length"]),
                    "unique_selected_layers": json.loads(row["unique_selected_layers"]),
                    "latency_seconds": float(row["latency_seconds"]),
                    "decoder_steps": int(row["decoder_steps"]),
                    "forward_passes": int(row["forward_passes"]),
                    "switch_rate": float(row["switch_rate"]) if row.get("switch_rate") not in {"", None} else None,
                }
            )

    group_order = {"mc1=1": 0, "mc1=0": 1}
    rows.sort(key=lambda row: (group_order.get(str(row["group"]), 99), int(row["group_rank"])))
    win_idx = 0
    loss_idx = 0
    for row in rows:
        if row["group"] == "mc1=1":
            row["color"] = WIN_COLORS[win_idx % len(WIN_COLORS)]
            win_idx += 1
        else:
            row["color"] = LOSS_COLORS[loss_idx % len(LOSS_COLORS)]
            loss_idx += 1
    return rows


def load_trace_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "source_run_id": row["source_run_id"],
                    "trace_run_id": row["trace_run_id"],
                    "decoder": row["decoder"],
                    "metric_name": row["metric_name"],
                    "group": row["group"],
                    "group_rank": int(row["group_rank"]),
                    "example_idx": int(row["example_idx"]),
                    "question": row["question"],
                    "prediction": row["prediction"],
                    "step": int(row["step"]),
                    "token_step": int(row["token_step"]),
                    "token_id": int(row["token_id"]),
                    "token_text": row["token_text"],
                    "selected_layer": int(row["selected_layer"]),
                    "divergence": float(row["divergence"]) if row.get("divergence") not in {"", None} else None,
                    "jsd_current": float(row["jsd_current"]) if row.get("jsd_current") not in {"", None} else None,
                    "selection_score": (
                        float(row["selection_score"]) if row.get("selection_score") not in {"", None} else None
                    ),
                }
            )
    return rows


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
        f'font-family="{FONT_FAMILY}" text-anchor="{anchor}" font-weight="{weight}">{escape(text)}</text>'
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
    stroke_opacity: float = 1.0,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" stroke-opacity="{stroke_opacity:.3f}"{dash_attr} />'
    )


def svg_circle(
    cx: float,
    cy: float,
    r: float,
    *,
    fill: str,
    stroke: str = "#ffffff",
    stroke_width: float = 0.8,
    fill_opacity: float = 0.92,
) -> str:
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


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def nice_x_ticks(max_step: int) -> list[int]:
    if max_step <= 12:
        step = 2
    elif max_step <= 24:
        step = 4
    elif max_step <= 48:
        step = 8
    else:
        step = 16
    ticks = [1]
    current = step
    while current < max_step:
        ticks.append(current)
        current += step
    if ticks[-1] != max_step:
        ticks.append(max_step)
    return sorted(set(ticks))


def draw_panel(
    svg: list[str],
    *,
    panel_left: float,
    panel_top: float,
    panel_width: float,
    panel_height: float,
    title: str,
    group_rows: list[dict[str, object]],
    cases: list[dict[str, object]],
    x_max: int,
    y_ticks: list[int],
) -> None:
    svg.append(
        svg_rect(
            panel_left,
            panel_top,
            panel_width,
            panel_height,
            fill=PANEL_BG,
            stroke=PANEL_STROKE,
            stroke_width=1.0,
            rx=14,
        )
    )
    svg.append(svg_text(panel_left + panel_width / 2, panel_top + 26, title, size=16, anchor="middle", weight="bold"))

    axis_left = panel_left + 66
    axis_right = panel_left + panel_width - 22
    axis_top = panel_top + 52
    axis_bottom = panel_top + panel_height - 96

    for tick in y_ticks:
        y = scale(float(tick), float(y_ticks[0]), float(y_ticks[-1]), axis_bottom, axis_top)
        svg.append(svg_line(axis_left, y, axis_right, y, stroke=GRID, stroke_width=1.0))
        svg.append(svg_text(axis_left - 10, y + 4, str(tick), size=10, fill=TEXT_MUTED, anchor="end"))

    x_ticks = nice_x_ticks(x_max)
    for tick in x_ticks:
        x = scale(float(tick), 1.0, float(x_max), axis_left, axis_right)
        svg.append(svg_line(x, axis_top, x, axis_bottom, stroke=GRID, stroke_width=1.0, dash="3 4"))
        svg.append(svg_text(x, axis_bottom + 22, str(tick), size=10, fill=TEXT_MUTED, anchor="middle"))

    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_line(axis_left, axis_bottom, axis_right, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_text(panel_left + panel_width / 2, panel_top + panel_height - 50, "Token timestep", size=12, anchor="middle"))
    svg.append(svg_text(panel_left + 16, panel_top + panel_height / 2, "Layer index", size=12))

    case_by_question = {str(case["question"]): case for case in cases}
    rows_by_question: dict[str, list[dict[str, object]]] = {}
    for row in group_rows:
        rows_by_question.setdefault(str(row["question"]), []).append(row)

    for question, rows in rows_by_question.items():
        rows = sorted(rows, key=lambda row: int(row["token_step"]))
        case = case_by_question[question]
        points: list[tuple[float, float]] = []
        for row in rows:
            x = scale(float(row["token_step"]), 1.0, float(x_max), axis_left, axis_right)
            y = scale(float(row["selected_layer"]), float(y_ticks[0]), float(y_ticks[-1]), axis_bottom, axis_top)
            points.append((x, y))
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            svg.append(
                svg_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    stroke=str(case["color"]),
                    stroke_width=1.0,
                    stroke_opacity=0.42,
                )
            )
        for x, y in points:
            svg.append(svg_circle(x, y, 3.6, fill=str(case["color"])))

    legend_y = panel_top + panel_height - 66
    for idx, case in enumerate(cases):
        x = panel_left + 24 + idx * ((panel_width - 48) / max(1, len(cases)))
        svg.append(svg_circle(x, legend_y, 4.6, fill=str(case["color"]), stroke="none"))
        label = f"Q{int(case['example_idx'])}: {str(case['short_question'])}"
        svg.append(svg_text(x + 10, legend_y + 4, label, size=10, fill=TEXT_MUTED))


def build_figure(
    trace_rows: list[dict[str, object]],
    case_rows: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    width = 1500
    height = 860
    body: list[str] = []
    body.append(svg_rect(0, 0, width, height, fill=BACKGROUND))
    body.append(svg_text(width / 2, 34, title, size=22, anchor="middle", weight="bold"))

    win_cases = [row for row in case_rows if row["group"] == "mc1=1"]
    loss_cases = [row for row in case_rows if row["group"] == "mc1=0"]
    win_rows = [row for row in trace_rows if row["group"] == "mc1=1"]
    loss_rows = [row for row in trace_rows if row["group"] == "mc1=0"]

    all_steps = [int(row["token_step"]) for row in trace_rows]
    all_layers = sorted({int(row["selected_layer"]) for row in trace_rows})
    x_max = max(all_steps) if all_steps else 1
    y_ticks = all_layers if all_layers else [0]

    panel_top = 72
    panel_height = 730
    panel_gap = 22
    panel_width = (width - 48 - panel_gap) / 2

    draw_panel(
        body,
        panel_left=24,
        panel_top=panel_top,
        panel_width=panel_width,
        panel_height=panel_height,
        title="Three strongest DoLa mc1=1 questions",
        group_rows=win_rows,
        cases=win_cases,
        x_max=x_max,
        y_ticks=y_ticks,
    )
    draw_panel(
        body,
        panel_left=24 + panel_width + panel_gap,
        panel_top=panel_top,
        panel_width=panel_width,
        panel_height=panel_height,
        title="Three worst DoLa mc1=0 questions",
        group_rows=loss_rows,
        cases=loss_cases,
        x_max=x_max,
        y_ticks=y_ticks,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(wrap_svg(body, width, height), encoding="utf-8")


def write_stats_json(
    path: Path,
    *,
    trace_csv: Path,
    selected_cases_csv: Path,
    case_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    payload = {
        "figure_path": str(output_path),
        "trace_csv_path": str(trace_csv),
        "selected_cases_csv_path": str(selected_cases_csv),
        "trace_mode": "plot_from_experiment_token_trace_csv",
        "cases": case_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    stats_output = args.stats_output or args.output.with_suffix(".json")
    case_rows = load_selected_cases(args.selected_cases_csv)
    trace_rows = load_trace_rows(args.trace_csv)
    build_figure(trace_rows, case_rows, args.output, args.title)
    write_stats_json(
        stats_output,
        trace_csv=args.trace_csv,
        selected_cases_csv=args.selected_cases_csv,
        case_rows=case_rows,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "stats_output": str(stats_output),
                "trace_csv": str(args.trace_csv),
                "selected_cases_csv": str(args.selected_cases_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
