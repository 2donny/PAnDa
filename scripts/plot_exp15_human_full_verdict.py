#!/usr/bin/env python3
"""Plot a simple human-verdict summary for exp15 full answers."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path


BACKGROUND = "#f7f6f3"
PANEL_BG = "#ffffff"
GRID = "#e4e4e4"
AXIS = "#555555"
TEXT = "#222222"
TEXT_MUTED = "#666666"
COLORS = {
    "frozen": "#6d597a",
    "update1": "#c8553d",
    "tie": "#7a7a7a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small human-verdict figure for exp15 full-answer review."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/experiments/exp15_prefix_probe/runs/run_01_default/"
            "run_01_default_assistant_full_verdict.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/exp15_human_full_verdict.svg"),
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("results/figures/exp15_human_full_verdict.json"),
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


def shorten(text: str, limit: int = 58) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_stats(rows: list[dict[str, str]]) -> dict[str, object]:
    counts = {"frozen": 0, "update1": 0, "tie": 0}
    clear_counts = {"frozen": 0, "update1": 0}
    probable_counts = {"frozen": 0, "update1": 0}
    for row in rows:
        verdict = row["verdict"]
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict in {"frozen", "update1"}:
            if row["confidence"] == "clear":
                clear_counts[verdict] += 1
            elif row["confidence"] == "probable":
                probable_counts[verdict] += 1
    return {
        "question_count": len(rows),
        "verdict_counts": counts,
        "clear_counts": clear_counts,
        "probable_counts": probable_counts,
        "rows": rows,
        "takeaway": (
            "Assistant full-answer review gives frozen a slight edge overall: 4 wins, "
            "update1 3 wins, and 1 tie."
        ),
    }


def draw_count_panel(svg: list[str], *, left: float, top: float, width: float, height: float, stats: dict[str, object]) -> None:
    counts = stats["verdict_counts"]
    svg.append(svg_rect(left, top, width, height, fill=PANEL_BG, stroke="#dddddd", stroke_width=1.0, rx=14))
    svg.append(svg_text(left + width / 2, top + 26, "Question-Level Verdicts", size=18, anchor="middle", weight="bold"))
    svg.append(svg_text(left + width / 2, top + 46, "Full-answer assistant review on 8 targeted exp15 questions", size=11, fill=TEXT_MUTED, anchor="middle"))

    axis_left = left + 60
    axis_right = left + width - 24
    axis_top = top + 70
    axis_bottom = top + height - 56
    max_count = max(int(v) for v in counts.values())

    for tick in range(max_count + 1):
        x = scale(float(tick), 0.0, float(max_count), axis_left, axis_right)
        svg.append(svg_line(x, axis_top, x, axis_bottom, stroke=GRID, stroke_width=1.0))
        svg.append(svg_text(x, axis_bottom + 20, str(tick), size=10, fill=TEXT_MUTED, anchor="middle"))

    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke=AXIS, stroke_width=1.1))
    row_gap = (axis_bottom - axis_top) / 3.0
    bar_h = 36.0
    labels = [("frozen", "frozen"), ("update1", "update1"), ("tie", "tie")]
    for idx, (key, label) in enumerate(labels):
        y = axis_top + idx * row_gap + 10
        bar_w = scale(float(counts[key]), 0.0, float(max_count), 0.0, axis_right - axis_left)
        svg.append(svg_text(axis_left - 12, y + 24, label, size=12, fill=TEXT_MUTED, anchor="end"))
        svg.append(svg_rect(axis_left, y, bar_w, bar_h, fill=COLORS[key], rx=10))
        svg.append(svg_text(axis_left + bar_w + 10, y + 24, str(counts[key]), size=12, weight="bold"))

    summary = f"Frozen leads overall ({counts['frozen']} wins) but not by much."
    svg.append(svg_text(left + width / 2, top + height - 18, summary, size=12, fill=TEXT_MUTED, anchor="middle"))


def draw_case_panel(svg: list[str], *, left: float, top: float, width: float, height: float, rows: list[dict[str, str]]) -> None:
    svg.append(svg_rect(left, top, width, height, fill=PANEL_BG, stroke="#dddddd", stroke_width=1.0, rx=14))
    svg.append(svg_text(left + width / 2, top + 26, "Per-Question Human Read", size=18, anchor="middle", weight="bold"))
    svg.append(svg_text(left + width / 2, top + 46, "Clear wins use solid emphasis; probable wins are softer calls", size=11, fill=TEXT_MUTED, anchor="middle"))

    start_y = top + 78
    row_h = 50
    for idx, row in enumerate(rows):
        y = start_y + idx * row_h
        if idx:
            svg.append(svg_line(left + 18, y - 14, left + width - 18, y - 14, stroke=GRID, stroke_width=1.0))
        verdict = row["verdict"]
        confidence = row["confidence"]
        fill = COLORS[verdict]
        stroke = "#222222" if confidence == "clear" else "none"
        stroke_w = 2.0 if confidence == "clear" else 1.0
        svg.append(svg_circle(left + 26, y, 8.0, fill=fill, stroke=stroke, stroke_width=stroke_w))
        svg.append(svg_text(left + 44, y + 4, f"{row['example_idx']}. {shorten(row['question'])}", size=12, weight="bold"))
        tag = verdict if verdict != "tie" else "tie / both messy"
        conf_text = confidence if confidence != "tie" else "no clean winner"
        svg.append(svg_text(left + width - 16, y + 4, f"{tag} ({conf_text})", size=11, fill=TEXT_MUTED, anchor="end"))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    stats = build_stats(rows)

    width = 1260
    height = 620
    body: list[str] = [svg_rect(0, 0, width, height, fill=BACKGROUND)]
    body.append(svg_text(36, 34, "exp15 Prefix Probe: Human Verdict", size=26, weight="bold"))
    body.append(
        svg_text(
            36,
            56,
            "Assistant full-answer review of frozen vs update1 on the 8 targeted disagreement questions.",
            size=12,
            fill=TEXT_MUTED,
        )
    )

    draw_count_panel(body, left=28, top=84, width=420, height=504, stats=stats)
    draw_case_panel(body, left=470, top=84, width=762, height=504, rows=rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(wrap_svg(body, width, height), encoding="utf-8")
    args.stats_output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print({"figure": str(args.output), "stats": str(args.stats_output)})


if __name__ == "__main__":
    main()
