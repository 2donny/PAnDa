#!/usr/bin/env python3
"""Create a presentation-friendly exp11 quality/latency figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from html import escape
from pathlib import Path

import pandas as pd


BASE_DECODER_ORDER = (
    "pure_greedy",
    "top_k",
    "top_p",
    "dola",
    "panda_switch",
)

V2_DECODER_ORDER = BASE_DECODER_ORDER + ("fanda_frozen",)

DISPLAY_LABELS = {
    "pure_greedy": "greedy",
    "top_k": "top k",
    "top_p": "top p",
    "dola": "dola",
    "panda_switch": "panda",
    "fanda_frozen": "fanda",
}

QUALITY_ORDER = (
    ("mc1", "MC1"),
    ("mc2", "MC2"),
    ("mc3", "MC3"),
)

COLORS = {
    "pure_greedy": "#0072B2",
    "top_k": "#E69F00",
    "top_p": "#009E73",
    "dola": "#CC79A7",
    "panda_switch": "#D55E00",
    "fanda_frozen": "#56B4E9",
}

BACKGROUND = "#ffffff"
PANEL_BG = "#ffffff"
GRID = "#d9d9d9"
AXIS = "#222222"
TEXT = "#222222"
TEXT_MUTED = "#444444"
PANEL_STROKE = "#dddddd"
FONT_FAMILY = "Times New Roman, Times, serif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an exp11 graph for MC scores and latency."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path(
            "results/experiments/exp11_core_decoder_comparison/runs/run_01_default/"
            "run_01_default_summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/exp11_core_decoder_mc_latency.svg"),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Core decoder quality vs latency",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=Path(
            "results/experiments/exp11_core_decoder_comparison/runs/run_01_default/"
            "run_01_default_raw_predictions.csv"
        ),
    )
    parser.add_argument(
        "--include-fanda-frozen",
        action="store_true",
        help="Append exp12's fanda_frozen result as a fanda bar using the matched question set.",
    )
    parser.add_argument(
        "--fanda-summary-csv",
        type=Path,
        default=Path(
            "results/experiments/exp12_state_persistence_diagnostics/runs/run_01_default/"
            "run_01_default_summary.csv"
        ),
    )
    parser.add_argument(
        "--fanda-raw-csv",
        type=Path,
        default=Path(
            "results/experiments/exp12_state_persistence_diagnostics/runs/run_01_default/"
            "run_01_default_raw_predictions.csv"
        ),
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


def clean_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def nice_latency_ticks(max_value: float) -> tuple[float, list[float]]:
    if max_value <= 50:
        step = 10.0
    elif max_value <= 100:
        step = 20.0
    else:
        step = 50.0
    upper = math.ceil(max_value / step) * step
    ticks = [step * idx for idx in range(int(upper // step) + 1)]
    return upper, ticks


def load_summary(path: Path, decoder_mapping: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[
        (df["benchmark"] == "truthfulqa")
        & (df["decoder"].isin(decoder_mapping))
        & (df["metric_name"].isin([metric for metric, _ in QUALITY_ORDER]))
    ].copy()
    df["decoder"] = df["decoder"].map(decoder_mapping)
    for column in ("score_mean", "score_sem", "latency_per_step_ms_mean", "latency_seconds_mean"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_missing_reasons(path: Path, decoder_mapping: dict[str, str]) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}

    grouped: dict[tuple[str, str], list[bool]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            decoder = row.get("decoder")
            metric_name = row.get("metric_name")
            benchmark = row.get("benchmark")
            if benchmark != "truthfulqa" or decoder not in decoder_mapping:
                continue
            if metric_name not in {metric for metric, _ in QUALITY_ORDER}:
                continue
            if not row.get("choice_scores"):
                continue
            try:
                choice_scores = json.loads(row["choice_scores"])
            except json.JSONDecodeError:
                continue
            all_truncated = bool(choice_scores) and all(
                value == float("-inf") for value in choice_scores.values()
            )
            canonical_decoder = decoder_mapping[decoder]
            grouped.setdefault((canonical_decoder, metric_name), []).append(all_truncated)

    reasons: dict[tuple[str, str], str] = {}
    for key, flags in grouped.items():
        if flags and all(flags):
            reasons[key] = "all choices truncated"
    return reasons


def build_stats(
    df: pd.DataFrame,
    source_paths: list[Path],
    raw_paths: list[Path],
    missing_reasons: dict[tuple[str, str], str],
    decoder_order: tuple[str, ...],
) -> dict[str, object]:
    quality: dict[str, dict[str, dict[str, float | None]]] = {}
    latency: dict[str, dict[str, float | str | None]] = {}
    for decoder in decoder_order:
        part = df[df["decoder"] == decoder]
        quality[decoder] = {}
        for metric_name, metric_label in QUALITY_ORDER:
            row = part[part["metric_name"] == metric_name]
            if row.empty:
                quality[decoder][metric_name] = {
                    "label": metric_label,
                    "score_mean": None,
                    "score_sem": None,
                    "missing_reason": missing_reasons.get((decoder, metric_name)),
                }
                continue
            first = row.iloc[0]
            quality[decoder][metric_name] = {
                "label": metric_label,
                "score_mean": clean_float(first["score_mean"]),
                "score_sem": clean_float(first["score_sem"]),
                "missing_reason": missing_reasons.get((decoder, metric_name)),
            }

        latency_value = clean_float(part["latency_per_step_ms_mean"].dropna().iloc[0] if not part["latency_per_step_ms_mean"].dropna().empty else None)
        latency_seconds = clean_float(part["latency_seconds_mean"].dropna().iloc[0] if not part["latency_seconds_mean"].dropna().empty else None)
        latency[decoder] = {
            "label": DISPLAY_LABELS[decoder],
            "latency_per_step_ms_mean": latency_value,
            "latency_seconds_mean": latency_seconds,
        }

    best_by_metric: dict[str, str | None] = {}
    for metric_name, _ in QUALITY_ORDER:
        scores = [
            (decoder, quality[decoder][metric_name]["score_mean"])
            for decoder in decoder_order
            if quality[decoder][metric_name]["score_mean"] is not None
        ]
        best_by_metric[metric_name] = max(scores, key=lambda item: float(item[1]))[0] if scores else None

    fastest_decoder = min(
        decoder_order,
        key=lambda decoder: float(latency[decoder]["latency_per_step_ms_mean"] or float("inf")),
    )
    if "fanda_frozen" in decoder_order:
        fanda_mc3 = clean_float(quality["fanda_frozen"]["mc3"]["score_mean"])
        fanda_latency = clean_float(latency["fanda_frozen"]["latency_per_step_ms_mean"])
        takeaway = (
            "V2 adds the matched exp12 frozen FAnDa result to the exp11 decoder comparison."
        )
        if fanda_mc3 is not None and fanda_latency is not None:
            takeaway = (
                f"Matched-question FAnDa posts MC3={fanda_mc3:.3f} at {fanda_latency:.1f} ms per decoding step "
                "when inserted into the exp11 comparison."
            )
    else:
        greedy_latency = clean_float(latency["pure_greedy"]["latency_per_step_ms_mean"])
        panda_latency = clean_float(latency["panda_switch"]["latency_per_step_ms_mean"])
        panda_mc2 = clean_float(quality["panda_switch"]["mc2"]["score_mean"])
        greedy_mc2 = clean_float(quality["pure_greedy"]["mc2"]["score_mean"])
        takeaway = (
            "Panda preserves the strong greedy-like MC profile in exp11 while paying a much larger "
            "step-time cost than the non-block baselines."
        )
        if all(value is not None for value in (greedy_latency, panda_latency, panda_mc2, greedy_mc2)):
            slowdown = float(panda_latency) / float(greedy_latency)
            takeaway = (
                f"Panda raises MC2 from {greedy_mc2:.3f} to {panda_mc2:.3f} over greedy in the saved exp11 run, "
                f"but it is {slowdown:.2f}x slower per decoding step."
            )

    return {
        "source_summary_csvs": [str(path) for path in source_paths],
        "source_raw_csvs": [str(path) for path in raw_paths],
        "decoders": [DISPLAY_LABELS[decoder] for decoder in decoder_order],
        "quality": quality,
        "latency": latency,
        "best_by_metric": best_by_metric,
        "fastest_decoder": fastest_decoder,
        "takeaway": takeaway,
        "notes": [
            "The plotted panda line is the original exp11 panda_switch decoder.",
            "Latency is mean milliseconds per decoding step from the saved summary CSV.",
            "When a missing metric can be traced to all candidate choice scores being -Infinity in the raw rows, the figure labels it as all choices truncated.",
        ],
    }


def draw_missing_label(svg: list[str], x: float, baseline_y: float, label: str) -> None:
    if label == "all choices truncated":
        svg.append(svg_text(x, baseline_y - 6, "all choices", size=9, fill=TEXT_MUTED, anchor="middle"))
        svg.append(svg_text(x, baseline_y + 6, "truncated", size=9, fill=TEXT_MUTED, anchor="middle"))
        return
    svg.append(svg_text(x, baseline_y, label, size=10, fill=TEXT_MUTED, anchor="middle"))


def draw_legend(svg: list[str], *, left: float, top: float, decoder_order: tuple[str, ...]) -> None:
    x = left
    for idx, decoder in enumerate(decoder_order):
        if idx == 3:
            x = left
            top += 22
        svg.append(svg_rect(x, top - 11, 14, 14, fill=COLORS[decoder], rx=3))
        svg.append(svg_text(x + 22, top, DISPLAY_LABELS[decoder], size=11))
        x += 116


def draw_quality_panel(
    svg: list[str],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    stats: dict[str, object],
) -> None:
    quality = stats["quality"]
    best_by_metric = stats["best_by_metric"]
    decoder_order = tuple(quality.keys())

    svg.append(svg_rect(left, top, width, height, fill=PANEL_BG, stroke=PANEL_STROKE, stroke_width=1.0, rx=14))
    svg.append(svg_text(left + width / 2, top + 24, "TruthfulQA MC results", size=17, anchor="middle", weight="bold"))
    draw_legend(svg, left=left + 24, top=top + 48, decoder_order=decoder_order)

    axis_left = left + 82
    axis_right = left + width - 26
    axis_top = top + 84
    axis_bottom = top + height - 76
    axis_width = axis_right - axis_left

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = scale(tick, 0.0, 1.0, axis_bottom, axis_top)
        svg.append(svg_line(axis_left, y, axis_right, y, stroke=GRID, stroke_width=1.0))
        svg.append(svg_text(axis_left - 10, y + 4, f"{tick:.2f}", size=10, fill=TEXT_MUTED, anchor="end"))

    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_line(axis_left, axis_bottom, axis_right, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_text(left + width / 2, top + height - 30, "Higher is better", size=11, fill=TEXT_MUTED, anchor="middle"))

    group_width = axis_width / len(QUALITY_ORDER)
    bar_width = 24.0
    gap = 8.0
    offset = (len(decoder_order) - 1) / 2.0

    for metric_idx, (metric_name, metric_label) in enumerate(QUALITY_ORDER):
        center_x = axis_left + group_width * (metric_idx + 0.5)
        for decoder_idx, decoder in enumerate(decoder_order):
            x = center_x + (decoder_idx - offset) * (bar_width + gap)
            score = quality[decoder][metric_name]["score_mean"]
            sem = quality[decoder][metric_name]["score_sem"]
            if score is None:
                missing_reason = quality[decoder][metric_name].get("missing_reason") or "n/a"
                svg.append(svg_line(x - bar_width / 2, axis_bottom - 2, x + bar_width / 2, axis_bottom - 2, stroke="#bbbbbb", stroke_width=1.1, dash="3 2"))
                draw_missing_label(svg, x, axis_bottom - 14, str(missing_reason))
                continue

            y = scale(float(score), 0.0, 1.0, axis_bottom, axis_top)
            is_leader = best_by_metric[metric_name] == decoder
            svg.append(
                svg_rect(
                    x - bar_width / 2,
                    y,
                    bar_width,
                    max(1.0, axis_bottom - y),
                    fill=COLORS[decoder],
                    stroke="#1f1f1f" if is_leader else "none",
                    stroke_width=1.2 if is_leader else 1.0,
                    rx=6,
                )
            )
            label_y = max(axis_top + 14, y - 8)
            svg.append(svg_text(x, label_y, f"{float(score):.3f}", size=10, fill=COLORS[decoder], anchor="middle"))

        svg.append(svg_text(center_x, axis_bottom + 24, metric_label, size=11, anchor="middle", weight="bold"))


def draw_latency_panel(
    svg: list[str],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    stats: dict[str, object],
) -> None:
    latency = stats["latency"]
    decoder_order = tuple(latency.keys())
    max_latency = max(float(latency[decoder]["latency_per_step_ms_mean"] or 0.0) for decoder in decoder_order)
    upper, ticks = nice_latency_ticks(max_latency)

    svg.append(svg_rect(left, top, width, height, fill=PANEL_BG, stroke=PANEL_STROKE, stroke_width=1.0, rx=14))
    svg.append(svg_text(left + width / 2, top + 24, "Latency", size=17, anchor="middle", weight="bold"))

    axis_left = left + 92
    axis_right = left + width - 30
    axis_top = top + 54
    axis_bottom = top + height - 84
    row_gap = (axis_bottom - axis_top) / len(decoder_order)
    bar_height = min(30.0, row_gap * 0.58)

    for tick in ticks:
        x = scale(float(tick), 0.0, upper, axis_left, axis_right)
        svg.append(svg_line(x, axis_top, x, axis_bottom, stroke=GRID, stroke_width=1.0))
        svg.append(svg_text(x, axis_bottom + 24, f"{int(tick)}", size=10, fill=TEXT_MUTED, anchor="middle"))

    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_line(axis_left, axis_bottom, axis_right, axis_bottom, stroke=AXIS, stroke_width=1.1))
    svg.append(svg_text((axis_left + axis_right) / 2, axis_bottom + 46, "Milliseconds", size=11, fill=TEXT_MUTED, anchor="middle"))

    for idx, decoder in enumerate(decoder_order):
        y_center = axis_top + row_gap * (idx + 0.5)
        y = y_center - bar_height / 2
        value = float(latency[decoder]["latency_per_step_ms_mean"] or 0.0)
        bar_right = scale(value, 0.0, upper, axis_left, axis_right)

        svg.append(svg_text(axis_left - 12, y_center + 4, DISPLAY_LABELS[decoder], size=11, fill=TEXT_MUTED, anchor="end"))
        svg.append(svg_rect(axis_left, y, bar_right - axis_left, bar_height, fill=COLORS[decoder], rx=8))
        svg.append(svg_text(bar_right + 8, y_center + 4, f"{value:.1f}", size=11, fill=COLORS[decoder], weight="bold"))


def build_figure(stats: dict[str, object], title: str) -> str:
    width = 1180
    height = 520
    body: list[str] = [svg_rect(0, 0, width, height, fill=BACKGROUND)]
    body.append(svg_text(width / 2, 26, title, size=18, anchor="middle", weight="bold"))

    draw_quality_panel(body, left=26, top=44, width=736, height=428, stats=stats)
    draw_latency_panel(body, left=784, top=44, width=370, height=428, stats=stats)
    return wrap_svg(body, width, height)


def main() -> None:
    args = parse_args()
    stats_output = args.stats_output or args.output.with_suffix(".json")
    exp11_mapping = {decoder: decoder for decoder in BASE_DECODER_ORDER}
    summary_frames = [load_summary(args.summary_csv, exp11_mapping)]
    missing_reasons = load_missing_reasons(args.raw_csv, exp11_mapping)
    source_summary_paths = [args.summary_csv]
    source_raw_paths = [args.raw_csv]
    decoder_order = BASE_DECODER_ORDER

    if args.include_fanda_frozen:
        fanda_mapping = {"fanda_frozen": "fanda_frozen"}
        summary_frames.append(load_summary(args.fanda_summary_csv, fanda_mapping))
        missing_reasons.update(load_missing_reasons(args.fanda_raw_csv, fanda_mapping))
        source_summary_paths.append(args.fanda_summary_csv)
        source_raw_paths.append(args.fanda_raw_csv)
        decoder_order = V2_DECODER_ORDER

    summary_df = pd.concat(summary_frames, ignore_index=True)
    stats = build_stats(summary_df, source_summary_paths, source_raw_paths, missing_reasons, decoder_order)
    svg = build_figure(stats, args.title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    stats_output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print({"figure": str(args.output), "stats": str(stats_output)})


if __name__ == "__main__":
    main()
