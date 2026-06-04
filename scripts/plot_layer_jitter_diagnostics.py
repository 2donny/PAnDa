#!/usr/bin/env python3
"""Plot saved decoder diagnostics that highlight selected-layer jitter."""

from __future__ import annotations

import argparse
import json
import math
import random
from html import escape
from pathlib import Path

import pandas as pd


EXP11_DECODERS = ("dola", "fanda")
EXP5_DECODERS = (
    "fanda",
    "panda_switch_update4",
    "panda_switch",
    "panda_fandas",
)
QUALITY_ORDER = ("mc1", "mc2", "mc3")

DISPLAY_LABELS = {
    "dola": "DoLa",
    "fanda": "fanda\n(update4)",
    "panda_switch_update4": "panda_switch\n(update4)",
    "panda_switch": "panda_switch\n(per-step)",
    "panda_fandas": "panda_fandas\n(per-step)",
}

COLORS = {
    "dola": "#c8553d",
    "fanda": "#2a9d8f",
    "panda_switch_update4": "#457b9d",
    "panda_switch": "#6d597a",
    "panda_fandas": "#b56576",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a figure that contrasts DoLa's per-token layer reselection "
            "against the update-every-4 fanda path using saved experiment outputs."
        )
    )
    parser.add_argument(
        "--exp11-raw",
        type=Path,
        default=Path(
            "results/experiments/exp11_core_decoder_comparison/runs/run_01_default/"
            "run_01_default_raw_predictions.csv"
        ),
    )
    parser.add_argument(
        "--exp5-raw",
        type=Path,
        default=Path(
            "results/experiments/exp5_block_ablation/runs/run_02_update4_default/"
            "run_02_update4_default_raw_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/dola_update4_jitter_diagnostics.svg"),
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional JSON path. Defaults to the image path with a .json suffix.",
    )
    return parser.parse_args()


def load_mc1_rows(path: Path, decoders: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = df[(df["metric_name"] == "mc1") & (df["decoder"].isin(decoders))].copy()
    rows["switch_count_est"] = rows["switch_rate"] * (rows["decoder_steps"] - 1).clip(lower=0)
    return rows


def load_quality_rows(path: Path, decoders: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(path)
    grouped = (
        df[df["decoder"].isin(decoders)]
        .groupby(["decoder", "metric_name"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "score_mean"})
    )
    return grouped


def summarize_switch_metrics(rows: pd.DataFrame, decoders: tuple[str, ...]) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for decoder in decoders:
        subset = rows[rows["decoder"] == decoder]
        values = subset["avg_instability"].dropna() if "avg_instability" in subset.columns else pd.Series(dtype=float)
        summary[decoder] = {
            "mean_switch_rate": float(subset["switch_rate"].mean()),
            "median_switch_rate": float(subset["switch_rate"].median()),
            "mean_switch_count_est": float(subset["switch_count_est"].mean()),
            "mean_avg_selected_layer": float(subset["avg_selected_layer"].mean()),
            "mean_avg_instability": float(values.mean()) if not values.empty else None,
            "num_examples": int(len(subset)),
        }
    return summary


def build_stats_payload(
    exp11_switch: pd.DataFrame,
    exp5_switch: pd.DataFrame,
    exp11_quality: pd.DataFrame,
) -> dict[str, object]:
    exp11_summary = summarize_switch_metrics(exp11_switch, EXP11_DECODERS)
    exp5_summary = summarize_switch_metrics(exp5_switch, EXP5_DECODERS)
    quality_summary = {
        decoder: {
            row.metric_name: float(row.score_mean)
            for row in exp11_quality[exp11_quality["decoder"] == decoder].itertuples()
        }
        for decoder in EXP11_DECODERS
    }

    dola_switch = float(exp11_summary["dola"]["mean_switch_rate"])
    update4_switch = float(exp11_summary["fanda"]["mean_switch_rate"])
    exp5_update4_switch = float(exp5_summary["panda_switch_update4"]["mean_switch_rate"])
    exp5_per_step_switch = float(exp5_summary["panda_switch"]["mean_switch_rate"])
    exp5_always_per_step_switch = float(exp5_summary["panda_fandas"]["mean_switch_rate"])

    return {
        "figure_claim": (
            "The saved results support the claim that per-token or per-position layer "
            "reselection is much more jittery than the update-every-4 path."
        ),
        "exp11": {
            "switch_metrics": exp11_summary,
            "quality": quality_summary,
            "mean_switch_rate_ratio_dola_over_update4": (
                dola_switch / update4_switch if update4_switch else None
            ),
            "mean_switch_rate_delta_dola_minus_update4": dola_switch - update4_switch,
        },
        "exp5": {
            "switch_metrics": exp5_summary,
            "mean_switch_rate_ratio_panda_switch_over_update4": (
                exp5_per_step_switch / exp5_update4_switch if exp5_update4_switch else None
            ),
            "mean_switch_rate_ratio_panda_fandas_over_update4": (
                exp5_always_per_step_switch / exp5_update4_switch if exp5_update4_switch else None
            ),
        },
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    fraction = position - low
    return float(sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction)


def svg_text(x: float, y: float, text: str, size: int = 12, fill: str = "#222222", anchor: str = "start",
             weight: str = "normal") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
        f'font-family="Arial, Helvetica, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}">{escape(text)}</text>'
    )


def svg_rect(x: float, y: float, width: float, height: float, fill: str, stroke: str = "none",
             stroke_width: float = 1.0, rx: float = 0.0, fill_opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="{fill}" fill-opacity="{fill_opacity:.3f}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.2f}" rx="{rx:.2f}" />'
    )


def svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#222222",
             stroke_width: float = 1.0, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"{dash_attr} />'
    )


def svg_circle(cx: float, cy: float, r: float, fill: str, stroke: str = "white",
               stroke_width: float = 0.7, fill_opacity: float = 0.78) -> str:
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
        f'fill-opacity="{fill_opacity:.3f}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" />'
    )


def value_to_y(value: float, panel_top: float, panel_height: float, y_min: float = 0.0, y_max: float = 1.0) -> float:
    ratio = 0.0 if y_max == y_min else (value - y_min) / (y_max - y_min)
    ratio = max(0.0, min(1.0, ratio))
    return panel_top + panel_height * (1.0 - ratio)


def draw_multiline_label(x: float, y: float, text: str, size: int = 12, fill: str = "#222222") -> list[str]:
    lines = text.split("\n")
    svg = []
    for idx, line in enumerate(lines):
        svg.append(svg_text(x, y + idx * (size + 2), line, size=size, fill=fill, anchor="middle"))
    return svg


def draw_switch_rate_panel(
    rows: pd.DataFrame,
    order: tuple[str, ...],
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
    title: str,
) -> list[str]:
    svg: list[str] = []
    axis_left = panel_x + 58
    axis_right = panel_x + panel_w - 18
    axis_top = panel_y + 34
    axis_bottom = panel_y + panel_h - 60
    axis_w = axis_right - axis_left
    axis_h = axis_bottom - axis_top

    svg.append(svg_rect(panel_x, panel_y, panel_w, panel_h, fill="#ffffff", stroke="#dddddd", rx=12))
    svg.append(svg_text(panel_x + panel_w / 2, panel_y + 20, title, size=14, anchor="middle", weight="bold"))

    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    for tick in ticks:
        y = value_to_y(tick, axis_top, axis_h)
        svg.append(svg_line(axis_left, y, axis_right, y, stroke="#e8e8e8", stroke_width=1.0))
        svg.append(svg_text(axis_left - 8, y + 4, f"{tick:.2f}", size=10, fill="#666666", anchor="end"))

    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke="#555555", stroke_width=1.2))
    svg.append(svg_line(axis_left, axis_bottom, axis_right, axis_bottom, stroke="#555555", stroke_width=1.2))
    svg.extend(draw_multiline_label(panel_x + 18, panel_y + panel_h / 2, "Switch\nrate", size=11, fill="#444444"))

    band = axis_w / len(order)
    for idx, decoder in enumerate(order):
        x_center = axis_left + band * (idx + 0.5)
        subset = rows[rows["decoder"] == decoder]["switch_rate"].dropna().tolist()
        values = sorted(float(v) for v in subset)
        rng = random.Random(17 + idx)
        for value in values:
            jitter = rng.uniform(-0.16, 0.16) * band
            svg.append(
                svg_circle(
                    x_center + jitter,
                    value_to_y(value, axis_top, axis_h),
                    4.0,
                    fill=COLORS[decoder],
                )
            )
        if values:
            q1 = percentile(values, 0.25)
            median = percentile(values, 0.50)
            q3 = percentile(values, 0.75)
            v_min = values[0]
            v_max = values[-1]
            box_w = min(38.0, band * 0.46)
            y_q1 = value_to_y(q1, axis_top, axis_h)
            y_q3 = value_to_y(q3, axis_top, axis_h)
            y_med = value_to_y(median, axis_top, axis_h)
            y_min = value_to_y(v_min, axis_top, axis_h)
            y_max = value_to_y(v_max, axis_top, axis_h)
            svg.append(svg_line(x_center, y_max, x_center, y_q3, stroke=COLORS[decoder], stroke_width=1.4))
            svg.append(svg_line(x_center, y_q1, x_center, y_min, stroke=COLORS[decoder], stroke_width=1.4))
            svg.append(svg_line(x_center - 11, y_max, x_center + 11, y_max, stroke=COLORS[decoder], stroke_width=1.4))
            svg.append(svg_line(x_center - 11, y_min, x_center + 11, y_min, stroke=COLORS[decoder], stroke_width=1.4))
            svg.append(
                svg_rect(
                    x_center - box_w / 2,
                    y_q3,
                    box_w,
                    max(1.0, y_q1 - y_q3),
                    fill=COLORS[decoder],
                    fill_opacity=0.20,
                    stroke=COLORS[decoder],
                    stroke_width=1.4,
                    rx=5,
                )
            )
            svg.append(svg_line(x_center - box_w / 2, y_med, x_center + box_w / 2, y_med, stroke="#222222", stroke_width=1.7))

            mean_value = sum(values) / len(values)
            y_mean = value_to_y(mean_value, axis_top, axis_h)
            svg.append(svg_line(x_center - box_w / 2, y_mean, x_center + box_w / 2, y_mean, stroke=COLORS[decoder], stroke_width=2.4))
            svg.append(svg_text(x_center, max(axis_top + 12, y_mean - 8), f"{mean_value:.3f}", size=10, fill=COLORS[decoder], anchor="middle", weight="bold"))

        svg.extend(draw_multiline_label(x_center, axis_bottom + 22, DISPLAY_LABELS[decoder], size=11))
    return svg


def draw_quality_panel(
    quality_rows: pd.DataFrame,
    order: tuple[str, ...],
    panel_x: float,
    panel_y: float,
    panel_w: float,
    panel_h: float,
) -> list[str]:
    svg: list[str] = []
    axis_left = panel_x + 58
    axis_right = panel_x + panel_w - 20
    axis_top = panel_y + 34
    axis_bottom = panel_y + panel_h - 60
    axis_w = axis_right - axis_left
    axis_h = axis_bottom - axis_top

    svg.append(svg_rect(panel_x, panel_y, panel_w, panel_h, fill="#ffffff", stroke="#dddddd", rx=12))
    svg.append(svg_text(panel_x + panel_w / 2, panel_y + 20, "Exp11: Quality On The Same Saved Run", size=14, anchor="middle", weight="bold"))

    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    for tick in ticks:
        y = value_to_y(tick, axis_top, axis_h)
        svg.append(svg_line(axis_left, y, axis_right, y, stroke="#e8e8e8", stroke_width=1.0))
        svg.append(svg_text(axis_left - 8, y + 4, f"{tick:.2f}", size=10, fill="#666666", anchor="end"))
    svg.append(svg_line(axis_left, axis_top, axis_left, axis_bottom, stroke="#555555", stroke_width=1.2))
    svg.append(svg_line(axis_left, axis_bottom, axis_right, axis_bottom, stroke="#555555", stroke_width=1.2))
    svg.extend(draw_multiline_label(panel_x + 18, panel_y + panel_h / 2, "TruthfulQA\nscore", size=11, fill="#444444"))

    group_w = axis_w / len(QUALITY_ORDER)
    bar_w = min(52.0, group_w * 0.28)
    center_offset = (len(order) - 1) / 2.0
    score_lookup = {
        (row.decoder, row.metric_name): float(row.score_mean)
        for row in quality_rows.itertuples()
    }

    for group_idx, metric in enumerate(QUALITY_ORDER):
        x_center = axis_left + group_w * (group_idx + 0.5)
        for decoder_idx, decoder in enumerate(order):
            value = score_lookup[(decoder, metric)]
            x = x_center + (decoder_idx - center_offset) * (bar_w + 8)
            y = value_to_y(value, axis_top, axis_h)
            svg.append(
                svg_rect(
                    x - bar_w / 2,
                    y,
                    bar_w,
                    axis_bottom - y,
                    fill=COLORS[decoder],
                    stroke="none",
                    rx=6,
                )
            )
            svg.append(svg_text(x, max(axis_top + 12, y - 8), f"{value:.3f}", size=10, fill=COLORS[decoder], anchor="middle"))
        svg.append(svg_text(x_center, axis_bottom + 22, metric.upper(), size=11, anchor="middle"))

    legend_x = axis_left + 6
    legend_y = panel_y + 48
    for idx, decoder in enumerate(order):
        y = legend_y + idx * 18
        svg.append(svg_rect(legend_x, y - 9, 12, 12, fill=COLORS[decoder], rx=3))
        svg.append(svg_text(legend_x + 18, y + 1, DISPLAY_LABELS[decoder].replace("\n", " "), size=10))
    return svg


def build_svg(
    exp11_switch: pd.DataFrame,
    exp5_switch: pd.DataFrame,
    exp11_quality: pd.DataFrame,
    stats: dict[str, object],
) -> str:
    width = 1500
    height = 620
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    svg.append(svg_rect(0, 0, width, height, fill="#f7f6f3"))
    svg.append(svg_text(width / 2, 34, "Selected-Layer Jitter Diagnostics From Completed Runs", size=22, anchor="middle", weight="bold"))
    svg.append(
        svg_text(
            width / 2,
            58,
            "Completed results only: DoLa reselects every token, while the default fanda path refreshes the layer every 4 steps.",
            size=12,
            fill="#555555",
            anchor="middle",
        )
    )

    panels = [
        (34, 84, 360, 472),
        (416, 84, 540, 472),
        (978, 84, 488, 472),
    ]
    svg.extend(
        draw_switch_rate_panel(
            exp11_switch,
            EXP11_DECODERS,
            panels[0][0],
            panels[0][1],
            panels[0][2],
            panels[0][3],
            "Exp11: DoLa Thrashes More Than update4",
        )
    )
    svg.extend(
        draw_switch_rate_panel(
            exp5_switch,
            EXP5_DECODERS,
            panels[1][0],
            panels[1][1],
            panels[1][2],
            panels[1][3],
            "Exp5: update4 Lowers Thrash Inside The Block Family",
        )
    )
    svg.extend(draw_quality_panel(exp11_quality, EXP11_DECODERS, panels[2][0], panels[2][1], panels[2][2], panels[2][3]))

    exp11_ratio = float(stats["exp11"]["mean_switch_rate_ratio_dola_over_update4"])
    exp5_ratio = float(stats["exp5"]["mean_switch_rate_ratio_panda_switch_over_update4"])
    exp5_ratio_contrast = float(stats["exp5"]["mean_switch_rate_ratio_panda_fandas_over_update4"])

    note_w = 282
    note_h = 48
    note1_x = panels[0][0] + 56
    note1_y = panels[0][1] + 56
    svg.append(svg_rect(note1_x, note1_y, note_w, note_h, fill="#ffffff", stroke="#dddddd", rx=10))
    svg.append(svg_text(note1_x + 14, note1_y + 21, f"DoLa / update4 mean switch-rate ratio: {exp11_ratio:.2f}x", size=11, weight="bold"))
    svg.append(svg_text(note1_x + 14, note1_y + 38, "The completed exp11 run already shows a large gap.", size=10, fill="#555555"))

    note2_x = panels[1][0] + 62
    note2_y = panels[1][1] + 56
    svg.append(svg_rect(note2_x, note2_y, 338, 62, fill="#ffffff", stroke="#dddddd", rx=10))
    svg.append(svg_text(note2_x + 14, note2_y + 21, f"panda_switch / update4: {exp5_ratio:.2f}x", size=11, weight="bold"))
    svg.append(svg_text(note2_x + 14, note2_y + 39, f"panda_fandas / update4: {exp5_ratio_contrast:.2f}x", size=11, weight="bold"))
    svg.append(svg_text(note2_x + 14, note2_y + 56, "The update-4 cadence stays low-switch across two scaffolds.", size=10, fill="#555555"))

    footer = (
        "Interpretation: this figure supports the stability claim from completed runs, but a stronger mechanism proof would still come from finishing exp12, which isolates update1/update2/update4/frozen directly."
    )
    svg.append(svg_text(width / 2, 594, footer, size=11, fill="#444444", anchor="middle"))
    svg.append("</svg>")
    return "\n".join(svg)


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output.suffix.lower() == ".svg" else args.output.with_suffix(".svg")

    exp11_switch = load_mc1_rows(args.exp11_raw, EXP11_DECODERS)
    exp5_switch = load_mc1_rows(args.exp5_raw, EXP5_DECODERS)
    exp11_quality = load_quality_rows(args.exp11_raw, EXP11_DECODERS)
    stats = build_stats_payload(exp11_switch, exp5_switch, exp11_quality)
    svg = build_svg(exp11_switch, exp5_switch, exp11_quality, stats)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")

    stats_output = args.stats_output or output_path.with_suffix(".json")
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "figure": str(output_path),
                "stats": str(stats_output),
                "exp11_mean_switch_rate_dola": stats["exp11"]["switch_metrics"]["dola"]["mean_switch_rate"],
                "exp11_mean_switch_rate_update4": stats["exp11"]["switch_metrics"]["fanda"][
                    "mean_switch_rate"
                ],
                "exp11_switch_rate_ratio_dola_over_update4": stats["exp11"][
                    "mean_switch_rate_ratio_dola_over_update4"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
