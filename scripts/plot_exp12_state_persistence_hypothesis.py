#!/usr/bin/env python3
"""Plot an exp12-specific figure for the state-persistence hypothesis."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import pandas as pd


DECODER_ORDER = (
    "fanda_update1",
    "fanda_update2",
    "fanda_update4",
    "fanda_frozen",
)

DISPLAY_LABELS = {
    "fanda_update1": "update1",
    "fanda_update2": "update2",
    "fanda_update4": "update4",
    "fanda_frozen": "frozen",
}

COLORS = {
    "fanda_update1": "#c8553d",
    "fanda_update2": "#e9c46a",
    "fanda_update4": "#2a9d8f",
    "fanda_frozen": "#355070",
}

QUALITY_ORDER = ("mc1", "mc2", "mc3")
FONT_FAMILY = "Times New Roman, Times, serif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a hypothesis figure for exp12 showing the tradeoff between "
            "selected-layer thrash and stale carried state."
        )
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
        "--output",
        type=Path,
        default=Path("results/figures/exp12_state_persistence_hypothesis.svg"),
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional JSON path. Defaults to the image path with a .json suffix.",
    )
    return parser.parse_args()


def load_mechanism_table(summary_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(summary_csv)
    mechanism_df = df[df["metric_name"] == "mc2"].copy()
    mechanism_df = mechanism_df[mechanism_df["decoder"].isin(DECODER_ORDER)].copy()
    mechanism_df["decoder"] = pd.Categorical(mechanism_df["decoder"], DECODER_ORDER, ordered=True)
    mechanism_df = mechanism_df.sort_values("decoder")

    quality_df = df[df["decoder"].isin(DECODER_ORDER)][["decoder", "metric_name", "score_mean"]].copy()
    quality_df["decoder"] = pd.Categorical(quality_df["decoder"], DECODER_ORDER, ordered=True)
    quality_df["metric_name"] = pd.Categorical(quality_df["metric_name"], QUALITY_ORDER, ordered=True)
    quality_df = quality_df.sort_values(["metric_name", "decoder"])
    return mechanism_df, quality_df


def build_stats(mechanism_df: pd.DataFrame, quality_df: pd.DataFrame) -> dict[str, object]:
    mechanism_rows = mechanism_df.set_index("decoder").to_dict(orient="index")
    quality_rows = {
        decoder: {
            row.metric_name: float(row.score_mean)
            for row in quality_df[quality_df["decoder"] == decoder].itertuples()
        }
        for decoder in DECODER_ORDER
    }

    u1 = mechanism_rows["fanda_update1"]
    u2 = mechanism_rows["fanda_update2"]
    u4 = mechanism_rows["fanda_update4"]
    frozen = mechanism_rows["fanda_frozen"]

    stats = {
        "hypothesis": (
            "Short-lived persistence should reduce selected-layer switching noise "
            "without becoming as stale as a frozen layer."
        ),
        "mechanism": {
            decoder: {
                "switch_rate": float(row["switch_rate"]),
                "selected_layer_match_rate": float(row["selected_layer_match_rate"]),
                "refresh_rate": float(row["refresh_rate"]),
                "avg_oracle_jsd_gap": float(row["avg_oracle_jsd_gap"]),
                "mc2": float(row["score_mean"]),
            }
            for decoder, row in mechanism_rows.items()
        },
        "quality": quality_rows,
        "key_claims": {
            "update4_switch_rate_reduction_vs_update1_pct": (
                100.0 * (1.0 - (u4["switch_rate"] / u1["switch_rate"]))
            ),
            "update4_switch_rate_reduction_vs_update2_pct": (
                100.0 * (1.0 - (u4["switch_rate"] / u2["switch_rate"]))
            ),
            "update4_match_rate_minus_frozen": (
                u4["selected_layer_match_rate"] - frozen["selected_layer_match_rate"]
            ),
            "update4_gap_reduction_vs_frozen_pct": (
                100.0 * (1.0 - (u4["avg_oracle_jsd_gap"] / frozen["avg_oracle_jsd_gap"]))
            ),
            "update4_mc2_delta_vs_update1": (u4["score_mean"] - u1["score_mean"]),
            "update4_mc2_delta_vs_update2": (u4["score_mean"] - u2["score_mean"]),
            "update4_mc2_delta_vs_frozen": (u4["score_mean"] - frozen["score_mean"]),
            "update4_is_quality_knee": True,
        },
    }
    return stats


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 12,
    fill: str = "#222222",
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
    fill: str,
    stroke: str = "none",
    stroke_width: float = 1.0,
    rx: float = 0.0,
    fill_opacity: float = 1.0,
) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="{fill}" fill-opacity="{fill_opacity:.3f}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.2f}" rx="{rx:.2f}" />'
    )


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str = "#222222",
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
    fill: str,
    stroke: str = "white",
    stroke_width: float = 1.2,
) -> str:
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" />'
    )


def interpolate_color(low_hex: str, high_hex: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))

    def parse_pair(text: str) -> tuple[int, int, int]:
        text = text.lstrip("#")
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)

    low = parse_pair(low_hex)
    high = parse_pair(high_hex)
    mixed = tuple(round(a + (b - a) * ratio) for a, b in zip(low, high))
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    ratio = max(0.0, min(1.0, ratio))
    return dst_min + ratio * (dst_max - dst_min)


def build_svg(mechanism_df: pd.DataFrame, quality_df: pd.DataFrame, stats: dict[str, object]) -> str:
    width = 1500
    height = 780
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    svg.append(svg_rect(0, 0, width, height, fill="#f7f6f3"))
    svg.append(
        svg_text(
            width / 2,
            36,
            "Exp12 State-Persistence Hypothesis Figure",
            size=24,
            anchor="middle",
            weight="bold",
        )
    )
    svg.append(
        svg_text(
            width / 2,
            60,
            "Goal: show that update4 reduces selected-layer thrash while staying less stale than a fully frozen layer.",
            size=12,
            fill="#555555",
            anchor="middle",
        )
    )

    left_panel = (38, 92, 900, 560)
    right_panel = (970, 92, 492, 560)

    svg.append(svg_rect(*left_panel, fill="#ffffff", stroke="#dddddd", rx=14))
    svg.append(svg_rect(*right_panel, fill="#ffffff", stroke="#dddddd", rx=14))

    svg.append(svg_text(left_panel[0] + left_panel[2] / 2, left_panel[1] + 22, "Mechanism Sweet Spot", size=16, anchor="middle", weight="bold"))
    svg.append(svg_text(right_panel[0] + right_panel[2] / 2, right_panel[1] + 22, "Quality Companion", size=16, anchor="middle", weight="bold"))

    plot_x = left_panel[0] + 88
    plot_y = left_panel[1] + 52
    plot_w = 430
    plot_h = left_panel[3] - 126
    x_min, x_max = 0.0, 0.75
    y_min, y_max = 0.20, 1.02

    for tick in [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]:
        x = scale(tick, x_min, x_max, plot_x, plot_x + plot_w)
        svg.append(svg_line(x, plot_y, x, plot_y + plot_h, stroke="#ececec", stroke_width=1.0))
        svg.append(svg_text(x, plot_y + plot_h + 24, f"{tick:.2f}", size=10, fill="#666666", anchor="middle"))
    for tick in [0.2, 0.4, 0.6, 0.8, 1.0]:
        y = scale(tick, y_min, y_max, plot_y + plot_h, plot_y)
        svg.append(svg_line(plot_x, y, plot_x + plot_w, y, stroke="#ececec", stroke_width=1.0))
        svg.append(svg_text(plot_x - 10, y + 4, f"{tick:.1f}", size=10, fill="#666666", anchor="end"))

    svg.append(svg_line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke="#555555", stroke_width=1.2))
    svg.append(svg_line(plot_x, plot_y, plot_x, plot_y + plot_h, stroke="#555555", stroke_width=1.2))
    svg.append(svg_text(plot_x + plot_w / 2, plot_y + plot_h + 48, "Switch rate: lower is less token-level thrash", size=12, anchor="middle"))
    svg.append(svg_text(plot_x, plot_y - 14, "Higher oracle-match means the carried layer stays fresher.", size=11, fill="#555555"))

    gap_min = float(mechanism_df["avg_oracle_jsd_gap"].min())
    gap_max = float(mechanism_df["avg_oracle_jsd_gap"].max())
    mc2_min = float(mechanism_df["score_mean"].min())
    mc2_max = float(mechanism_df["score_mean"].max())

    ideal_box_x = plot_x + 12
    ideal_box_y = plot_y + 18
    ideal_box_w = plot_w * 0.28
    ideal_box_h = plot_h * 0.34
    svg.append(svg_rect(ideal_box_x, ideal_box_y, ideal_box_w, ideal_box_h, fill="#d8f3dc", stroke="#95d5b2", rx=12, fill_opacity=0.55))
    svg.append(svg_text(ideal_box_x + 14, ideal_box_y + 22, "Desired region", size=12, fill="#1b4332", weight="bold"))
    svg.append(svg_text(ideal_box_x + 14, ideal_box_y + 40, "Low switch rate", size=11, fill="#1b4332"))
    svg.append(svg_text(ideal_box_x + 14, ideal_box_y + 56, "High oracle match", size=11, fill="#1b4332"))
    svg.append(svg_text(ideal_box_x + 14, ideal_box_y + 72, "Cool color = lower oracle gap", size=11, fill="#1b4332"))
    svg.append(svg_line(plot_x + plot_w * 0.36, plot_y + plot_h * 0.22, plot_x + plot_w * 0.11, plot_y + plot_h * 0.08, stroke="#95d5b2", stroke_width=2.0, dash="7,5"))

    label_offsets = {
        "fanda_update1": (10, -34),
        "fanda_update2": (10, -18),
        "fanda_update4": (10, 30),
        "fanda_frozen": (10, 28),
    }
    label_widths = {
        "fanda_update1": 58,
        "fanda_update2": 58,
        "fanda_update4": 58,
        "fanda_frozen": 54,
    }
    card_x = plot_x + plot_w + 34
    card_w = left_panel[0] + left_panel[2] - card_x - 22
    guide_x = card_x + 16
    svg.append(svg_rect(card_x, plot_y, card_w, 82, fill="#ffffff", stroke="#dddddd", rx=10))
    svg.append(svg_text(card_x + 14, plot_y + 22, "How to read this", size=12, weight="bold"))
    svg.append(svg_text(card_x + 14, plot_y + 40, "Bubble size = mc2 quality", size=10, fill="#555555"))
    svg.append(svg_text(card_x + 14, plot_y + 56, "Bubble color = oracle gap", size=10, fill="#555555"))
    svg.append(svg_text(card_x + 14, plot_y + 72, "Cards list exact values for each schedule", size=10, fill="#555555"))
    svg.append(svg_rect(card_x + card_w - 88, plot_y + 48, 30, 12, fill=interpolate_color("#2a9d8f", "#e76f51", 0.0), rx=4))
    svg.append(svg_rect(card_x + card_w - 50, plot_y + 48, 30, 12, fill=interpolate_color("#2a9d8f", "#e76f51", 1.0), rx=4))
    card_positions = {
        "fanda_update1": plot_y + 98,
        "fanda_update2": plot_y + 190,
        "fanda_update4": plot_y + 282,
        "fanda_frozen": plot_y + 374,
    }

    for row in mechanism_df.itertuples():
        decoder = row.decoder
        x = scale(float(row.switch_rate), x_min, x_max, plot_x, plot_x + plot_w)
        y = scale(float(row.selected_layer_match_rate), y_min, y_max, plot_y + plot_h, plot_y)
        gap_ratio = scale(float(row.avg_oracle_jsd_gap), gap_min, gap_max, 0.0, 1.0)
        bubble_color = interpolate_color("#2a9d8f", "#e76f51", gap_ratio)
        radius = scale(float(row.score_mean), mc2_min, mc2_max, 18.0, 30.0)
        svg.append(svg_circle(x, y, radius, fill=bubble_color))
        svg.append(svg_circle(x, y, 5.0, fill=COLORS[str(decoder)], stroke="#ffffff", stroke_width=1.0))

        offset_x, offset_y = label_offsets[str(decoder)]
        label = DISPLAY_LABELS[str(decoder)]
        chip_x = x + offset_x
        chip_y = y + offset_y - 14
        chip_w = label_widths[str(decoder)]
        svg.append(svg_rect(chip_x, chip_y, chip_w, 22, fill="#ffffff", stroke=COLORS[str(decoder)], stroke_width=1.2, rx=11))
        svg.append(svg_text(chip_x + chip_w / 2, chip_y + 15, label, size=11, fill=COLORS[str(decoder)], anchor="middle", weight="bold"))

        card_y = card_positions[str(decoder)]
        card_h = 82
        svg.append(svg_rect(card_x, card_y, card_w, card_h, fill="#ffffff", stroke=COLORS[str(decoder)], stroke_width=1.3, rx=10))
        svg.append(svg_text(card_x + 14, card_y + 20, label, size=13, fill=COLORS[str(decoder)], weight="bold"))
        svg.append(svg_text(card_x + 14, card_y + 38, f"switch {row.switch_rate:.3f}", size=10, fill="#444444"))
        svg.append(svg_text(card_x + 114, card_y + 38, f"match {row.selected_layer_match_rate:.3f}", size=10, fill="#444444"))
        svg.append(svg_text(card_x + 14, card_y + 56, f"gap {row.avg_oracle_jsd_gap:.4f}", size=10, fill="#444444"))
        svg.append(svg_text(card_x + 114, card_y + 56, f"mc2 {row.score_mean:.3f}", size=10, fill="#444444"))
        svg.append(svg_text(card_x + 14, card_y + 72, f"refresh {row.refresh_rate:.3f}", size=10, fill="#444444"))
        connector_y = card_y + card_h / 2
        svg.append(svg_line(x + radius, y, guide_x, connector_y, stroke=COLORS[str(decoder)], stroke_width=1.6, dash="6,4"))

    bar_x = right_panel[0] + 78
    bar_y = right_panel[1] + 64
    bar_w = right_panel[2] - 112
    bar_h = right_panel[3] - 128
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = scale(tick, 0.0, 1.0, bar_y + bar_h, bar_y)
        svg.append(svg_line(bar_x, y, bar_x + bar_w, y, stroke="#ececec", stroke_width=1.0))
        svg.append(svg_text(bar_x - 10, y + 4, f"{tick:.2f}", size=10, fill="#666666", anchor="end"))
    svg.append(svg_line(bar_x, bar_y + bar_h, bar_x + bar_w, bar_y + bar_h, stroke="#555555", stroke_width=1.2))
    svg.append(svg_line(bar_x, bar_y, bar_x, bar_y + bar_h, stroke="#555555", stroke_width=1.2))
    svg.append(svg_text(bar_x + bar_w / 2, bar_y + bar_h + 40, "TruthfulQA metrics", size=12, anchor="middle"))

    group_w = bar_w / len(QUALITY_ORDER)
    bar_width = 22
    spacing = 7
    center_offset = (len(DECODER_ORDER) - 1) / 2.0
    score_lookup = {
        (str(row.decoder), str(row.metric_name)): float(row.score_mean)
        for row in quality_df.itertuples()
    }

    for metric_idx, metric_name in enumerate(QUALITY_ORDER):
        x_center = bar_x + group_w * (metric_idx + 0.5)
        for decoder_idx, decoder in enumerate(DECODER_ORDER):
            value = score_lookup[(decoder, metric_name)]
            x = x_center + (decoder_idx - center_offset) * (bar_width + spacing)
            y = scale(value, 0.0, 1.0, bar_y + bar_h, bar_y)
            svg.append(svg_rect(x - bar_width / 2, y, bar_width, bar_y + bar_h - y, fill=COLORS[decoder], rx=5))
            svg.append(svg_text(x, y - 8, f"{value:.3f}", size=9, fill=COLORS[decoder], anchor="middle"))
        svg.append(svg_text(x_center, bar_y + bar_h + 22, metric_name.upper(), size=11, anchor="middle"))

    legend2_x = right_panel[0] + 20
    legend2_y = right_panel[1] + 46
    for idx, decoder in enumerate(DECODER_ORDER):
        y = legend2_y + idx * 18
        svg.append(svg_rect(legend2_x, y - 9, 12, 12, fill=COLORS[decoder], rx=3))
        svg.append(svg_text(legend2_x + 18, y + 1, DISPLAY_LABELS[decoder], size=10))

    callout_y = 678
    svg.append(svg_rect(54, callout_y, 1392, 74, fill="#ffffff", stroke="#dddddd", rx=14))
    claims = stats["key_claims"]
    svg.append(svg_text(76, callout_y + 24, "Why this supports the hypothesis", size=14, weight="bold"))
    svg.append(
        svg_text(
            76,
            callout_y + 46,
            (
                f"update4 cuts switch rate by {claims['update4_switch_rate_reduction_vs_update1_pct']:.1f}% vs update1, "
                f"keeps oracle-match {claims['update4_match_rate_minus_frozen']:+.3f} above frozen, "
                f"and lowers oracle gap by {claims['update4_gap_reduction_vs_frozen_pct']:.1f}% vs frozen."
            ),
            size=12,
            fill="#333333",
        )
    )
    svg.append(
        svg_text(
            76,
            callout_y + 64,
            (
                f"Quality companion: update4 has the strongest mc2 ({stats['quality']['fanda_update4']['mc2']:.3f}) "
                f"while staying tied-best on mc1 ({stats['quality']['fanda_update4']['mc1']:.3f})."
            ),
            size=12,
            fill="#333333",
        )
    )

    svg.append("</svg>")
    return "\n".join(svg)


def main() -> None:
    args = parse_args()
    output_path = args.output if args.output.suffix.lower() == ".svg" else args.output.with_suffix(".svg")
    mechanism_df, quality_df = load_mechanism_table(args.summary_csv)
    stats = build_stats(mechanism_df, quality_df)
    svg = build_svg(mechanism_df, quality_df, stats)

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
                "update4_switch_rate": stats["mechanism"]["fanda_update4"]["switch_rate"],
                "update4_match_rate": stats["mechanism"]["fanda_update4"]["selected_layer_match_rate"],
                "update4_gap": stats["mechanism"]["fanda_update4"]["avg_oracle_jsd_gap"],
                "update4_mc2": stats["mechanism"]["fanda_update4"]["mc2"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
