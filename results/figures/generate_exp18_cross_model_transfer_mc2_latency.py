#!/usr/bin/env python3
"""Generate a paper-style exp18 cross-model transfer summary figure."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "results" / "figures"

OUTPUT_SVG = FIGURES_DIR / "exp18_cross_model_transfer_mc2_latency.svg"
OUTPUT_JSON = FIGURES_DIR / "exp18_cross_model_transfer_mc2_latency.json"
PRIMARY_METRIC = "mc2"

SOURCE_SPECS = [
    {
        "model_key": "qwen_anchor",
        "model_label": "Qwen 2.5 3B",
        "axis_label": "Qwen 2.5\n3B",
        "summary_csv": ROOT
        / "results"
        / "experiments"
        / "exp11_core_decoder_comparison"
        / "runs"
        / "run_01_default"
        / "run_01_default_summary.csv",
        "decoder_map": {
            "pure_greedy": "pure_greedy",
            "dola": "dola",
            "fanda": "fanda",
        },
    },
    {
        "model_key": "falcon3_3b",
        "model_label": "Falcon 3 3B",
        "axis_label": "Falcon 3\n3B",
        "summary_csv": ROOT
        / "results"
        / "experiments"
        / "exp18_cross_model_transfer"
        / "runs"
        / "run_02_falcon3_3b_instruct"
        / "run_02_falcon3_3b_instruct_summary.csv",
        "decoder_map": {
            "pure_greedy": "pure_greedy",
            "dola": "dola",
            "fanda": "fanda",
        },
    },
    {
        "model_key": "gemma3_4b",
        "model_label": "Gemma 3 4B",
        "axis_label": "Gemma 3\n4B",
        "summary_csv": ROOT
        / "results"
        / "experiments"
        / "exp18_cross_model_transfer"
        / "runs"
        / "run_01_gemma3_4b_it"
        / "run_01_gemma3_4b_it_summary.csv",
        "decoder_map": {
            "pure_greedy": "pure_greedy",
            "dola": "dola",
            "fanda": "fanda",
        },
    },
    {
        "model_key": "hint_selfcal_qwen_1p5b",
        "model_label": "HINT self-cal Qwen 1.5B",
        "axis_label": "HINT self-cal\nQwen 1.5B",
        "summary_csv": ROOT
        / "results"
        / "experiments"
        / "exp18_cross_model_transfer"
        / "runs"
        / "run_04_hint_selfcal_qwen_1p5b"
        / "run_04_hint_selfcal_qwen_1p5b_summary.csv",
        "decoder_map": {
            "pure_greedy": "pure_greedy",
            "dola": "dola",
            "fanda": "fanda",
        },
    },
]

DECODER_ORDER = ["pure_greedy", "dola", "fanda"]
DECODER_LABELS = {
    "pure_greedy": "greedy",
    "dola": "dola",
    "fanda": "fanda",
}
DECODER_COLORS = {
    "pure_greedy": "#4C78A8",
    "dola": "#9C5A4B",
    "fanda": "#2A7F78",
}

FONT_FAMILY = "Times New Roman"
GRID_COLOR = "#D9D9D9"
TEXT_COLOR = "#222222"
EDGE_COLOR = "#333333"


def parse_float(value: str):
    value = str(value).strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"nan", "inf", "-inf"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def load_metric_rows(summary_csv: Path):
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_metric_decoder = {}
    for row in rows:
        metric_name = row["metric_name"]
        decoder_name = row["decoder"]
        by_metric_decoder[(metric_name, decoder_name)] = row
    return by_metric_decoder


def build_plot_payload():
    payload = {
        "metric_name": PRIMARY_METRIC,
        "latency_field": "latency_per_forward_ms_mean",
        "decoder_order": DECODER_ORDER,
        "source_summary_csvs": [str(spec["summary_csv"]) for spec in SOURCE_SPECS],
        "models": [],
    }

    for spec in SOURCE_SPECS:
        rows = load_metric_rows(spec["summary_csv"])
        model_entry = {
            "model_key": spec["model_key"],
            "model_label": spec["model_label"],
            "axis_label": spec["axis_label"],
            "decoders": {},
        }
        for decoder_name in DECODER_ORDER:
            source_decoder = spec["decoder_map"][decoder_name]
            metric_row = rows[(PRIMARY_METRIC, source_decoder)]
            model_entry["decoders"][decoder_name] = {
                "decoder_label": DECODER_LABELS[decoder_name],
                PRIMARY_METRIC: parse_float(metric_row["score_mean"]),
                f"{PRIMARY_METRIC}_sem": parse_float(metric_row["score_sem"]),
                "latency_ms": parse_float(metric_row["latency_per_forward_ms_mean"]),
                "latency_seconds": parse_float(metric_row["latency_seconds_mean"]),
            }
        payload["models"].append(model_entry)
    return payload


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT_FAMILY, "Times", "DejaVu Serif"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": EDGE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "svg.fonttype": "none",
        }
    )


def plot_payload(payload):
    configure_matplotlib()

    model_labels = [model["axis_label"] for model in payload["models"]]
    x = np.arange(len(model_labels), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, len(DECODER_ORDER))
    quality_winners = []
    latency_winners = []
    for model in payload["models"]:
        quality_values = {
            decoder_name: model["decoders"][decoder_name][PRIMARY_METRIC]
            for decoder_name in DECODER_ORDER
            if model["decoders"][decoder_name][PRIMARY_METRIC] is not None
        }
        best_quality = max(quality_values.values()) if quality_values else None
        quality_winners.append(
            {
                decoder_name
                for decoder_name, value in quality_values.items()
                if best_quality is not None and value == best_quality
            }
        )

        latency_by_decoder = {
            decoder_name: model["decoders"][decoder_name]["latency_ms"]
            for decoder_name in DECODER_ORDER
            if model["decoders"][decoder_name]["latency_ms"] is not None
        }
        best_latency = min(latency_by_decoder.values()) if latency_by_decoder else None
        latency_winners.append(
            {
                decoder_name
                for decoder_name, value in latency_by_decoder.items()
                if best_latency is not None and value == best_latency
            }
        )

    fig, (ax_quality, ax_latency) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    for decoder_idx, decoder_name in enumerate(DECODER_ORDER):
        values = [model["decoders"][decoder_name][PRIMARY_METRIC] for model in payload["models"]]
        errors = [model["decoders"][decoder_name][f"{PRIMARY_METRIC}_sem"] or 0.0 for model in payload["models"]]
        bars = ax_quality.bar(
            x + offsets[decoder_idx],
            values,
            width=width,
            label=DECODER_LABELS[decoder_name],
            color=DECODER_COLORS[decoder_name],
            edgecolor="none",
            linewidth=0.0,
            yerr=errors,
            error_kw={"elinewidth": 0.9, "ecolor": EDGE_COLOR, "capsize": 2.5},
            zorder=3,
        )
        for model_idx, (bar, value) in enumerate(zip(bars, values)):
            if value is None:
                continue
            if decoder_name in quality_winners[model_idx]:
                bar.set_edgecolor(EDGE_COLOR)
                bar.set_linewidth(1.1)
            ax_quality.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.018,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=DECODER_COLORS[decoder_name],
            )

    ax_quality.set_title(f"TruthfulQA {PRIMARY_METRIC.upper()}")
    ax_quality.set_ylabel("Score")
    ax_quality.set_xticks(x)
    ax_quality.set_xticklabels(model_labels)
    ax_quality.set_ylim(0.0, 0.72)
    ax_quality.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax_quality.set_axisbelow(True)
    ax_quality.spines["top"].set_visible(False)
    ax_quality.spines["right"].set_visible(False)

    latency_values = {
        decoder_name: [model["decoders"][decoder_name]["latency_ms"] for model in payload["models"]]
        for decoder_name in DECODER_ORDER
    }
    latency_max = max(value for values in latency_values.values() for value in values if value is not None)
    latency_limit = math.ceil((latency_max + 10.0) / 10.0) * 10.0

    for decoder_idx, decoder_name in enumerate(DECODER_ORDER):
        values = latency_values[decoder_name]
        bars = ax_latency.bar(
            x + offsets[decoder_idx],
            values,
            width=width,
            label=DECODER_LABELS[decoder_name],
            color=DECODER_COLORS[decoder_name],
            edgecolor="none",
            linewidth=0.0,
            zorder=3,
        )
        for model_idx, (bar, value) in enumerate(zip(bars, values)):
            if value is None:
                continue
            if decoder_name in latency_winners[model_idx]:
                bar.set_edgecolor(EDGE_COLOR)
                bar.set_linewidth(1.1)
            ax_latency.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + latency_limit * 0.015,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=DECODER_COLORS[decoder_name],
            )

    ax_latency.set_title("Latency per Forward Pass")
    ax_latency.set_ylabel("Milliseconds")
    ax_latency.set_xticks(x)
    ax_latency.set_xticklabels(model_labels)
    ax_latency.set_ylim(0.0, latency_limit)
    ax_latency.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax_latency.set_axisbelow(True)
    ax_latency.spines["top"].set_visible(False)
    ax_latency.spines["right"].set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=DECODER_COLORS[name], edgecolor="none")
        for name in DECODER_ORDER
    ]
    fig.legend(
        legend_handles,
        [DECODER_LABELS[name] for name in DECODER_ORDER],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    payload = build_plot_payload()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_payload(payload)
    print(json.dumps({"figure": str(OUTPUT_SVG), "data": str(OUTPUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
