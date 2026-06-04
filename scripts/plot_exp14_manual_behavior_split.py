#!/usr/bin/env python3
"""Split exp14 behavior diagnostics into two standalone figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_exp14_manual_behavior import (
    BACKGROUND,
    build_stats_payload,
    draw_decoder_legend,
    draw_scatter_panel,
    load_manual_eval,
    svg_rect,
    wrap_svg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create separate answer-length and switch-rate figures from exp14 behavior diagnostics."
    )
    parser.add_argument(
        "--manual-csv",
        type=Path,
        default=Path(
            "results/experiments/exp14_openended_factuality/runs/run_01_default/"
            "run_01_default_manual_eval.csv"
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
        default="exp14_manual_behavior",
    )
    return parser.parse_args()


def build_single_panel_svg(
    df,
    *,
    x_col: str,
    x_label: str,
    title: str,
    x_min: float,
    x_max: float,
    x_ticks: list[float],
) -> str:
    width = 720
    height = 700
    svg: list[str] = [svg_rect(0, 0, width, height, fill=BACKGROUND)]
    draw_scatter_panel(
        svg,
        left=36,
        top=28,
        width=648,
        height=592,
        df=df,
        x_col=x_col,
        x_label=x_label,
        title=title,
        subtitle="",
        x_min=x_min,
        x_max=x_max,
        x_ticks=x_ticks,
    )
    draw_decoder_legend(svg, left=94, top=664)
    return wrap_svg(svg, width, height)


def main() -> None:
    args = parse_args()
    df = load_manual_eval(args.manual_csv)
    payload = build_stats_payload(df)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    length_path = output_dir / f"{args.prefix}_answer_length.svg"
    switch_path = output_dir / f"{args.prefix}_switch_rate.svg"
    manifest_path = output_dir / f"{args.prefix}_split.json"

    length_path.write_text(
        build_single_panel_svg(
            df,
            x_col="proxy_answer_token_count",
            x_label="Generated answer length (tokens)",
            title="Manual score vs answer length",
            x_min=8.0,
            x_max=66.0,
            x_ticks=[10, 20, 30, 40, 50, 60],
        ),
        encoding="utf-8",
    )
    switch_path.write_text(
        build_single_panel_svg(
            df,
            x_col="switch_rate",
            x_label="Layer switch rate",
            title="Manual score vs switch rate",
            x_min=-0.02,
            x_max=0.98,
            x_ticks=[0.0, 0.2, 0.4, 0.6, 0.8],
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "source_manual_csv": str(args.manual_csv),
                "answer_length_figure": str(length_path),
                "switch_rate_figure": str(switch_path),
                "stats": payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "answer_length_figure": str(length_path),
                "switch_rate_figure": str(switch_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
