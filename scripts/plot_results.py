import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


QUALITY_ORDER = [
    ("truthfulqa", "mc1", "TruthfulQA MC1"),
    ("truthfulqa", "mc2", "TruthfulQA MC2"),
    ("truthfulqa", "mc3", "TruthfulQA MC3"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a compact evaluation overview from a saved summary CSV.")
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Evaluation Overview")
    return parser.parse_args()


def build_quality_table(summary_df):
    rows = []
    for benchmark, metric_name, label in QUALITY_ORDER:
        subset = summary_df[
            (summary_df["benchmark"] == benchmark)
            & (summary_df["metric_name"] == metric_name)
        ][["decoder_label", "score_mean"]].copy()
        if subset.empty:
            continue
        subset["metric_label"] = label
        rows.append(subset)
    if not rows:
        return pd.DataFrame(columns=["decoder_label", "score_mean", "metric_label"])
    return pd.concat(rows, ignore_index=True)


def build_latency_table(summary_df):
    agg_map = {
        "latency_per_step_ms_mean": ("latency_per_step_ms_mean", "mean"),
    }
    if "fallback_rate" in summary_df.columns:
        agg_map["fallback_rate"] = ("fallback_rate", "mean")
    latency_df = summary_df.groupby("decoder_label", as_index=False).agg(**agg_map)
    latency_df = latency_df.sort_values("latency_per_step_ms_mean", ascending=True)
    if "fallback_rate" not in latency_df.columns:
        latency_df["fallback_rate"] = pd.NA
    return latency_df


def plot_overview(summary_df, output_path, title):
    quality_df = build_quality_table(summary_df)
    latency_df = build_latency_table(summary_df)

    decoder_order = list(dict.fromkeys(summary_df["decoder_label"].tolist()))
    colors = {
        "greedy": "#355070",
        "Greedy": "#355070",
        "dola": "#6d597a",
        "DoLa": "#6d597a",
        "fixed alpha dola (0.1)": "#d08c60",
        "fixed alpha dola (0.5)": "#b56576",
        "fixed alpha dola (0.95)": "#82435d",
        "panda": "#5f0f40",
        "PAnDa": "#5f0f40",
    }

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.7], hspace=0.35, wspace=0.25)
    ax_quality = fig.add_subplot(gs[0, :])
    ax_latency = fig.add_subplot(gs[1, 0])
    ax_fallback = fig.add_subplot(gs[1, 1])

    if not quality_df.empty:
        metric_labels = list(dict.fromkeys(quality_df["metric_label"].tolist()))
        x = range(len(metric_labels))
        width = 0.18 if len(decoder_order) >= 4 else 0.22
        center_offset = (len(decoder_order) - 1) / 2.0
        for idx, decoder_label in enumerate(decoder_order):
            decoder_rows = quality_df[quality_df["decoder_label"] == decoder_label]
            score_map = {
                row["metric_label"]: float(row["score_mean"])
                for _, row in decoder_rows.iterrows()
            }
            values = [score_map.get(metric_label, float("nan")) for metric_label in metric_labels]
            positions = [pos + (idx - center_offset) * width for pos in x]
            ax_quality.bar(
                positions,
                values,
                width=width,
                label=decoder_label,
                color=colors.get(decoder_label, None),
            )
        ax_quality.set_xticks(list(x))
        ax_quality.set_xticklabels(metric_labels)
        ax_quality.set_ylim(0.0, 1.0)
        ax_quality.set_ylabel("Score")
        ax_quality.set_title("Quality Metrics")
        ax_quality.grid(axis="y", alpha=0.2)
        ax_quality.legend(ncol=2, frameon=False, fontsize=9)

    if not latency_df.empty:
        ax_latency.barh(
            latency_df["decoder_label"],
            latency_df["latency_per_step_ms_mean"],
            color=[colors.get(label, "#999999") for label in latency_df["decoder_label"]],
        )
        ax_latency.set_title("Average Latency")
        ax_latency.set_xlabel("ms / decoding step")
        ax_latency.grid(axis="x", alpha=0.2)

        fallback_rows = latency_df.dropna(subset=["fallback_rate"]).copy()
        if fallback_rows.empty:
            ax_fallback.text(0.5, 0.5, "No fallback metric", ha="center", va="center")
            ax_fallback.set_axis_off()
        else:
            ax_fallback.barh(
                fallback_rows["decoder_label"],
                fallback_rows["fallback_rate"],
                color=[colors.get(label, "#999999") for label in fallback_rows["decoder_label"]],
            )
            ax_fallback.set_xlim(0.0, 1.0)
            ax_fallback.set_title("Fallback Rate")
            ax_fallback.set_xlabel("fraction of decoding steps")
            ax_fallback.grid(axis="x", alpha=0.2)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    summary_df = pd.read_csv(args.summary_csv)
    plot_overview(summary_df, args.output, args.title)
    print({"output": str(args.output)})


if __name__ == "__main__":
    main()
