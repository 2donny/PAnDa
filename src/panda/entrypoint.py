"""Command-line entrypoint for benchmark evaluation."""

import json
import time
from pathlib import Path

import torch

try:
    import pandas as pd
except ImportError:
    pd = None

from .args import apply_comparison_preset, parse_args
from .benchmarks import load_truthfulqa_rows
from .evaluation import (
    assert_eval_sources,
    build_pairwise_summary,
    evaluate_truthfulqa,
)
from .evaluator import Stage4Evaluator
from .utils import make_sampling_rng, resolve_limit


def infer_artifact_prefix(args):
    if args.comparison_preset == "panda":
        return "panda_full_eval"
    return "comparison_full_eval"


def save_outputs(results_df, summary_df, pairwise_df, metadata, results_dir, artifact_prefix):
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / f"{artifact_prefix}_raw_predictions.csv"
    summary_path = results_dir / f"{artifact_prefix}_summary.csv"
    pairwise_path = results_dir / f"{artifact_prefix}_pairwise_summary.csv"
    metadata_path = results_dir / f"{artifact_prefix}_metadata.json"
    if pd is not None:
        results_df.to_csv(raw_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        pairwise_df.to_csv(pairwise_path, index=False)
    else:
        (results_dir / f"{artifact_prefix}_raw_predictions.json").write_text(
            json.dumps(results_df, indent=2), encoding="utf-8"
        )
        (results_dir / f"{artifact_prefix}_summary.json").write_text(
            json.dumps(summary_df, indent=2), encoding="utf-8"
        )
        (results_dir / f"{artifact_prefix}_pairwise_summary.json").write_text(
            json.dumps(pairwise_df, indent=2), encoding="utf-8"
        )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        {
            "raw_predictions_path": str(raw_path),
            "summary_path": str(summary_path),
            "pairwise_summary_path": str(pairwise_path),
            "metadata_path": str(metadata_path),
        }
    )


def main():
    args = apply_comparison_preset(parse_args())
    torch.manual_seed(args.seed)
    artifact_prefix = infer_artifact_prefix(args)

    truthfulqa_limit = resolve_limit(args.truthfulqa_limit, args.mode, 5)

    print(
        {
            "truthfulqa_limit": truthfulqa_limit,
            "save_results": args.save_results,
        }
    )

    evaluator = Stage4Evaluator(args)

    truthfulqa_rows, truthfulqa_source, truthfulqa_manifest = load_truthfulqa_rows(
        truthfulqa_limit,
        make_sampling_rng(args.seed, "truthfulqa"),
    )

    assert_eval_sources(args, truthfulqa_source)

    print(
        {
            "truthfulqa_source": truthfulqa_source,
            "truthfulqa_sampling": truthfulqa_manifest,
            "truthfulqa_examples": len(truthfulqa_rows),
        }
    )

    all_results = []
    start_time = time.perf_counter()
    if truthfulqa_rows:
        all_results.extend(evaluate_truthfulqa(evaluator, truthfulqa_rows, args.progress_every))
    elapsed = time.perf_counter() - start_time
    print({"evaluation_seconds": elapsed, "rows": len(all_results)})

    if pd is not None:
        results_df = pd.DataFrame(all_results)
        summary_group_columns = ["benchmark", "metric_name", "decoder", "decoder_label"]
        summary_df = (
            results_df.groupby(summary_group_columns, as_index=False)
            .agg(
                score_mean=("score", "mean"),
                score_std=("score", "std"),
                num_examples=("score", "size"),
                decision_margin_mean=("decision_margin", "mean"),
                correct_margin_mean=("correct_margin", "mean"),
                format_valid_rate=("format_valid", "mean"),
                latency_seconds_mean=("latency_seconds", "mean"),
                decoder_steps_mean=("decoder_steps", "mean"),
                forward_passes_mean=("forward_passes", "mean"),
                latency_per_step_ms_mean=("latency_per_step_ms", "mean"),
                latency_per_forward_ms_mean=("latency_per_forward_ms", "mean"),
                steps_per_second_mean=("steps_per_second", "mean"),
                tokens_per_forward_mean=("tokens_per_forward", "mean"),
                factual_speedup_mean=("factual_speedup", "mean"),
                avg_alpha=("avg_alpha", "mean"),
                alpha_std=("alpha_std", "mean"),
                avg_selected_layer=("avg_selected_layer", "mean"),
                switch_rate=("switch_rate", "mean"),
                avg_instability=("avg_instability", "mean"),
                avg_risk_score=("avg_risk_score", "mean"),
                trigger_rate=("trigger_rate", "mean"),
                avg_jsd_current=("avg_jsd_current", "mean"),
                avg_selection_margin=("avg_selection_margin", "mean"),
                avg_selection_score=("avg_selection_score", "mean"),
                fallback_rate=("fallback_rate", "mean"),
                avg_baseline_margin=("avg_baseline_margin", "mean"),
                avg_jacobi_passes=("avg_jacobi_passes", "mean"),
                avg_jacobi_window_size=("avg_jacobi_window_size", "mean"),
                avg_jacobi_stable_prefix=("avg_jacobi_stable_prefix", "mean"),
                avg_jacobi_commit_len=("avg_jacobi_commit_len", "mean"),
                jacobi_convergence_rate=("jacobi_convergence_rate", "mean"),
                panda_disagreement_rate=("panda_disagreement_rate", "mean"),
                panda_truth_selection_rate=("panda_truth_selection_rate", "mean"),
                avg_panda_divergence=("avg_panda_divergence", "mean"),
                avg_panda_safe_confidence=("avg_panda_safe_confidence", "mean"),
                avg_panda_truth_confidence=("avg_panda_truth_confidence", "mean"),
                avg_panda_agreement_prefix=("avg_panda_agreement_prefix", "mean"),
                panda_arbitration_rate=("panda_arbitration_rate", "mean"),
            )
            .sort_values(summary_group_columns)
        )
        summary_df["score_std"] = summary_df["score_std"].fillna(0.0)
        summary_df["score_sem"] = summary_df["score_std"] / summary_df["num_examples"].pow(0.5)
        pairwise_df = build_pairwise_summary(results_df)
        print("\nSummary")
        summary_print_columns = [
            "benchmark",
            "metric_name",
            "decoder_label",
            "score_mean",
            "score_sem",
            "num_examples",
            "decision_margin_mean",
            "latency_seconds_mean",
            "latency_per_step_ms_mean",
            "latency_per_forward_ms_mean",
            "steps_per_second_mean",
            "forward_passes_mean",
            "factual_speedup_mean",
            "avg_alpha",
            "alpha_std",
            "avg_selected_layer",
            "switch_rate",
            "avg_instability",
            "avg_risk_score",
            "trigger_rate",
            "avg_jsd_current",
            "avg_selection_margin",
            "avg_selection_score",
            "fallback_rate",
            "avg_baseline_margin",
            "avg_jacobi_passes",
            "avg_jacobi_window_size",
            "avg_jacobi_stable_prefix",
            "avg_jacobi_commit_len",
            "jacobi_convergence_rate",
            "panda_disagreement_rate",
            "panda_truth_selection_rate",
            "avg_panda_divergence",
            "avg_panda_safe_confidence",
            "avg_panda_truth_confidence",
            "avg_panda_agreement_prefix",
            "panda_arbitration_rate",
        ]
        print(summary_df[summary_print_columns].to_string(index=False))
        if not pairwise_df.empty:
            print("\nPairwise Score Deltas")
            pairwise_print_columns = [
                "benchmark",
                "metric_name",
                "left_decoder_label",
                "right_decoder_label",
                "mean_score_delta",
                "win_rate",
                "tie_rate",
                "loss_rate",
                "num_examples",
            ]
            print(pairwise_df[pairwise_print_columns].to_string(index=False))
    else:
        results_df = all_results
        summary_df = []
        pairwise_df = []
        print("pandas is missing; skipping DataFrame summary.")

    if args.save_results:
        metadata = {
            "model_name": args.model_name,
            "mode": args.mode,
            "seed": args.seed,
            "local_files_only": args.local_files_only,
            "strict_eval": args.strict_eval,
            "artifact_prefix": artifact_prefix,
            "dola_algorithm": "official_dynamic_dola",
            "dola_mature_layer": evaluator.mature_layer_index,
            "dola_relative_top": evaluator.dola_relative_top,
            "dola_relative_top_value": evaluator.dola_relative_top_value,
            "latency_measurement": "wall_clock_seconds_with_cuda_synchronize",
            "decoders": list(evaluator.decoder_names),
            "decoder_labels": evaluator.decoder_labels,
            "default_shallow_bucket": evaluator.default_bucket,
            "fixed_alpha_decoders": evaluator.fixed_alpha_decoders,
            "panda_low_alpha": evaluator.panda_low_alpha,
            "panda_high_alpha": evaluator.panda_high_alpha,
            "panda_binary_views": {
                "greedy_view": "final_logits",
                "contrast_subtracted_view": "final_logits - shallow_logits",
            },
            "jacobi_window_size": evaluator.jacobi_window_size,
            "jacobi_max_iters": evaluator.jacobi_max_iters,
            "jacobi_init_strategy": "repeat_last",
            "jacobi_commit_strategy": "stable_prefix_then_fallback_1",
            "panda_divergence_threshold": evaluator.panda_divergence_threshold,
            "panda_truth_bias": evaluator.panda_truth_bias,
            "panda_note": (
                "jacobi_block_decoding_with_shared_greedy_and_contrast_subtracted_views_and_local_truth_biased_arbitration"
            ),
            "panda_early_agreement_shortcut": evaluator.panda_early_agreement_shortcut,
            "evaluation_benchmark": "truthfulqa",
            "truthfulqa_metrics": ["mc1", "mc2", "mc3"],
            "truthfulqa_source": truthfulqa_source,
            "truthfulqa_sampling": truthfulqa_manifest,
            "truthfulqa_limit": truthfulqa_limit,
        }
        save_outputs(results_df, summary_df, pairwise_df, metadata, Path(args.results_dir), artifact_prefix)


__all__ = ["infer_artifact_prefix", "main", "save_outputs"]


if __name__ == "__main__":
    main()
