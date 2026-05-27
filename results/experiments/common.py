"""Shared helpers for isolated experiment runners."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional in the repo too
    pd = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from panda.evaluation import assert_eval_sources, build_pairwise_summary, evaluate_truthfulqa
from panda.utils import make_sampling_rng, resolve_limit


SUMMARY_GROUP_COLUMNS = ["benchmark", "metric_name", "decoder", "decoder_label"]


def read_run_matrix(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value, default=False):
    if value is None or value == "":
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value {value!r}.")


def build_evaluator_args(cli_args, run_spec, results_dir):
    return argparse.Namespace(
        model_name=cli_args.model_name,
        mode=cli_args.mode,
        local_files_only=bool(cli_args.local_files_only),
        no_chat_template=bool(cli_args.no_chat_template),
        results_dir=str(results_dir),
        save_results=True,
        progress_every=int(cli_args.progress_every),
        strict_eval=bool(cli_args.strict_eval),
        truthfulqa_limit=cli_args.truthfulqa_limit,
        comparison_preset=None,
        shallow_bucket=(run_spec.get("shallow_bucket") or cli_args.shallow_bucket or None),
        jacobi_window_size=int(run_spec.get("jacobi_window_size") or cli_args.jacobi_window_size),
        jacobi_max_iters=int(run_spec.get("jacobi_max_iters") or cli_args.jacobi_max_iters),
        panda_divergence_threshold=float(
            run_spec.get("panda_divergence_threshold") or cli_args.panda_divergence_threshold
        ),
        panda_truth_bias=float(run_spec.get("panda_truth_bias") or cli_args.panda_truth_bias),
        panda_early_agreement_shortcut=parse_bool(
            run_spec.get("panda_early_agreement_shortcut"),
            default=cli_args.panda_early_agreement_shortcut,
        ),
        dola_relative_top=float(run_spec.get("dola_relative_top") or cli_args.dola_relative_top),
        dola_relative_top_value=float(
            run_spec.get("dola_relative_top_value") or cli_args.dola_relative_top_value
        ),
        seed=int(run_spec.get("seed") or cli_args.seed),
    )


def build_summary_frames(all_results):
    if pd is None:
        return all_results, [], []

    results_df = pd.DataFrame(all_results)
    summary_df = (
        results_df.groupby(SUMMARY_GROUP_COLUMNS, as_index=False)
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
        .sort_values(SUMMARY_GROUP_COLUMNS)
    )
    summary_df["score_std"] = summary_df["score_std"].fillna(0.0)
    summary_df["score_sem"] = summary_df["score_std"] / summary_df["num_examples"].pow(0.5)
    pairwise_df = build_pairwise_summary(results_df)
    return results_df, summary_df, pairwise_df


def save_experiment_outputs(results_df, summary_df, pairwise_df, metadata, results_dir, artifact_prefix):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / f"{artifact_prefix}_raw_predictions.csv"
    summary_path = results_dir / f"{artifact_prefix}_summary.csv"
    pairwise_path = results_dir / f"{artifact_prefix}_pairwise_summary.csv"
    metadata_path = results_dir / f"{artifact_prefix}_metadata.json"
    if pd is not None:
        results_df.to_csv(raw_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        pairwise_df.to_csv(pairwise_path, index=False)
    else:  # pragma: no cover - fallback mirrors the main CLI
        (results_dir / f"{artifact_prefix}_raw_predictions.json").write_text(
            json.dumps(results_df, indent=2),
            encoding="utf-8",
        )
        (results_dir / f"{artifact_prefix}_summary.json").write_text(
            json.dumps(summary_df, indent=2),
            encoding="utf-8",
        )
        (results_dir / f"{artifact_prefix}_pairwise_summary.json").write_text(
            json.dumps(pairwise_df, indent=2),
            encoding="utf-8",
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


def run_truthfulqa_suite(evaluator, decoder_names, cli_args, artifact_prefix, results_dir, metadata):
    from panda.benchmarks import load_truthfulqa_rows

    truthfulqa_limit = resolve_limit(cli_args.truthfulqa_limit, cli_args.mode, 5)
    truthfulqa_rows, truthfulqa_source, truthfulqa_manifest = load_truthfulqa_rows(
        truthfulqa_limit,
        make_sampling_rng(cli_args.seed, "truthfulqa"),
    )
    assert_eval_sources(cli_args, truthfulqa_source)

    print(
        {
            "truthfulqa_source": truthfulqa_source,
            "truthfulqa_sampling": truthfulqa_manifest,
            "truthfulqa_examples": len(truthfulqa_rows),
            "decoder_names": list(decoder_names),
        }
    )

    start_time = time.perf_counter()
    all_results = evaluate_truthfulqa(
        evaluator,
        truthfulqa_rows,
        progress_every=cli_args.progress_every,
        decoder_names=decoder_names,
    )
    elapsed = time.perf_counter() - start_time
    print({"evaluation_seconds": elapsed, "rows": len(all_results)})

    results_df, summary_df, pairwise_df = build_summary_frames(all_results)
    metadata = dict(metadata)
    metadata.update(
        {
            "evaluation_benchmark": "truthfulqa",
            "truthfulqa_source": truthfulqa_source,
            "truthfulqa_sampling": truthfulqa_manifest,
            "truthfulqa_limit": truthfulqa_limit,
            "truthfulqa_metrics": ["mc1", "mc2", "mc3"],
            "latency_measurement": "wall_clock_seconds_with_cuda_synchronize",
        }
    )
    save_experiment_outputs(results_df, summary_df, pairwise_df, metadata, Path(results_dir), artifact_prefix)
    return results_df, summary_df, pairwise_df
