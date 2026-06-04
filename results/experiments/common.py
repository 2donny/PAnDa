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


def _write_progress_snapshot(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _append_progress_event(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


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
        cd_amateur_model_name=(
            run_spec.get("cd_amateur_model_name")
            or getattr(cli_args, "cd_amateur_model_name", None)
            or "Qwen/Qwen2.5-0.5B-Instruct"
        ),
        cd_plausibility_alpha=float(
            run_spec.get("cd_plausibility_alpha")
            or getattr(cli_args, "cd_plausibility_alpha", 0.1)
        ),
        cd_amateur_temperature=float(
            run_spec.get("cd_amateur_temperature")
            or getattr(cli_args, "cd_amateur_temperature", 0.5)
        ),
        top_k_value=int(run_spec.get("top_k_value") or getattr(cli_args, "top_k_value", 50)),
        top_p_value=float(run_spec.get("top_p_value") or getattr(cli_args, "top_p_value", 0.9)),
        exp6_guarded_top_k=int(
            run_spec.get("exp6_guarded_top_k") or getattr(cli_args, "exp6_guarded_top_k", 2)
        ),
        exp6_sticky_hold_steps=int(
            run_spec.get("exp6_sticky_hold_steps") or getattr(cli_args, "exp6_sticky_hold_steps", 1)
        ),
        exp6_margin_threshold=float(
            run_spec.get("exp6_margin_threshold") or getattr(cli_args, "exp6_margin_threshold", 0.5)
        ),
        exp9_lambda_min=float(
            run_spec.get("exp9_lambda_min") or getattr(cli_args, "exp9_lambda_min", 0.5)
        ),
        exp9_lambda_max=float(
            run_spec.get("exp9_lambda_max") or getattr(cli_args, "exp9_lambda_max", 1.0)
        ),
        exp9_uncertainty_weight=float(
            run_spec.get("exp9_uncertainty_weight")
            or getattr(cli_args, "exp9_uncertainty_weight", 1.0)
        ),
        exp9_confidence_gap_weight=float(
            run_spec.get("exp9_confidence_gap_weight")
            or getattr(cli_args, "exp9_confidence_gap_weight", 2.0)
        ),
        exp10_risk_beta=float(
            run_spec.get("exp10_risk_beta") or getattr(cli_args, "exp10_risk_beta", 0.8)
        ),
        exp10_entropy_weight=float(
            run_spec.get("exp10_entropy_weight") or getattr(cli_args, "exp10_entropy_weight", 1.0)
        ),
        exp10_margin_weight=float(
            run_spec.get("exp10_margin_weight") or getattr(cli_args, "exp10_margin_weight", 1.0)
        ),
        exp10_layer_jsd_weight=float(
            run_spec.get("exp10_layer_jsd_weight") or getattr(cli_args, "exp10_layer_jsd_weight", 1.0)
        ),
        exp10_risk_threshold=float(
            run_spec.get("exp10_risk_threshold") or getattr(cli_args, "exp10_risk_threshold", 0.55)
        ),
        exp10_sticky_hold_steps=int(
            run_spec.get("exp10_sticky_hold_steps")
            or getattr(cli_args, "exp10_sticky_hold_steps", 0)
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
            selected_layer_match_rate=("selected_layer_match_rate", "mean"),
            refresh_rate=("refresh_rate", "mean"),
            avg_instability=("avg_instability", "mean"),
            avg_risk_score=("avg_risk_score", "mean"),
            trigger_rate=("trigger_rate", "mean"),
            avg_jsd_current=("avg_jsd_current", "mean"),
            avg_oracle_jsd_gap=("avg_oracle_jsd_gap", "mean"),
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


def run_truthfulqa_suite_on_rows(
    evaluator,
    decoder_names,
    cli_args,
    artifact_prefix,
    results_dir,
    metadata,
    truthfulqa_rows,
    truthfulqa_source,
    truthfulqa_manifest,
    truthfulqa_limit,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_json_path = results_dir / "progress.json"
    progress_events_path = results_dir / "progress.ndjson"
    assert_eval_sources(cli_args, truthfulqa_source)

    print(
        {
            "truthfulqa_source": truthfulqa_source,
            "truthfulqa_sampling": truthfulqa_manifest,
            "truthfulqa_examples": len(truthfulqa_rows),
            "decoder_names": list(decoder_names),
        }
    )

    total_examples = len(truthfulqa_rows)
    total_decoder_evals = total_examples * len(decoder_names)
    progress_state = {
        "status": "running",
        "experiment": metadata.get("experiment_name"),
        "run_id": metadata.get("run_id"),
        "results_dir": str(results_dir),
        "benchmark": "truthfulqa",
        "total_examples": total_examples,
        "total_decoders": len(decoder_names),
        "total_decoder_evals": total_decoder_evals,
        "completed_examples": 0,
        "completed_decoder_evals": 0,
        "current_example_idx": None,
        "current_decoder_name": None,
        "current_decoder_idx": None,
        "latest_metrics": None,
        "started_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
    }

    def progress_callback(event):
        progress_state["updated_at_epoch"] = time.time()
        event_type = event["event"]
        progress_state["last_event"] = event_type
        if event_type == "example_started":
            progress_state["current_example_idx"] = int(event["example_idx"])
            progress_state["current_question"] = event.get("question")
        elif event_type == "decoder_finished":
            progress_state["current_example_idx"] = int(event["example_idx"])
            progress_state["current_decoder_idx"] = int(event["decoder_idx"])
            progress_state["current_decoder_name"] = event["decoder_name"]
            progress_state["completed_decoder_evals"] = int(
                (int(event["example_idx"]) - 1) * len(decoder_names) + int(event["decoder_idx"])
            )
            progress_state["latest_metrics"] = {
                "decoder_name": event["decoder_name"],
                "mc1": event["mc1"],
                "mc2": event["mc2"],
                "mc3": event["mc3"],
                "latency_seconds": event["latency_seconds"],
            }
            print(
                {
                    "progress": f"{progress_state['completed_decoder_evals']}/{total_decoder_evals}",
                    "example": f"{event['example_idx']}/{total_examples}",
                    "decoder": event["decoder_name"],
                    "mc1": event["mc1"],
                    "mc2": event["mc2"],
                    "mc3": event["mc3"],
                    "latency_seconds": event["latency_seconds"],
                },
                flush=True,
            )
        elif event_type == "example_finished":
            progress_state["completed_examples"] = int(event["example_idx"])
        elif event_type == "evaluation_finished":
            progress_state["status"] = "completed"
            progress_state["completed_examples"] = total_examples
            progress_state["completed_decoder_evals"] = total_decoder_evals
            progress_state["result_rows"] = int(event["result_rows"])
        snapshot = dict(progress_state)
        snapshot["percent_complete"] = (
            100.0 * snapshot["completed_decoder_evals"] / total_decoder_evals if total_decoder_evals else 100.0
        )
        _write_progress_snapshot(progress_json_path, snapshot)
        event_record = {"timestamp_epoch": time.time(), **event}
        _append_progress_event(progress_events_path, event_record)

    _write_progress_snapshot(progress_json_path, dict(progress_state, percent_complete=0.0))

    start_time = time.perf_counter()
    try:
        all_results = evaluate_truthfulqa(
            evaluator,
            truthfulqa_rows,
            progress_every=cli_args.progress_every,
            decoder_names=decoder_names,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        progress_state["status"] = "failed"
        progress_state["updated_at_epoch"] = time.time()
        progress_state["error"] = repr(exc)
        _write_progress_snapshot(
            progress_json_path,
            dict(
                progress_state,
                percent_complete=(
                    100.0 * progress_state["completed_decoder_evals"] / total_decoder_evals
                    if total_decoder_evals
                    else 0.0
                ),
            ),
        )
        _append_progress_event(
            progress_events_path,
            {"timestamp_epoch": time.time(), "event": "evaluation_failed", "error": repr(exc)},
        )
        raise
    elapsed = time.perf_counter() - start_time
    print({"evaluation_seconds": elapsed, "rows": len(all_results)}, flush=True)

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


def run_truthfulqa_suite(evaluator, decoder_names, cli_args, artifact_prefix, results_dir, metadata):
    from panda.benchmarks import load_truthfulqa_rows

    truthfulqa_limit = resolve_limit(cli_args.truthfulqa_limit, cli_args.mode, 5)
    truthfulqa_rows, truthfulqa_source, truthfulqa_manifest = load_truthfulqa_rows(
        truthfulqa_limit,
        make_sampling_rng(cli_args.seed, "truthfulqa"),
    )
    return run_truthfulqa_suite_on_rows(
        evaluator=evaluator,
        decoder_names=decoder_names,
        cli_args=cli_args,
        artifact_prefix=artifact_prefix,
        results_dir=results_dir,
        metadata=metadata,
        truthfulqa_rows=truthfulqa_rows,
        truthfulqa_source=truthfulqa_source,
        truthfulqa_manifest=truthfulqa_manifest,
        truthfulqa_limit=truthfulqa_limit,
    )
