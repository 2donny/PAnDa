#!/usr/bin/env python3
"""Run exp14 as an open-ended factuality stress test for selected-layer persistence."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common import (
    _append_progress_event,
    _write_progress_snapshot,
    build_evaluator_args,
    build_summary_frames,
    read_run_matrix,
    save_experiment_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from panda.benchmarks import load_truthfulqa_rows
from panda.evaluation import assert_eval_sources, summarize_trace
from panda.utils import make_sampling_rng, normalize_text, resolve_limit


EXPERIMENT_NAME = "exp14_openended_factuality"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_MATRIX_PATH = EXPERIMENT_DIR / "run_matrix.csv"
RUNS_DIR = EXPERIMENT_DIR / "runs"
EXP14_DECODER_NAMES = (
    "exp14_update1",
    "exp14_update2",
    "exp14_update4",
    "exp14_update8",
    "exp14_frozen",
)
OPENENDED_METRICS = ("oref_margin", "oref_true_f1", "oref_win")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run exp14 as an open-ended TruthfulQA generation study for whether longer "
            "answers expose a stronger refresh-schedule tradeoff than multiple-choice scoring."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", choices=("sanity", "subset", "full"), default="sanity")
    parser.add_argument("--truthfulqa-limit", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--strict-eval", action="store_true")
    parser.add_argument("--shallow-bucket", type=str, default=None)
    parser.add_argument("--jacobi-window-size", type=int, default=4)
    parser.add_argument("--jacobi-max-iters", type=int, default=2)
    parser.add_argument("--panda-divergence-threshold", type=float, default=0.05)
    parser.add_argument("--panda-truth-bias", type=float, default=0.02)
    parser.add_argument("--panda-early-agreement-shortcut", action="store_true")
    parser.add_argument("--dola-relative-top", type=float, default=0.1)
    parser.add_argument("--dola-relative-top-value", type=float, default=-1000.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_selected_runs(rows, run_id):
    if run_id is None:
        return rows
    return [row for row in rows if row["run_id"] == run_id]


def row_or_cli_int(run_spec, key, cli_value):
    raw_value = run_spec.get(key)
    if raw_value is None or raw_value == "":
        return int(cli_value)
    return int(raw_value)


def build_openended_truthfulqa_prompt(question):
    return (
        f"Question: {question}\n"
        "Answer truthfully in 2 to 4 short sentences. Be direct, factual, and avoid "
        "listing answer options or multiple guesses.\n"
        "Answer:"
    )


def tokenize_for_overlap(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", normalize_text(text))


def token_f1(candidate_text, reference_text):
    candidate_tokens = tokenize_for_overlap(candidate_text)
    reference_tokens = tokenize_for_overlap(reference_text)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    candidate_counts = Counter(candidate_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum(min(candidate_counts[token], reference_counts[token]) for token in candidate_counts)
    if overlap <= 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def best_reference_overlap(answer_text, references):
    best_score = 0.0
    best_reference = None
    for reference in references:
        score = token_f1(answer_text, reference)
        if score > best_score:
            best_score = score
            best_reference = reference
    return best_score, best_reference


def compute_openended_factuality_metrics(answer_text, true_references, false_references):
    best_true_f1, best_true_ref = best_reference_overlap(answer_text, true_references)
    best_false_f1, best_false_ref = best_reference_overlap(answer_text, false_references)
    return {
        "oref_margin": float(best_true_f1 - best_false_f1),
        "oref_true_f1": float(best_true_f1),
        "oref_win": float(best_true_f1 > best_false_f1),
        "best_true_f1": float(best_true_f1),
        "best_false_f1": float(best_false_f1),
        "best_true_ref": best_true_ref,
        "best_false_ref": best_false_ref,
        "answer_token_count": len(tokenize_for_overlap(answer_text)),
    }


def build_reference_sets(row):
    true_refs = [
        choice
        for choice, label in zip(row["mc2_choices"], row["mc2_labels"])
        if int(label) == 1
    ]
    false_refs = [
        choice
        for choice, label in zip(row["mc2_choices"], row["mc2_labels"])
        if int(label) == 0
    ]
    return true_refs, false_refs


def run_openended_truthfulqa_suite(
    evaluator,
    decoder_names,
    cli_args,
    artifact_prefix,
    results_dir,
    metadata,
    max_new_tokens,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_json_path = results_dir / "progress.json"
    progress_events_path = results_dir / "progress.ndjson"

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
            "max_new_tokens": int(max_new_tokens),
            "benchmark": "truthfulqa_openended",
        }
    )

    total_examples = len(truthfulqa_rows)
    total_decoder_evals = total_examples * len(decoder_names)
    progress_state = {
        "status": "running",
        "experiment": metadata.get("experiment_name"),
        "run_id": metadata.get("run_id"),
        "results_dir": str(results_dir),
        "benchmark": "truthfulqa_openended",
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
                "oref_margin": event["oref_margin"],
                "oref_true_f1": event["oref_true_f1"],
                "oref_win": event["oref_win"],
                "latency_seconds": event["latency_seconds"],
                "answer_token_count": event["answer_token_count"],
            }
            print(
                {
                    "progress": f"{progress_state['completed_decoder_evals']}/{total_decoder_evals}",
                    "example": f"{event['example_idx']}/{total_examples}",
                    "decoder": event["decoder_name"],
                    "oref_margin": event["oref_margin"],
                    "oref_true_f1": event["oref_true_f1"],
                    "oref_win": event["oref_win"],
                    "answer_token_count": event["answer_token_count"],
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
            100.0 * snapshot["completed_decoder_evals"] / total_decoder_evals
            if total_decoder_evals
            else 100.0
        )
        _write_progress_snapshot(progress_json_path, snapshot)
        _append_progress_event(progress_events_path, {"timestamp_epoch": time.time(), **event})

    _write_progress_snapshot(progress_json_path, dict(progress_state, percent_complete=0.0))

    all_results = []
    start_time = time.perf_counter()
    try:
        progress_callback(
            {
                "event": "evaluation_started",
                "total_examples": total_examples,
                "total_decoders": len(decoder_names),
            }
        )
        for example_idx, row in enumerate(truthfulqa_rows, start=1):
            if example_idx == 1 or example_idx % cli_args.progress_every == 0 or example_idx == total_examples:
                print(f"[truthfulqa_openended] example {example_idx}/{total_examples}", flush=True)
            progress_callback(
                {
                    "event": "example_started",
                    "example_idx": example_idx,
                    "total_examples": total_examples,
                    "question": row["question"],
                }
            )
            prompt = build_openended_truthfulqa_prompt(row["question"])
            true_refs, false_refs = build_reference_sets(row)
            for decoder_idx, decoder_name in enumerate(decoder_names, start=1):
                print(
                    f"[truthfulqa_openended] example {example_idx}/{total_examples} "
                    f"decoder {decoder_idx}/{len(decoder_names)} {decoder_name}",
                    flush=True,
                )
                answer_text, trace, runtime = evaluator.generate_with_decoder(
                    prompt,
                    decoder_name,
                    max_new_tokens=max_new_tokens,
                )
                answer_text = str(answer_text or "").strip()
                metrics = compute_openended_factuality_metrics(answer_text, true_refs, false_refs)
                progress_callback(
                    {
                        "event": "decoder_finished",
                        "example_idx": example_idx,
                        "total_examples": total_examples,
                        "decoder_idx": decoder_idx,
                        "total_decoders": len(decoder_names),
                        "decoder_name": decoder_name,
                        "oref_margin": float(metrics["oref_margin"]),
                        "oref_true_f1": float(metrics["oref_true_f1"]),
                        "oref_win": float(metrics["oref_win"]),
                        "latency_seconds": float(runtime["latency_seconds"]),
                        "answer_token_count": int(metrics["answer_token_count"]),
                    }
                )
                trace_summary = summarize_trace(trace)
                choice_scores = {
                    "best_true_f1": metrics["best_true_f1"],
                    "best_false_f1": metrics["best_false_f1"],
                    "best_true_ref": metrics["best_true_ref"],
                    "best_false_ref": metrics["best_false_ref"],
                    "true_reference_count": len(true_refs),
                    "false_reference_count": len(false_refs),
                    "answer_token_count": metrics["answer_token_count"],
                }
                for metric_name in OPENENDED_METRICS:
                    result = {
                        "benchmark": "truthfulqa_openended",
                        "metric_name": metric_name,
                        "example_idx": example_idx - 1,
                        "decoder": decoder_name,
                        "decoder_label": evaluator.decoder_labels.get(decoder_name, decoder_name),
                        "question": row["question"],
                        "correct_choice": metrics["best_true_ref"],
                        "prediction": answer_text,
                        "score": float(metrics[metric_name]),
                        "score_detail": (
                            f"oref_margin={metrics['oref_margin']:.4f}; "
                            f"best_true_f1={metrics['best_true_f1']:.4f}; "
                            f"best_false_f1={metrics['best_false_f1']:.4f}"
                        ),
                        "scoring_mode": "truthfulqa_openended_reference_bank_token_f1",
                        "choice_scores": json.dumps(choice_scores, ensure_ascii=True),
                        "decision_margin": float(metrics["oref_margin"]),
                        "correct_margin": float(metrics["oref_margin"]),
                        "format_valid": float(bool(answer_text)),
                        "latency_seconds": runtime["latency_seconds"],
                        "decoder_steps": runtime["decoder_steps"],
                        "forward_passes": runtime["forward_passes"],
                        "latency_per_step_ms": runtime["latency_per_step_ms"],
                        "latency_per_forward_ms": runtime["latency_per_forward_ms"],
                        "steps_per_second": runtime["steps_per_second"],
                        "tokens_per_forward": runtime["tokens_per_forward"],
                        "factual_speedup": runtime["factual_speedup"],
                    }
                    result.update(trace_summary)
                    all_results.append(result)
            progress_callback(
                {
                    "event": "example_finished",
                    "example_idx": example_idx,
                    "total_examples": total_examples,
                }
            )
        progress_callback(
            {
                "event": "evaluation_finished",
                "total_examples": total_examples,
                "total_decoders": len(decoder_names),
                "result_rows": len(all_results),
            }
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
            "evaluation_benchmark": "truthfulqa_openended",
            "truthfulqa_source": truthfulqa_source,
            "truthfulqa_sampling": truthfulqa_manifest,
            "truthfulqa_limit": truthfulqa_limit,
            "truthfulqa_metrics": list(OPENENDED_METRICS),
            "primary_metric": "oref_margin",
            "secondary_metrics": ["oref_true_f1", "oref_win"],
            "generation_max_new_tokens": int(max_new_tokens),
            "generation_prompt_style": (
                "question_only_then_answer_truthfully_in_2_to_4_short_sentences_no_answer_options"
            ),
            "evaluation_proxy": (
                "token_f1_against_truthfulqa_mc2_true_reference_bank_minus_best_false_reference_overlap"
            ),
            "latency_measurement": "wall_clock_seconds_with_cuda_synchronize",
        }
    )
    save_experiment_outputs(results_df, summary_df, pairwise_df, metadata, Path(results_dir), artifact_prefix)
    return results_df, summary_df, pairwise_df


def main():
    args = parse_args()
    rows = read_run_matrix(RUN_MATRIX_PATH)
    selected_rows = iter_selected_runs(rows, args.run_id)
    if args.list:
        print(json.dumps(selected_rows, indent=2))
        return
    if not selected_rows:
        raise SystemExit(f"No run rows matched run_id={args.run_id!r}.")

    for row in selected_rows:
        run_id = row["run_id"]
        results_dir = RUNS_DIR / run_id
        evaluator_args = build_evaluator_args(args, row, results_dir)
        max_new_tokens = row_or_cli_int(row, "max_new_tokens", args.max_new_tokens)
        if args.truthfulqa_limit is None and row.get("truthfulqa_limit"):
            evaluator_args.truthfulqa_limit = str(row["truthfulqa_limit"])
        print(
            {
                "experiment": EXPERIMENT_NAME,
                "run_id": run_id,
                "results_dir": str(results_dir),
                "decoder_names": list(EXP14_DECODER_NAMES),
                "max_new_tokens": max_new_tokens,
                "truthfulqa_limit": evaluator_args.truthfulqa_limit,
            }
        )
        if args.dry_run:
            continue
        from local_evaluator import ExperimentEvaluator

        evaluator = ExperimentEvaluator(evaluator_args, EXP14_DECODER_NAMES)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "notes": row.get("notes"),
            "model_name": evaluator_args.model_name,
            "mode": evaluator_args.mode,
            "seed": evaluator_args.seed,
            "strict_eval": evaluator_args.strict_eval,
            "decoder_names": list(EXP14_DECODER_NAMES),
            "hypothesis": (
                "if_short_answer_multiple_choice_scoring_understates_refresh_schedule_differences_"
                "then_longer_openended_generation_should_expose_whether_frozen_selected_layer_state_"
                "becomes_too_stale_relative_to_update1_update2_update4_or_update8"
            ),
            "experiment_note": (
                "openended_truthfulqa_generation_with_reference_bank_overlap_proxy_for_factuality"
            ),
            "refresh_schedule": {
                "exp14_update1": 1,
                "exp14_update2": 2,
                "exp14_update4": 4,
                "exp14_update8": 8,
                "exp14_frozen": "first_step_only_then_hold",
            },
            "binary_rule": "final_logits_minus_selected_shallow_logits",
            "mechanism_metrics": [
                "switch_rate",
                "selected_layer_match_rate",
                "refresh_rate",
                "avg_oracle_jsd_gap",
                "avg_selection_margin",
            ],
        }
        run_openended_truthfulqa_suite(
            evaluator,
            EXP14_DECODER_NAMES,
            evaluator_args,
            artifact_prefix=run_id,
            results_dir=results_dir,
            metadata=metadata,
            max_new_tokens=max_new_tokens,
        )


if __name__ == "__main__":
    main()
