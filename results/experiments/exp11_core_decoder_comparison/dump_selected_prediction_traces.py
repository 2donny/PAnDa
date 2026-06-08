#!/usr/bin/env python3
"""Dump real token-level DoLa traces for selected exp11 MC predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = EXPERIMENT_DIR.parents[0]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common import build_evaluator_args, read_run_matrix
from local_evaluator import ExperimentEvaluator
from panda.evaluation import summarize_trace
from panda.prompts import build_truthfulqa_prompt


RUN_MATRIX_PATH = EXPERIMENT_DIR / "run_matrix.csv"
RUNS_DIR = EXPERIMENT_DIR / "runs"
DEFAULT_SOURCE_RUN_ID = "run_01_default"
DEFAULT_TRACE_RUN_ID = "trace_dola_mc1_extremes"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rerun selected exp11 MC predictions through the real ExperimentEvaluator and "
            "dump per-token selected-layer traces."
        )
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--top-k-value", type=int, default=50)
    parser.add_argument("--top-p-value", type=float, default=0.9)
    parser.add_argument("--mode", choices=("sanity", "subset", "full"), default="subset")
    parser.add_argument("--truthfulqa-limit", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only locally cached model files by default.",
    )
    parser.add_argument("--no-chat-template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shallow-bucket", type=str, default=None)
    parser.add_argument("--jacobi-window-size", type=int, default=4)
    parser.add_argument("--jacobi-max-iters", type=int, default=2)
    parser.add_argument("--panda-divergence-threshold", type=float, default=0.05)
    parser.add_argument("--panda-truth-bias", type=float, default=0.02)
    parser.add_argument("--panda-early-agreement-shortcut", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dola-relative-top", type=float, default=0.1)
    parser.add_argument("--dola-relative-top-value", type=float, default=-1000.0)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--trace-run-id", default=DEFAULT_TRACE_RUN_ID)
    parser.add_argument("--decoder-name", default="dola")
    parser.add_argument("--metric-name", default="mc1")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--wrong-k", type=int, default=3)
    parser.add_argument(
        "--example-indices",
        type=str,
        default=None,
        help="Comma-separated explicit example_idx list. Overrides the top/worst auto-selection.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    values = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values


def load_source_rows(raw_csv: Path, *, decoder_name: str, metric_name: str) -> list[dict[str, str]]:
    rows = []
    with raw_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["decoder"] != decoder_name or row["metric_name"] != metric_name:
                continue
            rows.append(row)
    return rows


def select_cases(
    rows: list[dict[str, str]],
    *,
    example_indices: list[int],
    top_k: int,
    wrong_k: int,
) -> list[dict[str, object]]:
    if example_indices:
        by_idx = {int(row["example_idx"]): row for row in rows}
        selected = []
        for rank, example_idx in enumerate(example_indices, start=1):
            if example_idx not in by_idx:
                raise SystemExit(f"example_idx={example_idx} was not found in the filtered source rows.")
            row = by_idx[example_idx]
            selected.append(
                {
                    "group": f"{row['metric_name']}={int(float(row['score']))}",
                    "group_rank": rank,
                    "example_idx": int(row["example_idx"]),
                    "question": row["question"],
                    "prediction": row["prediction"],
                    "score": float(row["score"]),
                    "decision_margin": float(row["decision_margin"]),
                    "score_detail": row["score_detail"],
                }
            )
        return selected

    correct_rows = [row for row in rows if float(row["score"]) == 1.0]
    wrong_rows = [row for row in rows if float(row["score"]) == 0.0]
    correct_rows.sort(key=lambda row: float(row["decision_margin"]), reverse=True)
    wrong_rows.sort(key=lambda row: float(row["decision_margin"]))

    selected = []
    for rank, row in enumerate(correct_rows[:top_k], start=1):
        selected.append(
            {
                "group": "mc1=1",
                "group_rank": rank,
                "example_idx": int(row["example_idx"]),
                "question": row["question"],
                "prediction": row["prediction"],
                "score": float(row["score"]),
                "decision_margin": float(row["decision_margin"]),
                "score_detail": row["score_detail"],
            }
        )
    for rank, row in enumerate(wrong_rows[:wrong_k], start=1):
        selected.append(
            {
                "group": "mc1=0",
                "group_rank": rank,
                "example_idx": int(row["example_idx"]),
                "question": row["question"],
                "prediction": row["prediction"],
                "score": float(row["score"]),
                "decision_margin": float(row["decision_margin"]),
                "score_detail": row["score_detail"],
            }
        )
    return selected


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    source_results_dir = RUNS_DIR / args.source_run_id
    raw_csv = source_results_dir / f"{args.source_run_id}_raw_predictions.csv"
    source_metadata_path = source_results_dir / f"{args.source_run_id}_metadata.json"
    run_rows = read_run_matrix(RUN_MATRIX_PATH)
    matching_runs = [row for row in run_rows if row["run_id"] == args.source_run_id]
    if not matching_runs:
        raise SystemExit(f"source run_id={args.source_run_id!r} is not present in {RUN_MATRIX_PATH}.")
    run_spec = matching_runs[0]

    source_metadata = {}
    if source_metadata_path.exists():
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if args.model_name is None:
        args.model_name = source_metadata.get("model_name") or "Qwen/Qwen2.5-3B-Instruct"

    filtered_rows = load_source_rows(raw_csv, decoder_name=args.decoder_name, metric_name=args.metric_name)
    selected_cases = select_cases(
        filtered_rows,
        example_indices=parse_int_list(args.example_indices),
        top_k=args.top_k,
        wrong_k=args.wrong_k,
    )
    if args.list:
        print(json.dumps(selected_cases, indent=2))
        return
    if not selected_cases:
        raise SystemExit("No selected cases were produced.")

    results_dir = RUNS_DIR / args.trace_run_id
    evaluator_args = build_evaluator_args(args, run_spec, results_dir)
    print(
        {
            "source_run_id": args.source_run_id,
            "trace_run_id": args.trace_run_id,
            "results_dir": str(results_dir),
            "decoder_name": args.decoder_name,
            "selected_example_indices": [int(case["example_idx"]) for case in selected_cases],
            "local_files_only": bool(args.local_files_only),
        },
        flush=True,
    )
    if args.dry_run:
        return

    evaluator = ExperimentEvaluator(evaluator_args, (args.decoder_name,))
    token_rows: list[dict[str, object]] = []
    selected_case_rows: list[dict[str, object]] = []

    for case_idx, case in enumerate(selected_cases, start=1):
        print(
            {
                "case_progress": f"{case_idx}/{len(selected_cases)}",
                "example_idx": int(case["example_idx"]),
                "group": case["group"],
                "question": case["question"],
            },
            flush=True,
        )
        prompt = build_truthfulqa_prompt(str(case["question"]))
        sequence_logprob, trace, runtime = evaluator.score_candidate_with_decoder(
            prompt,
            args.decoder_name,
            str(case["prediction"]),
        )
        trace_summary = summarize_trace(trace)
        unique_selected_layers = sorted(
            {
                int(row["selected_layer"])
                for row in trace
                if row.get("selected_layer") is not None
            }
        )
        selected_case_rows.append(
            {
                "source_run_id": args.source_run_id,
                "trace_run_id": args.trace_run_id,
                "decoder": args.decoder_name,
                "metric_name": args.metric_name,
                "group": case["group"],
                "group_rank": int(case["group_rank"]),
                "example_idx": int(case["example_idx"]),
                "question": case["question"],
                "prediction": case["prediction"],
                "score": float(case["score"]),
                "decision_margin": float(case["decision_margin"]),
                "score_detail": case["score_detail"],
                "sequence_logprob": float(sequence_logprob),
                "trace_length": int(len(trace)),
                "unique_selected_layers": json.dumps(unique_selected_layers),
                "latency_seconds": float(runtime["latency_seconds"]),
                "decoder_steps": int(runtime["decoder_steps"]),
                "forward_passes": int(runtime["forward_passes"]),
                **trace_summary,
            }
        )
        for token_row in trace:
            row = dict(token_row)
            step = int(row.get("step", 0))
            row.update(
                {
                    "source_run_id": args.source_run_id,
                    "trace_run_id": args.trace_run_id,
                    "decoder": args.decoder_name,
                    "metric_name": args.metric_name,
                    "group": case["group"],
                    "group_rank": int(case["group_rank"]),
                    "example_idx": int(case["example_idx"]),
                    "question": case["question"],
                    "prediction": case["prediction"],
                    "score": float(case["score"]),
                    "decision_margin": float(case["decision_margin"]),
                    "token_step": step + 1,
                }
            )
            token_rows.append(row)

    token_trace_path = results_dir / f"{args.trace_run_id}_token_trace.csv"
    selected_cases_path = results_dir / f"{args.trace_run_id}_selected_cases.csv"
    metadata_path = results_dir / f"{args.trace_run_id}_metadata.json"
    write_csv(token_trace_path, token_rows)
    write_csv(selected_cases_path, selected_case_rows)

    metadata = {
        "experiment_name": "exp11_core_decoder_comparison_trace_dump",
        "source_run_id": args.source_run_id,
        "trace_run_id": args.trace_run_id,
        "source_raw_predictions_csv": str(raw_csv),
        "source_metadata_json": str(source_metadata_path),
        "results_dir": str(results_dir),
        "model_name": evaluator_args.model_name,
        "decoder_name": args.decoder_name,
        "metric_name": args.metric_name,
        "selection_rule": {
            "top_k": int(args.top_k),
            "wrong_k": int(args.wrong_k),
            "example_indices_override": parse_int_list(args.example_indices),
        },
        "trace_mode": "teacher_forced_trace_of_saved_prediction_text_from_exp11_raw_predictions",
        "prompt_style": "truthfulqa_mc_question_only_prompt_used_by_score_candidate_with_decoder",
        "artifact_paths": {
            "token_trace_csv": str(token_trace_path),
            "selected_cases_csv": str(selected_cases_path),
        },
        "selected_example_indices": [int(case["example_idx"]) for case in selected_cases],
        "dola_relative_top": float(evaluator.dola_relative_top),
        "dola_relative_top_value": float(evaluator.dola_relative_top_value),
        "default_shallow_bucket": list(evaluator.default_bucket),
        "dola_mature_layer": int(evaluator.mature_layer_index),
        "local_files_only": bool(evaluator_args.local_files_only),
        "source_run_metadata": source_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        {
            "token_trace_csv": str(token_trace_path),
            "selected_cases_csv": str(selected_cases_path),
            "metadata_json": str(metadata_path),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
