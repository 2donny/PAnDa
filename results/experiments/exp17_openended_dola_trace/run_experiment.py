#!/usr/bin/env python3
"""Replay selected exp14 update1 open-ended generations and dump token-level traces."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path


EXPERIMENT_NAME = "exp17_openended_dola_trace"
EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = EXPERIMENT_DIR.parent
RUN_MATRIX_PATH = EXPERIMENT_DIR / "run_matrix.csv"
RUNS_DIR = EXPERIMENT_DIR / "runs"

if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common import (
    _append_progress_event,
    _write_progress_snapshot,
    build_evaluator_args,
    read_run_matrix,
)

PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from panda.evaluation import summarize_trace


SOURCE_EXPERIMENT_DIR = EXPERIMENTS_ROOT / "exp14_openended_factuality"
DEFAULT_SOURCE_RUN_ID = "run_01_default"
DEFAULT_SOURCE_DECODER = "exp14_update1"
DEFAULT_SOURCE_MANUAL_EVAL_CSV = (
    SOURCE_EXPERIMENT_DIR
    / "runs"
    / DEFAULT_SOURCE_RUN_ID
    / f"{DEFAULT_SOURCE_RUN_ID}_manual_eval.csv"
)
DEFAULT_SOURCE_METADATA_JSON = (
    SOURCE_EXPERIMENT_DIR
    / "runs"
    / DEFAULT_SOURCE_RUN_ID
    / f"{DEFAULT_SOURCE_RUN_ID}_metadata.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run exp17 as a selected-case open-ended trace dump using the exp14 "
            "codex-manual-eval sheet to choose strong and weak exp14_update1 answers."
        )
    )
    parser.add_argument("--model-name", default=None)
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
    parser.add_argument("--source-decoder", default=DEFAULT_SOURCE_DECODER)
    parser.add_argument("--source-manual-eval-csv", type=Path, default=DEFAULT_SOURCE_MANUAL_EVAL_CSV)
    parser.add_argument("--source-metadata-json", type=Path, default=DEFAULT_SOURCE_METADATA_JSON)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--manual-positive-score", type=int, default=2)
    parser.add_argument("--manual-negative-score", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--wrong-k", type=int, default=3)
    parser.add_argument(
        "--example-indices",
        type=str,
        default=None,
        help="Comma-separated explicit example_idx list. Overrides the top/worst auto-selection.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_int_list(text: str | None) -> list[int]:
    if not text:
        return []
    values = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    return values


def iter_selected_runs(rows, run_id):
    if run_id is None:
        return rows
    return [row for row in rows if row["run_id"] == run_id]


def build_openended_truthfulqa_prompt(question):
    return (
        f"Question: {question}\n"
        "Answer truthfully in 2 to 4 short sentences. Be direct, factual, and avoid "
        "listing answer options or multiple guesses.\n"
        "Answer:"
    )


def word_count(text):
    return len(re.findall(r"\S+", str(text or "").strip()))


def normalize_answer(text):
    return " ".join(str(text or "").split())


def add_cumulative_word_counts(trace):
    running = ""
    augmented = []
    for row in trace:
        item = dict(row)
        running += str(item.get("token_text") or "")
        item["cumulative_word_count"] = word_count(running)
        item["cumulative_char_count"] = len(running.strip())
        augmented.append(item)
    return augmented


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


def load_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_manual_eval_rows(path: Path, *, decoder_name: str) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["decoder"] != decoder_name:
                continue
            rows.append(row)
    return rows


def manual_score(row):
    return int(row["manual_score_0_2"])


def proxy_margin(row):
    return float(row["proxy_oref_margin"])


def select_cases(
    rows: list[dict[str, str]],
    *,
    example_indices: list[int],
    manual_positive_score: int,
    manual_negative_score: int,
    top_k: int,
    wrong_k: int,
) -> list[dict[str, object]]:
    if example_indices:
        by_idx = {int(row["example_idx"]): row for row in rows}
        selected = []
        for rank, example_idx in enumerate(example_indices, start=1):
            if example_idx not in by_idx:
                raise SystemExit(f"example_idx={example_idx} was not found in the filtered manual-eval rows.")
            row = by_idx[example_idx]
            selected.append(build_case_row(row, group_rank=rank))
        return selected

    positive_rows = [row for row in rows if manual_score(row) == int(manual_positive_score)]
    negative_rows = [row for row in rows if manual_score(row) == int(manual_negative_score)]
    positive_rows.sort(key=proxy_margin, reverse=True)
    negative_rows.sort(key=proxy_margin)

    selected = []
    for rank, row in enumerate(positive_rows[:top_k], start=1):
        selected.append(build_case_row(row, group_rank=rank))
    for rank, row in enumerate(negative_rows[:wrong_k], start=1):
        selected.append(build_case_row(row, group_rank=rank))
    return selected


def build_case_row(row: dict[str, str], *, group_rank: int) -> dict[str, object]:
    score = manual_score(row)
    return {
        "group": f"manual_score_0_2={score}",
        "group_rank": int(group_rank),
        "example_idx": int(row["example_idx"]),
        "decoder": row["decoder"],
        "decoder_label": row["decoder_label"],
        "question": row["question"],
        "source_prediction": row["prediction"],
        "manual_score_0_2": score,
        "manual_label": row["manual_label"],
        "issue_tags": row["issue_tags"],
        "manual_notes": row["manual_notes"],
        "review_status": row["review_status"],
        "reviewer": row["reviewer"],
        "proxy_oref_margin": float(row["proxy_oref_margin"]),
        "proxy_best_true_f1": float(row["proxy_best_true_f1"]),
        "proxy_best_false_f1": float(row["proxy_best_false_f1"]),
        "proxy_best_true_ref": row["proxy_best_true_ref"],
        "proxy_best_false_ref": row["proxy_best_false_ref"],
        "proxy_answer_token_count": int(float(row["proxy_answer_token_count"] or 0)),
        "source_decoder_steps": int(float(row["decoder_steps"] or 0)),
        "source_switch_rate": float(row["switch_rate"] or 0.0),
        "source_selected_layer_match_rate": float(row["selected_layer_match_rate"] or 0.0),
        "source_avg_oracle_jsd_gap": float(row["avg_oracle_jsd_gap"] or 0.0),
    }


def run_trace_dump(
    evaluator,
    *,
    decoder_name: str,
    selected_cases: list[dict[str, object]],
    results_dir: Path,
    artifact_prefix: str,
    metadata: dict[str, object],
    max_new_tokens: int,
):
    progress_json_path = results_dir / "progress.json"
    progress_events_path = results_dir / "progress.ndjson"
    selected_cases_path = results_dir / f"{artifact_prefix}_selected_cases.csv"
    token_trace_path = results_dir / f"{artifact_prefix}_token_trace.csv"
    metadata_path = results_dir / f"{artifact_prefix}_metadata.json"

    progress_state = {
        "status": "running",
        "experiment": metadata["experiment_name"],
        "run_id": metadata["run_id"],
        "results_dir": str(results_dir),
        "source_decoder": decoder_name,
        "total_cases": len(selected_cases),
        "completed_cases": 0,
        "current_example_idx": None,
        "current_group": None,
        "started_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
    }
    _write_progress_snapshot(progress_json_path, dict(progress_state, percent_complete=0.0))

    selected_case_rows: list[dict[str, object]] = []
    token_rows: list[dict[str, object]] = []
    start_time = time.perf_counter()

    for case_idx, case in enumerate(selected_cases, start=1):
        progress_state["current_example_idx"] = int(case["example_idx"])
        progress_state["current_group"] = case["group"]
        progress_state["updated_at_epoch"] = time.time()
        _append_progress_event(
            progress_events_path,
            {
                "timestamp_epoch": time.time(),
                "event": "case_started",
                "case_idx": case_idx,
                "total_cases": len(selected_cases),
                "example_idx": int(case["example_idx"]),
                "group": case["group"],
                "question": case["question"],
            },
        )
        print(
            {
                "case_progress": f"{case_idx}/{len(selected_cases)}",
                "example_idx": int(case["example_idx"]),
                "group": case["group"],
                "question": case["question"],
            },
            flush=True,
        )

        prompt = build_openended_truthfulqa_prompt(str(case["question"]))
        regenerated_prediction, trace, runtime = evaluator.generate_with_decoder(
            prompt,
            decoder_name,
            max_new_tokens=max_new_tokens,
        )
        regenerated_prediction = str(regenerated_prediction or "").strip()
        trace_summary = summarize_trace(trace)
        trace_with_counts = add_cumulative_word_counts(trace)
        unique_selected_layers = sorted(
            {
                int(token_row["selected_layer"])
                for token_row in trace
                if token_row.get("selected_layer") is not None
            }
        )
        source_prediction = str(case["source_prediction"])
        exact_match = regenerated_prediction == source_prediction
        normalized_match = normalize_answer(regenerated_prediction) == normalize_answer(source_prediction)

        selected_case_rows.append(
            {
                "source_run_id": metadata["source_run_id"],
                "run_id": metadata["run_id"],
                "decoder": decoder_name,
                "decoder_label": case["decoder_label"],
                "group": case["group"],
                "group_rank": int(case["group_rank"]),
                "example_idx": int(case["example_idx"]),
                "question": case["question"],
                "manual_score_0_2": int(case["manual_score_0_2"]),
                "manual_label": case["manual_label"],
                "issue_tags": case["issue_tags"],
                "manual_notes": case["manual_notes"],
                "review_status": case["review_status"],
                "reviewer": case["reviewer"],
                "proxy_oref_margin": float(case["proxy_oref_margin"]),
                "proxy_best_true_f1": float(case["proxy_best_true_f1"]),
                "proxy_best_false_f1": float(case["proxy_best_false_f1"]),
                "proxy_best_true_ref": case["proxy_best_true_ref"],
                "proxy_best_false_ref": case["proxy_best_false_ref"],
                "source_prediction": source_prediction,
                "regenerated_prediction": regenerated_prediction,
                "source_decoder_steps": int(case["source_decoder_steps"]),
                "regenerated_decoder_steps": int(runtime["decoder_steps"]),
                "source_switch_rate": float(case["source_switch_rate"]),
                "regenerated_switch_rate": float(trace_summary.get("switch_rate", 0.0)),
                "source_selected_layer_match_rate": float(case["source_selected_layer_match_rate"]),
                "regenerated_selected_layer_match_rate": float(
                    trace_summary.get("selected_layer_match_rate", 0.0)
                ),
                "source_avg_oracle_jsd_gap": float(case["source_avg_oracle_jsd_gap"]),
                "regenerated_avg_oracle_jsd_gap": float(trace_summary.get("avg_oracle_jsd_gap", 0.0)),
                "trace_length": int(len(trace)),
                "unique_selected_layers": json.dumps(unique_selected_layers),
                "exact_prediction_match": int(exact_match),
                "normalized_prediction_match": int(normalized_match),
                "source_answer_token_count": int(case["proxy_answer_token_count"]),
                "regenerated_answer_token_count": int(word_count(regenerated_prediction)),
                "latency_seconds": float(runtime["latency_seconds"]),
                "forward_passes": int(runtime["forward_passes"]),
                "latency_per_step_ms": float(runtime["latency_per_step_ms"]),
                "steps_per_second": float(runtime["steps_per_second"]),
                **trace_summary,
            }
        )

        for token_row in trace_with_counts:
            token_rows.append(
                {
                    "source_run_id": metadata["source_run_id"],
                    "run_id": metadata["run_id"],
                    "decoder": decoder_name,
                    "decoder_label": case["decoder_label"],
                    "group": case["group"],
                    "group_rank": int(case["group_rank"]),
                    "example_idx": int(case["example_idx"]),
                    "question": case["question"],
                    "manual_score_0_2": int(case["manual_score_0_2"]),
                    "manual_label": case["manual_label"],
                    "review_status": case["review_status"],
                    "reviewer": case["reviewer"],
                    "proxy_oref_margin": float(case["proxy_oref_margin"]),
                    "source_prediction": source_prediction,
                    "regenerated_prediction": regenerated_prediction,
                    "token_step": int(token_row["step"]) + 1,
                    **token_row,
                }
            )

        progress_state["completed_cases"] = case_idx
        progress_state["updated_at_epoch"] = time.time()
        snapshot = dict(progress_state)
        snapshot["percent_complete"] = 100.0 * case_idx / max(1, len(selected_cases))
        snapshot["latest_case"] = {
            "example_idx": int(case["example_idx"]),
            "group": case["group"],
            "normalized_prediction_match": int(normalized_match),
            "trace_length": int(len(trace)),
        }
        _write_progress_snapshot(progress_json_path, snapshot)
        _append_progress_event(
            progress_events_path,
            {
                "timestamp_epoch": time.time(),
                "event": "case_finished",
                "case_idx": case_idx,
                "total_cases": len(selected_cases),
                "example_idx": int(case["example_idx"]),
                "group": case["group"],
                "normalized_prediction_match": int(normalized_match),
                "trace_length": int(len(trace)),
                "latency_seconds": float(runtime["latency_seconds"]),
            },
        )

    elapsed = time.perf_counter() - start_time
    progress_state["status"] = "completed"
    progress_state["updated_at_epoch"] = time.time()
    _write_progress_snapshot(progress_json_path, dict(progress_state, percent_complete=100.0))
    _append_progress_event(
        progress_events_path,
        {
            "timestamp_epoch": time.time(),
            "event": "trace_dump_finished",
            "cases": len(selected_cases),
            "elapsed_seconds": elapsed,
        },
    )

    write_csv(selected_cases_path, selected_case_rows)
    write_csv(token_trace_path, token_rows)
    metadata = dict(metadata)
    metadata.update(
        {
            "elapsed_seconds": elapsed,
            "artifact_paths": {
                "selected_cases_csv": str(selected_cases_path),
                "token_trace_csv": str(token_trace_path),
            },
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        {
            "selected_cases_csv": str(selected_cases_path),
            "token_trace_csv": str(token_trace_path),
            "metadata_path": str(metadata_path),
            "elapsed_seconds": elapsed,
        },
        flush=True,
    )


def main():
    args = parse_args()
    run_rows = read_run_matrix(RUN_MATRIX_PATH)
    selected_runs = iter_selected_runs(run_rows, args.run_id)
    if not selected_runs:
        raise SystemExit(f"No run rows matched run_id={args.run_id!r}.")

    source_metadata = load_json_if_exists(args.source_metadata_json)
    if args.model_name is None:
        args.model_name = str(source_metadata.get("model_name") or "Qwen/Qwen2.5-3B-Instruct")

    source_manual_rows = load_manual_eval_rows(args.source_manual_eval_csv, decoder_name=args.source_decoder)
    if not source_manual_rows:
        raise SystemExit(
            f"No rows were found in {args.source_manual_eval_csv} for decoder={args.source_decoder!r}."
        )
    source_review_status_counts = Counter(row["review_status"] for row in source_manual_rows)
    source_reviewer_counts = Counter(row["reviewer"] for row in source_manual_rows)

    for run_row in selected_runs:
        run_id = run_row["run_id"]
        results_dir = RUNS_DIR / run_id
        evaluator_args = build_evaluator_args(args, run_row, results_dir)
        max_new_tokens = int(run_row.get("max_new_tokens") or args.max_new_tokens)
        source_decoder = run_row.get("source_decoder") or args.source_decoder
        manual_positive_score = int(run_row.get("manual_positive_score") or args.manual_positive_score)
        manual_negative_score = int(run_row.get("manual_negative_score") or args.manual_negative_score)
        top_k = int(run_row.get("top_k") or args.top_k)
        wrong_k = int(run_row.get("wrong_k") or args.wrong_k)
        selected_cases = select_cases(
            source_manual_rows,
            example_indices=parse_int_list(args.example_indices),
            manual_positive_score=manual_positive_score,
            manual_negative_score=manual_negative_score,
            top_k=top_k,
            wrong_k=wrong_k,
        )

        if args.list:
            print(
                json.dumps(
                    {
                        "experiment": EXPERIMENT_NAME,
                        "run_id": run_id,
                        "source_run_id": args.source_run_id,
                        "source_decoder": source_decoder,
                        "source_manual_eval_csv": str(args.source_manual_eval_csv),
                        "source_review_status_counts": dict(source_review_status_counts),
                        "source_reviewer_counts": dict(source_reviewer_counts),
                        "selected_cases": selected_cases,
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )
            continue

        print(
            {
                "experiment": EXPERIMENT_NAME,
                "run_id": run_id,
                "results_dir": str(results_dir),
                "source_run_id": args.source_run_id,
                "source_decoder": source_decoder,
                "source_manual_eval_csv": str(args.source_manual_eval_csv),
                "selected_example_indices": [int(case["example_idx"]) for case in selected_cases],
                "manual_positive_score": manual_positive_score,
                "manual_negative_score": manual_negative_score,
                "max_new_tokens": max_new_tokens,
                "local_files_only": bool(args.local_files_only),
            },
            flush=True,
        )
        if args.dry_run:
            continue

        from local_evaluator import ExperimentEvaluator

        evaluator = ExperimentEvaluator(evaluator_args, (source_decoder,))
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "notes": run_row.get("notes"),
            "model_name": evaluator_args.model_name,
            "mode": evaluator_args.mode,
            "seed": evaluator_args.seed,
            "strict_eval": evaluator_args.strict_eval,
            "source_experiment_name": "exp14_openended_factuality",
            "source_run_id": args.source_run_id,
            "source_decoder": source_decoder,
            "source_manual_eval_csv": str(args.source_manual_eval_csv),
            "source_metadata_json": str(args.source_metadata_json),
            "source_review_status_counts": dict(source_review_status_counts),
            "source_reviewer_counts": dict(source_reviewer_counts),
            "selection_rule": (
                "pick_top_manual_score_cases_by_proxy_margin_desc_and_bottom_manual_score_cases_"
                "by_proxy_margin_asc_from_exp14_update1_codex_manual_eval_sheet"
            ),
            "manual_positive_score": manual_positive_score,
            "manual_negative_score": manual_negative_score,
            "top_k": top_k,
            "wrong_k": wrong_k,
            "max_new_tokens": max_new_tokens,
            "selected_example_indices": [int(case["example_idx"]) for case in selected_cases],
        }
        run_trace_dump(
            evaluator,
            decoder_name=source_decoder,
            selected_cases=selected_cases,
            results_dir=results_dir,
            artifact_prefix=run_id,
            metadata=metadata,
            max_new_tokens=max_new_tokens,
        )


if __name__ == "__main__":
    main()
