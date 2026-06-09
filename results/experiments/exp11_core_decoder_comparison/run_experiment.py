#!/usr/bin/env python3
"""Run exp11 as a direct comparison across core decoder baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common import build_evaluator_args, read_run_matrix, run_truthfulqa_suite


EXPERIMENT_NAME = "exp11_core_decoder_comparison"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_MATRIX_PATH = EXPERIMENT_DIR / "run_matrix.csv"
RUNS_DIR = EXPERIMENT_DIR / "runs"
EXP11_BASE_DECODER_NAMES = (
    "pure_greedy",
    "top_k",
    "top_p",
    "dola",
    "fanda",
    "panda_switch",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run exp11 across pure_greedy, top_k, top_p, dola, fanda, and panda_switch."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--top-k-value", type=int, default=50)
    parser.add_argument("--top-p-value", type=float, default=0.9)
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
    parser.add_argument(
        "--include-top-p-backoff",
        action="store_true",
        help="Also evaluate the legacy top_p_backoff alias for explicit fallback tracing alongside top_p.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_selected_runs(rows, run_id):
    if run_id is None:
        return rows
    return [row for row in rows if row["run_id"] == run_id]


def resolve_decoder_names(include_top_p_backoff):
    decoder_names = list(EXP11_BASE_DECODER_NAMES)
    if include_top_p_backoff:
        decoder_names.insert(3, "top_p_backoff")
    return tuple(decoder_names)


def main():
    args = parse_args()
    decoder_names = resolve_decoder_names(args.include_top_p_backoff)
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
        print(
            {
                "experiment": EXPERIMENT_NAME,
                "run_id": run_id,
                "results_dir": str(results_dir),
                "decoder_names": list(decoder_names),
            }
        )
        if args.dry_run:
            continue
        from local_evaluator import ExperimentEvaluator

        evaluator = ExperimentEvaluator(evaluator_args, decoder_names)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "notes": row.get("notes"),
            "model_name": evaluator_args.model_name,
            "mode": evaluator_args.mode,
            "seed": evaluator_args.seed,
            "strict_eval": evaluator_args.strict_eval,
            "decoder_names": list(decoder_names),
            "hypothesis": (
                "fanda_should_remain_the_strongest_accuracy_baseline_unless_original_"
                "panda_switch_recovers_value_over_classic_truncation_and_dola_families"
            ),
            "experiment_note": (
                "pure_greedy_vs_topk_topp_vs_dola_vs_fanda_vs_original_panda_switch"
            ),
            "top_k_value": evaluator_args.top_k_value,
            "top_p_value": evaluator_args.top_p_value,
            "top_p_backoff_enabled": bool(args.include_top_p_backoff),
            "binary_rule": "final_logits_vs_final_logits_minus_shallow_logits",
            "panda_divergence_threshold": evaluator_args.panda_divergence_threshold,
            "panda_truth_bias": evaluator_args.panda_truth_bias,
            "jacobi_max_iters": evaluator_args.jacobi_max_iters,
            "base_jacobi_window_size": evaluator_args.jacobi_window_size,
        }
        run_truthfulqa_suite(
            evaluator,
            decoder_names,
            evaluator_args,
            artifact_prefix=run_id,
            results_dir=results_dir,
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
