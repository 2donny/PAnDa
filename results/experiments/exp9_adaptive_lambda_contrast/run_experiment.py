#!/usr/bin/env python3
"""Run the fixed-contrast vs adaptive-lambda contrast experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common import build_evaluator_args, read_run_matrix, run_truthfulqa_suite


EXPERIMENT_NAME = "exp9_adaptive_lambda_contrast"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_MATRIX_PATH = EXPERIMENT_DIR / "run_matrix.csv"
RUNS_DIR = EXPERIMENT_DIR / "runs"
EXP9_DECODER_NAMES = (
    "always_contrast",
    "adaptive_lambda_contrast",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fixed-contrast vs adaptive-lambda contrast experiment.")
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
    parser.add_argument("--exp9-lambda-min", type=float, default=0.5)
    parser.add_argument("--exp9-lambda-max", type=float, default=1.0)
    parser.add_argument("--exp9-uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--exp9-confidence-gap-weight", type=float, default=2.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_selected_runs(rows, run_id):
    if run_id is None:
        return rows
    return [row for row in rows if row["run_id"] == run_id]


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
        print(
            {
                "experiment": EXPERIMENT_NAME,
                "run_id": run_id,
                "results_dir": str(results_dir),
                "decoder_names": list(EXP9_DECODER_NAMES),
            }
        )
        if args.dry_run:
            continue
        from local_evaluator import ExperimentEvaluator

        evaluator = ExperimentEvaluator(evaluator_args, EXP9_DECODER_NAMES)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "notes": row.get("notes"),
            "model_name": evaluator_args.model_name,
            "mode": evaluator_args.mode,
            "seed": evaluator_args.seed,
            "strict_eval": evaluator_args.strict_eval,
            "decoder_names": list(EXP9_DECODER_NAMES),
            "hypothesis": (
                "if_full_contrast_is_too_blunt_a_conservative_adaptive_lambda_should_match_or_beat_it"
            ),
            "experiment_note": (
                "stateful_fixed_contrast_vs_stateful_entropy_and_confidence_gap_adaptive_contrast"
            ),
            "binary_rule": "final_logits_vs_final_logits_minus_lambda_t_times_shallow_logits",
            "panda_divergence_threshold": evaluator_args.panda_divergence_threshold,
            "panda_truth_bias": evaluator_args.panda_truth_bias,
            "exp9_lambda_min": evaluator_args.exp9_lambda_min,
            "exp9_lambda_max": evaluator_args.exp9_lambda_max,
            "exp9_uncertainty_weight": evaluator_args.exp9_uncertainty_weight,
            "exp9_confidence_gap_weight": evaluator_args.exp9_confidence_gap_weight,
        }
        run_truthfulqa_suite(
            evaluator,
            EXP9_DECODER_NAMES,
            evaluator_args,
            artifact_prefix=run_id,
            results_dir=results_dir,
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
