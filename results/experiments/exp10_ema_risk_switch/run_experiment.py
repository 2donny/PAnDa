#!/usr/bin/env python3
"""Run the fixed-contrast vs EMA-risk switch experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common import build_evaluator_args, read_run_matrix, run_truthfulqa_suite


EXPERIMENT_NAME = "exp10_ema_risk_switch"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_MATRIX_PATH = EXPERIMENT_DIR / "run_matrix.csv"
RUNS_DIR = EXPERIMENT_DIR / "runs"
EXP10_DECODER_NAMES = (
    "always_contrast",
    "ema_risk_switch",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fixed-contrast vs EMA-risk switch experiment.")
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
    parser.add_argument("--exp10-risk-beta", type=float, default=0.8)
    parser.add_argument("--exp10-entropy-weight", type=float, default=1.0)
    parser.add_argument("--exp10-margin-weight", type=float, default=1.0)
    parser.add_argument("--exp10-layer-jsd-weight", type=float, default=1.0)
    parser.add_argument("--exp10-risk-threshold", type=float, default=0.55)
    parser.add_argument("--exp10-sticky-hold-steps", type=int, default=0)
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
                "decoder_names": list(EXP10_DECODER_NAMES),
            }
        )
        if args.dry_run:
            continue
        from local_evaluator import ExperimentEvaluator

        evaluator = ExperimentEvaluator(evaluator_args, EXP10_DECODER_NAMES)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "notes": row.get("notes"),
            "model_name": evaluator_args.model_name,
            "mode": evaluator_args.mode,
            "seed": evaluator_args.seed,
            "strict_eval": evaluator_args.strict_eval,
            "decoder_names": list(EXP10_DECODER_NAMES),
            "hypothesis": (
                "if_hallucination_prone_contrast_errors_cluster_temporally_a_smoothed_risk_gate_"
                "can_keep_useful_contrast_while_skipping_bad_spans"
            ),
            "experiment_note": (
                "single_pass_fixed_layer_ema_risk_switch_vs_matched_always_contrast_control"
            ),
            "binary_rule": "final_logits_vs_final_logits_minus_shallow_logits",
            "panda_divergence_threshold": evaluator_args.panda_divergence_threshold,
            "panda_truth_bias": evaluator_args.panda_truth_bias,
            "shallow_bucket": evaluator_args.shallow_bucket,
            "exp10_risk_beta": evaluator_args.exp10_risk_beta,
            "exp10_entropy_weight": evaluator_args.exp10_entropy_weight,
            "exp10_margin_weight": evaluator_args.exp10_margin_weight,
            "exp10_layer_jsd_weight": evaluator_args.exp10_layer_jsd_weight,
            "exp10_risk_threshold": evaluator_args.exp10_risk_threshold,
            "exp10_sticky_hold_steps": evaluator_args.exp10_sticky_hold_steps,
        }
        run_truthfulqa_suite(
            evaluator,
            EXP10_DECODER_NAMES,
            evaluator_args,
            artifact_prefix=run_id,
            results_dir=results_dir,
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
