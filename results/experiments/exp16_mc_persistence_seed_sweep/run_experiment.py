#!/usr/bin/env python3
"""Run exp16 as a multi-subset MC sweep over selected-layer persistence schedules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common import build_evaluator_args, read_run_matrix, run_truthfulqa_suite_on_rows


EXPERIMENT_NAME = "exp16_mc_persistence_seed_sweep"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_MATRIX_PATH = EXPERIMENT_DIR / "run_matrix.csv"
RUNS_DIR = EXPERIMENT_DIR / "runs"
EXP12_ANCHOR_METADATA_PATH = (
    EXPERIMENTS_ROOT
    / "exp12_state_persistence_diagnostics"
    / "runs"
    / "run_01_default"
    / "run_01_default_metadata.json"
)
EXP16_DECODER_NAMES = (
    "fanda_update1",
    "fanda_update2",
    "fanda_update4",
    "fanda_update8",
    "fanda_frozen",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run exp16 as one MC seed-sweep experiment over selected-layer persistence schedules, "
            "using one anchor subset from exp12 plus four fresh random subsets."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", choices=("sanity", "subset", "full"), default="subset")
    parser.add_argument("--truthfulqa-limit", type=str, default="50")
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
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_selected_runs(rows, run_id):
    if run_id is None:
        return rows
    return [row for row in rows if row["run_id"] == run_id]


def _load_exp12_anchor_source_indices():
    metadata = json.loads(EXP12_ANCHOR_METADATA_PATH.read_text(encoding="utf-8"))
    sampling_manifest = metadata.get("truthfulqa_sampling") or {}
    source_indices = sampling_manifest.get("selected_source_indices") or []
    if not source_indices:
        raise ValueError(
            "exp16 anchor subset expected selected_source_indices in exp12 metadata at "
            f"{EXP12_ANCHOR_METADATA_PATH}"
        )
    return [int(source_idx) for source_idx in source_indices]


def _load_truthfulqa_subset(evaluator_args, run_spec):
    from panda.benchmarks import load_truthfulqa_rows, load_truthfulqa_rows_from_source_indices
    from panda.utils import make_sampling_rng, resolve_limit

    subset_strategy = (run_spec.get("subset_strategy") or "random_seed_subset").strip()
    requested_limit = resolve_limit(evaluator_args.truthfulqa_limit, evaluator_args.mode, 5)
    if subset_strategy == "anchor_exp12_run_01":
        source_indices = _load_exp12_anchor_source_indices()
        rows, source, manifest = load_truthfulqa_rows_from_source_indices(source_indices)
        manifest = {
            **manifest,
            "anchor_experiment": "exp12_state_persistence_diagnostics",
            "anchor_run_id": "run_01_default",
            "anchor_metadata_path": str(EXP12_ANCHOR_METADATA_PATH),
        }
        return rows, source, manifest, len(source_indices)
    if subset_strategy == "random_seed_subset":
        rows, source, manifest = load_truthfulqa_rows(
            requested_limit,
            make_sampling_rng(evaluator_args.seed, "truthfulqa"),
        )
        return rows, source, manifest, requested_limit
    raise ValueError(f"Unknown subset_strategy {subset_strategy!r}.")


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
        truthfulqa_rows, truthfulqa_source, truthfulqa_manifest, truthfulqa_limit = _load_truthfulqa_subset(
            evaluator_args,
            row,
        )
        print(
            {
                "experiment": EXPERIMENT_NAME,
                "run_id": run_id,
                "results_dir": str(results_dir),
                "subset_strategy": row.get("subset_strategy"),
                "truthfulqa_examples": len(truthfulqa_rows),
                "decoder_names": list(EXP16_DECODER_NAMES),
            }
        )
        if args.dry_run:
            continue
        from local_evaluator import ExperimentEvaluator

        evaluator = ExperimentEvaluator(evaluator_args, EXP16_DECODER_NAMES)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "notes": row.get("notes"),
            "model_name": evaluator_args.model_name,
            "mode": evaluator_args.mode,
            "seed": evaluator_args.seed,
            "strict_eval": evaluator_args.strict_eval,
            "decoder_names": list(EXP16_DECODER_NAMES),
            "primary_metric": "mc2",
            "secondary_metrics": ["mc1", "mc3"],
            "subset_strategy": row.get("subset_strategy"),
            "anchor_subset_reference": (
                "exp12_state_persistence_diagnostics/run_01_default"
                if row.get("subset_strategy") == "anchor_exp12_run_01"
                else None
            ),
            "experiment_question": (
                "across_different_truthfulqa_subsets_is_the_mc_ranking_of_update1_update2_"
                "update4_update8_and_frozen_stable"
            ),
            "experiment_note": (
                "one_anchor_subset_matches_exp12_exactly_and_four_additional_runs_use_fresh_"
                "seeded_random_subsets_of_equal_size"
            ),
            "refresh_schedule": {
                "fanda_update1": 1,
                "fanda_update2": 2,
                "fanda_update4": 4,
                "fanda_update8": 8,
                "fanda_frozen": "first_step_only_then_hold",
            },
            "binary_rule": "final_logits_minus_selected_shallow_logits",
        }
        run_truthfulqa_suite_on_rows(
            evaluator=evaluator,
            decoder_names=EXP16_DECODER_NAMES,
            cli_args=evaluator_args,
            artifact_prefix=run_id,
            results_dir=results_dir,
            metadata=metadata,
            truthfulqa_rows=truthfulqa_rows,
            truthfulqa_source=truthfulqa_source,
            truthfulqa_manifest=truthfulqa_manifest,
            truthfulqa_limit=truthfulqa_limit,
        )


if __name__ == "__main__":
    main()
