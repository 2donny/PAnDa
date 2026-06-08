#!/usr/bin/env python3
"""Run exp18 as a fixed-subset cross-model transfer sweep over core decoders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common import build_evaluator_args, parse_bool, read_run_matrix, run_truthfulqa_suite_on_rows


EXPERIMENT_NAME = "exp18_cross_model_transfer"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_MATRIX_PATH = EXPERIMENT_DIR / "run_matrix.csv"
RUNS_DIR = EXPERIMENT_DIR / "runs"
EXP11_ANCHOR_METADATA_PATH = (
    EXPERIMENTS_ROOT
    / "exp11_core_decoder_comparison"
    / "runs"
    / "run_01_default"
    / "run_01_default_metadata.json"
)
EXP18_DECODER_NAMES = (
    "pure_greedy",
    "dola",
    "fanda",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run exp18 as a locked-subset multi-model transfer sweep over pure_greedy, "
            "dola, and fanda."
        )
    )
    parser.add_argument("--model-name", default=None)
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


def filter_enabled_rows(rows):
    return [row for row in rows if parse_bool(row.get("enabled"), default=True)]


def _load_exp11_anchor_source_indices():
    metadata = json.loads(EXP11_ANCHOR_METADATA_PATH.read_text(encoding="utf-8"))
    sampling_manifest = metadata.get("truthfulqa_sampling") or {}
    source_indices = sampling_manifest.get("selected_source_indices") or []
    if not source_indices:
        raise ValueError(
            "exp18 anchor subset expected selected_source_indices in exp11 metadata at "
            f"{EXP11_ANCHOR_METADATA_PATH}"
        )
    return [int(source_idx) for source_idx in source_indices]


def _load_truthfulqa_subset(evaluator_args, run_spec):
    from panda.benchmarks import load_truthfulqa_rows, load_truthfulqa_rows_from_source_indices
    from panda.utils import make_sampling_rng, resolve_limit

    subset_strategy = (run_spec.get("subset_strategy") or "anchor_exp11_run_01").strip()
    requested_limit = resolve_limit(evaluator_args.truthfulqa_limit, evaluator_args.mode, 5)
    if subset_strategy == "anchor_exp11_run_01":
        source_indices = _load_exp11_anchor_source_indices()
        rows, source, manifest = load_truthfulqa_rows_from_source_indices(source_indices)
        manifest = {
            **manifest,
            "anchor_experiment": "exp11_core_decoder_comparison",
            "anchor_run_id": "run_01_default",
            "anchor_metadata_path": str(EXP11_ANCHOR_METADATA_PATH),
        }
        return rows, source, manifest, len(source_indices)
    if subset_strategy == "random_seed_subset":
        rows, source, manifest = load_truthfulqa_rows(
            requested_limit,
            make_sampling_rng(evaluator_args.seed, "truthfulqa"),
        )
        return rows, source, manifest, requested_limit
    raise ValueError(f"Unknown subset_strategy {subset_strategy!r}.")


def _resolve_model_name(row, cli_args):
    return str(row.get("model_name") or cli_args.model_name or "").strip()


def _resolve_model_label(row, model_name):
    return str(row.get("model_label") or model_name).strip()


def main():
    args = parse_args()
    rows = read_run_matrix(RUN_MATRIX_PATH)
    selected_rows = iter_selected_runs(rows, args.run_id)
    active_rows = filter_enabled_rows(selected_rows)
    if args.list:
        print(json.dumps(active_rows, indent=2))
        return
    if not selected_rows:
        raise SystemExit(f"No run rows matched run_id={args.run_id!r}.")
    if not active_rows:
        disabled_run_ids = [row["run_id"] for row in selected_rows]
        raise SystemExit(
            "Matched run rows were found, but all are disabled. "
            f"Edit run_matrix.csv and set enabled=true for one of: {disabled_run_ids}"
        )

    for row in active_rows:
        run_id = row["run_id"]
        results_dir = RUNS_DIR / run_id
        evaluator_args = build_evaluator_args(args, row, results_dir)
        model_name = _resolve_model_name(row, args)
        if not model_name:
            raise SystemExit(f"run_id={run_id!r} is missing model_name in run_matrix.csv.")
        model_label = _resolve_model_label(row, model_name)
        evaluator_args.model_name = model_name
        truthfulqa_rows, truthfulqa_source, truthfulqa_manifest, truthfulqa_limit = _load_truthfulqa_subset(
            evaluator_args,
            row,
        )
        print(
            {
                "experiment": EXPERIMENT_NAME,
                "run_id": run_id,
                "results_dir": str(results_dir),
                "model_name": model_name,
                "model_label": model_label,
                "subset_strategy": row.get("subset_strategy"),
                "truthfulqa_examples": len(truthfulqa_rows),
                "decoder_names": list(EXP18_DECODER_NAMES),
            }
        )
        if args.dry_run:
            continue
        from local_evaluator import ExperimentEvaluator

        evaluator = ExperimentEvaluator(evaluator_args, EXP18_DECODER_NAMES)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "notes": row.get("notes"),
            "model_name": evaluator_args.model_name,
            "model_label": model_label,
            "mode": evaluator_args.mode,
            "seed": evaluator_args.seed,
            "strict_eval": evaluator_args.strict_eval,
            "decoder_names": list(EXP18_DECODER_NAMES),
            "primary_metric": "mc2",
            "secondary_metrics": ["mc1", "mc3"],
            "subset_strategy": row.get("subset_strategy"),
            "anchor_subset_reference": (
                "exp11_core_decoder_comparison/run_01_default"
                if row.get("subset_strategy") == "anchor_exp11_run_01"
                else None
            ),
            "experiment_question": (
                "on_the_exact_same_50_truthfulqa_mc_questions_does_fanda_still_beat_"
                "pure_greedy_and_dola_when_the_base_model_changes"
            ),
            "experiment_note": (
                "cross_model_transfer_sweep_over_one_locked_exp11_subset_using_only_the_"
                "minimal_core_decoder_family"
            ),
            "transfer_axis": "model_family",
            "target_decoder": "fanda",
            "reference_decoders": ["pure_greedy", "dola"],
            "binary_rule": "final_logits_vs_final_logits_minus_shallow_logits",
            "subset_lock_reason": "direct_model_to_model_comparability_without_question_difficulty_drift",
        }
        run_truthfulqa_suite_on_rows(
            evaluator=evaluator,
            decoder_names=EXP18_DECODER_NAMES,
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
