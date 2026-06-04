"""Benchmark loaders kept in sync with the public TruthfulQA evaluator."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset

from .utils import sample_candidate_rows


def _find_truthfulqa_arrow_cache():
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "truthful_qa" / "multiple_choice" / "0.0.0"
    if not cache_root.exists():
        return None
    arrow_candidates = sorted(cache_root.glob("*/truthful_qa-validation.arrow"))
    if not arrow_candidates:
        return None
    return arrow_candidates[-1]


def _load_truthfulqa_validation_dataset():
    arrow_path = _find_truthfulqa_arrow_cache()
    if arrow_path is not None:
        return Dataset.from_file(str(arrow_path))
    try:
        return load_dataset("truthful_qa", "multiple_choice", split="validation")
    except Exception:
        arrow_path = _find_truthfulqa_arrow_cache()
        if arrow_path is None:
            raise
        return Dataset.from_file(str(arrow_path))


def _build_truthfulqa_candidates(dataset):
    candidates = []
    for source_idx, row in enumerate(dataset):
        mc1_targets = row.get("mc1_targets") or {}
        mc2_targets = row.get("mc2_targets") or {}
        mc1_choices = list(mc1_targets.get("choices", [])) if isinstance(mc1_targets, dict) else []
        mc1_labels = list(mc1_targets.get("labels", [])) if isinstance(mc1_targets, dict) else []
        mc2_choices = list(mc2_targets.get("choices", [])) if isinstance(mc2_targets, dict) else []
        mc2_labels = list(mc2_targets.get("labels", [])) if isinstance(mc2_targets, dict) else []
        if not mc1_choices or not mc1_labels or not mc2_choices or not mc2_labels:
            continue
        if sum(int(label) == 1 for label in mc1_labels) != 1:
            continue
        if sum(int(label) == 1 for label in mc2_labels) < 1:
            continue
        candidates.append(
            {
                "source_idx": int(source_idx),
                "benchmark": "truthfulqa",
                "question": row["question"],
                "mc1_choices": mc1_choices,
                "mc1_labels": mc1_labels,
                "mc2_choices": mc2_choices,
                "mc2_labels": mc2_labels,
            }
        )
    return candidates


def load_truthfulqa_rows(limit, rng):
    dataset = _load_truthfulqa_validation_dataset()
    candidates = _build_truthfulqa_candidates(dataset)
    rows, manifest = sample_candidate_rows(candidates, limit, rng)
    return rows, "truthful_qa/multiple_choice", manifest


def load_truthfulqa_rows_from_source_indices(source_indices):
    dataset = _load_truthfulqa_validation_dataset()
    candidates = _build_truthfulqa_candidates(dataset)
    by_source_idx = {int(row["source_idx"]): row for row in candidates}
    selected_rows = []
    missing_source_indices = []
    for source_idx in source_indices:
        source_idx = int(source_idx)
        row = by_source_idx.get(source_idx)
        if row is None:
            missing_source_indices.append(source_idx)
            continue
        selected_rows.append(dict(row))
    if missing_source_indices:
        raise ValueError(
            "TruthfulQA source indices were requested that are not available after candidate "
            f"filtering: {missing_source_indices}"
        )
    manifest = {
        "total_candidates": len(candidates),
        "requested_limit": len(selected_rows),
        "resolved_limit": len(selected_rows),
        "sampling_mode": "fixed_source_indices",
        "selected_source_indices": [int(source_idx) for source_idx in source_indices],
    }
    return selected_rows, "truthful_qa/multiple_choice", manifest
