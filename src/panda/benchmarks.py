"""Benchmark loaders kept in sync with the public TruthfulQA evaluator."""

from pathlib import Path

from datasets import Dataset, load_dataset

from .utils import sample_candidate_rows


def _load_truthfulqa_validation_dataset():
    try:
        return load_dataset("truthful_qa", "multiple_choice", split="validation")
    except Exception:
        cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "truthful_qa" / "multiple_choice" / "0.0.0"
        if not cache_root.exists():
            raise
        arrow_candidates = sorted(cache_root.glob("*/truthful_qa-validation.arrow"))
        if not arrow_candidates:
            raise
        return Dataset.from_file(str(arrow_candidates[-1]))


def load_truthfulqa_rows(limit, rng):
    dataset = _load_truthfulqa_validation_dataset()
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
    rows, manifest = sample_candidate_rows(candidates, limit, rng)
    return rows, "truthful_qa/multiple_choice", manifest
