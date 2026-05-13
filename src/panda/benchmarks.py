"""Benchmark loaders kept in sync with the complete evaluator implementation."""

import json
import re
from pathlib import Path

from datasets import load_dataset

from .config import (
    DEFAULT_ALPACAEVAL_CONFIG,
    DEFAULT_ALPACAEVAL_DATASET,
    DEFAULT_ALPACAEVAL_SOURCE,
    DEFAULT_ALPACAEVAL_SPLIT,
    DEFAULT_STRATEGYQA_CONFIG,
    DEFAULT_STRATEGYQA_DATASET,
    DEFAULT_STRATEGYQA_SOURCE,
    DEFAULT_STRATEGYQA_SPLIT,
)
from .prompts import build_halueval_prompt
from .utils import canonicalize_yes_no_label, sample_candidate_rows


def load_truthfulqa_rows(limit, rng):
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
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


def load_strategyqa_rows(limit, rng, dataset=None, config=None, split=None):
    forced_dataset = dataset if dataset is not None else getattr(load_strategyqa_rows, "_forced_dataset", None)
    forced_config = config if config is not None else getattr(load_strategyqa_rows, "_forced_config", None)
    forced_split = split if split is not None else getattr(load_strategyqa_rows, "_forced_split", None)

    if forced_dataset is not None:
        dataset_name = forced_dataset
        config_name = forced_config
        split_name = forced_split or DEFAULT_STRATEGYQA_SPLIT
    else:
        dataset_name = DEFAULT_STRATEGYQA_DATASET
        config_name = DEFAULT_STRATEGYQA_CONFIG
        split_name = DEFAULT_STRATEGYQA_SPLIT

    try:
        kwargs = {"path": dataset_name}
        if config_name is not None:
            kwargs["name"] = config_name
        dataset_obj = load_dataset(**kwargs, split=split_name)
    except Exception as exc:
        raise RuntimeError(
            "StrategyQA loader failed for the configured source: "
            f"dataset={dataset_name!r} config={config_name!r} split={split_name!r}: {exc}"
        ) from exc

    candidates = []
    for source_idx, row in enumerate(dataset_obj):
        question = row.get("inputs") or row.get("question")
        target_norm = canonicalize_yes_no_label(row.get("targets"))
        if target_norm is None:
            target_norm = canonicalize_yes_no_label(row.get("answer"))
        if target_norm is None:
            target_norm = canonicalize_yes_no_label(row.get("label"))
        if target_norm is None:
            mc_targets = row.get("multiple_choice_targets")
            mc_scores = row.get("multiple_choice_scores")
            if isinstance(mc_targets, list) and isinstance(mc_scores, list) and mc_targets and mc_scores:
                best_idx = max(range(min(len(mc_targets), len(mc_scores))), key=lambda idx: float(mc_scores[idx]))
                target_norm = canonicalize_yes_no_label(mc_targets[best_idx])
        if target_norm not in {"yes", "no"} or not str(question).strip():
            continue
        candidates.append(
            {
                "source_idx": int(source_idx),
                "benchmark": "strategyqa",
                "question": str(question).strip(),
                "choices": ["yes", "no"],
                "correct_choice": target_norm,
            }
        )

    if forced_dataset is not None and not candidates:
        raise RuntimeError(
            "StrategyQA loader found the forced dataset but produced 0 usable rows. "
            f"dataset={forced_dataset!r} config={forced_config!r} split={forced_split or 'validation'!r}"
        )
    if not candidates:
        raise RuntimeError(
            "StrategyQA loader produced 0 usable rows for the configured source. "
            f"dataset={dataset_name!r} config={config_name!r} split={split_name!r}"
        )
    source_name = dataset_name if config_name is None else f"{dataset_name}/{config_name}"
    rows, manifest = sample_candidate_rows(candidates, limit, rng)
    return rows, f"{source_name}:{split_name}", manifest


def load_gsm8k_rows(limit, rng):
    dataset = load_dataset("gsm8k", "main", split="test")
    candidates = []
    for source_idx, row in enumerate(dataset):
        answer_text = row.get("answer", "")
        match = re.findall(r"####\s*([-0-9.,]+)", answer_text)
        if not match:
            continue
        candidates.append(
            {
                "source_idx": int(source_idx),
                "benchmark": "gsm8k",
                "question": row["question"],
                "correct_choice": match[-1].replace(",", "").strip(),
            }
        )
    rows, manifest = sample_candidate_rows(candidates, limit, rng)
    return rows, "gsm8k/main", manifest


def load_alpacaeval_rows(limit, rng):
    dataset = load_dataset(
        DEFAULT_ALPACAEVAL_DATASET,
        DEFAULT_ALPACAEVAL_CONFIG,
        split=DEFAULT_ALPACAEVAL_SPLIT,
    )
    candidates = []
    for source_idx, row in enumerate(dataset):
        instruction = str(row.get("instruction") or "").strip()
        reference_output = str(row.get("output") or "").strip()
        dataset_name = str(row.get("dataset") or "alpaca_eval").strip()
        generator_name = str(row.get("generator") or "gpt4_turbo").strip()
        if not instruction or not reference_output:
            continue
        candidates.append(
            {
                "source_idx": int(source_idx),
                "benchmark": "alpacaeval",
                "instruction": instruction,
                "reference_output": reference_output,
                "reference_generator": generator_name,
                "dataset_name": dataset_name,
            }
        )
    rows, manifest = sample_candidate_rows(candidates, limit, rng)
    return rows, DEFAULT_ALPACAEVAL_SOURCE, manifest


def load_halueval_rows(limit, rng, root_path, tasks):
    root = Path(root_path).expanduser()
    if not root.exists():
        raise RuntimeError(f"HaluEval root does not exist: {root}")

    task_to_filename = {
        "qa": "qa_data.json",
        "dialogue": "dialogue_data.json",
        "summarization": "summarization_data.json",
        "general": "general_data.json",
    }

    candidates = []
    for task_name in tasks:
        if task_name not in task_to_filename:
            raise RuntimeError(f"Unsupported HaluEval task: {task_name!r}")
        data_path = root / task_to_filename[task_name]
        if not data_path.exists():
            raise RuntimeError(f"Missing HaluEval file for task {task_name!r}: {data_path}")
        raw_text = data_path.read_text(encoding="utf-8")
        try:
            rows = json.loads(raw_text)
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
        for source_idx, row in enumerate(rows):
            knowledge = str(row.get("knowledge") or row.get("document") or row.get("context") or "").strip()
            user_input = str(
                row.get("question") or row.get("dialogue_history") or row.get("user_query") or row.get("input") or ""
            ).strip()
            right_response = str(
                row.get("right_answer") or row.get("right_response") or row.get("right_summary") or ""
            ).strip()
            hallucinated_response = str(
                row.get("hallucinated_answer")
                or row.get("hallucinated_response")
                or row.get("hallucinated_summary")
                or ""
            ).strip()
            if not right_response or not hallucinated_response:
                continue
            use_hallucinated = bool(rng.random() < 0.5)
            response_text = hallucinated_response if use_hallucinated else right_response
            yes_label, no_label = ("A", "B") if bool(rng.random() < 0.5) else ("B", "A")
            correct_choice = yes_label if use_hallucinated else no_label
            prompt_text = build_halueval_prompt(
                task_name,
                knowledge,
                user_input,
                response_text,
                label_yes=yes_label,
                label_no=no_label,
            )
            candidates.append(
                {
                    "source_idx": int(source_idx),
                    "benchmark": f"halueval_{task_name}",
                    "task_name": task_name,
                    "question": prompt_text,
                    "choices": ["A", "B"],
                    "correct_choice": correct_choice,
                    "halueval_yes_label": yes_label,
                    "halueval_no_label": no_label,
                    "halueval_has_hallucination": use_hallucinated,
                    "knowledge": knowledge,
                    "user_input": user_input,
                    "response_text": response_text,
                    "halueval_source_file": str(data_path),
                }
            )

    rows, manifest = sample_candidate_rows(candidates, limit, rng)
    source_name = f"local:{root}"
    return rows, source_name, manifest
