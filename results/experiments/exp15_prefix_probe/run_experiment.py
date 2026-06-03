#!/usr/bin/env python3
"""Run exp15 as a prefix-drift case study for frozen vs update1."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common import (
    _append_progress_event,
    _write_progress_snapshot,
    build_evaluator_args,
    read_run_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from panda.benchmarks import load_truthfulqa_rows
from panda.evaluation import assert_eval_sources, summarize_trace
from panda.utils import make_sampling_rng, normalize_text, resolve_limit


EXPERIMENT_NAME = "exp15_prefix_probe"
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_MATRIX_PATH = EXPERIMENT_DIR / "run_matrix.csv"
RUNS_DIR = EXPERIMENT_DIR / "runs"
EXP15_DECODER_NAMES = ("exp14_frozen", "exp14_update1")
DEFAULT_MANUAL_EVAL_CSV = Path(
    "results/experiments/exp14_openended_factuality/runs/run_01_default/run_01_default_manual_eval.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run exp15 as a small open-ended prefix probe that compares exp14_frozen "
            "against exp14_update1 on targeted questions from the completed exp14 manual eval."
        )
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", choices=("sanity", "subset", "full"), default="subset")
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
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--manual-eval-csv", type=Path, default=DEFAULT_MANUAL_EVAL_CSV)
    parser.add_argument("--frozen-win-count", type=int, default=3)
    parser.add_argument("--update1-win-count", type=int, default=3)
    parser.add_argument("--both-bad-count", type=int, default=2)
    parser.add_argument("--prefix-words", type=str, default="10,30,60,100")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_selected_runs(rows, run_id):
    if run_id is None:
        return rows
    return [row for row in rows if row["run_id"] == run_id]


def row_or_cli_int(run_spec, key, cli_value):
    raw_value = run_spec.get(key)
    if raw_value is None or raw_value == "":
        return int(cli_value)
    return int(raw_value)


def row_or_cli_text(run_spec, key, cli_value):
    raw_value = run_spec.get(key)
    if raw_value is None or raw_value == "":
        return str(cli_value)
    return str(raw_value)


def build_openended_truthfulqa_prompt(question):
    return (
        f"Question: {question}\n"
        "Answer truthfully in 2 to 4 short sentences. Be direct, factual, and avoid "
        "listing answer options or multiple guesses.\n"
        "Answer:"
    )


def tokenize_for_overlap(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", normalize_text(text))


def word_count(text):
    return len(re.findall(r"\S+", str(text or "").strip()))


def token_f1(candidate_text, reference_text):
    candidate_tokens = tokenize_for_overlap(candidate_text)
    reference_tokens = tokenize_for_overlap(reference_text)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    candidate_counts = Counter(candidate_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum(min(candidate_counts[token], reference_counts[token]) for token in candidate_counts)
    if overlap <= 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def best_reference_overlap(answer_text, references):
    best_score = 0.0
    best_reference = None
    for reference in references:
        score = token_f1(answer_text, reference)
        if score > best_score:
            best_score = score
            best_reference = reference
    return best_score, best_reference


def compute_openended_factuality_metrics(answer_text, true_references, false_references):
    best_true_f1, best_true_ref = best_reference_overlap(answer_text, true_references)
    best_false_f1, best_false_ref = best_reference_overlap(answer_text, false_references)
    return {
        "oref_margin": float(best_true_f1 - best_false_f1),
        "oref_true_f1": float(best_true_f1),
        "oref_win": float(best_true_f1 > best_false_f1),
        "best_true_f1": float(best_true_f1),
        "best_false_f1": float(best_false_f1),
        "best_true_ref": best_true_ref,
        "best_false_ref": best_false_ref,
        "answer_token_count": word_count(answer_text),
    }


def build_reference_sets(row):
    true_refs = [
        choice
        for choice, label in zip(row["mc2_choices"], row["mc2_labels"])
        if int(label) == 1
    ]
    false_refs = [
        choice
        for choice, label in zip(row["mc2_choices"], row["mc2_labels"])
        if int(label) == 0
    ]
    return true_refs, false_refs


def parse_prefix_words(text):
    values = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def prefix_text_by_word_budget(text, budget_words):
    text = str(text or "").strip()
    if not text or budget_words <= 0:
        return ""
    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return ""
    if len(matches) <= budget_words:
        return text
    end = matches[budget_words - 1].end()
    return text[:end].rstrip()


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


def first_step_for_word_budget(trace_with_counts, budget_words):
    if budget_words <= 0 or not trace_with_counts:
        return None
    for idx, row in enumerate(trace_with_counts):
        if int(row.get("cumulative_word_count") or 0) >= int(budget_words):
            return idx
    return len(trace_with_counts) - 1


def load_manual_eval(path):
    df = pd.read_csv(path, keep_default_na=False)
    df["manual_score_0_2"] = pd.to_numeric(df["manual_score_0_2"])
    df["proxy_answer_token_count"] = pd.to_numeric(df["proxy_answer_token_count"], errors="coerce")
    return df


def select_questions(manual_df, frozen_win_count, update1_win_count, both_bad_count):
    pair = manual_df[manual_df["decoder"].isin(EXP15_DECODER_NAMES)].copy()
    score_pivot = pair.pivot(index="example_idx", columns="decoder", values="manual_score_0_2")
    question_map = pair.groupby("example_idx")["question"].first()
    token_means = pair.groupby("example_idx")["proxy_answer_token_count"].mean()

    rows = []
    for example_idx in score_pivot.index:
        frozen_score = float(score_pivot.loc[example_idx, "exp14_frozen"])
        update1_score = float(score_pivot.loc[example_idx, "exp14_update1"])
        rows.append(
            {
                "example_idx": int(example_idx),
                "question": str(question_map.loc[example_idx]),
                "frozen_score": frozen_score,
                "update1_score": update1_score,
                "score_delta_frozen_minus_update1": float(frozen_score - update1_score),
                "mean_answer_token_count": float(token_means.loc[example_idx]),
            }
        )
    selected_df = pd.DataFrame(rows).sort_values("example_idx").reset_index(drop=True)

    frozen_wins = selected_df[selected_df["frozen_score"] > selected_df["update1_score"]].copy()
    frozen_wins = frozen_wins.sort_values(
        ["score_delta_frozen_minus_update1", "frozen_score", "mean_answer_token_count", "example_idx"],
        ascending=[False, False, False, True],
    )

    update1_wins = selected_df[selected_df["update1_score"] > selected_df["frozen_score"]].copy()
    update1_wins = update1_wins.sort_values(
        ["score_delta_frozen_minus_update1", "update1_score", "mean_answer_token_count", "example_idx"],
        ascending=[True, False, False, True],
    )

    chosen_rows = []
    used = set()

    for rank, row in enumerate(frozen_wins.head(frozen_win_count).itertuples(index=False), start=1):
        chosen_rows.append(
            {
                **row._asdict(),
                "selection_group": "frozen_gt_update1",
                "selection_rank_within_group": rank,
            }
        )
        used.add(int(row.example_idx))

    for rank, row in enumerate(update1_wins.head(update1_win_count).itertuples(index=False), start=1):
        chosen_rows.append(
            {
                **row._asdict(),
                "selection_group": "update1_gt_frozen",
                "selection_rank_within_group": rank,
            }
        )
        used.add(int(row.example_idx))

    both_bad = selected_df[
        (selected_df["frozen_score"] <= 1) & (selected_df["update1_score"] <= 1)
    ].copy()
    both_bad = both_bad[~both_bad["example_idx"].isin(used)]
    both_bad = both_bad.sort_values(
        ["mean_answer_token_count", "frozen_score", "update1_score", "example_idx"],
        ascending=[False, True, True, True],
    )
    for rank, row in enumerate(both_bad.head(both_bad_count).itertuples(index=False), start=1):
        chosen_rows.append(
            {
                **row._asdict(),
                "selection_group": "both_bad_or_noisy",
                "selection_rank_within_group": rank,
            }
        )

    chosen_df = pd.DataFrame(chosen_rows).sort_values(
        ["selection_group", "selection_rank_within_group", "example_idx"]
    ).reset_index(drop=True)
    chosen_df["selection_index"] = range(1, len(chosen_df) + 1)
    return chosen_df


def verify_selected_questions(truthfulqa_rows, chosen_df):
    mismatches = []
    for row in chosen_df.itertuples(index=False):
        example_idx = int(row.example_idx)
        if example_idx < 0 or example_idx >= len(truthfulqa_rows):
            mismatches.append(
                f"example_idx {example_idx} is out of range for truthfulqa_rows size {len(truthfulqa_rows)}"
            )
            continue
        truth_row = truthfulqa_rows[example_idx]
        if str(truth_row["question"]).strip() != str(row.question).strip():
            mismatches.append(
                f"example_idx {example_idx} question mismatch: manual_eval={row.question!r} truthfulqa={truth_row['question']!r}"
            )
    if mismatches:
        joined = "\n".join(mismatches[:8])
        raise ValueError(
            "Selected questions from the manual eval sheet do not match the loaded TruthfulQA subset. "
            "Use a compatible seed/limit/manual-eval source.\n"
            + joined
        )


def build_prefix_rows(
    *,
    example_idx,
    question,
    selection_group,
    decoder_name,
    decoder_label,
    answer_text,
    trace_with_counts,
    true_refs,
    false_refs,
    prefix_words,
):
    rows = []
    full_word_count = word_count(answer_text)
    variants = [(budget, prefix_text_by_word_budget(answer_text, budget)) for budget in prefix_words]
    variants.append((None, str(answer_text or "").strip()))

    for budget, prefix_text in variants:
        prefix_text = str(prefix_text or "").strip()
        actual_word_count = word_count(prefix_text)
        cutoff_step = None
        prefix_trace = []
        if actual_word_count > 0 and trace_with_counts:
            cutoff_step = first_step_for_word_budget(trace_with_counts, actual_word_count)
            if cutoff_step is not None:
                prefix_trace = trace_with_counts[: cutoff_step + 1]
        prefix_summary = summarize_trace(prefix_trace)
        metrics = compute_openended_factuality_metrics(prefix_text, true_refs, false_refs)
        rows.append(
            {
                "example_idx": int(example_idx),
                "question": question,
                "selection_group": selection_group,
                "decoder": decoder_name,
                "decoder_label": decoder_label,
                "prefix_variant": "full" if budget is None else f"word_{int(budget)}",
                "prefix_budget_words": "" if budget is None else int(budget),
                "prefix_word_count": int(actual_word_count),
                "prefix_cutoff_step": "" if cutoff_step is None else int(cutoff_step),
                "prefix_trace_steps": int(len(prefix_trace)),
                "prediction": prefix_text,
                "proxy_oref_margin": float(metrics["oref_margin"]),
                "proxy_best_true_f1": float(metrics["best_true_f1"]),
                "proxy_best_false_f1": float(metrics["best_false_f1"]),
                "proxy_best_true_ref": metrics["best_true_ref"] or "",
                "proxy_best_false_ref": metrics["best_false_ref"] or "",
                "proxy_answer_token_count": int(metrics["answer_token_count"]),
                "manual_score_0_2": "",
                "manual_label": "",
                "issue_tags": "",
                "manual_notes": "",
                "review_status": "unreviewed",
                "reviewer": "",
                "core_claim_correct": "",
                "first_unsupported_span_seen_by_here": "",
                "unsupported_span_notes": "",
                **prefix_summary,
            }
        )
    return rows


def run_prefix_probe_suite(
    evaluator,
    decoder_names,
    cli_args,
    results_dir,
    metadata,
    max_new_tokens,
    chosen_df,
    prefix_words,
    manual_eval_source_csv,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    progress_json_path = results_dir / "progress.json"
    progress_events_path = results_dir / "progress.ndjson"

    truthfulqa_limit = resolve_limit(cli_args.truthfulqa_limit, cli_args.mode, 50)
    truthfulqa_rows, truthfulqa_source, truthfulqa_manifest = load_truthfulqa_rows(
        truthfulqa_limit,
        make_sampling_rng(cli_args.seed, "truthfulqa"),
    )
    assert_eval_sources(cli_args, truthfulqa_source)
    verify_selected_questions(truthfulqa_rows, chosen_df)

    total_examples = len(chosen_df)
    total_decoder_evals = total_examples * len(decoder_names)
    progress_state = {
        "status": "running",
        "experiment": metadata.get("experiment_name"),
        "run_id": metadata.get("run_id"),
        "results_dir": str(results_dir),
        "benchmark": "truthfulqa_prefix_probe",
        "total_examples": total_examples,
        "total_decoders": len(decoder_names),
        "total_decoder_evals": total_decoder_evals,
        "completed_examples": 0,
        "completed_decoder_evals": 0,
        "started_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
    }

    def progress_callback(event):
        progress_state["updated_at_epoch"] = time.time()
        progress_state["last_event"] = event["event"]
        if event["event"] == "decoder_finished":
            progress_state["completed_decoder_evals"] = int(event["completed_decoder_evals"])
            progress_state["latest_metrics"] = {
                "decoder_name": event["decoder_name"],
                "answer_word_count": event["answer_word_count"],
                "latency_seconds": event["latency_seconds"],
            }
        elif event["event"] == "example_finished":
            progress_state["completed_examples"] = int(event["completed_examples"])
        elif event["event"] == "evaluation_finished":
            progress_state["status"] = "completed"
            progress_state["completed_examples"] = total_examples
            progress_state["completed_decoder_evals"] = total_decoder_evals
        snapshot = dict(progress_state)
        snapshot["percent_complete"] = (
            100.0 * snapshot["completed_decoder_evals"] / total_decoder_evals
            if total_decoder_evals
            else 100.0
        )
        _write_progress_snapshot(progress_json_path, snapshot)
        _append_progress_event(progress_events_path, {"timestamp_epoch": time.time(), **event})

    _write_progress_snapshot(progress_json_path, dict(progress_state, percent_complete=0.0))

    generation_rows = []
    prefix_rows = []
    token_rows = []

    try:
        progress_callback(
            {
                "event": "evaluation_started",
                "total_examples": total_examples,
                "total_decoders": len(decoder_names),
                "prefix_words": list(prefix_words),
            }
        )
        for chosen_idx, chosen in enumerate(chosen_df.itertuples(index=False), start=1):
            example_idx = int(chosen.example_idx)
            truth_row = truthfulqa_rows[example_idx]
            question = str(chosen.question)
            prompt = build_openended_truthfulqa_prompt(question)
            true_refs, false_refs = build_reference_sets(truth_row)
            progress_callback(
                {
                    "event": "example_started",
                    "example_idx": example_idx,
                    "selection_index": chosen_idx,
                    "selection_group": chosen.selection_group,
                    "question": question,
                }
            )
            for decoder_offset, decoder_name in enumerate(decoder_names, start=1):
                answer_text, trace, runtime = evaluator.generate_with_decoder(
                    prompt,
                    decoder_name,
                    max_new_tokens=max_new_tokens,
                )
                answer_text = str(answer_text or "").strip()
                trace_with_counts = add_cumulative_word_counts(trace)
                trace_summary = summarize_trace(trace)
                metrics = compute_openended_factuality_metrics(answer_text, true_refs, false_refs)
                decoder_label = evaluator.decoder_labels.get(decoder_name, decoder_name)
                generation_rows.append(
                    {
                        "example_idx": example_idx,
                        "question": question,
                        "selection_group": chosen.selection_group,
                        "selection_index": int(chosen.selection_index),
                        "decoder": decoder_name,
                        "decoder_label": decoder_label,
                        "prompt": prompt,
                        "prediction": answer_text,
                        "answer_word_count": int(word_count(answer_text)),
                        "latency_seconds": float(runtime["latency_seconds"]),
                        "decoder_steps": int(runtime["decoder_steps"]),
                        "forward_passes": int(runtime["forward_passes"]),
                        "latency_per_step_ms": float(runtime["latency_per_step_ms"]),
                        "latency_per_forward_ms": float(runtime["latency_per_forward_ms"]),
                        "steps_per_second": float(runtime["steps_per_second"]),
                        "tokens_per_forward": float(runtime["tokens_per_forward"]),
                        "factual_speedup": float(runtime["factual_speedup"]),
                        "proxy_oref_margin_full": float(metrics["oref_margin"]),
                        "proxy_best_true_f1_full": float(metrics["best_true_f1"]),
                        "proxy_best_false_f1_full": float(metrics["best_false_f1"]),
                        **trace_summary,
                    }
                )
                for step_idx, token_row in enumerate(trace_with_counts):
                    token_rows.append(
                        {
                            "example_idx": example_idx,
                            "question": question,
                            "selection_group": chosen.selection_group,
                            "selection_index": int(chosen.selection_index),
                            "decoder": decoder_name,
                            "decoder_label": decoder_label,
                            "step": int(step_idx),
                            **token_row,
                        }
                    )
                prefix_rows.extend(
                    build_prefix_rows(
                        example_idx=example_idx,
                        question=question,
                        selection_group=chosen.selection_group,
                        decoder_name=decoder_name,
                        decoder_label=decoder_label,
                        answer_text=answer_text,
                        trace_with_counts=trace_with_counts,
                        true_refs=true_refs,
                        false_refs=false_refs,
                        prefix_words=prefix_words,
                    )
                )
                progress_callback(
                    {
                        "event": "decoder_finished",
                        "example_idx": example_idx,
                        "selection_index": chosen_idx,
                        "decoder_name": decoder_name,
                        "completed_decoder_evals": (chosen_idx - 1) * len(decoder_names) + decoder_offset,
                        "answer_word_count": int(word_count(answer_text)),
                        "latency_seconds": float(runtime["latency_seconds"]),
                    }
                )
            progress_callback(
                {
                    "event": "example_finished",
                    "example_idx": example_idx,
                    "completed_examples": chosen_idx,
                }
            )
        progress_callback({"event": "evaluation_finished"})
    except Exception as exc:
        progress_state["status"] = "failed"
        progress_state["updated_at_epoch"] = time.time()
        progress_state["error"] = repr(exc)
        _write_progress_snapshot(
            progress_json_path,
            dict(
                progress_state,
                percent_complete=(
                    100.0 * progress_state["completed_decoder_evals"] / total_decoder_evals
                    if total_decoder_evals
                    else 0.0
                ),
            ),
        )
        _append_progress_event(
            progress_events_path,
            {"timestamp_epoch": time.time(), "event": "evaluation_failed", "error": repr(exc)},
        )
        raise

    selected_path = results_dir / f"{metadata['run_id']}_selected_questions.csv"
    generations_path = results_dir / f"{metadata['run_id']}_full_generations.csv"
    token_trace_path = results_dir / f"{metadata['run_id']}_token_trace.csv"
    prefix_manual_path = results_dir / f"{metadata['run_id']}_prefix_manual_eval.csv"
    metadata_path = results_dir / f"{metadata['run_id']}_metadata.json"

    chosen_df.to_csv(selected_path, index=False)
    pd.DataFrame(generation_rows).to_csv(generations_path, index=False)
    pd.DataFrame(token_rows).to_csv(token_trace_path, index=False)
    pd.DataFrame(prefix_rows).to_csv(prefix_manual_path, index=False)

    metadata = dict(metadata)
    metadata.update(
        {
            "evaluation_benchmark": "truthfulqa_prefix_probe",
            "truthfulqa_source": truthfulqa_source,
            "truthfulqa_sampling": truthfulqa_manifest,
            "truthfulqa_limit": truthfulqa_limit,
            "question_count": int(total_examples),
            "prefix_words": list(prefix_words),
            "generation_max_new_tokens": int(max_new_tokens),
            "manual_eval_source_csv": str(manual_eval_source_csv),
            "selected_question_groups": chosen_df["selection_group"].value_counts().to_dict(),
            "artifact_paths": {
                "selected_questions_csv": str(selected_path),
                "full_generations_csv": str(generations_path),
                "token_trace_csv": str(token_trace_path),
                "prefix_manual_eval_csv": str(prefix_manual_path),
            },
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        {
            "selected_questions_path": str(selected_path),
            "full_generations_path": str(generations_path),
            "token_trace_path": str(token_trace_path),
            "prefix_manual_eval_path": str(prefix_manual_path),
            "metadata_path": str(metadata_path),
        }
    )


def main():
    args = parse_args()
    rows = read_run_matrix(RUN_MATRIX_PATH)
    selected_rows = iter_selected_runs(rows, args.run_id)
    if args.list:
        print(json.dumps(selected_rows, indent=2))
        return
    if not selected_rows:
        raise SystemExit(f"No run rows matched run_id={args.run_id!r}.")

    manual_eval_df = load_manual_eval(args.manual_eval_csv)

    for row in selected_rows:
        run_id = row["run_id"]
        results_dir = RUNS_DIR / run_id
        evaluator_args = build_evaluator_args(args, row, results_dir)
        max_new_tokens = row_or_cli_int(row, "max_new_tokens", args.max_new_tokens)
        if args.truthfulqa_limit is None and row.get("truthfulqa_limit"):
            evaluator_args.truthfulqa_limit = str(row["truthfulqa_limit"])
        frozen_win_count = row_or_cli_int(row, "frozen_win_count", args.frozen_win_count)
        update1_win_count = row_or_cli_int(row, "update1_win_count", args.update1_win_count)
        both_bad_count = row_or_cli_int(row, "both_bad_count", args.both_bad_count)
        prefix_words = parse_prefix_words(row_or_cli_text(row, "prefix_words", args.prefix_words))
        chosen_df = select_questions(
            manual_eval_df,
            frozen_win_count=frozen_win_count,
            update1_win_count=update1_win_count,
            both_bad_count=both_bad_count,
        )

        print(
            {
                "experiment": EXPERIMENT_NAME,
                "run_id": run_id,
                "results_dir": str(results_dir),
                "decoder_names": list(EXP15_DECODER_NAMES),
                "question_count": int(len(chosen_df)),
                "prefix_words": list(prefix_words),
                "max_new_tokens": max_new_tokens,
                "truthfulqa_limit": evaluator_args.truthfulqa_limit,
                "selected_questions": chosen_df[
                    [
                        "example_idx",
                        "selection_group",
                        "selection_rank_within_group",
                        "frozen_score",
                        "update1_score",
                        "question",
                    ]
                ].to_dict(orient="records"),
            }
        )
        if args.dry_run:
            continue

        from local_evaluator import ExperimentEvaluator

        evaluator = ExperimentEvaluator(evaluator_args, EXP15_DECODER_NAMES)
        metadata = {
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "notes": row.get("notes"),
            "model_name": evaluator_args.model_name,
            "mode": evaluator_args.mode,
            "seed": evaluator_args.seed,
            "strict_eval": evaluator_args.strict_eval,
            "decoder_names": list(EXP15_DECODER_NAMES),
            "hypothesis": (
                "if_frozen_helps_as_an_early_question_level_anchor_but_update1_helps_later_by_avoiding_stale_drift_"
                "then_prefix_level_manual_review_should_show_when_frozen_and_update1_begin_to_diverge"
            ),
            "experiment_note": (
                "targeted_prefix_probe_on_exp14_manual_disagreement_questions_with_full_generation_prefix_slices"
            ),
            "question_selection_counts": {
                "frozen_win_count": frozen_win_count,
                "update1_win_count": update1_win_count,
                "both_bad_count": both_bad_count,
            },
            "refresh_schedule": {
                "exp14_update1": 1,
                "exp14_frozen": "first_step_only_then_hold",
            },
            "prefix_probe_rule": "score_prefixes_of_same_long_generation_instead_of_prompting_separate_short_answers",
        }
        run_prefix_probe_suite(
            evaluator,
            EXP15_DECODER_NAMES,
            evaluator_args,
            results_dir=results_dir,
            metadata=metadata,
            max_new_tokens=max_new_tokens,
            chosen_df=chosen_df,
            prefix_words=prefix_words,
            manual_eval_source_csv=args.manual_eval_csv,
        )


if __name__ == "__main__":
    main()
