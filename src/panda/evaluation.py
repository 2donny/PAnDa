"""Evaluation helpers and benchmark loops for the current decoder set."""

import json
import statistics
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

from .config import DECODER_LABELS
from .core import merge_runtime_summaries
from .prompts import build_truthfulqa_prompt
from .utils import (
    get_decoder_label,
    mean_or_none,
    normalize_text,
    softmax_over_scores,
)


def summarize_trace(trace):
    if not trace:
        return {
            "dyn_steps": 0,
            "avg_alpha": None,
            "alpha_std": None,
            "avg_selected_layer": None,
            "switch_rate": None,
            "avg_instability": None,
            "avg_risk_score": None,
            "trigger_rate": None,
            "avg_jsd_current": None,
            "avg_selection_margin": None,
            "avg_selection_score": None,
            "fallback_rate": None,
            "avg_baseline_margin": None,
            "avg_jacobi_passes": None,
            "avg_jacobi_window_size": None,
            "avg_jacobi_stable_prefix": None,
            "avg_jacobi_commit_len": None,
            "jacobi_convergence_rate": None,
            "panda_disagreement_rate": None,
            "panda_truth_selection_rate": None,
            "avg_panda_divergence": None,
            "avg_panda_safe_confidence": None,
            "avg_panda_truth_confidence": None,
            "avg_panda_agreement_prefix": None,
            "panda_arbitration_rate": None,
        }
    alphas = [float(row["alpha"]) for row in trace if row.get("alpha") is not None]
    layers = [int(row["selected_layer"]) for row in trace if row.get("selected_layer") is not None]
    instabilities = [float(row["instability"]) for row in trace if row.get("instability") is not None]
    switches = sum(int(layers[i] != layers[i - 1]) for i in range(1, len(layers))) if len(layers) > 1 else 0
    switch_rate = switches / max(1, len(layers) - 1) if layers else None
    return {
        "dyn_steps": len(trace),
        "avg_alpha": statistics.mean(alphas) if alphas else None,
        "alpha_std": statistics.pstdev(alphas) if len(alphas) > 1 else 0.0 if alphas else None,
        "avg_selected_layer": statistics.mean(layers) if layers else None,
        "switch_rate": switch_rate,
        "avg_instability": statistics.mean(instabilities) if instabilities else None,
        "avg_risk_score": mean_or_none(row.get("risk_score") for row in trace),
        "trigger_rate": mean_or_none(row.get("risk_triggered") for row in trace),
        "avg_jsd_current": mean_or_none(row.get("jsd_current") for row in trace),
        "avg_selection_margin": mean_or_none(row.get("selection_margin") for row in trace),
        "avg_selection_score": mean_or_none(row.get("selection_score") for row in trace),
        "fallback_rate": mean_or_none(row.get("fallback_used") for row in trace),
        "avg_baseline_margin": mean_or_none(row.get("baseline_margin") for row in trace),
        "avg_jacobi_passes": mean_or_none(row.get("jacobi_passes_used") for row in trace),
        "avg_jacobi_window_size": mean_or_none(row.get("jacobi_window_size") for row in trace),
        "avg_jacobi_stable_prefix": mean_or_none(row.get("jacobi_stable_prefix_len") for row in trace),
        "avg_jacobi_commit_len": mean_or_none(row.get("jacobi_commit_len") for row in trace),
        "jacobi_convergence_rate": mean_or_none(row.get("jacobi_converged") for row in trace),
        "panda_disagreement_rate": mean_or_none(row.get("panda_disagreement") for row in trace),
        "panda_truth_selection_rate": mean_or_none(row.get("panda_selected_truth") for row in trace),
        "avg_panda_divergence": mean_or_none(row.get("panda_divergence") for row in trace),
        "avg_panda_safe_confidence": mean_or_none(row.get("panda_safe_confidence") for row in trace),
        "avg_panda_truth_confidence": mean_or_none(row.get("panda_truth_confidence") for row in trace),
        "avg_panda_agreement_prefix": mean_or_none(row.get("panda_agreement_prefix_len") for row in trace),
        "panda_arbitration_rate": mean_or_none(row.get("panda_arbitration_active") for row in trace),
    }


def compute_choice_score_details(choice_scores, correct_choice=None):
    sorted_rows = sorted(
        ((str(choice), float(score)) for choice, score in choice_scores.items()),
        key=lambda row: row[1],
        reverse=True,
    )
    prediction = sorted_rows[0][0]
    prediction_score = sorted_rows[0][1]
    runner_up_score = sorted_rows[1][1] if len(sorted_rows) > 1 else None
    decision_margin = prediction_score - runner_up_score if runner_up_score is not None else None

    details = {
        "prediction": prediction,
        "prediction_score": prediction_score,
        "runner_up_score": runner_up_score,
        "decision_margin": decision_margin,
        "correct_choice_score": None,
        "best_false_score": None,
        "correct_margin": None,
    }
    if correct_choice is None:
        return details

    correct_choice_norm = normalize_text(correct_choice)
    matching_scores = [score for choice, score in sorted_rows if normalize_text(choice) == correct_choice_norm]
    false_scores = [score for choice, score in sorted_rows if normalize_text(choice) != correct_choice_norm]
    if matching_scores:
        details["correct_choice_score"] = matching_scores[0]
    if false_scores:
        details["best_false_score"] = max(false_scores)
    if details["correct_choice_score"] is not None and details["best_false_score"] is not None:
        details["correct_margin"] = details["correct_choice_score"] - details["best_false_score"]
    return details


def score_choices_with_decoder(evaluator, prompt, choices, decoder_name):
    scored_rows = []
    runtime_summaries = []
    for choice in choices:
        sequence_logprob, trace, runtime_summary = evaluator.score_candidate_with_decoder(prompt, decoder_name, choice)
        scored_rows.append({"choice": choice, "sequence_logprob": sequence_logprob, "trace": trace})
        runtime_summaries.append(runtime_summary)
    best_row = max(scored_rows, key=lambda row: row["sequence_logprob"])
    return scored_rows, best_row["trace"], merge_runtime_summaries(runtime_summaries)


def compute_truthfulqa_mc_metrics(mc1_rows, mc1_labels, mc2_rows, mc2_labels):
    mc1_scores = [row["sequence_logprob"] for row in mc1_rows]
    mc1_true_scores = [score for score, label in zip(mc1_scores, mc1_labels) if int(label) == 1]
    mc1_false_scores = [score for score, label in zip(mc1_scores, mc1_labels) if int(label) == 0]
    if not mc1_true_scores or not mc1_false_scores:
        raise ValueError("TruthfulQA MC1 requires at least one true and one false answer.")
    mc1 = float(max(mc1_true_scores) > max(mc1_false_scores))
    mc1_margin = max(mc1_true_scores) - max(mc1_false_scores)

    mc2_scores = [row["sequence_logprob"] for row in mc2_rows]
    mc2_probs = softmax_over_scores(mc2_scores)
    mc2_true_indices = [idx for idx, label in enumerate(mc2_labels) if int(label) == 1]
    mc2_false_scores = [score for score, label in zip(mc2_scores, mc2_labels) if int(label) == 0]
    if not mc2_true_indices or not mc2_false_scores:
        raise ValueError("TruthfulQA MC2/MC3 requires at least one true and one false answer.")
    mc2 = sum(mc2_probs[idx] for idx in mc2_true_indices)
    false_cutoff = max(mc2_false_scores)
    true_scores = [mc2_scores[idx] for idx in mc2_true_indices]
    mc3 = statistics.mean(float(score > false_cutoff) for score in true_scores)
    mc2_margin = max(true_scores) - false_cutoff
    mc3_margin = statistics.mean(true_scores) - false_cutoff

    sorted_rows = sorted(mc2_rows, key=lambda row: row["sequence_logprob"], reverse=True)
    return {
        "prediction": sorted_rows[0]["choice"],
        "mc1": mc1,
        "mc2": mc2,
        "mc3": mc3,
        "mc1_margin": mc1_margin,
        "mc2_margin": mc2_margin,
        "mc3_margin": mc3_margin,
        "choice_scores": {row["choice"]: row["sequence_logprob"] for row in sorted_rows},
    }


def predict_truthfulqa_mc(evaluator, question, mc1_choices, mc1_labels, mc2_choices, mc2_labels, decoder_name):
    prompt = build_truthfulqa_prompt(question)
    mc1_rows, best_trace, mc1_runtime = score_choices_with_decoder(evaluator, prompt, mc1_choices, decoder_name)
    mc2_rows, _, mc2_runtime = score_choices_with_decoder(evaluator, prompt, mc2_choices, decoder_name)
    metrics = compute_truthfulqa_mc_metrics(mc1_rows, mc1_labels, mc2_rows, mc2_labels)
    extra = {"scoring_mode": "truthfulqa_mc_sequence_logprob", "choice_scores": metrics["choice_scores"]}
    extra.update(merge_runtime_summaries((mc1_runtime, mc2_runtime)))
    return metrics, best_trace, extra


def assert_eval_sources(args, truthfulqa_source):
    if not args.strict_eval:
        return

    expected_truthfulqa = "truthful_qa/multiple_choice"
    if truthfulqa_source != expected_truthfulqa:
        raise RuntimeError(
            f"Strict evaluation requires {expected_truthfulqa} for TruthfulQA, got {truthfulqa_source!r}."
        )


def evaluate_truthfulqa(evaluator, rows, progress_every=1, decoder_names=None, progress_callback=None):
    results = []
    total = len(rows)
    decoder_names = tuple(decoder_names or evaluator.decoder_names)
    total_decoders = len(decoder_names)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "evaluation_started",
                "total_examples": total,
                "total_decoders": total_decoders,
            }
        )
    for example_idx, row in enumerate(rows, start=1):
        if example_idx == 1 or example_idx % progress_every == 0 or example_idx == total:
            print(f"[truthfulqa] example {example_idx}/{total}", flush=True)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "example_started",
                    "example_idx": example_idx,
                    "total_examples": total,
                    "question": row["question"],
                }
            )
        for decoder_idx, decoder_name in enumerate(decoder_names, start=1):
            print(
                f"[truthfulqa] example {example_idx}/{total} decoder {decoder_idx}/{total_decoders} {decoder_name}",
                flush=True,
            )
            metrics, trace, extra = predict_truthfulqa_mc(
                evaluator,
                row["question"],
                row["mc1_choices"],
                row["mc1_labels"],
                row["mc2_choices"],
                row["mc2_labels"],
                decoder_name,
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "decoder_finished",
                        "example_idx": example_idx,
                        "total_examples": total,
                        "decoder_idx": decoder_idx,
                        "total_decoders": total_decoders,
                        "decoder_name": decoder_name,
                        "mc1": float(metrics["mc1"]),
                        "mc2": float(metrics["mc2"]),
                        "mc3": float(metrics["mc3"]),
                        "latency_seconds": float(extra["latency_seconds"]),
                    }
                )
            trace_summary = summarize_trace(trace)
            metric_margins = {
                "mc1": metrics["mc1_margin"],
                "mc2": metrics["mc2_margin"],
                "mc3": metrics["mc3_margin"],
            }
            for metric_name in ("mc1", "mc2", "mc3"):
                result = {
                    "benchmark": "truthfulqa",
                    "metric_name": metric_name,
                    "example_idx": example_idx - 1,
                    "decoder": decoder_name,
                    "decoder_label": get_decoder_label(decoder_name),
                    "question": row["question"],
                    "correct_choice": None,
                    "prediction": metrics["prediction"],
                    "score": float(metrics[metric_name]),
                    "score_detail": f"{metric_name}={metrics[metric_name]:.4f}",
                    "scoring_mode": extra["scoring_mode"],
                    "choice_scores": json.dumps(extra["choice_scores"], ensure_ascii=True),
                    "decision_margin": float(metric_margins[metric_name]),
                    "correct_margin": float(metric_margins[metric_name]),
                    "format_valid": None,
                    "latency_seconds": extra["latency_seconds"],
                    "decoder_steps": extra["decoder_steps"],
                    "forward_passes": extra["forward_passes"],
                    "latency_per_step_ms": extra["latency_per_step_ms"],
                    "latency_per_forward_ms": extra["latency_per_forward_ms"],
                    "steps_per_second": extra["steps_per_second"],
                    "tokens_per_forward": extra["tokens_per_forward"],
                    "factual_speedup": extra["factual_speedup"],
                }
                result.update(trace_summary)
                results.append(result)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "example_finished",
                    "example_idx": example_idx,
                    "total_examples": total,
                }
            )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "evaluation_finished",
                "total_examples": total,
                "total_decoders": total_decoders,
                "result_rows": len(results),
            }
        )
    return results


def build_pairwise_summary(results_df):
    if pd is None:
        return []
    pairwise_rows = []
    score_df = results_df[["benchmark", "metric_name", "example_idx", "decoder_label", "score"]].copy()
    group_columns = ["benchmark", "metric_name"]
    if "sweep_id" in results_df.columns:
        score_df["sweep_id"] = results_df["sweep_id"]
        group_columns = ["sweep_id"] + group_columns
    for group_key, group_df in score_df.groupby(group_columns):
        if len(group_columns) == 3:
            sweep_id, benchmark, metric_name = group_key
        else:
            benchmark, metric_name = group_key
            sweep_id = None
        wide_df = group_df.pivot_table(index="example_idx", columns="decoder_label", values="score", aggfunc="first")
        decoder_labels = list(wide_df.columns)
        for left_label in decoder_labels:
            for right_label in decoder_labels:
                if left_label == right_label:
                    continue
                delta = (wide_df[left_label] - wide_df[right_label]).dropna()
                if delta.empty:
                    continue
                pairwise_rows.append(
                    {
                        "sweep_id": sweep_id,
                        "benchmark": benchmark,
                        "metric_name": metric_name,
                        "left_decoder_label": left_label,
                        "right_decoder_label": right_label,
                        "num_examples": int(delta.shape[0]),
                        "mean_score_delta": float(delta.mean()),
                        "median_score_delta": float(delta.median()),
                        "win_rate": float((delta > 0).mean()),
                        "tie_rate": float((delta == 0).mean()),
                        "loss_rate": float((delta < 0).mean()),
                    }
                )
    if not pairwise_rows:
        return pd.DataFrame()
    sort_columns = ["benchmark", "metric_name", "left_decoder_label", "right_decoder_label"]
    if any(row["sweep_id"] is not None for row in pairwise_rows):
        sort_columns = ["sweep_id"] + sort_columns
    return pd.DataFrame(pairwise_rows).sort_values(sort_columns)


def aggregate_results(results_by_decoder):
    aggregated = {}
    for decoder_name, results in results_by_decoder.items():
        if results:
            aggregated[decoder_name] = {
                metric: mean_or_none([r.get(metric) for r in results if r.get(metric) is not None])
                for metric in set().union(*[set(r.keys()) for r in results])
            }
        else:
            aggregated[decoder_name] = {}
    return aggregated


def save_outputs(output_list, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_list, indent=2, default=str),
        encoding="utf-8",
    )


def save_pairwise_summary(pairwise_summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["decoder", "mean_score", "count", "std_dev"]
    lines = [",".join(header)]
    for row in pairwise_summary:
        lines.append(f"{row['decoder']},{row['mean_score']},{row['count']},{row['std_dev']}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary(results_by_decoder, summary_path, benchmark_name):
    del benchmark_name

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    aggregated = aggregate_results(results_by_decoder)
    all_metrics = sorted({metric for metrics in aggregated.values() for metric in metrics.keys()})

    header = ["decoder"] + all_metrics
    lines = [",".join(header)]
    for decoder_name in sorted(aggregated.keys()):
        metrics = aggregated[decoder_name]
        values = [DECODER_LABELS.get(decoder_name, decoder_name)]
        for metric in all_metrics:
            value = metrics.get(metric)
            if value is None:
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append(",".join(values))

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "aggregate_results",
    "assert_eval_sources",
    "build_pairwise_summary",
    "compute_choice_score_details",
    "compute_truthfulqa_mc_metrics",
    "evaluate_truthfulqa",
    "predict_truthfulqa_mc",
    "save_outputs",
    "save_pairwise_summary",
    "save_summary",
    "score_choices_with_decoder",
    "summarize_trace",
]
