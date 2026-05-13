"""Evaluation helpers and benchmark loops for the current decoder set."""

import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

from .config import DEFAULT_ALPACAEVAL_SOURCE, DEFAULT_STRATEGYQA_SOURCE, DECODER_LABELS
from .evaluator import merge_runtime_summaries
from .prompts import (
    build_gsm8k_prompt,
    build_gsm8k_sequence_prompt,
    build_pairwise_preference_prompt,
    build_strategyqa_prompt,
    build_truthfulqa_prompt,
)
from .utils import (
    canonicalize_number_text,
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


def query_pairwise_candidate_preference(evaluator, prompt, candidate_a, candidate_b):
    preference_prompt = build_pairwise_preference_prompt(prompt, candidate_a, candidate_b)
    label_choices = ["A", "B"]
    scored_rows, trace, runtime_summary = score_choices_with_decoder(
        evaluator,
        preference_prompt,
        label_choices,
        "greedy",
    )
    sorted_rows = sorted(scored_rows, key=lambda row: row["sequence_logprob"], reverse=True)
    scores = [row["sequence_logprob"] for row in scored_rows]
    probs = softmax_over_scores(scores)
    prob_map = {row["choice"]: float(prob) for row, prob in zip(scored_rows, probs)}
    score_map = {row["choice"]: row["sequence_logprob"] for row in scored_rows}
    selected_choice = sorted_rows[0]["choice"]
    return {
        "selected_choice": selected_choice,
        "prob_a": prob_map["A"],
        "prob_b": prob_map["B"],
        "prob_tie": 0.0,
        "choice_scores": score_map,
        "choice_probs": prob_map,
        "trace": trace,
        "runtime_summary": runtime_summary,
    }


def choose_pairwise_rerank_branch(
    evaluator,
    prompt,
    candidate_a_prediction,
    candidate_b_prediction,
    candidate_a_decoder,
    candidate_b_decoder,
    default_decoder,
):
    candidate_a_norm = normalize_text(candidate_a_prediction)
    candidate_b_norm = normalize_text(candidate_b_prediction)
    same_prediction = candidate_a_norm == candidate_b_norm
    confidence_base = None
    confidence_fixed = None
    confidence_base_raw = None
    confidence_fixed_raw = None
    confidence_base_valid = None
    confidence_fixed_valid = None
    confidence_runtime = None
    pairwise_choice = None
    pairwise_choice_scores = None
    pairwise_choice_probs = None
    pairwise_tie_prob = None

    selected_decoder = default_decoder
    if not same_prediction:
        preference_query = query_pairwise_candidate_preference(
            evaluator,
            prompt,
            candidate_a_prediction,
            candidate_b_prediction,
        )
        confidence_base = float(preference_query["prob_a"])
        confidence_fixed = float(preference_query["prob_b"])
        confidence_base_raw = f"{confidence_base:.6f}"
        confidence_fixed_raw = f"{confidence_fixed:.6f}"
        confidence_base_valid = 1.0
        confidence_fixed_valid = 1.0
        confidence_runtime = preference_query["runtime_summary"]
        pairwise_choice = preference_query["selected_choice"]
        pairwise_choice_scores = json.dumps(preference_query["choice_scores"], ensure_ascii=True)
        pairwise_choice_probs = json.dumps(preference_query["choice_probs"], ensure_ascii=True)
        pairwise_tie_prob = float(preference_query["prob_tie"])
        if pairwise_choice == "A":
            selected_decoder = candidate_a_decoder
        elif pairwise_choice == "B":
            selected_decoder = candidate_b_decoder

    return {
        "selected_decoder": selected_decoder,
        "selected_base": float(selected_decoder == candidate_a_decoder),
        "same_prediction": float(same_prediction),
        "confidence_base": confidence_base,
        "confidence_fixed_alpha": confidence_fixed,
        "confidence_base_raw": confidence_base_raw,
        "confidence_fixed_alpha_raw": confidence_fixed_raw,
        "confidence_base_valid": confidence_base_valid,
        "confidence_fixed_alpha_valid": confidence_fixed_valid,
        "confidence_runtime": confidence_runtime,
        "pairwise_choice": pairwise_choice,
        "pairwise_choice_scores": pairwise_choice_scores,
        "pairwise_choice_probs": pairwise_choice_probs,
        "pairwise_tie_prob": pairwise_tie_prob,
    }


def choose_tbasco_branch(evaluator, prompt, low_prediction, high_prediction):
    return choose_pairwise_rerank_branch(
        evaluator,
        prompt,
        low_prediction,
        high_prediction,
        "tbasco_low",
        "tbasco_high",
        "tbasco_low",
    )


def generate_with_tbasco_decoder(evaluator, prompt, max_new_tokens=96, stop_on_eos=True):
    low_prediction, low_trace, low_runtime = evaluator.generate_with_decoder(
        prompt,
        "tbasco_low",
        max_new_tokens=max_new_tokens,
        stop_on_eos=stop_on_eos,
    )
    high_prediction, high_trace, high_runtime = evaluator.generate_with_decoder(
        prompt,
        "tbasco_high",
        max_new_tokens=max_new_tokens,
        stop_on_eos=stop_on_eos,
    )
    branch = choose_tbasco_branch(
        evaluator,
        prompt,
        low_prediction,
        high_prediction,
    )
    use_low = branch["selected_decoder"] == "tbasco_low"
    prediction = low_prediction if use_low else high_prediction
    trace = low_trace if use_low else high_trace
    extra = {
        "scoring_mode": "generation_tbasco_rerank",
        "choice_scores": None,
        "tbasco_selected_decoder": branch["selected_decoder"],
        "tbasco_selected_low": branch["selected_base"],
        "tbasco_same_prediction": branch["same_prediction"],
        "tbasco_confidence_low": branch["confidence_base"],
        "tbasco_confidence_high": branch["confidence_fixed_alpha"],
        "tbasco_confidence_low_valid": branch["confidence_base_valid"],
        "tbasco_confidence_high_valid": branch["confidence_fixed_alpha_valid"],
        "tbasco_confidence_low_raw": branch["confidence_base_raw"],
        "tbasco_confidence_high_raw": branch["confidence_fixed_alpha_raw"],
        "tbasco_pairwise_choice": branch["pairwise_choice"],
        "tbasco_pairwise_choice_scores": branch["pairwise_choice_scores"],
        "tbasco_pairwise_choice_probs": branch["pairwise_choice_probs"],
        "tbasco_pairwise_tie_prob": branch["pairwise_tie_prob"],
    }
    extra.update(
        merge_runtime_summaries(
            (low_runtime, high_runtime, branch["confidence_runtime"])
        )
    )
    return prediction, trace, extra


def predict_choice(evaluator, question, choices, decoder_name, prompt_builder):
    prompt = prompt_builder(question)
    if decoder_name == "tbasco":
        low_rows, low_trace, low_runtime = score_choices_with_decoder(
            evaluator, prompt, choices, "tbasco_low"
        )
        high_rows, high_trace, high_runtime = score_choices_with_decoder(
            evaluator, prompt, choices, "tbasco_high"
        )
        low_sorted_rows = sorted(low_rows, key=lambda row: row["sequence_logprob"], reverse=True)
        high_sorted_rows = sorted(high_rows, key=lambda row: row["sequence_logprob"], reverse=True)
        branch = choose_tbasco_branch(
            evaluator,
            prompt,
            low_sorted_rows[0]["choice"],
            high_sorted_rows[0]["choice"],
        )
        use_low = branch["selected_decoder"] == "tbasco_low"
        chosen_rows = low_sorted_rows if use_low else high_sorted_rows
        best_trace = low_trace if use_low else high_trace
        runtime_summary = merge_runtime_summaries(
            (low_runtime, high_runtime, branch["confidence_runtime"])
        )
        prediction = chosen_rows[0]["choice"]
        choice_scores = {row["choice"]: row["sequence_logprob"] for row in chosen_rows}
        extra = {
            "scoring_mode": "choice_sequence_logprob_tbasco_rerank",
            "choice_scores": choice_scores,
            "tbasco_selected_decoder": branch["selected_decoder"],
            "tbasco_selected_low": branch["selected_base"],
            "tbasco_same_prediction": branch["same_prediction"],
            "tbasco_confidence_low": branch["confidence_base"],
            "tbasco_confidence_high": branch["confidence_fixed_alpha"],
            "tbasco_confidence_low_valid": branch["confidence_base_valid"],
            "tbasco_confidence_high_valid": branch["confidence_fixed_alpha_valid"],
            "tbasco_confidence_low_raw": branch["confidence_base_raw"],
            "tbasco_confidence_high_raw": branch["confidence_fixed_alpha_raw"],
            "tbasco_pairwise_choice": branch["pairwise_choice"],
            "tbasco_pairwise_choice_scores": branch["pairwise_choice_scores"],
            "tbasco_pairwise_choice_probs": branch["pairwise_choice_probs"],
            "tbasco_pairwise_tie_prob": branch["pairwise_tie_prob"],
        }
        extra.update(runtime_summary)
        return prediction, best_trace, extra
    scored_rows, best_trace, runtime_summary = score_choices_with_decoder(evaluator, prompt, choices, decoder_name)
    sorted_rows = sorted(scored_rows, key=lambda row: row["sequence_logprob"], reverse=True)
    prediction = sorted_rows[0]["choice"]
    choice_scores = {row["choice"]: row["sequence_logprob"] for row in sorted_rows}
    extra = {"scoring_mode": "choice_sequence_logprob", "choice_scores": choice_scores}
    extra.update(runtime_summary)
    return prediction, best_trace, extra


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
    if decoder_name == "tbasco":
        mc1_rows_low, mc1_trace_low, mc1_runtime_low = score_choices_with_decoder(
            evaluator, prompt, mc1_choices, "tbasco_low"
        )
        mc2_rows_low, _, mc2_runtime_low = score_choices_with_decoder(
            evaluator, prompt, mc2_choices, "tbasco_low"
        )
        metrics_low = compute_truthfulqa_mc_metrics(mc1_rows_low, mc1_labels, mc2_rows_low, mc2_labels)

        mc1_rows_high, mc1_trace_high, mc1_runtime_high = score_choices_with_decoder(
            evaluator, prompt, mc1_choices, "tbasco_high"
        )
        mc2_rows_high, _, mc2_runtime_high = score_choices_with_decoder(
            evaluator, prompt, mc2_choices, "tbasco_high"
        )
        metrics_high = compute_truthfulqa_mc_metrics(mc1_rows_high, mc1_labels, mc2_rows_high, mc2_labels)

        if normalize_text(metrics_low["prediction"]) == normalize_text(metrics_high["prediction"]):
            branch = {
                "selected_decoder": "tbasco_high",
                "selected_base": 0.0,
                "same_prediction": 1.0,
                "confidence_base": None,
                "confidence_fixed_alpha": None,
                "confidence_base_raw": None,
                "confidence_fixed_alpha_raw": None,
                "confidence_base_valid": None,
                "confidence_fixed_alpha_valid": None,
                "confidence_runtime": None,
                "pairwise_choice": None,
                "pairwise_choice_scores": None,
                "pairwise_choice_probs": None,
                "pairwise_tie_prob": None,
            }
        else:
            branch = choose_tbasco_branch(
                evaluator,
                prompt,
                metrics_low["prediction"],
                metrics_high["prediction"],
            )
        use_low = branch["selected_decoder"] == "tbasco_low"
        metrics = metrics_low if use_low else metrics_high
        best_trace = mc1_trace_low if use_low else mc1_trace_high
        extra = {
            "scoring_mode": "truthfulqa_mc_sequence_logprob_tbasco_rerank",
            "choice_scores": metrics["choice_scores"],
            "tbasco_selected_decoder": branch["selected_decoder"],
            "tbasco_selected_low": branch["selected_base"],
            "tbasco_same_prediction": branch["same_prediction"],
            "tbasco_confidence_low": branch["confidence_base"],
            "tbasco_confidence_high": branch["confidence_fixed_alpha"],
            "tbasco_confidence_low_valid": branch["confidence_base_valid"],
            "tbasco_confidence_high_valid": branch["confidence_fixed_alpha_valid"],
            "tbasco_confidence_low_raw": branch["confidence_base_raw"],
            "tbasco_confidence_high_raw": branch["confidence_fixed_alpha_raw"],
            "tbasco_pairwise_choice": branch["pairwise_choice"],
            "tbasco_pairwise_choice_scores": branch["pairwise_choice_scores"],
            "tbasco_pairwise_choice_probs": branch["pairwise_choice_probs"],
            "tbasco_pairwise_tie_prob": branch["pairwise_tie_prob"],
        }
        extra.update(
            merge_runtime_summaries(
                (
                    mc1_runtime_low,
                    mc2_runtime_low,
                    mc1_runtime_high,
                    mc2_runtime_high,
                    branch["confidence_runtime"],
                )
            )
        )
        return metrics, best_trace, extra
    mc1_rows, best_trace, mc1_runtime = score_choices_with_decoder(evaluator, prompt, mc1_choices, decoder_name)
    mc2_rows, _, mc2_runtime = score_choices_with_decoder(evaluator, prompt, mc2_choices, decoder_name)
    metrics = compute_truthfulqa_mc_metrics(mc1_rows, mc1_labels, mc2_rows, mc2_labels)
    extra = {"scoring_mode": "truthfulqa_mc_sequence_logprob", "choice_scores": metrics["choice_scores"]}
    extra.update(merge_runtime_summaries((mc1_runtime, mc2_runtime)))
    return metrics, best_trace, extra


def predict_gsm8k(evaluator, question, decoder_name):
    prompt = build_gsm8k_prompt(question)
    if decoder_name == "tbasco":
        return generate_with_tbasco_decoder(evaluator, prompt, max_new_tokens=96, stop_on_eos=True)
    prediction, trace, runtime_summary = evaluator.generate_with_decoder(prompt, decoder_name)
    extra = {"scoring_mode": "strict_generation", "choice_scores": None}
    extra.update(runtime_summary)
    return prediction, trace, extra


def predict_gsm8k_sequence(evaluator, question, decoder_name, max_new_tokens):
    prompt = build_gsm8k_sequence_prompt(question)
    if decoder_name == "tbasco":
        return generate_with_tbasco_decoder(
            evaluator,
            prompt,
            max_new_tokens=max_new_tokens,
            stop_on_eos=True,
        )
    prediction, trace, runtime_summary = evaluator.generate_with_decoder(
        prompt,
        decoder_name,
        max_new_tokens=max_new_tokens,
    )
    extra = {"scoring_mode": "sequence_generation", "choice_scores": None}
    extra.update(runtime_summary)
    return prediction, trace, extra


def assert_eval_sources(args, truthfulqa_source, strategyqa_source, gsm8k_source, alpacaeval_source="disabled"):
    if not args.strict_eval:
        return

    expected_truthfulqa = "truthful_qa/multiple_choice"
    expected_gsm8k = "gsm8k/main"

    if not args.skip_truthfulqa and truthfulqa_source != expected_truthfulqa:
        raise RuntimeError(
            f"Strict evaluation requires {expected_truthfulqa} for TruthfulQA, got {truthfulqa_source!r}."
        )
    if not args.skip_gsm8k and gsm8k_source != expected_gsm8k:
        raise RuntimeError(f"Strict evaluation requires {expected_gsm8k} for GSM8K, got {gsm8k_source!r}.")
    if not args.skip_strategyqa and strategyqa_source != DEFAULT_STRATEGYQA_SOURCE:
        raise RuntimeError(
            f"Strict evaluation requires {DEFAULT_STRATEGYQA_SOURCE} for StrategyQA, "
            f"got {strategyqa_source!r}."
        )
    if args.include_alpacaeval and alpacaeval_source != DEFAULT_ALPACAEVAL_SOURCE:
        raise RuntimeError(
            f"Strict evaluation requires {DEFAULT_ALPACAEVAL_SOURCE} for AlpacaEval, "
            f"got {alpacaeval_source!r}."
        )


def export_alpacaeval_outputs(evaluator, rows, decoder_names, results_dir, artifact_prefix):
    results_dir.mkdir(parents=True, exist_ok=True)
    reference_rows = [
        {
            "instruction": row["instruction"],
            "output": row["reference_output"],
            "generator": row["reference_generator"],
            "dataset": row["dataset_name"],
        }
        for row in rows
    ]
    reference_path = results_dir / f"{artifact_prefix}_alpacaeval_reference_outputs.json"
    reference_path.write_text(json.dumps(reference_rows, indent=2, ensure_ascii=True), encoding="utf-8")

    generated_paths = {}
    for decoder_name in decoder_names:
        exported_rows = []
        for example_idx, row in enumerate(rows, start=1):
            if example_idx == 1 or example_idx == len(rows):
                print(f"[alpacaeval:{decoder_name}] {example_idx}/{len(rows)}")
            if decoder_name == "tbasco":
                output_text, _, extra = generate_with_tbasco_decoder(
                    evaluator,
                    row["instruction"],
                    max_new_tokens=evaluator.alpacaeval_max_new_tokens,
                    stop_on_eos=True,
                )
            else:
                output_text, _, extra = evaluator.generate_with_decoder(
                    row["instruction"],
                    decoder_name,
                    max_new_tokens=evaluator.alpacaeval_max_new_tokens,
                    stop_on_eos=True,
                )
            exported_rows.append(
                {
                    "instruction": row["instruction"],
                    "output": output_text,
                    "generator": get_decoder_label(decoder_name),
                    "dataset": row["dataset_name"],
                    "latency_seconds": extra.get("latency_seconds"),
                }
            )
        output_path = results_dir / f"{artifact_prefix}_alpacaeval_{decoder_name}_model_outputs.json"
        output_path.write_text(json.dumps(exported_rows, indent=2, ensure_ascii=True), encoding="utf-8")
        generated_paths[decoder_name] = str(output_path)

    command_lines = [
        "# Official AlpacaEval 2.0 scoring uses the alpaca_eval package.",
        "# Default annotator: weighted_alpaca_eval_gpt4_turbo",
    ]
    for decoder_name, output_path in generated_paths.items():
        command_lines.append(
            "alpaca_eval "
            f"--model_outputs '{output_path}' "
            f"--reference_outputs '{reference_path}' "
            f"--output_path '{results_dir / f'{artifact_prefix}_alpacaeval_{decoder_name}_scored'}'"
        )
    command_path = results_dir / f"{artifact_prefix}_alpacaeval_commands.txt"
    command_path.write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    return {
        "reference_outputs_path": str(reference_path),
        "model_outputs_paths": generated_paths,
        "command_path": str(command_path),
        "official_scorer_installed": importlib.util.find_spec("alpaca_eval") is not None,
    }


def evaluate_truthfulqa(evaluator, rows, progress_every=1, decoder_names=None):
    results = []
    total = len(rows)
    decoder_names = tuple(decoder_names or evaluator.decoder_names)
    for example_idx, row in enumerate(rows, start=1):
        if example_idx == 1 or example_idx % progress_every == 0 or example_idx == total:
            print(f"[truthfulqa] {example_idx}/{total}")
        for decoder_name in decoder_names:
            metrics, trace, extra = predict_truthfulqa_mc(
                evaluator,
                row["question"],
                row["mc1_choices"],
                row["mc1_labels"],
                row["mc2_choices"],
                row["mc2_labels"],
                decoder_name,
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
                    "tbasco_selected_decoder": extra.get("tbasco_selected_decoder"),
                    "tbasco_selected_low": extra.get("tbasco_selected_low"),
                    "tbasco_same_prediction": extra.get("tbasco_same_prediction"),
                    "tbasco_confidence_low": extra.get("tbasco_confidence_low"),
                    "tbasco_confidence_high": extra.get("tbasco_confidence_high"),
                    "tbasco_confidence_low_valid": extra.get("tbasco_confidence_low_valid"),
                    "tbasco_confidence_high_valid": extra.get("tbasco_confidence_high_valid"),
                    "tbasco_confidence_low_raw": extra.get("tbasco_confidence_low_raw"),
                    "tbasco_confidence_high_raw": extra.get("tbasco_confidence_high_raw"),
                    "tbasco_pairwise_choice": extra.get("tbasco_pairwise_choice"),
                    "tbasco_pairwise_choice_scores": extra.get("tbasco_pairwise_choice_scores"),
                    "tbasco_pairwise_choice_probs": extra.get("tbasco_pairwise_choice_probs"),
                    "tbasco_pairwise_tie_prob": extra.get("tbasco_pairwise_tie_prob"),
                }
                result.update(trace_summary)
                results.append(result)
    return results


def evaluate_strategyqa(evaluator, rows, progress_every=1, decoder_names=None):
    results = []
    total = len(rows)
    decoder_names = tuple(decoder_names or evaluator.decoder_names)
    for example_idx, row in enumerate(rows, start=1):
        if example_idx == 1 or example_idx % progress_every == 0 or example_idx == total:
            print(f"[strategyqa] {example_idx}/{total}")
        for decoder_name in decoder_names:
            prediction, trace, extra = predict_choice(
                evaluator, row["question"], row["choices"], decoder_name, build_strategyqa_prompt
            )
            score_details = compute_choice_score_details(extra["choice_scores"], row["correct_choice"])
            score = float(normalize_text(prediction) == normalize_text(row["correct_choice"]))
            result = {
                "benchmark": "strategyqa",
                "metric_name": "accuracy",
                "example_idx": example_idx - 1,
                "decoder": decoder_name,
                "decoder_label": get_decoder_label(decoder_name),
                "question": row["question"],
                "correct_choice": row["correct_choice"],
                "prediction": prediction,
                "score": score,
                "score_detail": f"pred={normalize_text(prediction)} gold={normalize_text(row['correct_choice'])}",
                "scoring_mode": extra["scoring_mode"],
                "choice_scores": json.dumps(extra["choice_scores"], ensure_ascii=True),
                "decision_margin": score_details["decision_margin"],
                "correct_margin": score_details["correct_margin"],
                "format_valid": None,
                "latency_seconds": extra["latency_seconds"],
                "decoder_steps": extra["decoder_steps"],
                "forward_passes": extra["forward_passes"],
                "latency_per_step_ms": extra["latency_per_step_ms"],
                "latency_per_forward_ms": extra["latency_per_forward_ms"],
                "steps_per_second": extra["steps_per_second"],
                "tokens_per_forward": extra["tokens_per_forward"],
                "factual_speedup": extra["factual_speedup"],
                "tbasco_selected_decoder": extra.get("tbasco_selected_decoder"),
                "tbasco_selected_low": extra.get("tbasco_selected_low"),
                "tbasco_same_prediction": extra.get("tbasco_same_prediction"),
                "tbasco_confidence_low": extra.get("tbasco_confidence_low"),
                "tbasco_confidence_high": extra.get("tbasco_confidence_high"),
                "tbasco_confidence_low_valid": extra.get("tbasco_confidence_low_valid"),
                "tbasco_confidence_high_valid": extra.get("tbasco_confidence_high_valid"),
                "tbasco_confidence_low_raw": extra.get("tbasco_confidence_low_raw"),
                "tbasco_confidence_high_raw": extra.get("tbasco_confidence_high_raw"),
                "tbasco_pairwise_choice": extra.get("tbasco_pairwise_choice"),
                "tbasco_pairwise_choice_scores": extra.get("tbasco_pairwise_choice_scores"),
                "tbasco_pairwise_choice_probs": extra.get("tbasco_pairwise_choice_probs"),
                "tbasco_pairwise_tie_prob": extra.get("tbasco_pairwise_tie_prob"),
            }
            result.update(summarize_trace(trace))
            results.append(result)
    return results


def evaluate_halueval(evaluator, rows, progress_every=1, decoder_names=None):
    results = []
    total = len(rows)
    decoder_names = tuple(decoder_names or evaluator.decoder_names)
    for example_idx, row in enumerate(rows, start=1):
        if example_idx == 1 or example_idx % progress_every == 0 or example_idx == total:
            print(f"[{row['benchmark']}] {example_idx}/{total}")
        for decoder_name in decoder_names:
            prediction, trace, extra = predict_choice(
                evaluator,
                row["question"],
                row["choices"],
                decoder_name,
                lambda text: text,
            )
            score_details = compute_choice_score_details(extra["choice_scores"], row["correct_choice"])
            score = float(normalize_text(prediction) == normalize_text(row["correct_choice"]))
            result = {
                "benchmark": row["benchmark"],
                "metric_name": "accuracy",
                "example_idx": example_idx - 1,
                "decoder": decoder_name,
                "decoder_label": get_decoder_label(decoder_name),
                "question": row["question"],
                "correct_choice": row["correct_choice"],
                "prediction": prediction,
                "halueval_yes_label": row.get("halueval_yes_label"),
                "halueval_no_label": row.get("halueval_no_label"),
                "halueval_has_hallucination": row.get("halueval_has_hallucination"),
                "score": score,
                "score_detail": f"pred={normalize_text(prediction)} gold={normalize_text(row['correct_choice'])}",
                "scoring_mode": extra["scoring_mode"],
                "choice_scores": json.dumps(extra["choice_scores"], ensure_ascii=True),
                "decision_margin": score_details["decision_margin"],
                "correct_margin": score_details["correct_margin"],
                "format_valid": None,
                "latency_seconds": extra["latency_seconds"],
                "decoder_steps": extra["decoder_steps"],
                "forward_passes": extra["forward_passes"],
                "latency_per_step_ms": extra["latency_per_step_ms"],
                "latency_per_forward_ms": extra["latency_per_forward_ms"],
                "steps_per_second": extra["steps_per_second"],
                "tokens_per_forward": extra["tokens_per_forward"],
                "factual_speedup": extra["factual_speedup"],
                "tbasco_selected_decoder": extra.get("tbasco_selected_decoder"),
                "tbasco_selected_low": extra.get("tbasco_selected_low"),
                "tbasco_same_prediction": extra.get("tbasco_same_prediction"),
                "tbasco_confidence_low": extra.get("tbasco_confidence_low"),
                "tbasco_confidence_high": extra.get("tbasco_confidence_high"),
                "tbasco_confidence_low_valid": extra.get("tbasco_confidence_low_valid"),
                "tbasco_confidence_high_valid": extra.get("tbasco_confidence_high_valid"),
                "tbasco_confidence_low_raw": extra.get("tbasco_confidence_low_raw"),
                "tbasco_confidence_high_raw": extra.get("tbasco_confidence_high_raw"),
                "tbasco_pairwise_choice": extra.get("tbasco_pairwise_choice"),
                "tbasco_pairwise_choice_scores": extra.get("tbasco_pairwise_choice_scores"),
                "tbasco_pairwise_choice_probs": extra.get("tbasco_pairwise_choice_probs"),
                "tbasco_pairwise_tie_prob": extra.get("tbasco_pairwise_tie_prob"),
            }
            result.update(summarize_trace(trace))
            results.append(result)
    return results


def evaluate_gsm8k(evaluator, rows, progress_every=1, decoder_names=None):
    results = []
    total = len(rows)
    decoder_names = tuple(decoder_names or evaluator.decoder_names)
    for example_idx, row in enumerate(rows, start=1):
        if example_idx == 1 or example_idx % progress_every == 0 or example_idx == total:
            print(f"[gsm8k] {example_idx}/{total}")
        for decoder_name in decoder_names:
            prediction, trace, extra = predict_gsm8k(evaluator, row["question"], decoder_name)
            pred_num = canonicalize_number_text(prediction)
            gold_num = canonicalize_number_text(row["correct_choice"])
            score = float(pred_num == gold_num)
            format_valid = float(pred_num is not None)
            result = {
                "benchmark": "gsm8k",
                "metric_name": "accuracy",
                "example_idx": example_idx - 1,
                "decoder": decoder_name,
                "decoder_label": get_decoder_label(decoder_name),
                "question": row["question"],
                "correct_choice": row["correct_choice"],
                "prediction": prediction,
                "score": score,
                "score_detail": f"pred={pred_num} gold={gold_num} format_valid={bool(format_valid)}",
                "scoring_mode": extra["scoring_mode"],
                "choice_scores": None,
                "decision_margin": None,
                "correct_margin": None,
                "format_valid": format_valid,
                "latency_seconds": extra["latency_seconds"],
                "decoder_steps": extra["decoder_steps"],
                "forward_passes": extra["forward_passes"],
                "latency_per_step_ms": extra["latency_per_step_ms"],
                "latency_per_forward_ms": extra["latency_per_forward_ms"],
                "steps_per_second": extra["steps_per_second"],
                "tokens_per_forward": extra["tokens_per_forward"],
                "factual_speedup": extra["factual_speedup"],
                "tbasco_selected_decoder": extra.get("tbasco_selected_decoder"),
                "tbasco_selected_low": extra.get("tbasco_selected_low"),
                "tbasco_same_prediction": extra.get("tbasco_same_prediction"),
                "tbasco_confidence_low": extra.get("tbasco_confidence_low"),
                "tbasco_confidence_high": extra.get("tbasco_confidence_high"),
                "tbasco_confidence_low_valid": extra.get("tbasco_confidence_low_valid"),
                "tbasco_confidence_high_valid": extra.get("tbasco_confidence_high_valid"),
                "tbasco_confidence_low_raw": extra.get("tbasco_confidence_low_raw"),
                "tbasco_confidence_high_raw": extra.get("tbasco_confidence_high_raw"),
                "tbasco_pairwise_choice": extra.get("tbasco_pairwise_choice"),
                "tbasco_pairwise_choice_scores": extra.get("tbasco_pairwise_choice_scores"),
                "tbasco_pairwise_choice_probs": extra.get("tbasco_pairwise_choice_probs"),
                "tbasco_pairwise_tie_prob": extra.get("tbasco_pairwise_tie_prob"),
            }
            result.update(summarize_trace(trace))
            results.append(result)
    return results


def evaluate_gsm8k_sequence(evaluator, rows, progress_every=1, decoder_names=None, max_new_tokens=160):
    results = []
    total = len(rows)
    decoder_names = tuple(decoder_names or evaluator.decoder_names)
    for example_idx, row in enumerate(rows, start=1):
        if example_idx == 1 or example_idx % progress_every == 0 or example_idx == total:
            print(f"[gsm8k_sequence] {example_idx}/{total}")
        for decoder_name in decoder_names:
            prediction, trace, extra = predict_gsm8k_sequence(
                evaluator,
                row["question"],
                decoder_name,
                max_new_tokens=max_new_tokens,
            )
            pred_num = canonicalize_number_text(prediction)
            gold_num = canonicalize_number_text(row["correct_choice"])
            score = float(pred_num == gold_num)
            format_valid = float(pred_num is not None)
            result = {
                "benchmark": "gsm8k_sequence",
                "metric_name": "accuracy",
                "example_idx": example_idx - 1,
                "decoder": decoder_name,
                "decoder_label": get_decoder_label(decoder_name),
                "question": row["question"],
                "correct_choice": row["correct_choice"],
                "prediction": prediction,
                "score": score,
                "score_detail": f"pred={pred_num} gold={gold_num} format_valid={bool(format_valid)}",
                "scoring_mode": extra["scoring_mode"],
                "choice_scores": None,
                "decision_margin": None,
                "correct_margin": None,
                "format_valid": format_valid,
                "latency_seconds": extra["latency_seconds"],
                "decoder_steps": extra["decoder_steps"],
                "forward_passes": extra["forward_passes"],
                "latency_per_step_ms": extra["latency_per_step_ms"],
                "latency_per_forward_ms": extra["latency_per_forward_ms"],
                "steps_per_second": extra["steps_per_second"],
                "tokens_per_forward": extra["tokens_per_forward"],
                "factual_speedup": extra["factual_speedup"],
                "tbasco_selected_decoder": extra.get("tbasco_selected_decoder"),
                "tbasco_selected_low": extra.get("tbasco_selected_low"),
                "tbasco_same_prediction": extra.get("tbasco_same_prediction"),
                "tbasco_confidence_low": extra.get("tbasco_confidence_low"),
                "tbasco_confidence_high": extra.get("tbasco_confidence_high"),
                "tbasco_confidence_low_valid": extra.get("tbasco_confidence_low_valid"),
                "tbasco_confidence_high_valid": extra.get("tbasco_confidence_high_valid"),
                "tbasco_confidence_low_raw": extra.get("tbasco_confidence_low_raw"),
                "tbasco_confidence_high_raw": extra.get("tbasco_confidence_high_raw"),
                "tbasco_pairwise_choice": extra.get("tbasco_pairwise_choice"),
                "tbasco_pairwise_choice_scores": extra.get("tbasco_pairwise_choice_scores"),
                "tbasco_pairwise_choice_probs": extra.get("tbasco_pairwise_choice_probs"),
                "tbasco_pairwise_tie_prob": extra.get("tbasco_pairwise_tie_prob"),
            }
            result.update(summarize_trace(trace))
            results.append(result)
    return results


def evaluate_alpacaeval(evaluator, rows, decoder_names, max_new_tokens=2048):
    pairwise_results = defaultdict(dict)
    outputs = []

    for example_idx, row in enumerate(rows):
        source_idx = row.get("source_idx", example_idx)
        for decoder_name in decoder_names:
            if decoder_name == "tbasco":
                candidate_output, _, _ = generate_with_tbasco_decoder(
                    evaluator,
                    row["instruction"],
                    max_new_tokens=max_new_tokens,
                    stop_on_eos=True,
                )
            else:
                candidate_output, _, _ = evaluator.generate_with_decoder(
                    row["instruction"],
                    decoder_name,
                    max_new_tokens=max_new_tokens,
                    stop_on_eos=True,
                )

            outputs.append(
                {
                    "instruction": row["instruction"],
                    "output": candidate_output,
                    "generator": decoder_name,
                    "dataset": row.get("dataset_name", "alpaca_eval"),
                    "source_index": source_idx,
                }
            )
            pairwise_results[decoder_name][source_idx] = {
                "candidate": candidate_output,
                "reference": row["reference_output"],
            }

    return pairwise_results, outputs


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
    "choose_pairwise_rerank_branch",
    "choose_tbasco_branch",
    "compute_choice_score_details",
    "compute_truthfulqa_mc_metrics",
    "evaluate_alpacaeval",
    "evaluate_gsm8k",
    "evaluate_gsm8k_sequence",
    "evaluate_halueval",
    "evaluate_strategyqa",
    "evaluate_truthfulqa",
    "export_alpacaeval_outputs",
    "generate_with_tbasco_decoder",
    "predict_choice",
    "predict_gsm8k",
    "predict_gsm8k_sequence",
    "predict_truthfulqa_mc",
    "query_pairwise_candidate_preference",
    "save_outputs",
    "save_pairwise_summary",
    "save_summary",
    "score_choices_with_decoder",
    "summarize_trace",
]
