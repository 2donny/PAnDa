"""Experiment-local decoder variants kept outside the main package."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn.functional as F

from panda.import_shims import suppress_problematic_optional_dependency_detection

suppress_problematic_optional_dependency_detection()

try:
    from transformers import AutoModelForCausalLM
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    if exc.name == "transformers":
        raise SystemExit(
            "Missing Python dependency 'transformers'. "
            "This repo has a project virtualenv, so run experiment scripts with "
            "'./.venv/bin/python ...' from the repo root, or install the project dependencies first."
        ) from exc
    raise

from panda.core import make_runtime_summary
from panda.evaluator import Stage4Evaluator


EXP1_DECODER_NAMES = (
    "official_dola",
    "matched_alpha_dola_0p10",
    "matched_alpha_dola_0p50",
    "matched_alpha_dola_0p95",
)

EXP2_DECODER_NAMES = (
    "simple_panda_h1",
    "simple_panda_h2",
)

EXP5_PANDA_WINDOW_SIZES = {
    "panda_w4": 4,
    "panda_w2": 2,
    "panda_w1": 1,
}

EXP5_SIMPLE_ALIAS = {
    "panda_no_block_h1": "simple_panda_h1",
}

EXP5_DECODER_NAMES = tuple(EXP5_PANDA_WINDOW_SIZES) + tuple(EXP5_SIMPLE_ALIAS)

EXP6_DECODER_NAMES = (
    "pure_argmax_switch",
    "guarded_argmax_switch",
    "sticky_contrast_switch",
    "contrast_margin_switch",
)

EXP7_FIXED_VIEW_DECODERS = {
    "fanda_greedy": {"use_contrast": False, "update_every": 10**9},
    # FanDa now means the full packaged recipe:
    # raw-logit contrast, no relative-top filter, and a frozen shallow layer.
    "fanda": {"use_contrast": True, "update_every": 10**9},
}

EXP12_STATEFUL_FIXED_VIEW_DECODERS = {
    "fanda_update1": {"use_contrast": True, "update_every": 1},
    "fanda_update2": {"use_contrast": True, "update_every": 2},
    "fanda_update4": {"use_contrast": True, "update_every": 4},
    "fanda_update8": {"use_contrast": True, "update_every": 8},
    # Step 0 still refreshes once, then the layer is held for the rest of the answer.
    "fanda_frozen": {"use_contrast": True, "update_every": 10**9},
}

EXP13_FACTORIAL_DECODERS = {
    "exp13_logprob_top": {"score_space": "logprob", "relative_top": True},
    "exp13_logprob_no_top": {"score_space": "logprob", "relative_top": False},
    "exp13_logit_top": {"score_space": "logit", "relative_top": True},
    "exp13_logit_no_top": {"score_space": "logit", "relative_top": False},
}

EXP14_OPEN_FACTUAL_DECODERS = {
    "exp14_update1": {"use_contrast": True, "update_every": 1},
    "exp14_update2": {"use_contrast": True, "update_every": 2},
    "exp14_update4": {"use_contrast": True, "update_every": 4},
    "exp14_update8": {"use_contrast": True, "update_every": 8},
    "exp14_frozen": {"use_contrast": True, "update_every": 10**9},
}

EXP7_STATEFUL_SWITCH_DECODERS = (
    "pure_argmax_switchv2",
)

EXP8_ORACLE_DECODERS = (
    "oracle_token_switch",
)

EXP9_ADAPTIVE_DECODERS = (
    "adaptive_lambda_contrast",
)

EXP10_EMA_RISK_DECODERS = (
    "ema_risk_switch",
)

PANDA_SWITCH_ALIAS_DECODERS = {
    "panda_switch": "panda",
}

PANDA_SWITCH_UPDATE_DECODERS = (
    "panda_switch_update4",
)

BLOCK_REFINED_CONTRAST_DECODERS = (
    "panda_fandas",
    "block_refined_fanda",
)

EXP7_ALIAS_DECODERS = {
    "pure_greedy": "greedy",
}

EXP4_DECODER_NAMES = (
    "contrastive_decoding",
)

TRUNCATION_DECODER_NAMES = (
    "top_k",
    "top_p",
    "top_p_backoff",
)

MATCHED_ALPHA_VALUES = {
    "matched_alpha_dola_0p10": 0.10,
    "matched_alpha_dola_0p50": 0.50,
    "matched_alpha_dola_0p95": 0.95,
}

EXPERIMENT_DECODER_LABELS = {
    "official_dola": "official_dola",
    "matched_alpha_dola_0p10": "matched_alpha_dola_0p10",
    "matched_alpha_dola_0p50": "matched_alpha_dola_0p50",
    "matched_alpha_dola_0p95": "matched_alpha_dola_0p95",
    "simple_panda_h1": "simple_panda_h1",
    "simple_panda_h2": "simple_panda_h2",
    "contrastive_decoding": "contrastive_decoding",
    "top_k": "top_k",
    "top_p": "top_p",
    "top_p_backoff": "top_p_backoff",
    "panda_w4": "panda_w4",
    "panda_w2": "panda_w2",
    "panda_w1": "panda_w1",
    "panda_no_block_h1": "panda_no_block_h1",
    "dola": "dola",
    "pure_greedy": "pure_greedy",
    "fanda_greedy": "fanda_greedy",
    "fanda": "fanda",
    "fanda_update1": "fanda_update1",
    "fanda_update2": "fanda_update2",
    "fanda_update4": "fanda_update4",
    "fanda_update8": "fanda_update8",
    "fanda_frozen": "fanda_frozen",
    "exp13_logprob_top": "logprob + relative-top (u1)",
    "exp13_logprob_no_top": "logprob + no-filter (u1)",
    "exp13_logit_top": "raw-logit + relative-top (u1)",
    "exp13_logit_no_top": "raw-logit + no-filter (u1)",
    "exp14_update1": "exp14 update1",
    "exp14_update2": "exp14 update2",
    "exp14_update4": "exp14 update4",
    "exp14_update8": "exp14 update8",
    "exp14_frozen": "exp14 frozen",
    "pure_argmax_switch": "pure_argmax_switch",
    "pure_argmax_switchv2": "pure_argmax_switchv2",
    "oracle_token_switch": "oracle_token_switch",
    "adaptive_lambda_contrast": "adaptive_lambda_contrast",
    "ema_risk_switch": "ema_risk_switch",
    "panda_switch": "panda_switch",
    "panda_switch_update4": "panda_switch_update4",
    "panda_fandas": "panda_fandas",
    "block_refined_fanda": "block_refined_fanda",
    "guarded_argmax_switch": "guarded_argmax_switch",
    "sticky_contrast_switch": "sticky_contrast_switch",
    "contrast_margin_switch": "contrast_margin_switch",
}


class ExperimentEvaluator(Stage4Evaluator):
    def __init__(self, args, decoder_names):
        self._experiment_decoder_names = tuple(decoder_names)
        self.cd_amateur_model = None
        self.cd_amateur_input_device = None
        self.cd_amateur_model_name = getattr(args, "cd_amateur_model_name", None)
        self.cd_plausibility_alpha = float(getattr(args, "cd_plausibility_alpha", 0.1))
        self.cd_amateur_temperature = float(getattr(args, "cd_amateur_temperature", 0.5))
        self.top_k_value = int(getattr(args, "top_k_value", 50))
        self.top_p_value = float(getattr(args, "top_p_value", 0.9))
        self.exp6_guarded_top_k = int(getattr(args, "exp6_guarded_top_k", 2))
        self.exp6_sticky_hold_steps = int(getattr(args, "exp6_sticky_hold_steps", 1))
        self.exp6_margin_threshold = float(getattr(args, "exp6_margin_threshold", 0.5))
        self.exp9_lambda_min = float(getattr(args, "exp9_lambda_min", 0.5))
        self.exp9_lambda_max = float(getattr(args, "exp9_lambda_max", 1.0))
        self.exp9_uncertainty_weight = float(getattr(args, "exp9_uncertainty_weight", 1.0))
        self.exp9_confidence_gap_weight = float(getattr(args, "exp9_confidence_gap_weight", 2.0))
        self.exp10_risk_beta = float(getattr(args, "exp10_risk_beta", 0.8))
        self.exp10_entropy_weight = float(getattr(args, "exp10_entropy_weight", 1.0))
        self.exp10_margin_weight = float(getattr(args, "exp10_margin_weight", 1.0))
        self.exp10_layer_jsd_weight = float(getattr(args, "exp10_layer_jsd_weight", 1.0))
        self.exp10_risk_threshold = float(getattr(args, "exp10_risk_threshold", 0.55))
        self.exp10_sticky_hold_steps = int(getattr(args, "exp10_sticky_hold_steps", 0))
        super().__init__(args)
        self.decoder_names = self._experiment_decoder_names
        self.decoder_labels = {
            name: EXPERIMENT_DECODER_LABELS.get(name, name) for name in self.decoder_names
        }
        if not (0.0 < self.cd_plausibility_alpha <= 1.0):
            raise ValueError("contrastive_decoding requires 0 < cd_plausibility_alpha <= 1.")
        if self.cd_amateur_temperature <= 0.0:
            raise ValueError("contrastive_decoding requires cd_amateur_temperature > 0.")
        if self.top_k_value < 1:
            raise ValueError("top_k requires top_k_value >= 1.")
        if not (0.0 < self.top_p_value <= 1.0):
            raise ValueError("top_p and top_p_backoff require 0 < top_p_value <= 1.")
        if self.exp6_guarded_top_k < 1:
            raise ValueError("exp6_guarded_top_k must be >= 1.")
        if self.exp6_sticky_hold_steps < 0:
            raise ValueError("exp6_sticky_hold_steps must be >= 0.")
        if self.exp6_margin_threshold < 0.0:
            raise ValueError("exp6_margin_threshold must be >= 0.")
        if not (0.0 <= self.exp9_lambda_min <= self.exp9_lambda_max <= 1.0):
            raise ValueError("exp9 requires 0 <= exp9_lambda_min <= exp9_lambda_max <= 1.")
        if self.exp9_uncertainty_weight < 0.0:
            raise ValueError("exp9_uncertainty_weight must be >= 0.")
        if self.exp9_confidence_gap_weight < 0.0:
            raise ValueError("exp9_confidence_gap_weight must be >= 0.")
        if not (0.0 <= self.exp10_risk_beta < 1.0):
            raise ValueError("exp10_risk_beta must satisfy 0 <= exp10_risk_beta < 1.")
        if self.exp10_entropy_weight < 0.0:
            raise ValueError("exp10_entropy_weight must be >= 0.")
        if self.exp10_margin_weight < 0.0:
            raise ValueError("exp10_margin_weight must be >= 0.")
        if self.exp10_layer_jsd_weight < 0.0:
            raise ValueError("exp10_layer_jsd_weight must be >= 0.")
        if (
            self.exp10_entropy_weight + self.exp10_margin_weight + self.exp10_layer_jsd_weight
            <= 0.0
        ):
            raise ValueError("exp10 requires at least one positive risk-component weight.")
        if not (0.0 <= self.exp10_risk_threshold <= 1.0):
            raise ValueError("exp10_risk_threshold must satisfy 0 <= threshold <= 1.")
        if self.exp10_sticky_hold_steps < 0:
            raise ValueError("exp10_sticky_hold_steps must be >= 0.")
        if any(name in EXP4_DECODER_NAMES for name in self.decoder_names):
            self._load_cd_amateur_model()

    def _print_run_config(self):
        config = {
            "model_name": self.args.model_name,
            "mode": self.args.mode,
            "strict_eval": self.args.strict_eval,
            "experiment_decoders": list(self._experiment_decoder_names),
            "panda_config": {
                "divergence_threshold": float(self.args.panda_divergence_threshold),
                "truth_bias": float(self.args.panda_truth_bias),
                "binary_views": {
                    "greedy_view": "final_logits",
                    "contrast_subtracted_view": "final_logits - shallow_logits",
                },
            },
            "dola_relative_top": float(self.args.dola_relative_top),
            "dola_relative_top_value": float(self.args.dola_relative_top_value),
        }
        if any(name in EXP4_DECODER_NAMES for name in self._experiment_decoder_names):
            config["contrastive_decoding_config"] = {
                "amateur_model_name": getattr(self.args, "cd_amateur_model_name", None),
                "plausibility_alpha": float(getattr(self.args, "cd_plausibility_alpha", 0.1)),
                "amateur_temperature": float(getattr(self.args, "cd_amateur_temperature", 0.5)),
            }
        if any(name in TRUNCATION_DECODER_NAMES for name in self._experiment_decoder_names):
            config["truncation_baseline_config"] = {
                "top_k_value": int(getattr(self.args, "top_k_value", 50)),
                "top_p_value": float(getattr(self.args, "top_p_value", 0.9)),
            }
            if "top_p_backoff" in self._experiment_decoder_names:
                config["truncation_baseline_config"]["top_p_backoff_note"] = (
                    "teacher_forced_gold_tokens_outside_the_nucleus_fall_back_to_full_distribution_logprob"
                )
        if any(name in EXP5_DECODER_NAMES for name in self._experiment_decoder_names):
            config["exp5_block_ablation_config"] = {
                "decoder_names": [name for name in self._experiment_decoder_names if name in EXP5_DECODER_NAMES],
                "experiment_question": (
                    "whether_speculative_block_refinement_or_slower_carried_layer_refresh_still_"
                    "helps_after_fanda_became_the_strongest_fixed_baseline"
                ),
                "jacobi_window_size": int(getattr(self.args, "jacobi_window_size", 0)),
                "jacobi_max_iters": int(getattr(self.args, "jacobi_max_iters", 0)),
            }
        if any(name in EXP6_DECODER_NAMES for name in self._experiment_decoder_names):
            config["exp6_switch_variants_config"] = {
                "decoder_names": [name for name in self._experiment_decoder_names if name in EXP6_DECODER_NAMES],
                "binary_rule": "greedy_view_vs_contrast_subtracted_view",
                "guarded_top_k": self.exp6_guarded_top_k,
                "sticky_hold_steps": self.exp6_sticky_hold_steps,
                "margin_threshold": self.exp6_margin_threshold,
            }
        if any(
            name in EXP7_FIXED_VIEW_DECODERS or name in EXP7_STATEFUL_SWITCH_DECODERS or name == "dola"
            for name in self._experiment_decoder_names
        ):
            config["exp7_fixed_vs_switch_config"] = {
                "alias_decoders": {
                    name: EXP7_ALIAS_DECODERS[name]
                    for name in self._experiment_decoder_names
                    if name in EXP7_ALIAS_DECODERS
                },
                "fixed_view_decoders": {
                    name: {
                        "view": (
                            "contrast_subtracted_view"
                            if EXP7_FIXED_VIEW_DECODERS[name]["use_contrast"]
                            else "greedy_view"
                        ),
                        "layer_update_every": (
                            "first_step_only_then_hold"
                            if int(EXP7_FIXED_VIEW_DECODERS[name]["update_every"]) >= 10**9
                            else int(EXP7_FIXED_VIEW_DECODERS[name]["update_every"])
                        ),
                    }
                    for name in self._experiment_decoder_names
                    if name in EXP7_FIXED_VIEW_DECODERS
                },
                "switch_decoders": [
                    name
                    for name in self._experiment_decoder_names
                    if name in ("pure_argmax_switch",) + EXP7_STATEFUL_SWITCH_DECODERS
                ],
                "external_baselines": [name for name in self._experiment_decoder_names if name == "dola"],
            }
        if any(name in EXP8_ORACLE_DECODERS for name in self._experiment_decoder_names):
            config["exp8_oracle_switch_config"] = {
                "oracle_decoders": [
                    name for name in self._experiment_decoder_names if name in EXP8_ORACLE_DECODERS
                ],
                "oracle_rule": (
                    "at_each_teacher_forced_token_choose_the_view_with_higher_logprob_for_the_actual_token"
                ),
            }
        if any(name in EXP9_ADAPTIVE_DECODERS for name in self._experiment_decoder_names):
            config["exp9_adaptive_lambda_config"] = {
                "adaptive_decoders": [
                    name for name in self._experiment_decoder_names if name in EXP9_ADAPTIVE_DECODERS
                ],
                "adaptive_rule": (
                    "lambda_t = clamp(lambda_min + (lambda_max - lambda_min) * "
                    "(uncertainty_weight * normalized_final_entropy - "
                    "confidence_gap_weight * max(0, greedy_confidence - contrast_confidence)))"
                ),
                "lambda_min": self.exp9_lambda_min,
                "lambda_max": self.exp9_lambda_max,
                "uncertainty_weight": self.exp9_uncertainty_weight,
                "confidence_gap_weight": self.exp9_confidence_gap_weight,
            }
        if any(name in EXP12_STATEFUL_FIXED_VIEW_DECODERS for name in self._experiment_decoder_names):
            config["exp12_state_persistence_config"] = {
                "decoder_variants": {
                    name: EXP12_STATEFUL_FIXED_VIEW_DECODERS[name]
                    for name in self._experiment_decoder_names
                    if name in EXP12_STATEFUL_FIXED_VIEW_DECODERS
                },
                "mechanism_question": (
                    "does_holding_the_same_selected_layer_for_a_short_span_reduce_correction_signal_"
                    "thrash_without_becoming_too_stale"
                ),
                "primary_diagnostics": [
                    "switch_rate",
                    "selected_layer_match_rate",
                    "avg_oracle_jsd_gap",
                    "avg_selection_margin",
                    "avg_risk_score",
                ],
            }
        if any(name in EXP13_FACTORIAL_DECODERS for name in self._experiment_decoder_names):
            config["exp13_factorial_config"] = {
                "decoder_variants": {
                    name: EXP13_FACTORIAL_DECODERS[name]
                    for name in self._experiment_decoder_names
                    if name in EXP13_FACTORIAL_DECODERS
                },
                "layer_selection_rule": "official_dola_step_local_jsd_argmax",
                "matching_rule": "all_four_cells_refresh_selected_layer_every_token",
                "binary_question": "does_raw_logit_vs_logprob_and_relative_top_on_vs_off_explain_part_of_the_gain",
            }
        if any(name in EXP14_OPEN_FACTUAL_DECODERS for name in self._experiment_decoder_names):
            config["exp14_openended_factuality_config"] = {
                "decoder_variants": {
                    name: EXP14_OPEN_FACTUAL_DECODERS[name]
                    for name in self._experiment_decoder_names
                    if name in EXP14_OPEN_FACTUAL_DECODERS
                },
                "benchmark_mode": "free_form_generation_on_truthfulqa_mc_questions",
                "evaluation_mode": "reference_bank_token_overlap_against_true_vs_false_answer_sets",
            }
        if any(name in EXP10_EMA_RISK_DECODERS for name in self._experiment_decoder_names):
            config["exp10_ema_risk_switch_config"] = {
                "adaptive_decoders": [
                    name for name in self._experiment_decoder_names if name in EXP10_EMA_RISK_DECODERS
                ],
                "fixed_layer_rule": (
                    "freeze_selected_layer_after_the_first_token; if multiple shallow candidates are "
                    "provided, choose the first-token best-JSD layer once and hold it"
                ),
                "risk_rule": (
                    "ema_risk_t = beta * ema_risk_{t-1} + (1 - beta) * weighted_mean("
                    "normalized_final_entropy, inverse_top1_top2_prob_gap, normalized_final_vs_shallow_jsd)"
                ),
                "switch_rule": (
                    "use_contrast iff token_mismatch and ema_risk_t >= risk_threshold, "
                    "with optional sticky_hold_steps"
                ),
                "risk_beta": self.exp10_risk_beta,
                "entropy_weight": self.exp10_entropy_weight,
                "margin_weight": self.exp10_margin_weight,
                "layer_jsd_weight": self.exp10_layer_jsd_weight,
                "risk_threshold": self.exp10_risk_threshold,
                "sticky_hold_steps": self.exp10_sticky_hold_steps,
            }
        if any(
            name in BLOCK_REFINED_CONTRAST_DECODERS
            or name in PANDA_SWITCH_ALIAS_DECODERS
            or name in PANDA_SWITCH_UPDATE_DECODERS
            for name in self._experiment_decoder_names
        ):
            config["exp5_block_family_config"] = {
                "hybrid_decoders": [
                    name
                    for name in self._experiment_decoder_names
                    if name in BLOCK_REFINED_CONTRAST_DECODERS
                ],
                "switch_aliases": {
                    name: PANDA_SWITCH_ALIAS_DECODERS[name]
                    for name in self._experiment_decoder_names
                    if name in PANDA_SWITCH_ALIAS_DECODERS
                },
                "update4_decoders": [
                    name
                    for name in self._experiment_decoder_names
                    if name in PANDA_SWITCH_UPDATE_DECODERS
                ],
                "block_rule": (
                    "jacobi_block_refinement_with_per_position_dynamic_shallow_layer_selection_"
                    "and_forced_contrast_subtracted_view_at_every_position"
                ),
                "update4_rule": (
                    "refresh_one_shared_shallow_layer_every_4_teacher_forced_tokens_then_reuse_"
                    "that_layer_across_all_positions_inside_each_panda_block"
                ),
                "commit_rule": "stable_prefix_then_fallback_1",
                "jacobi_window_size": int(self.jacobi_window_size),
                "jacobi_max_iters": int(self.jacobi_max_iters),
                "fixed_layer_update_every": 4,
            }
        print(config)

    def _load_cd_amateur_model(self):
        if not self.cd_amateur_model_name:
            raise ValueError("contrastive_decoding requires --cd-amateur-model-name.")
        print({"loading_contrastive_amateur_model": self.cd_amateur_model_name})
        amateur_model = AutoModelForCausalLM.from_pretrained(
            self.cd_amateur_model_name,
            dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            token=self.hf_token,
            local_files_only=self.args.local_files_only,
        )
        amateur_model.eval()
        if amateur_model.config.vocab_size != self.model.config.vocab_size:
            raise ValueError(
                "contrastive_decoding requires expert and amateur models with matching vocab sizes; "
                f"got expert={self.model.config.vocab_size} amateur={amateur_model.config.vocab_size}."
            )
        self.cd_amateur_model = amateur_model
        self.cd_amateur_input_device = next(amateur_model.parameters()).device

    def score_candidate_with_decoder(self, prompt, decoder_name, choice_text):
        if decoder_name == "official_dola":
            total_logprob, trace, runtime = super().score_candidate_with_decoder(prompt, "dola", choice_text)
            for row in trace:
                row["ablation_mode"] = "official_dola"
            return total_logprob, trace, runtime

        if decoder_name in EXP7_ALIAS_DECODERS:
            total_logprob, trace, runtime = super().score_candidate_with_decoder(
                prompt,
                EXP7_ALIAS_DECODERS[decoder_name],
                choice_text,
            )
            for row in trace:
                row["ablation_mode"] = decoder_name
            return total_logprob, trace, runtime

        if decoder_name in PANDA_SWITCH_ALIAS_DECODERS:
            total_logprob, trace, runtime = super().score_candidate_with_decoder(
                prompt,
                PANDA_SWITCH_ALIAS_DECODERS[decoder_name],
                choice_text,
            )
            for row in trace:
                row["ablation_mode"] = decoder_name
            return total_logprob, trace, runtime

        if decoder_name in PANDA_SWITCH_UPDATE_DECODERS:
            return self._score_candidate_with_panda_window_update4(
                prompt,
                choice_text,
                decoder_name,
            )

        if decoder_name in TRUNCATION_DECODER_NAMES:
            return self._score_candidate_with_truncation_backoff(prompt, choice_text, decoder_name)

        if decoder_name in EXP4_DECODER_NAMES:
            return self._score_candidate_with_custom_step(prompt, choice_text, decoder_name)

        if decoder_name in MATCHED_ALPHA_VALUES:
            return self._score_candidate_with_custom_step(prompt, choice_text, decoder_name)

        if decoder_name in EXP2_DECODER_NAMES:
            return self._score_candidate_with_custom_step(prompt, choice_text, decoder_name)

        if decoder_name in EXP5_DECODER_NAMES:
            return self._score_candidate_with_custom_step(
                prompt,
                choice_text,
                decoder_name,
            )

        if decoder_name in EXP6_DECODER_NAMES:
            return self._score_candidate_with_custom_step(
                prompt,
                choice_text,
                decoder_name,
            )

        if decoder_name in EXP7_FIXED_VIEW_DECODERS:
            return self._score_candidate_with_custom_step(
                prompt,
                choice_text,
                decoder_name,
            )

        if decoder_name in EXP12_STATEFUL_FIXED_VIEW_DECODERS:
            return self._score_candidate_with_custom_step(
                prompt,
                choice_text,
                decoder_name,
            )

        if decoder_name in EXP13_FACTORIAL_DECODERS:
            return self._score_candidate_with_custom_step(
                prompt,
                choice_text,
                decoder_name,
            )

        if decoder_name in EXP14_OPEN_FACTUAL_DECODERS:
            return self._score_candidate_with_custom_step(
                prompt,
                choice_text,
                decoder_name,
            )

        if decoder_name in EXP7_STATEFUL_SWITCH_DECODERS:
            return self._score_candidate_with_custom_step(
                prompt,
                choice_text,
                decoder_name,
            )

        if decoder_name in EXP8_ORACLE_DECODERS:
            return self._score_candidate_with_oracle_switch(
                prompt,
                choice_text,
                decoder_name,
            )

        if decoder_name in EXP9_ADAPTIVE_DECODERS:
            return self._score_candidate_with_custom_step(
                prompt,
                choice_text,
                decoder_name,
            )

        if decoder_name in EXP10_EMA_RISK_DECODERS:
            return self._score_candidate_with_custom_step(
                prompt,
                choice_text,
                decoder_name,
            )

        if decoder_name in BLOCK_REFINED_CONTRAST_DECODERS:
            return self._score_candidate_with_block_refined_contrast(
                prompt,
                choice_text,
                decoder_name,
            )

        return super().score_candidate_with_decoder(prompt, decoder_name, choice_text)

    def generate_with_decoder(self, prompt, decoder_name, max_new_tokens=96, stop_on_eos=True):
        if decoder_name in EXP14_OPEN_FACTUAL_DECODERS:
            return self._generate_with_custom_step(
                prompt,
                decoder_name,
                max_new_tokens=max_new_tokens,
                stop_on_eos=stop_on_eos,
            )
        return super().generate_with_decoder(
            prompt,
            decoder_name,
            max_new_tokens=max_new_tokens,
            stop_on_eos=stop_on_eos,
        )

    def _generate_with_custom_step(self, prompt, decoder_name, max_new_tokens=96, stop_on_eos=True):
        generated = self.prepare_prompt(prompt)
        prompt_length = generated.shape[1]
        eos_token_id = self.tokenizer.eos_token_id
        trace = []
        generated_steps = 0
        forward_passes = 0
        decoder_state = None

        self.synchronize_cuda()
        start_time = time.perf_counter()
        for _ in range(max_new_tokens):
            scores, scores_are_logprobs, trace_row, step_forward_passes, decoder_state = self._custom_decoder_step_scores(
                generated,
                decoder_name,
                decoder_state,
            )
            del scores_are_logprobs
            forward_passes += int(step_forward_passes)
            next_token = torch.argmax(scores, dim=-1, keepdim=True)
            generated_steps += 1
            row = dict(trace_row)
            row["step"] = len(trace)
            row["token_id"] = int(next_token.item())
            row["token_text"] = self.decode_token(next_token.item())
            trace.append(row)
            generated = torch.cat([generated, next_token.to(generated.device)], dim=-1)
            if stop_on_eos and eos_token_id is not None and next_token.item() == eos_token_id:
                break
        self.synchronize_cuda()
        elapsed = time.perf_counter() - start_time

        return self.decode_continuation(generated, prompt_length), trace, make_runtime_summary(
            elapsed,
            generated_steps,
            forward_passes=forward_passes,
            generated_tokens=generated_steps,
        )

    def _select_step_local_dola_layer(self, final_logits, layer_logits):
        candidate_metrics = []
        for candidate_idx in self.cfg.shallow_bucket:
            if candidate_idx >= len(layer_logits) or candidate_idx == self.mature_layer_index:
                continue
            candidate_logits = layer_logits[candidate_idx]
            candidate_score = self.official_dola_js_divergence(final_logits, candidate_logits)
            candidate_metrics.append(
                {
                    "layer": int(candidate_idx),
                    "jsd_current": float(candidate_score),
                }
            )
        if not candidate_metrics:
            raise ValueError("No candidate shallow layers were available for the matched exp13 comparison.")
        return max(candidate_metrics, key=lambda row: row["jsd_current"])

    def build_exp13_factorial_scores(self, final_logits, layer_logits, decoder_name):
        variant = EXP13_FACTORIAL_DECODERS[decoder_name]
        best_candidate = self._select_step_local_dola_layer(final_logits, layer_logits)
        selected_layer = int(best_candidate["layer"])
        shallow_logits = layer_logits[selected_layer]
        mature_log_probs = F.log_softmax(final_logits, dim=-1)

        if variant["score_space"] == "logprob":
            shallow_view = F.log_softmax(shallow_logits, dim=-1)
            scores = mature_log_probs - shallow_view
            scores = F.log_softmax(scores, dim=-1)
            scores_are_logprobs = True
        else:
            scores = final_logits - shallow_logits
            scores_are_logprobs = False

        if variant["relative_top"]:
            relative_top_mask = self.get_relative_top_filter(mature_log_probs, self.dola_relative_top)
            scores = torch.where(
                relative_top_mask,
                torch.full_like(scores, self.dola_relative_top_value),
                scores,
            )

        trace_row = {
            "step": None,
            "selected_layer": selected_layer,
            "divergence": float(best_candidate["jsd_current"]),
            "margin": None,
            "instability": None,
            "alpha": None,
            "ablation_mode": decoder_name,
            "jsd_current": float(best_candidate["jsd_current"]),
            "selection_score": float(best_candidate["jsd_current"]),
            "exp13_score_space": variant["score_space"],
            "exp13_relative_top": bool(variant["relative_top"]),
        }
        return scores, scores_are_logprobs, trace_row

    def _score_candidate_with_custom_step(self, prompt, choice_text, decoder_name):
        generated = self.prepare_prompt(prompt)
        total_logprob = 0.0
        token_ids = self.candidate_to_ids(choice_text)
        trace = []
        forward_passes = 0
        decoder_state = None

        self.synchronize_cuda()
        start_time = time.perf_counter()
        for token_id in token_ids:
            scores, scores_are_logprobs, trace_row, step_forward_passes, decoder_state = self._custom_decoder_step_scores(
                generated,
                decoder_name,
                decoder_state,
            )
            forward_passes += int(step_forward_passes)
            if scores_are_logprobs:
                total_logprob += float(scores[0, token_id].item())
            else:
                logprobs = torch.log_softmax(scores, dim=-1)
                total_logprob += float(logprobs[0, token_id].item())
            next_token = torch.tensor([[token_id]], device=generated.device)
            row = dict(trace_row)
            row["step"] = len(trace)
            row["token_id"] = int(token_id)
            row["token_text"] = self.decode_token(token_id)
            trace.append(row)
            generated = torch.cat([generated, next_token], dim=-1)
        self.synchronize_cuda()
        elapsed = time.perf_counter() - start_time

        return total_logprob, trace, make_runtime_summary(
            elapsed,
            len(token_ids),
            forward_passes=forward_passes,
            generated_tokens=len(token_ids),
        )

    def _score_candidate_with_truncation_backoff(self, prompt, choice_text, decoder_name):
        generated = self.prepare_prompt(prompt)
        total_logprob = 0.0
        token_ids = self.candidate_to_ids(choice_text)
        trace = []
        forward_passes = 0

        self.synchronize_cuda()
        start_time = time.perf_counter()
        for token_id in token_ids:
            truncated_log_probs, full_log_probs, keep_mask, trace_row = self.build_truncation_scores(
                generated,
                decoder_name,
                return_keep_mask=True,
            )
            forward_passes += 1
            gold_token_kept = bool(keep_mask[0, token_id].item())
            if gold_token_kept:
                selected_logprob = float(truncated_log_probs[0, token_id].item())
            else:
                selected_logprob = float(full_log_probs[0, token_id].item())
            total_logprob += selected_logprob

            next_token = torch.tensor([[token_id]], device=generated.device)
            row = dict(trace_row)
            row["step"] = len(trace)
            row["token_id"] = int(token_id)
            row["token_text"] = self.decode_token(token_id)
            row["fallback_used"] = float(not gold_token_kept)
            row["truncation_gold_token_kept"] = float(gold_token_kept)
            row["truncation_backoff_used"] = float(not gold_token_kept)
            row["truncation_selected_logprob"] = selected_logprob
            row["truncation_full_logprob"] = float(full_log_probs[0, token_id].item())
            if decoder_name in ("top_p", "top_p_backoff"):
                row["top_p_gold_token_kept"] = float(gold_token_kept)
                row["top_p_backoff_used"] = float(not gold_token_kept)
                row["top_p_selected_logprob"] = selected_logprob
                row["top_p_full_logprob"] = float(full_log_probs[0, token_id].item())
            trace.append(row)
            generated = torch.cat([generated, next_token], dim=-1)
        self.synchronize_cuda()
        elapsed = time.perf_counter() - start_time

        return total_logprob, trace, make_runtime_summary(
            elapsed,
            len(token_ids),
            forward_passes=forward_passes,
            generated_tokens=len(token_ids),
        )

    def _score_candidate_with_oracle_switch(self, prompt, choice_text, decoder_name):
        generated = self.prepare_prompt(prompt)
        total_logprob = 0.0
        token_ids = self.candidate_to_ids(choice_text)
        trace = []
        forward_passes = 0
        decoder_state = None

        self.synchronize_cuda()
        start_time = time.perf_counter()
        for token_id in token_ids:
            (
                selected_scores,
                selected_logprob,
                trace_row,
                decoder_state,
            ) = self.compute_stateful_oracle_binary_view_step(
                generated,
                int(token_id),
                decoder_name,
                decoder_state,
            )
            forward_passes += 1
            total_logprob += float(selected_logprob)
            next_token = torch.tensor([[token_id]], device=generated.device)
            row = dict(trace_row)
            row["step"] = len(trace)
            row["token_id"] = int(token_id)
            row["token_text"] = self.decode_token(token_id)
            trace.append(row)
            generated = torch.cat([generated, next_token], dim=-1)
        self.synchronize_cuda()
        elapsed = time.perf_counter() - start_time

        return total_logprob, trace, make_runtime_summary(
            elapsed,
            len(token_ids),
            forward_passes=forward_passes,
            generated_tokens=len(token_ids),
        )

    def _custom_decoder_step_scores(self, generated, decoder_name, decoder_state=None):
        if decoder_name in TRUNCATION_DECODER_NAMES:
            scores, trace_row = self.build_truncation_scores(
                generated,
                decoder_name=decoder_name,
            )
            return scores, True, trace_row, 1, None

        if decoder_name in EXP4_DECODER_NAMES:
            scores, trace_row = self.build_contrastive_decoding_scores(
                generated,
                decoder_name=decoder_name,
            )
            return scores, True, trace_row, 2, None

        if decoder_name in MATCHED_ALPHA_VALUES:
            layer_logits, final_logits = self.forward_with_layer_logits(generated)
            scores, trace_row = self.build_matched_alpha_dola_scores(
                final_logits,
                layer_logits,
                alpha=MATCHED_ALPHA_VALUES[decoder_name],
                decoder_name=decoder_name,
            )
            return scores, True, trace_row, 1, None

        if decoder_name in EXP13_FACTORIAL_DECODERS:
            layer_logits, final_logits = self.forward_with_layer_logits(generated)
            scores, scores_are_logprobs, trace_row = self.build_exp13_factorial_scores(
                final_logits,
                layer_logits,
                decoder_name,
            )
            return scores, scores_are_logprobs, trace_row, 1, None

        if decoder_name in EXP14_OPEN_FACTUAL_DECODERS:
            variant = EXP14_OPEN_FACTUAL_DECODERS[decoder_name]
            selected_scores, trace_row, next_state = self.compute_stateful_fixed_binary_view_step(
                generated,
                decoder_name,
                decoder_state=decoder_state,
                use_contrast=bool(variant["use_contrast"]),
                update_every_override=int(variant["update_every"]),
            )
            return selected_scores, False, trace_row, 1, next_state

        if decoder_name in EXP2_DECODER_NAMES:
            step_info = self.build_simple_panda_step(generated)
            selected_scores, trace_row, extra_forward_passes = self.choose_simple_panda_branch(
                generated,
                step_info,
                decoder_name,
            )
            return selected_scores, False, trace_row, 1 + int(extra_forward_passes), None

        if decoder_name in EXP5_DECODER_NAMES:
            selected_scores, trace_row, next_state = self.compute_stateful_fixed_binary_view_step(
                generated,
                decoder_name,
                decoder_state=decoder_state,
                use_contrast=True,
            )
            return selected_scores, False, trace_row, 1, next_state

        if decoder_name in EXP5_SIMPLE_ALIAS:
            step_info = self.build_simple_panda_step(generated)
            selected_scores, trace_row, extra_forward_passes = self.choose_simple_panda_branch(
                generated,
                step_info,
                EXP5_SIMPLE_ALIAS[decoder_name],
            )
            trace_row = dict(trace_row)
            trace_row["ablation_mode"] = decoder_name
            return selected_scores, False, trace_row, 1 + int(extra_forward_passes), None

        if decoder_name in EXP6_DECODER_NAMES:
            step_info = self.build_simple_panda_step(generated)
            selected_scores, trace_row, next_state = self.choose_exp6_switch_branch(
                step_info,
                decoder_name,
                decoder_state=decoder_state,
            )
            return selected_scores, False, trace_row, 1, next_state

        if decoder_name in EXP12_STATEFUL_FIXED_VIEW_DECODERS:
            variant = EXP12_STATEFUL_FIXED_VIEW_DECODERS[decoder_name]
            selected_scores, trace_row, next_state = self.compute_stateful_fixed_binary_view_step(
                generated,
                decoder_name,
                decoder_state=decoder_state,
                use_contrast=bool(variant["use_contrast"]),
                update_every_override=int(variant["update_every"]),
            )
            return selected_scores, False, trace_row, 1, next_state

        if decoder_name in EXP7_FIXED_VIEW_DECODERS:
            variant = EXP7_FIXED_VIEW_DECODERS[decoder_name]
            selected_scores, trace_row, next_state = self.compute_stateful_fixed_binary_view_step(
                generated,
                decoder_name,
                decoder_state=decoder_state,
                use_contrast=bool(variant["use_contrast"]),
                update_every_override=int(variant["update_every"]),
            )
            return selected_scores, False, trace_row, 1, next_state

        if decoder_name in EXP7_STATEFUL_SWITCH_DECODERS:
            selected_scores, trace_row, next_state = self.compute_stateful_switch_binary_view_step(
                generated,
                decoder_name,
                decoder_state=decoder_state,
            )
            return selected_scores, False, trace_row, 1, next_state

        if decoder_name in EXP9_ADAPTIVE_DECODERS:
            selected_scores, trace_row, next_state = self.compute_stateful_adaptive_lambda_step(
                generated,
                decoder_name,
                decoder_state=decoder_state,
            )
            return selected_scores, False, trace_row, 1, next_state

        if decoder_name in EXP10_EMA_RISK_DECODERS:
            selected_scores, trace_row, next_state = self.compute_stateful_ema_risk_switch_step(
                generated,
                decoder_name,
                decoder_state=decoder_state,
            )
            return selected_scores, False, trace_row, 1, next_state

        raise ValueError(f"Unknown experiment-local decoder {decoder_name!r}.")

    def _score_candidate_with_panda_window(self, prompt, choice_text, decoder_name, window_size):
        generated = self.prepare_prompt(prompt)
        total_logprob = 0.0
        token_ids = self.candidate_to_ids(choice_text)
        trace = []
        forward_passes = 0

        self.synchronize_cuda()
        start_time = time.perf_counter()
        for token_idx, token_id in enumerate(token_ids):
            remaining_tokens = len(token_ids) - token_idx
            block_window_size = min(int(window_size), remaining_tokens)
            block_result = self.run_panda_block(generated, block_window_size)
            forward_passes += int(block_result["forward_passes"])
            logprobs = torch.log_softmax(block_result["first_scores"], dim=-1)
            total_logprob += float(logprobs[0, token_id].item())
            row = dict(block_result["position_rows"][0])
            row["step"] = len(trace)
            row["token_id"] = int(token_id)
            row["token_text"] = self.decode_token(token_id)
            row["jacobi_block_index"] = int(token_idx)
            row["ablation_mode"] = decoder_name
            trace.append(row)
            next_token = torch.tensor([[token_id]], device=generated.device)
            generated = torch.cat([generated, next_token], dim=-1)
        self.synchronize_cuda()
        elapsed = time.perf_counter() - start_time

        return total_logprob, trace, make_runtime_summary(
            elapsed,
            len(token_ids),
            forward_passes=forward_passes,
            generated_tokens=len(token_ids),
        )

    def run_panda_block_with_fixed_layer_refresh(self, generated, window_size, decoder_name, decoder_state):
        if decoder_state is None:
            decoder_state = self.init_fixed_alpha_state()

        step = int(decoder_state["step"])
        selected_layer = int(decoder_state["selected_layer"])
        window_size = int(window_size)
        buffer = self.repeat_last_token_buffer(generated, window_size)
        previous_buffer = buffer.clone()
        final_rows = []
        first_scores = None
        converged = False
        passes_used = 0
        agreement_prefix_len = 0
        first_divergence_idx = None

        for iteration_idx in range(self.jacobi_max_iters):
            input_ids = torch.cat([generated, buffer], dim=-1)
            layer_logits, final_logits = self.forward_with_window_layer_logits(input_ids, window_size)
            if iteration_idx == 0:
                p_final_first = F.softmax(final_logits[:, 0, :] / self.cfg.tau, dim=-1)
                selected_layer = self.select_dynamic_layer(
                    step,
                    selected_layer,
                    [layer[:, 0, :] for layer in layer_logits],
                    p_final_first,
                )

            candidate_rows = []
            first_divergence_idx = None
            for position_idx in range(window_size):
                final_logits_pos = final_logits[:, position_idx, :]
                p_final = F.softmax(final_logits_pos / self.cfg.tau, dim=-1)
                shallow_logits = layer_logits[selected_layer][:, position_idx, :]
                shallow_probs = F.softmax(shallow_logits / self.cfg.tau, dim=-1)
                jsd_score = self.js_divergence(p_final, shallow_probs)
                divergence, margin, instability = self.compute_instability_terms(
                    final_logits_pos,
                    shallow_logits,
                    p_final,
                )
                greedy_scores, contrast_scores = self.build_binary_views(final_logits_pos, shallow_logits)
                greedy_token = torch.argmax(greedy_scores, dim=-1)
                contrast_token = torch.argmax(contrast_scores, dim=-1)
                greedy_confidence = self.top1_confidence(greedy_scores)
                contrast_confidence = self.top1_confidence(contrast_scores)
                token_mismatch = int(int(greedy_token.item()) != int(contrast_token.item()))
                if self.panda_early_agreement_shortcut and not token_mismatch:
                    regime_jsd = 0.0
                    disagreement = 0
                else:
                    greedy_probs = F.softmax(greedy_scores / self.cfg.tau, dim=-1)
                    contrast_probs = F.softmax(contrast_scores / self.cfg.tau, dim=-1)
                    regime_jsd = self.js_divergence(greedy_probs, contrast_probs)
                    disagreement = int(
                        token_mismatch and regime_jsd >= float(self.panda_divergence_threshold)
                    )
                if disagreement and first_divergence_idx is None:
                    first_divergence_idx = int(position_idx)
                candidate_rows.append(
                    {
                        "position_idx": int(position_idx),
                        "selected_layer": int(selected_layer),
                        "divergence": float(divergence),
                        "margin": float(margin),
                        "instability": float(instability),
                        "jsd_current": float(jsd_score),
                        "greedy_scores": greedy_scores,
                        "contrast_scores": contrast_scores,
                        "greedy_token": greedy_token,
                        "contrast_token": contrast_token,
                        "panda_token_mismatch": float(token_mismatch),
                        "panda_divergence": float(regime_jsd),
                        "panda_greedy_confidence": float(greedy_confidence),
                        "panda_contrast_confidence": float(contrast_confidence),
                        "panda_safe_confidence": float(greedy_confidence),
                        "panda_truth_confidence": float(contrast_confidence),
                        "panda_disagreement": float(disagreement),
                    }
                )

            agreement_prefix_len = 0
            for row in candidate_rows:
                if int(row["panda_disagreement"]) != 0:
                    break
                agreement_prefix_len += 1

            next_tokens = []
            current_rows = []
            current_first_scores = None
            for row in candidate_rows:
                position_idx = int(row["position_idx"])
                arbitration_active = first_divergence_idx is not None and position_idx >= first_divergence_idx
                use_contrast = arbitration_active and (
                    float(row["panda_contrast_confidence"])
                    > float(row["panda_greedy_confidence"]) - float(self.panda_truth_bias)
                )
                selected_scores = row["contrast_scores"] if use_contrast else row["greedy_scores"]
                selected_token = row["contrast_token"] if use_contrast else row["greedy_token"]
                if position_idx == 0:
                    current_first_scores = selected_scores
                next_tokens.append(selected_token)
                current_rows.append(
                    {
                        "step": None,
                        "selected_layer": int(row["selected_layer"]),
                        "divergence": float(row["divergence"]),
                        "margin": float(row["margin"]),
                        "instability": float(row["instability"]),
                        "alpha": float(1.0 if use_contrast else 0.0),
                        "ablation_mode": decoder_name,
                        "risk_triggered": float(row["panda_disagreement"]),
                        "risk_score": float(row["panda_divergence"]),
                        "jsd_current": float(row["jsd_current"]),
                        "selection_margin": float(
                            float(row["panda_contrast_confidence"]) - float(row["panda_greedy_confidence"])
                        ),
                        "selection_score": float(
                            row["panda_contrast_confidence"] if use_contrast else row["panda_greedy_confidence"]
                        ),
                        "fallback_used": float(not use_contrast),
                        "baseline_margin": float(row["margin"]),
                        "jacobi_position": int(position_idx),
                        "jacobi_window_size": int(window_size),
                        "jacobi_pass_index": int(iteration_idx),
                        "panda_divergence": float(row["panda_divergence"]),
                        "panda_greedy_confidence": float(row["panda_greedy_confidence"]),
                        "panda_contrast_confidence": float(row["panda_contrast_confidence"]),
                        "panda_safe_confidence": float(row["panda_safe_confidence"]),
                        "panda_truth_confidence": float(row["panda_truth_confidence"]),
                        "panda_token_mismatch": float(row["panda_token_mismatch"]),
                        "panda_disagreement": float(row["panda_disagreement"]),
                        "panda_selected_contrast": float(use_contrast),
                        "panda_selected_truth": float(use_contrast),
                        "panda_arbitration_active": float(arbitration_active),
                        "panda_first_divergence_position": (
                            int(first_divergence_idx) if first_divergence_idx is not None else None
                        ),
                        "panda_agreement_prefix_len": int(agreement_prefix_len),
                        "layer_refresh_step": int(step),
                        "layer_update_every": int(self.cfg.update_every),
                    }
                )

            new_buffer = torch.stack(next_tokens, dim=1)
            passes_used = iteration_idx + 1
            final_rows = current_rows
            first_scores = current_first_scores
            previous_buffer = buffer
            converged = torch.equal(new_buffer, buffer)
            buffer = new_buffer
            if converged:
                break

        stable_prefix_len = window_size if converged else self.common_prefix_length(previous_buffer, buffer)
        if self.panda_early_agreement_shortcut:
            commit_len = max(
                stable_prefix_len if stable_prefix_len > 0 else 1,
                agreement_prefix_len if agreement_prefix_len > 0 else 1,
            )
        else:
            commit_len = stable_prefix_len if stable_prefix_len > 0 else 1
        for row in final_rows:
            row["jacobi_passes_used"] = int(passes_used)
            row["jacobi_converged"] = float(converged)
            row["jacobi_stable_prefix_len"] = int(stable_prefix_len)
            row["jacobi_commit_len"] = int(commit_len)

        next_state = {"selected_layer": int(selected_layer), "step": step + 1}
        return {
            "buffer": buffer,
            "first_scores": first_scores,
            "position_rows": final_rows,
            "forward_passes": int(passes_used),
            "converged": bool(converged),
            "stable_prefix_len": int(stable_prefix_len),
            "commit_len": int(commit_len),
            "decoder_state": next_state,
        }

    def _score_candidate_with_panda_window_update4(self, prompt, choice_text, decoder_name):
        generated = self.prepare_prompt(prompt)
        total_logprob = 0.0
        token_ids = self.candidate_to_ids(choice_text)
        trace = []
        forward_passes = 0
        decoder_state = None

        self.synchronize_cuda()
        start_time = time.perf_counter()
        for token_idx, token_id in enumerate(token_ids):
            remaining_tokens = len(token_ids) - token_idx
            block_window_size = min(self.jacobi_window_size, remaining_tokens)
            block_result = self.run_panda_block_with_fixed_layer_refresh(
                generated,
                block_window_size,
                decoder_name,
                decoder_state,
            )
            decoder_state = block_result["decoder_state"]
            forward_passes += int(block_result["forward_passes"])
            logprobs = torch.log_softmax(block_result["first_scores"], dim=-1)
            total_logprob += float(logprobs[0, token_id].item())
            row = dict(block_result["position_rows"][0])
            row["step"] = len(trace)
            row["token_id"] = int(token_id)
            row["token_text"] = self.decode_token(token_id)
            row["jacobi_block_index"] = int(token_idx)
            trace.append(row)
            next_token = torch.tensor([[token_id]], device=generated.device)
            generated = torch.cat([generated, next_token], dim=-1)
        self.synchronize_cuda()
        elapsed = time.perf_counter() - start_time

        return total_logprob, trace, make_runtime_summary(
            elapsed,
            len(token_ids),
            forward_passes=forward_passes,
            generated_tokens=len(token_ids),
        )

    def run_block_refined_fanda(self, generated, window_size, decoder_name):
        window_size = int(window_size)
        buffer = self.repeat_last_token_buffer(generated, window_size)
        previous_buffer = buffer.clone()
        final_rows = []
        first_scores = None
        converged = False
        passes_used = 0
        agreement_prefix_len = 0
        first_divergence_idx = None

        for iteration_idx in range(self.jacobi_max_iters):
            input_ids = torch.cat([generated, buffer], dim=-1)
            layer_logits, final_logits = self.forward_with_window_layer_logits(input_ids, window_size)
            selected_layers, jsd_scores = self.select_dynamic_layers_for_window(final_logits, layer_logits)

            candidate_rows = []
            first_divergence_idx = None
            for position_idx, (selected_layer, jsd_score) in enumerate(zip(selected_layers, jsd_scores)):
                final_logits_pos = final_logits[:, position_idx, :]
                p_final = F.softmax(final_logits_pos / self.cfg.tau, dim=-1)
                shallow_logits = layer_logits[selected_layer][:, position_idx, :]
                divergence, margin, instability = self.compute_instability_terms(
                    final_logits_pos,
                    shallow_logits,
                    p_final,
                )
                greedy_scores, contrast_scores = self.build_binary_views(final_logits_pos, shallow_logits)
                greedy_token = torch.argmax(greedy_scores, dim=-1)
                contrast_token = torch.argmax(contrast_scores, dim=-1)
                greedy_confidence = self.top1_confidence(greedy_scores)
                contrast_confidence = self.top1_confidence(contrast_scores)
                token_mismatch = int(int(greedy_token.item()) != int(contrast_token.item()))
                if self.panda_early_agreement_shortcut and not token_mismatch:
                    regime_jsd = 0.0
                    disagreement = 0
                else:
                    greedy_probs = F.softmax(greedy_scores / self.cfg.tau, dim=-1)
                    contrast_probs = F.softmax(contrast_scores / self.cfg.tau, dim=-1)
                    regime_jsd = self.js_divergence(greedy_probs, contrast_probs)
                    disagreement = int(
                        token_mismatch and regime_jsd >= float(self.panda_divergence_threshold)
                    )
                if disagreement and first_divergence_idx is None:
                    first_divergence_idx = int(position_idx)
                candidate_rows.append(
                    {
                        "position_idx": int(position_idx),
                        "selected_layer": int(selected_layer),
                        "divergence": float(divergence),
                        "margin": float(margin),
                        "instability": float(instability),
                        "jsd_current": float(jsd_score),
                        "greedy_scores": greedy_scores,
                        "contrast_scores": contrast_scores,
                        "greedy_token": greedy_token,
                        "contrast_token": contrast_token,
                        "panda_token_mismatch": float(token_mismatch),
                        "panda_divergence": float(regime_jsd),
                        "panda_greedy_confidence": float(greedy_confidence),
                        "panda_contrast_confidence": float(contrast_confidence),
                        "panda_safe_confidence": float(greedy_confidence),
                        "panda_truth_confidence": float(contrast_confidence),
                        "panda_disagreement": float(disagreement),
                    }
                )

            agreement_prefix_len = 0
            for row in candidate_rows:
                if int(row["panda_disagreement"]) != 0:
                    break
                agreement_prefix_len += 1

            next_tokens = []
            current_rows = []
            current_first_scores = None
            for row in candidate_rows:
                position_idx = int(row["position_idx"])
                selected_scores = row["contrast_scores"]
                selected_token = row["contrast_token"]
                if position_idx == 0:
                    current_first_scores = selected_scores
                next_tokens.append(selected_token)
                current_rows.append(
                    {
                        "step": None,
                        "selected_layer": int(row["selected_layer"]),
                        "divergence": float(row["divergence"]),
                        "margin": float(row["margin"]),
                        "instability": float(row["instability"]),
                        "alpha": 1.0,
                        "ablation_mode": decoder_name,
                        "risk_triggered": float(row["panda_disagreement"]),
                        "risk_score": float(row["panda_divergence"]),
                        "jsd_current": float(row["jsd_current"]),
                        "selection_margin": float(
                            float(row["panda_contrast_confidence"])
                            - float(row["panda_greedy_confidence"])
                        ),
                        "selection_score": float(row["panda_contrast_confidence"]),
                        "fallback_used": 0.0,
                        "baseline_margin": float(row["margin"]),
                        "jacobi_position": int(position_idx),
                        "jacobi_window_size": int(window_size),
                        "jacobi_pass_index": int(iteration_idx),
                        "panda_divergence": float(row["panda_divergence"]),
                        "panda_greedy_confidence": float(row["panda_greedy_confidence"]),
                        "panda_contrast_confidence": float(row["panda_contrast_confidence"]),
                        "panda_safe_confidence": float(row["panda_safe_confidence"]),
                        "panda_truth_confidence": float(row["panda_truth_confidence"]),
                        "panda_token_mismatch": float(row["panda_token_mismatch"]),
                        "panda_disagreement": float(row["panda_disagreement"]),
                        "panda_selected_contrast": 1.0,
                        "panda_selected_truth": 1.0,
                        "panda_arbitration_active": 0.0,
                        "panda_first_divergence_position": (
                            int(first_divergence_idx) if first_divergence_idx is not None else None
                        ),
                        "panda_agreement_prefix_len": int(agreement_prefix_len),
                    }
                )

            new_buffer = torch.stack(next_tokens, dim=1)
            passes_used = iteration_idx + 1
            final_rows = current_rows
            first_scores = current_first_scores
            previous_buffer = buffer
            converged = torch.equal(new_buffer, buffer)
            buffer = new_buffer
            if converged:
                break

        stable_prefix_len = (
            window_size if converged else self.common_prefix_length(previous_buffer, buffer)
        )
        commit_len = stable_prefix_len if stable_prefix_len > 0 else 1
        for row in final_rows:
            row["jacobi_passes_used"] = int(passes_used)
            row["jacobi_converged"] = float(converged)
            row["jacobi_stable_prefix_len"] = int(stable_prefix_len)
            row["jacobi_commit_len"] = int(commit_len)

        return {
            "buffer": buffer,
            "first_scores": first_scores,
            "position_rows": final_rows,
            "forward_passes": int(passes_used),
            "converged": bool(converged),
            "stable_prefix_len": int(stable_prefix_len),
            "commit_len": int(commit_len),
        }

    def _score_candidate_with_block_refined_contrast(self, prompt, choice_text, decoder_name):
        generated = self.prepare_prompt(prompt)
        total_logprob = 0.0
        token_ids = self.candidate_to_ids(choice_text)
        trace = []
        forward_passes = 0

        self.synchronize_cuda()
        start_time = time.perf_counter()
        for token_idx, token_id in enumerate(token_ids):
            remaining_tokens = len(token_ids) - token_idx
            block_window_size = min(self.jacobi_window_size, remaining_tokens)
            block_result = self.run_block_refined_fanda(
                generated,
                block_window_size,
                decoder_name,
            )
            forward_passes += int(block_result["forward_passes"])
            logprobs = torch.log_softmax(block_result["first_scores"], dim=-1)
            total_logprob += float(logprobs[0, token_id].item())
            row = dict(block_result["position_rows"][0])
            row["step"] = len(trace)
            row["token_id"] = int(token_id)
            row["token_text"] = self.decode_token(token_id)
            row["jacobi_block_index"] = int(token_idx)
            trace.append(row)
            next_token = torch.tensor([[token_id]], device=generated.device)
            generated = torch.cat([generated, next_token], dim=-1)
        self.synchronize_cuda()
        elapsed = time.perf_counter() - start_time

        return total_logprob, trace, make_runtime_summary(
            elapsed,
            len(token_ids),
            forward_passes=forward_passes,
            generated_tokens=len(token_ids),
        )

    def build_top_p_backoff_scores(self, generated):
        return self.build_truncation_scores(generated, "top_p_backoff", return_keep_mask=True)

    def build_truncation_scores(self, generated, decoder_name, return_keep_mask=False):
        outputs = self.model(
            input_ids=generated,
            use_cache=False,
            return_dict=True,
        )
        final_logits = outputs.logits[:, -1, :].float()
        full_log_probs = F.log_softmax(final_logits, dim=-1)

        if decoder_name == "top_k":
            truncated_log_probs, kept_fraction, keep_mask = self.apply_top_k_truncation(
                full_log_probs,
                self.top_k_value,
                return_keep_mask=True,
            )
        elif decoder_name in ("top_p", "top_p_backoff"):
            truncated_log_probs, kept_fraction, keep_mask = self.apply_top_p_truncation(
                full_log_probs,
                self.top_p_value,
                return_keep_mask=True,
            )
        else:  # pragma: no cover - protected by dispatch
            raise ValueError(f"Unsupported truncation decoder {decoder_name!r}.")

        trace_row = {
            "step": None,
            "selected_layer": None,
            "divergence": None,
            "margin": None,
            "instability": None,
            "alpha": None,
            "ablation_mode": decoder_name,
            "risk_triggered": None,
            "risk_score": None,
            "jsd_current": None,
            "selection_margin": None,
            "selection_score": kept_fraction,
            "fallback_used": None,
            "baseline_margin": None,
            "truncation_kept_fraction": kept_fraction,
        }
        if return_keep_mask:
            return truncated_log_probs, full_log_probs, keep_mask, trace_row
        return truncated_log_probs, trace_row

    @staticmethod
    def apply_top_k_truncation(log_probs, k, return_keep_mask=False):
        vocab_size = log_probs.shape[-1]
        k = min(int(k), int(vocab_size))
        topk_values, _ = torch.topk(log_probs, k=k, dim=-1)
        threshold = topk_values[..., -1].unsqueeze(-1)
        keep_mask = log_probs >= threshold
        filtered = torch.where(keep_mask, log_probs, torch.full_like(log_probs, float("-inf")))
        truncated_log_probs = F.log_softmax(filtered, dim=-1)
        kept_fraction = float(keep_mask.float().mean().item())
        if return_keep_mask:
            return truncated_log_probs, kept_fraction, keep_mask
        return truncated_log_probs, kept_fraction

    @staticmethod
    def apply_top_p_truncation(log_probs, p, return_keep_mask=False):
        sorted_log_probs, sorted_indices = torch.sort(log_probs, dim=-1, descending=True)
        sorted_probs = torch.exp(sorted_log_probs)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_keep_mask = cumulative_probs <= float(p)
        sorted_keep_mask[..., 0] = True
        first_over_mask = cumulative_probs > float(p)
        first_over_idx = first_over_mask.float().argmax(dim=-1, keepdim=True)
        sorted_keep_mask.scatter_(dim=-1, index=first_over_idx, value=True)
        keep_mask = torch.zeros_like(sorted_keep_mask, dtype=torch.bool)
        keep_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_keep_mask)
        filtered = torch.where(keep_mask, log_probs, torch.full_like(log_probs, float("-inf")))
        truncated_log_probs = F.log_softmax(filtered, dim=-1)
        kept_fraction = float(keep_mask.float().mean().item())
        if return_keep_mask:
            return truncated_log_probs, kept_fraction, keep_mask
        return truncated_log_probs, kept_fraction

    def build_contrastive_decoding_scores(self, generated, decoder_name):
        if self.cd_amateur_model is None or self.cd_amateur_input_device is None:
            raise ValueError("contrastive_decoding amateur model was not loaded.")
        expert_outputs = self.model(
            input_ids=generated,
            use_cache=False,
            return_dict=True,
        )
        final_logits = expert_outputs.logits[:, -1, :].float()
        amateur_generated = generated.to(self.cd_amateur_input_device)
        amateur_outputs = self.cd_amateur_model(
            input_ids=amateur_generated,
            use_cache=False,
            return_dict=True,
        )
        amateur_logits = amateur_outputs.logits[:, -1, :].float().to(final_logits.device)
        expert_log_probs = F.log_softmax(final_logits, dim=-1)
        amateur_log_probs = F.log_softmax(amateur_logits / self.cd_amateur_temperature, dim=-1)
        contrast_scores = expert_log_probs - amateur_log_probs
        contrast_scores = F.log_softmax(contrast_scores, dim=-1)
        plausibility_threshold = torch.max(expert_log_probs, dim=-1, keepdim=True).values + torch.log(
            torch.tensor(self.cd_plausibility_alpha, device=expert_log_probs.device, dtype=expert_log_probs.dtype)
        )
        invalid_mask = expert_log_probs < plausibility_threshold
        contrast_scores = torch.where(
            invalid_mask,
            torch.full_like(contrast_scores, float("-inf")),
            contrast_scores,
        )
        divergence = self.official_dola_js_divergence(final_logits, amateur_logits)
        trace_row = {
            "step": None,
            "selected_layer": None,
            "divergence": float(divergence),
            "margin": None,
            "instability": None,
            "alpha": None,
            "ablation_mode": decoder_name,
            "risk_triggered": None,
            "risk_score": None,
            "jsd_current": float(divergence),
            "selection_margin": None,
            "selection_score": float(divergence),
            "fallback_used": None,
            "baseline_margin": None,
            "cd_invalid_fraction": float(invalid_mask.float().mean().item()),
        }
        return contrast_scores, trace_row

    def build_matched_alpha_dola_scores(self, final_logits, layer_logits, alpha, decoder_name):
        candidate_metrics = []
        for candidate_idx in self.cfg.shallow_bucket:
            if candidate_idx >= len(layer_logits) or candidate_idx == self.mature_layer_index:
                continue
            candidate_logits = layer_logits[candidate_idx]
            candidate_score = self.official_dola_js_divergence(final_logits, candidate_logits)
            candidate_metrics.append({"layer": int(candidate_idx), "jsd_current": float(candidate_score)})
        if not candidate_metrics:
            raise ValueError("No valid candidate layers were available for matched-alpha DoLa.")

        best_candidate = max(candidate_metrics, key=lambda row: row["jsd_current"])
        selected_layer = int(best_candidate["layer"])
        premature_logits = layer_logits[selected_layer]
        mature_log_probs = F.log_softmax(final_logits, dim=-1)
        premature_log_probs = F.log_softmax(premature_logits, dim=-1)
        contrast_scores = mature_log_probs - float(alpha) * premature_log_probs
        contrast_scores = F.log_softmax(contrast_scores, dim=-1)
        if self.dola_relative_top > 0.0:
            relative_top_mask = self.get_relative_top_filter(mature_log_probs, self.dola_relative_top)
            contrast_scores = torch.where(
                relative_top_mask,
                torch.full_like(contrast_scores, self.dola_relative_top_value),
                contrast_scores,
            )
        trace_row = {
            "step": None,
            "selected_layer": selected_layer,
            "divergence": float(best_candidate["jsd_current"]),
            "margin": None,
            "instability": None,
            "alpha": float(alpha),
            "ablation_mode": decoder_name,
            "risk_triggered": None,
            "risk_score": None,
            "jsd_current": float(best_candidate["jsd_current"]),
            "selection_margin": None,
            "selection_score": float(best_candidate["jsd_current"]),
            "fallback_used": None,
            "baseline_margin": None,
        }
        return contrast_scores, trace_row

    def select_best_layer_by_jsd(self, final_logits, layer_logits):
        p_final = F.softmax(final_logits / self.cfg.tau, dim=-1)
        best_score = -float("inf")
        selected_layer = self.cfg.shallow_bucket[0]
        for candidate_idx in self.cfg.shallow_bucket:
            if candidate_idx >= len(layer_logits):
                continue
            candidate_logits = layer_logits[candidate_idx]
            candidate_probs = F.softmax(candidate_logits / self.cfg.tau, dim=-1)
            score = self.js_divergence(p_final, candidate_probs)
            if score > best_score:
                best_score = score
                selected_layer = candidate_idx
        return int(selected_layer), float(best_score), p_final

    @staticmethod
    def top1_top2_probability_gap(logits):
        probs = F.softmax(logits, dim=-1)
        top_k = min(2, int(probs.shape[-1]))
        top_values = torch.topk(probs, k=top_k, dim=-1).values[0]
        if top_k < 2:
            return 1.0
        return float((top_values[0] - top_values[1]).item())

    def build_simple_panda_step(self, generated):
        layer_logits, final_logits = self.forward_with_layer_logits(generated)
        selected_layer, jsd_score, p_final = self.select_best_layer_by_jsd(final_logits, layer_logits)
        shallow_logits = layer_logits[selected_layer]
        divergence, margin, instability = self.compute_instability_terms(final_logits, shallow_logits, p_final)
        greedy_scores, contrast_scores = self.build_binary_views(final_logits, shallow_logits)
        greedy_token = torch.argmax(greedy_scores, dim=-1)
        contrast_token = torch.argmax(contrast_scores, dim=-1)
        greedy_confidence = self.top1_confidence(greedy_scores)
        contrast_confidence = self.top1_confidence(contrast_scores)
        token_mismatch = int(int(greedy_token.item()) != int(contrast_token.item()))
        greedy_probs = F.softmax(greedy_scores / self.cfg.tau, dim=-1)
        contrast_probs = F.softmax(contrast_scores / self.cfg.tau, dim=-1)
        regime_jsd = self.js_divergence(greedy_probs, contrast_probs)
        disagreement = int(token_mismatch and regime_jsd >= float(self.panda_divergence_threshold))
        greedy_logprobs = F.log_softmax(greedy_scores, dim=-1)
        contrast_logprobs = F.log_softmax(contrast_scores, dim=-1)
        return {
            "selected_layer": int(selected_layer),
            "jsd_current": float(jsd_score),
            "divergence": float(divergence),
            "margin": float(margin),
            "instability": float(instability),
            "greedy_scores": greedy_scores,
            "contrast_scores": contrast_scores,
            "greedy_token": greedy_token,
            "contrast_token": contrast_token,
            # Preserve the historical keys so downstream trace aggregation still works.
            "low_scores": greedy_scores,
            "high_scores": contrast_scores,
            "low_token": greedy_token,
            "high_token": contrast_token,
            "greedy_token_logprob": float(greedy_logprobs[0, int(greedy_token.item())].item()),
            "contrast_token_logprob": float(contrast_logprobs[0, int(contrast_token.item())].item()),
            "low_token_logprob": float(greedy_logprobs[0, int(greedy_token.item())].item()),
            "high_token_logprob": float(contrast_logprobs[0, int(contrast_token.item())].item()),
            "panda_token_mismatch": float(token_mismatch),
            "panda_divergence": float(regime_jsd),
            "panda_greedy_confidence": float(greedy_confidence),
            "panda_contrast_confidence": float(contrast_confidence),
            "panda_safe_confidence": float(greedy_confidence),
            "panda_truth_confidence": float(contrast_confidence),
            "panda_disagreement": float(disagreement),
        }

    def compute_simple_view_scores(self, generated, use_contrast):
        layer_logits, final_logits = self.forward_with_layer_logits(generated)
        selected_layer, jsd_score, p_final = self.select_best_layer_by_jsd(final_logits, layer_logits)
        shallow_logits = layer_logits[selected_layer]
        divergence, margin, instability = self.compute_instability_terms(final_logits, shallow_logits, p_final)
        greedy_scores, contrast_scores = self.build_binary_views(final_logits, shallow_logits)
        scores = contrast_scores if use_contrast else greedy_scores
        trace_meta = {
            "selected_layer": int(selected_layer),
            "jsd_current": float(jsd_score),
            "divergence": float(divergence),
            "margin": float(margin),
            "instability": float(instability),
        }
        return scores, trace_meta

    def rollout_branch_total(self, generated, first_token, first_scores, use_contrast, horizon):
        token_id = int(first_token.item())
        total = float(torch.log_softmax(first_scores, dim=-1)[0, token_id].item())
        branch_tokens = [token_id]
        branch_prefix = torch.cat(
            [generated, first_token.view(1, 1).to(generated.device)],
            dim=-1,
        )
        extra_forward_passes = 0
        for _ in range(max(0, int(horizon) - 1)):
            scores, _ = self.compute_simple_view_scores(branch_prefix, use_contrast=use_contrast)
            logprobs = torch.log_softmax(scores, dim=-1)
            next_token = torch.argmax(scores, dim=-1, keepdim=True)
            next_token_id = int(next_token.item())
            total += float(logprobs[0, next_token_id].item())
            branch_tokens.append(next_token_id)
            branch_prefix = torch.cat([branch_prefix, next_token.to(branch_prefix.device)], dim=-1)
            extra_forward_passes += 1
        return total, extra_forward_passes, branch_tokens

    def choose_simple_panda_branch(self, generated, step_info, decoder_name):
        disagreement = bool(step_info["panda_disagreement"])
        if not disagreement:
            use_contrast = False
            greedy_total = None
            contrast_total = None
            selected_total = step_info["greedy_token_logprob"]
            extra_forward_passes = 0
        elif decoder_name == "simple_panda_h1":
            greedy_total = float(step_info["greedy_token_logprob"])
            contrast_total = float(step_info["contrast_token_logprob"])
            use_contrast = bool(contrast_total > greedy_total)
            selected_total = contrast_total if use_contrast else greedy_total
            extra_forward_passes = 0
        elif decoder_name == "simple_panda_h2":
            greedy_total, greedy_extra, _ = self.rollout_branch_total(
                generated,
                step_info["greedy_token"],
                step_info["greedy_scores"],
                use_contrast=False,
                horizon=2,
            )
            contrast_total, contrast_extra, _ = self.rollout_branch_total(
                generated,
                step_info["contrast_token"],
                step_info["contrast_scores"],
                use_contrast=True,
                horizon=2,
            )
            use_contrast = bool(contrast_total > greedy_total)
            selected_total = contrast_total if use_contrast else greedy_total
            extra_forward_passes = greedy_extra + contrast_extra
        else:  # pragma: no cover - protected by dispatch
            raise ValueError(f"Unsupported simple panda decoder {decoder_name!r}.")

        selected_scores = step_info["contrast_scores"] if use_contrast else step_info["greedy_scores"]
        trace_row = {
            "step": None,
            "selected_layer": int(step_info["selected_layer"]),
            "divergence": float(step_info["divergence"]),
            "margin": float(step_info["margin"]),
            "instability": float(step_info["instability"]),
            # Keep alpha-like values for backward compatibility: 0.0 = greedy, 1.0 = contrast.
            "alpha": float(1.0 if use_contrast else 0.0),
            "ablation_mode": decoder_name,
            "risk_triggered": float(step_info["panda_disagreement"]),
            "risk_score": float(step_info["panda_divergence"]),
            "jsd_current": float(step_info["jsd_current"]),
            "selection_margin": (
                float(contrast_total - greedy_total)
                if (greedy_total is not None and contrast_total is not None)
                else None
            ),
            "selection_score": float(selected_total),
            "fallback_used": float(disagreement and not use_contrast),
            "baseline_margin": float(step_info["margin"]),
            "panda_divergence": float(step_info["panda_divergence"]),
            "panda_greedy_confidence": float(step_info["panda_greedy_confidence"]),
            "panda_contrast_confidence": float(step_info["panda_contrast_confidence"]),
            "panda_safe_confidence": float(step_info["panda_safe_confidence"]),
            "panda_truth_confidence": float(step_info["panda_truth_confidence"]),
            "panda_token_mismatch": float(step_info["panda_token_mismatch"]),
            "panda_disagreement": float(step_info["panda_disagreement"]),
            "panda_selected_contrast": float(use_contrast),
            "panda_selected_truth": float(use_contrast),
            "panda_arbitration_active": float(disagreement),
            "simple_panda_horizon": 1 if decoder_name == "simple_panda_h1" else 2,
        }
        return selected_scores, trace_row, extra_forward_passes

    @staticmethod
    def token_in_top_k(scores, token_id, k):
        top_k = min(int(k), int(scores.shape[-1]))
        top_indices = torch.topk(scores, k=top_k, dim=-1).indices[0]
        return any(int(idx.item()) == int(token_id) for idx in top_indices)

    def choose_exp6_switch_branch(self, step_info, decoder_name, decoder_state=None):
        decoder_state = dict(decoder_state or {})
        greedy_token_id = int(step_info["greedy_token"].item())
        contrast_token_id = int(step_info["contrast_token"].item())
        token_mismatch = bool(greedy_token_id != contrast_token_id)
        greedy_near_contrast_top = self.token_in_top_k(
            step_info["contrast_scores"],
            greedy_token_id,
            self.exp6_guarded_top_k,
        )
        hard_disagreement = bool(token_mismatch and not greedy_near_contrast_top)
        contrast_margin_gap = float(
            step_info["contrast_scores"][0, contrast_token_id].item()
            - step_info["contrast_scores"][0, greedy_token_id].item()
        )
        sticky_remaining = int(decoder_state.get("sticky_remaining", 0))
        sticky_active = bool(sticky_remaining > 0)

        if decoder_name == "pure_argmax_switch":
            use_contrast = bool(token_mismatch)
            next_state = {}
            arbitration_active = token_mismatch
            trigger_active = token_mismatch
        elif decoder_name == "guarded_argmax_switch":
            use_contrast = bool(token_mismatch and not greedy_near_contrast_top)
            next_state = {}
            arbitration_active = token_mismatch
            trigger_active = hard_disagreement
        elif decoder_name == "sticky_contrast_switch":
            if sticky_active:
                use_contrast = True
                next_state = {"sticky_remaining": max(0, sticky_remaining - 1)}
            elif hard_disagreement:
                use_contrast = True
                next_state = {"sticky_remaining": max(0, self.exp6_sticky_hold_steps)}
            else:
                use_contrast = False
                next_state = {}
            arbitration_active = bool(token_mismatch or sticky_active)
            trigger_active = bool(hard_disagreement or sticky_active)
        elif decoder_name == "contrast_margin_switch":
            use_contrast = bool(token_mismatch and contrast_margin_gap > self.exp6_margin_threshold)
            next_state = {}
            arbitration_active = token_mismatch
            trigger_active = bool(token_mismatch and use_contrast)
        else:  # pragma: no cover - protected by dispatch
            raise ValueError(f"Unsupported exp6 decoder {decoder_name!r}.")

        selected_scores = step_info["contrast_scores"] if use_contrast else step_info["greedy_scores"]
        greedy_total = float(step_info["greedy_token_logprob"])
        contrast_total = float(step_info["contrast_token_logprob"])
        selected_total = contrast_total if use_contrast else greedy_total

        trace_row = {
            "step": None,
            "selected_layer": int(step_info["selected_layer"]),
            "divergence": float(step_info["divergence"]),
            "margin": float(step_info["margin"]),
            "instability": float(step_info["instability"]),
            "alpha": float(1.0 if use_contrast else 0.0),
            "ablation_mode": decoder_name,
            "risk_triggered": float(trigger_active),
            "risk_score": float(step_info["panda_divergence"]),
            "jsd_current": float(step_info["jsd_current"]),
            "selection_margin": float(contrast_total - greedy_total),
            "selection_score": float(selected_total),
            "fallback_used": float(arbitration_active and not use_contrast),
            "baseline_margin": float(step_info["margin"]),
            "panda_divergence": float(step_info["panda_divergence"]),
            "panda_greedy_confidence": float(step_info["panda_greedy_confidence"]),
            "panda_contrast_confidence": float(step_info["panda_contrast_confidence"]),
            "panda_safe_confidence": float(step_info["panda_safe_confidence"]),
            "panda_truth_confidence": float(step_info["panda_truth_confidence"]),
            "panda_token_mismatch": float(step_info["panda_token_mismatch"]),
            "panda_disagreement": float(step_info["panda_disagreement"]),
            "panda_selected_contrast": float(use_contrast),
            "panda_selected_truth": float(use_contrast),
            "panda_arbitration_active": float(arbitration_active),
            "exp6_hard_disagreement": float(hard_disagreement),
            "exp6_contrast_margin_gap": float(contrast_margin_gap),
            "exp6_greedy_in_contrast_topk": float(greedy_near_contrast_top),
            "exp6_sticky_active": float(sticky_active),
            "exp6_sticky_remaining": float(next_state.get("sticky_remaining", 0)),
        }
        return selected_scores, trace_row, next_state

    def select_dynamic_layer_with_interval(
        self,
        step,
        selected_layer,
        layer_logits,
        p_final,
        update_every_override=None,
    ):
        update_every = int(self.cfg.update_every if update_every_override is None else update_every_override)
        if update_every < 1:
            raise ValueError("update_every must be >= 1.")
        if step % update_every != 0:
            return int(selected_layer), False, update_every

        best_score = -float("inf")
        refreshed_layer = int(selected_layer)
        for candidate_idx in self.cfg.shallow_bucket:
            if candidate_idx >= len(layer_logits):
                continue
            candidate_logits = layer_logits[candidate_idx]
            candidate_probs = F.softmax(candidate_logits / self.cfg.tau, dim=-1)
            score = self.js_divergence(p_final, candidate_probs)
            if score > best_score:
                best_score = score
                refreshed_layer = int(candidate_idx)
        return refreshed_layer, True, update_every

    def compute_stateful_fixed_binary_view_step(
        self,
        generated,
        decoder_name,
        decoder_state,
        use_contrast,
        update_every_override=None,
    ):
        if decoder_state is None:
            decoder_state = self.init_fixed_alpha_state()

        step = int(decoder_state["step"])
        selected_layer = int(decoder_state["selected_layer"])
        layer_logits, final_logits = self.forward_with_layer_logits(generated)
        p_final = F.softmax(final_logits / self.cfg.tau, dim=-1)
        oracle_best_layer, oracle_best_jsd, _ = self.select_best_layer_by_jsd(final_logits, layer_logits)
        selected_layer, layer_refreshed, layer_update_every = self.select_dynamic_layer_with_interval(
            step,
            selected_layer,
            layer_logits,
            p_final,
            update_every_override=update_every_override,
        )

        shallow_logits = layer_logits[selected_layer]
        shallow_probs = F.softmax(shallow_logits / self.cfg.tau, dim=-1)
        current_layer_jsd = self.js_divergence(p_final, shallow_probs)
        divergence, margin, instability = self.compute_instability_terms(
            final_logits,
            shallow_logits,
            p_final,
        )
        greedy_scores, contrast_scores = self.build_binary_views(final_logits, shallow_logits)
        greedy_token = torch.argmax(greedy_scores, dim=-1)
        contrast_token = torch.argmax(contrast_scores, dim=-1)
        greedy_confidence = self.top1_confidence(greedy_scores)
        contrast_confidence = self.top1_confidence(contrast_scores)
        token_mismatch = int(int(greedy_token.item()) != int(contrast_token.item()))
        greedy_probs = F.softmax(greedy_scores / self.cfg.tau, dim=-1)
        contrast_probs = F.softmax(contrast_scores / self.cfg.tau, dim=-1)
        regime_jsd = self.js_divergence(greedy_probs, contrast_probs)
        disagreement = int(token_mismatch and regime_jsd >= float(self.panda_divergence_threshold))
        greedy_logprobs = F.log_softmax(greedy_scores, dim=-1)
        contrast_logprobs = F.log_softmax(contrast_scores, dim=-1)

        selected_scores = contrast_scores if use_contrast else greedy_scores
        selected_total = (
            float(contrast_logprobs[0, int(contrast_token.item())].item())
            if use_contrast
            else float(greedy_logprobs[0, int(greedy_token.item())].item())
        )
        greedy_total = float(greedy_logprobs[0, int(greedy_token.item())].item())
        contrast_total = float(contrast_logprobs[0, int(contrast_token.item())].item())

        trace_row = {
            "step": None,
            "selected_layer": int(selected_layer),
            "oracle_best_layer": int(oracle_best_layer),
            "selected_layer_matches_oracle": float(int(selected_layer == int(oracle_best_layer))),
            "oracle_best_layer_jsd": float(oracle_best_jsd),
            "oracle_jsd_gap": float(oracle_best_jsd - current_layer_jsd),
            "layer_refreshed": float(layer_refreshed),
            "layer_update_every": float(layer_update_every),
            "divergence": float(divergence),
            "margin": float(margin),
            "instability": float(instability),
            "alpha": float(1.0 if use_contrast else 0.0),
            "ablation_mode": decoder_name,
            "risk_triggered": float(disagreement),
            "risk_score": float(regime_jsd),
            "jsd_current": float(current_layer_jsd),
            "selection_margin": float(contrast_total - greedy_total),
            "selection_score": float(selected_total),
            "fallback_used": 0.0,
            "baseline_margin": float(margin),
            "panda_divergence": float(regime_jsd),
            "panda_greedy_confidence": float(greedy_confidence),
            "panda_contrast_confidence": float(contrast_confidence),
            "panda_safe_confidence": float(greedy_confidence),
            "panda_truth_confidence": float(contrast_confidence),
            "panda_token_mismatch": float(token_mismatch),
            "panda_disagreement": float(disagreement),
            "panda_selected_contrast": float(use_contrast),
            "panda_selected_truth": float(use_contrast),
            "panda_arbitration_active": 0.0,
        }
        next_state = {"selected_layer": int(selected_layer), "step": step + 1}
        return selected_scores, trace_row, next_state

    def compute_stateful_switch_binary_view_step(self, generated, decoder_name, decoder_state):
        if decoder_state is None:
            decoder_state = self.init_fixed_alpha_state()

        step = int(decoder_state["step"])
        selected_layer = int(decoder_state["selected_layer"])
        layer_logits, final_logits = self.forward_with_layer_logits(generated)
        p_final = F.softmax(final_logits / self.cfg.tau, dim=-1)
        selected_layer = self.select_dynamic_layer(step, selected_layer, layer_logits, p_final)

        shallow_logits = layer_logits[selected_layer]
        shallow_probs = F.softmax(shallow_logits / self.cfg.tau, dim=-1)
        current_layer_jsd = self.js_divergence(p_final, shallow_probs)
        divergence, margin, instability = self.compute_instability_terms(
            final_logits,
            shallow_logits,
            p_final,
        )
        greedy_scores, contrast_scores = self.build_binary_views(final_logits, shallow_logits)
        greedy_token = torch.argmax(greedy_scores, dim=-1)
        contrast_token = torch.argmax(contrast_scores, dim=-1)
        greedy_confidence = self.top1_confidence(greedy_scores)
        contrast_confidence = self.top1_confidence(contrast_scores)
        token_mismatch = int(int(greedy_token.item()) != int(contrast_token.item()))
        greedy_probs = F.softmax(greedy_scores / self.cfg.tau, dim=-1)
        contrast_probs = F.softmax(contrast_scores / self.cfg.tau, dim=-1)
        regime_jsd = self.js_divergence(greedy_probs, contrast_probs)
        disagreement = int(token_mismatch and regime_jsd >= float(self.panda_divergence_threshold))
        greedy_logprobs = F.log_softmax(greedy_scores, dim=-1)
        contrast_logprobs = F.log_softmax(contrast_scores, dim=-1)

        use_contrast = bool(token_mismatch)
        selected_scores = contrast_scores if use_contrast else greedy_scores
        selected_total = (
            float(contrast_logprobs[0, int(contrast_token.item())].item())
            if use_contrast
            else float(greedy_logprobs[0, int(greedy_token.item())].item())
        )
        greedy_total = float(greedy_logprobs[0, int(greedy_token.item())].item())
        contrast_total = float(contrast_logprobs[0, int(contrast_token.item())].item())

        trace_row = {
            "step": None,
            "selected_layer": int(selected_layer),
            "divergence": float(divergence),
            "margin": float(margin),
            "instability": float(instability),
            "alpha": float(1.0 if use_contrast else 0.0),
            "ablation_mode": decoder_name,
            "risk_triggered": float(token_mismatch),
            "risk_score": float(regime_jsd),
            "jsd_current": float(current_layer_jsd),
            "selection_margin": float(contrast_total - greedy_total),
            "selection_score": float(selected_total),
            "fallback_used": 0.0,
            "baseline_margin": float(margin),
            "panda_divergence": float(regime_jsd),
            "panda_greedy_confidence": float(greedy_confidence),
            "panda_contrast_confidence": float(contrast_confidence),
            "panda_safe_confidence": float(greedy_confidence),
            "panda_truth_confidence": float(contrast_confidence),
            "panda_token_mismatch": float(token_mismatch),
            "panda_disagreement": float(disagreement),
            "panda_selected_contrast": float(use_contrast),
            "panda_selected_truth": float(use_contrast),
            "panda_arbitration_active": float(token_mismatch),
        }
        next_state = {"selected_layer": int(selected_layer), "step": step + 1}
        return selected_scores, trace_row, next_state

    def compute_stateful_adaptive_lambda_step(self, generated, decoder_name, decoder_state):
        if decoder_state is None:
            decoder_state = self.init_fixed_alpha_state()

        step = int(decoder_state["step"])
        selected_layer = int(decoder_state["selected_layer"])
        layer_logits, final_logits = self.forward_with_layer_logits(generated)
        p_final = F.softmax(final_logits / self.cfg.tau, dim=-1)
        selected_layer = self.select_dynamic_layer(step, selected_layer, layer_logits, p_final)

        shallow_logits = layer_logits[selected_layer]
        shallow_probs = F.softmax(shallow_logits / self.cfg.tau, dim=-1)
        current_layer_jsd = self.js_divergence(p_final, shallow_probs)
        divergence, margin, instability = self.compute_instability_terms(
            final_logits,
            shallow_logits,
            p_final,
        )
        greedy_scores, contrast_scores = self.build_binary_views(final_logits, shallow_logits)
        greedy_token = torch.argmax(greedy_scores, dim=-1)
        contrast_token = torch.argmax(contrast_scores, dim=-1)
        greedy_confidence = self.top1_confidence(greedy_scores)
        contrast_confidence = self.top1_confidence(contrast_scores)
        greedy_entropy = self.logits_entropy(greedy_scores)
        contrast_entropy = self.logits_entropy(contrast_scores)
        final_entropy = self.logits_entropy(final_logits)
        max_entropy = float(torch.log(torch.tensor(final_logits.shape[-1], dtype=final_logits.dtype)).item())
        normalized_final_entropy = 0.0 if max_entropy <= 0.0 else min(1.0, max(0.0, final_entropy / max_entropy))
        confidence_gap = max(0.0, greedy_confidence - contrast_confidence)
        adaptive_signal = (
            self.exp9_uncertainty_weight * normalized_final_entropy
            - self.exp9_confidence_gap_weight * confidence_gap
        )
        lambda_span = self.exp9_lambda_max - self.exp9_lambda_min
        lambda_t = self.exp9_lambda_min + lambda_span * adaptive_signal
        lambda_t = max(self.exp9_lambda_min, min(self.exp9_lambda_max, lambda_t))
        adaptive_scores = final_logits - float(lambda_t) * shallow_logits

        greedy_logprobs = F.log_softmax(greedy_scores, dim=-1)
        contrast_logprobs = F.log_softmax(contrast_scores, dim=-1)
        adaptive_logprobs = F.log_softmax(adaptive_scores, dim=-1)
        adaptive_token = torch.argmax(adaptive_scores, dim=-1)
        adaptive_confidence = self.top1_confidence(adaptive_scores)
        adaptive_entropy = self.logits_entropy(adaptive_scores)
        token_mismatch = int(int(greedy_token.item()) != int(contrast_token.item()))
        greedy_probs = F.softmax(greedy_scores / self.cfg.tau, dim=-1)
        contrast_probs = F.softmax(contrast_scores / self.cfg.tau, dim=-1)
        regime_jsd = self.js_divergence(greedy_probs, contrast_probs)
        contrast_reduced = float(lambda_t < (self.exp9_lambda_max - 1e-8))

        selected_total = float(adaptive_logprobs[0, int(adaptive_token.item())].item())
        greedy_total = float(greedy_logprobs[0, int(greedy_token.item())].item())
        contrast_total = float(contrast_logprobs[0, int(contrast_token.item())].item())

        trace_row = {
            "step": None,
            "selected_layer": int(selected_layer),
            "divergence": float(divergence),
            "margin": float(margin),
            "instability": float(instability),
            "alpha": float(lambda_t),
            "ablation_mode": decoder_name,
            "risk_triggered": contrast_reduced,
            "risk_score": float(adaptive_signal),
            "jsd_current": float(current_layer_jsd),
            "selection_margin": float(selected_total - contrast_total),
            "selection_score": float(lambda_t),
            "fallback_used": 0.0,
            "baseline_margin": float(margin),
            "panda_divergence": float(regime_jsd),
            "panda_greedy_confidence": float(greedy_confidence),
            "panda_contrast_confidence": float(contrast_confidence),
            "panda_safe_confidence": float(greedy_confidence),
            "panda_truth_confidence": float(contrast_confidence),
            "panda_token_mismatch": float(token_mismatch),
            "panda_disagreement": float(
                token_mismatch and regime_jsd >= float(self.panda_divergence_threshold)
            ),
            "panda_selected_contrast": 1.0,
            "panda_selected_truth": 1.0,
            "panda_arbitration_active": 0.0,
            "exp9_lambda": float(lambda_t),
            "exp9_adaptive_signal": float(adaptive_signal),
            "exp9_normalized_final_entropy": float(normalized_final_entropy),
            "exp9_confidence_gap": float(confidence_gap),
            "exp9_greedy_entropy": float(greedy_entropy),
            "exp9_contrast_entropy": float(contrast_entropy),
            "exp9_adaptive_entropy": float(adaptive_entropy),
            "exp9_adaptive_confidence": float(adaptive_confidence),
            "exp9_contrast_reduced": contrast_reduced,
        }
        next_state = {"selected_layer": int(selected_layer), "step": step + 1}
        return adaptive_scores, trace_row, next_state

    def compute_stateful_ema_risk_switch_step(self, generated, decoder_name, decoder_state):
        if decoder_state is None:
            decoder_state = {
                **self.init_fixed_alpha_state(),
                "ema_risk": 0.0,
                "sticky_remaining": 0,
            }

        step = int(decoder_state["step"])
        selected_layer = int(decoder_state["selected_layer"])
        ema_prev = float(decoder_state.get("ema_risk", 0.0))
        sticky_remaining = int(decoder_state.get("sticky_remaining", 0))

        layer_logits, final_logits = self.forward_with_layer_logits(generated)
        p_final = F.softmax(final_logits / self.cfg.tau, dim=-1)
        if step == 0:
            selected_layer = self.select_dynamic_layer(step, selected_layer, layer_logits, p_final)

        shallow_logits = layer_logits[selected_layer]
        shallow_probs = F.softmax(shallow_logits / self.cfg.tau, dim=-1)
        current_layer_jsd = self.js_divergence(p_final, shallow_probs)
        divergence, margin, instability = self.compute_instability_terms(
            final_logits,
            shallow_logits,
            p_final,
        )
        greedy_scores, contrast_scores = self.build_binary_views(final_logits, shallow_logits)
        greedy_token = torch.argmax(greedy_scores, dim=-1)
        contrast_token = torch.argmax(contrast_scores, dim=-1)
        greedy_confidence = self.top1_confidence(greedy_scores)
        contrast_confidence = self.top1_confidence(contrast_scores)
        greedy_entropy = self.logits_entropy(greedy_scores)
        max_entropy = float(torch.log(torch.tensor(final_logits.shape[-1], dtype=final_logits.dtype)).item())
        normalized_final_entropy = 0.0 if max_entropy <= 0.0 else min(1.0, max(0.0, greedy_entropy / max_entropy))
        probability_gap = self.top1_top2_probability_gap(greedy_scores)
        margin_risk = 1.0 - min(1.0, max(0.0, probability_gap))
        normalized_layer_jsd = min(1.0, max(0.0, current_layer_jsd / math.log(2.0)))
        risk_weight_sum = self.exp10_entropy_weight + self.exp10_margin_weight + self.exp10_layer_jsd_weight
        instantaneous_risk = (
            self.exp10_entropy_weight * normalized_final_entropy
            + self.exp10_margin_weight * margin_risk
            + self.exp10_layer_jsd_weight * normalized_layer_jsd
        ) / risk_weight_sum
        ema_risk = self.exp10_risk_beta * ema_prev + (1.0 - self.exp10_risk_beta) * instantaneous_risk

        token_mismatch = int(int(greedy_token.item()) != int(contrast_token.item()))
        greedy_probs = F.softmax(greedy_scores / self.cfg.tau, dim=-1)
        contrast_probs = F.softmax(contrast_scores / self.cfg.tau, dim=-1)
        regime_jsd = self.js_divergence(greedy_probs, contrast_probs)
        disagreement = int(token_mismatch and regime_jsd >= float(self.panda_divergence_threshold))
        greedy_logprobs = F.log_softmax(greedy_scores, dim=-1)
        contrast_logprobs = F.log_softmax(contrast_scores, dim=-1)

        threshold_active = bool(ema_risk >= self.exp10_risk_threshold)
        sticky_active = bool(sticky_remaining > 0)
        if sticky_active:
            use_contrast = True
            next_sticky_remaining = max(0, sticky_remaining - 1)
        elif threshold_active and token_mismatch:
            use_contrast = True
            next_sticky_remaining = max(0, self.exp10_sticky_hold_steps)
        else:
            use_contrast = False
            next_sticky_remaining = 0

        trigger_active = bool(sticky_active or (threshold_active and token_mismatch))
        arbitration_active = bool(token_mismatch or sticky_active)
        selected_scores = contrast_scores if use_contrast else greedy_scores
        selected_total = (
            float(contrast_logprobs[0, int(contrast_token.item())].item())
            if use_contrast
            else float(greedy_logprobs[0, int(greedy_token.item())].item())
        )
        greedy_total = float(greedy_logprobs[0, int(greedy_token.item())].item())
        contrast_total = float(contrast_logprobs[0, int(contrast_token.item())].item())

        trace_row = {
            "step": None,
            "selected_layer": int(selected_layer),
            "divergence": float(divergence),
            "margin": float(margin),
            "instability": float(instability),
            "alpha": float(1.0 if use_contrast else 0.0),
            "ablation_mode": decoder_name,
            "risk_triggered": float(trigger_active),
            "risk_score": float(ema_risk),
            "jsd_current": float(current_layer_jsd),
            "selection_margin": float(contrast_total - greedy_total),
            "selection_score": float(ema_risk),
            "fallback_used": float(token_mismatch and not use_contrast),
            "baseline_margin": float(margin),
            "panda_divergence": float(regime_jsd),
            "panda_greedy_confidence": float(greedy_confidence),
            "panda_contrast_confidence": float(contrast_confidence),
            "panda_safe_confidence": float(greedy_confidence),
            "panda_truth_confidence": float(contrast_confidence),
            "panda_token_mismatch": float(token_mismatch),
            "panda_disagreement": float(disagreement),
            "panda_selected_contrast": float(use_contrast),
            "panda_selected_truth": float(use_contrast),
            "panda_arbitration_active": float(arbitration_active),
            "exp10_selected_logprob": float(selected_total),
            "exp10_instantaneous_risk": float(instantaneous_risk),
            "exp10_ema_risk": float(ema_risk),
            "exp10_normalized_final_entropy": float(normalized_final_entropy),
            "exp10_margin_risk": float(margin_risk),
            "exp10_probability_gap": float(probability_gap),
            "exp10_normalized_layer_jsd": float(normalized_layer_jsd),
            "exp10_threshold_active": float(threshold_active),
            "exp10_sticky_active": float(sticky_active),
            "exp10_sticky_remaining": float(next_sticky_remaining),
        }
        next_state = {
            "selected_layer": int(selected_layer),
            "step": step + 1,
            "ema_risk": float(ema_risk),
            "sticky_remaining": int(next_sticky_remaining),
        }
        return selected_scores, trace_row, next_state

    def compute_stateful_oracle_binary_view_step(self, generated, token_id, decoder_name, decoder_state):
        if decoder_state is None:
            decoder_state = self.init_fixed_alpha_state()

        step = int(decoder_state["step"])
        selected_layer = int(decoder_state["selected_layer"])
        layer_logits, final_logits = self.forward_with_layer_logits(generated)
        p_final = F.softmax(final_logits / self.cfg.tau, dim=-1)
        selected_layer = self.select_dynamic_layer(step, selected_layer, layer_logits, p_final)

        shallow_logits = layer_logits[selected_layer]
        shallow_probs = F.softmax(shallow_logits / self.cfg.tau, dim=-1)
        current_layer_jsd = self.js_divergence(p_final, shallow_probs)
        divergence, margin, instability = self.compute_instability_terms(
            final_logits,
            shallow_logits,
            p_final,
        )
        greedy_scores, contrast_scores = self.build_binary_views(final_logits, shallow_logits)
        greedy_token = torch.argmax(greedy_scores, dim=-1)
        contrast_token = torch.argmax(contrast_scores, dim=-1)
        greedy_confidence = self.top1_confidence(greedy_scores)
        contrast_confidence = self.top1_confidence(contrast_scores)
        token_mismatch = int(int(greedy_token.item()) != int(contrast_token.item()))
        greedy_probs = F.softmax(greedy_scores / self.cfg.tau, dim=-1)
        contrast_probs = F.softmax(contrast_scores / self.cfg.tau, dim=-1)
        regime_jsd = self.js_divergence(greedy_probs, contrast_probs)
        disagreement = int(token_mismatch and regime_jsd >= float(self.panda_divergence_threshold))
        greedy_logprobs = F.log_softmax(greedy_scores, dim=-1)
        contrast_logprobs = F.log_softmax(contrast_scores, dim=-1)

        greedy_target_logprob = float(greedy_logprobs[0, int(token_id)].item())
        contrast_target_logprob = float(contrast_logprobs[0, int(token_id)].item())
        use_contrast = bool(contrast_target_logprob > greedy_target_logprob)
        selected_scores = contrast_scores if use_contrast else greedy_scores
        selected_total = contrast_target_logprob if use_contrast else greedy_target_logprob
        selection_margin = contrast_target_logprob - greedy_target_logprob

        trace_row = {
            "step": None,
            "selected_layer": int(selected_layer),
            "divergence": float(divergence),
            "margin": float(margin),
            "instability": float(instability),
            "alpha": float(1.0 if use_contrast else 0.0),
            "ablation_mode": decoder_name,
            "risk_triggered": float(use_contrast),
            "risk_score": float(regime_jsd),
            "jsd_current": float(current_layer_jsd),
            "selection_margin": float(selection_margin),
            "selection_score": float(selected_total),
            "fallback_used": 0.0,
            "baseline_margin": float(margin),
            "panda_divergence": float(regime_jsd),
            "panda_greedy_confidence": float(greedy_confidence),
            "panda_contrast_confidence": float(contrast_confidence),
            "panda_safe_confidence": float(greedy_confidence),
            "panda_truth_confidence": float(contrast_confidence),
            "panda_token_mismatch": float(token_mismatch),
            "panda_disagreement": float(disagreement),
            "panda_selected_contrast": float(use_contrast),
            "panda_selected_truth": float(use_contrast),
            "panda_arbitration_active": float(use_contrast),
            "oracle_target_margin": float(selection_margin),
            "oracle_same_top1_switch": float(use_contrast and not token_mismatch),
        }
        next_state = {"selected_layer": int(selected_layer), "step": step + 1}
        return selected_scores, selected_total, trace_row, next_state
