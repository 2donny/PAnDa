"""Experiment-local decoder variants kept outside the main package."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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

MATCHED_ALPHA_VALUES = {
    "matched_alpha_dola_0p10": 0.10,
    "matched_alpha_dola_0p50": 0.50,
    "matched_alpha_dola_0p95": 0.95,
}


class ExperimentEvaluator(Stage4Evaluator):
    def __init__(self, args, decoder_names):
        self._experiment_decoder_names = tuple(decoder_names)
        super().__init__(args)
        self.decoder_names = self._experiment_decoder_names
        self.decoder_labels = {name: name for name in self.decoder_names}

    def _print_run_config(self):
        print(
            {
                "model_name": self.args.model_name,
                "mode": self.args.mode,
                "strict_eval": self.args.strict_eval,
                "experiment_decoders": list(self._experiment_decoder_names),
                "panda_config": {
                    "divergence_threshold": float(self.args.panda_divergence_threshold),
                    "truth_bias": float(self.args.panda_truth_bias),
                    "low_alpha": 0.1,
                    "high_alpha": 0.95,
                },
                "dola_relative_top": float(self.args.dola_relative_top),
                "dola_relative_top_value": float(self.args.dola_relative_top_value),
            }
        )

    def score_candidate_with_decoder(self, prompt, decoder_name, choice_text):
        if decoder_name == "official_dola":
            total_logprob, trace, runtime = super().score_candidate_with_decoder(prompt, "dola", choice_text)
            for row in trace:
                row["ablation_mode"] = "official_dola"
            return total_logprob, trace, runtime

        if decoder_name in MATCHED_ALPHA_VALUES:
            return self._score_candidate_with_custom_step(prompt, choice_text, decoder_name)

        if decoder_name in EXP2_DECODER_NAMES:
            return self._score_candidate_with_custom_step(prompt, choice_text, decoder_name)

        return super().score_candidate_with_decoder(prompt, decoder_name, choice_text)

    def _score_candidate_with_custom_step(self, prompt, choice_text, decoder_name):
        generated = self.prepare_prompt(prompt)
        total_logprob = 0.0
        token_ids = self.candidate_to_ids(choice_text)
        trace = []
        forward_passes = 0

        self.synchronize_cuda()
        start_time = time.perf_counter()
        for token_id in token_ids:
            scores, scores_are_logprobs, trace_row, step_forward_passes = self._custom_decoder_step_scores(
                generated,
                decoder_name,
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

    def _custom_decoder_step_scores(self, generated, decoder_name):
        if decoder_name in MATCHED_ALPHA_VALUES:
            layer_logits, final_logits = self.forward_with_layer_logits(generated)
            scores, trace_row = self.build_matched_alpha_dola_scores(
                final_logits,
                layer_logits,
                alpha=MATCHED_ALPHA_VALUES[decoder_name],
                decoder_name=decoder_name,
            )
            return scores, True, trace_row, 1

        if decoder_name in EXP2_DECODER_NAMES:
            step_info = self.build_simple_panda_step(generated)
            selected_scores, trace_row, extra_forward_passes = self.choose_simple_panda_branch(
                generated,
                step_info,
                decoder_name,
            )
            return selected_scores, False, trace_row, 1 + int(extra_forward_passes)

        raise ValueError(f"Unknown experiment-local decoder {decoder_name!r}.")

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

    def build_simple_panda_step(self, generated):
        low_alpha = self.panda_low_alpha
        high_alpha = self.panda_high_alpha
        layer_logits, final_logits = self.forward_with_layer_logits(generated)
        selected_layer, jsd_score, p_final = self.select_best_layer_by_jsd(final_logits, layer_logits)
        shallow_logits = layer_logits[selected_layer]
        divergence, margin, instability = self.compute_instability_terms(final_logits, shallow_logits, p_final)
        low_scores = final_logits - low_alpha * shallow_logits
        high_scores = final_logits - high_alpha * shallow_logits
        low_token = torch.argmax(low_scores, dim=-1)
        high_token = torch.argmax(high_scores, dim=-1)
        safe_confidence = self.top1_confidence(low_scores)
        truth_confidence = self.top1_confidence(high_scores)
        token_mismatch = int(int(low_token.item()) != int(high_token.item()))
        low_probs = F.softmax(low_scores / self.cfg.tau, dim=-1)
        high_probs = F.softmax(high_scores / self.cfg.tau, dim=-1)
        regime_jsd = self.js_divergence(low_probs, high_probs)
        disagreement = int(token_mismatch and regime_jsd >= float(self.panda_divergence_threshold))
        low_logprobs = F.log_softmax(low_scores, dim=-1)
        high_logprobs = F.log_softmax(high_scores, dim=-1)
        return {
            "selected_layer": int(selected_layer),
            "jsd_current": float(jsd_score),
            "divergence": float(divergence),
            "margin": float(margin),
            "instability": float(instability),
            "low_scores": low_scores,
            "high_scores": high_scores,
            "low_token": low_token,
            "high_token": high_token,
            "low_token_logprob": float(low_logprobs[0, int(low_token.item())].item()),
            "high_token_logprob": float(high_logprobs[0, int(high_token.item())].item()),
            "panda_token_mismatch": float(token_mismatch),
            "panda_divergence": float(regime_jsd),
            "panda_safe_confidence": float(safe_confidence),
            "panda_truth_confidence": float(truth_confidence),
            "panda_disagreement": float(disagreement),
        }

    def compute_simple_view_scores(self, generated, alpha):
        layer_logits, final_logits = self.forward_with_layer_logits(generated)
        selected_layer, jsd_score, p_final = self.select_best_layer_by_jsd(final_logits, layer_logits)
        shallow_logits = layer_logits[selected_layer]
        divergence, margin, instability = self.compute_instability_terms(final_logits, shallow_logits, p_final)
        scores = final_logits - float(alpha) * shallow_logits
        trace_meta = {
            "selected_layer": int(selected_layer),
            "jsd_current": float(jsd_score),
            "divergence": float(divergence),
            "margin": float(margin),
            "instability": float(instability),
        }
        return scores, trace_meta

    def rollout_branch_total(self, generated, first_token, first_scores, alpha, horizon):
        token_id = int(first_token.item())
        total = float(torch.log_softmax(first_scores, dim=-1)[0, token_id].item())
        branch_tokens = [token_id]
        branch_prefix = torch.cat(
            [generated, first_token.view(1, 1).to(generated.device)],
            dim=-1,
        )
        extra_forward_passes = 0
        for _ in range(max(0, int(horizon) - 1)):
            scores, _ = self.compute_simple_view_scores(branch_prefix, alpha)
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
            choose_high = False
            low_total = None
            high_total = None
            selected_total = step_info["low_token_logprob"]
            extra_forward_passes = 0
        elif decoder_name == "simple_panda_h1":
            low_total = float(step_info["low_token_logprob"])
            high_total = float(step_info["high_token_logprob"])
            choose_high = bool(high_total > low_total)
            selected_total = high_total if choose_high else low_total
            extra_forward_passes = 0
        elif decoder_name == "simple_panda_h2":
            low_total, low_extra, _ = self.rollout_branch_total(
                generated,
                step_info["low_token"],
                step_info["low_scores"],
                self.panda_low_alpha,
                horizon=2,
            )
            high_total, high_extra, _ = self.rollout_branch_total(
                generated,
                step_info["high_token"],
                step_info["high_scores"],
                self.panda_high_alpha,
                horizon=2,
            )
            choose_high = bool(high_total > low_total)
            selected_total = high_total if choose_high else low_total
            extra_forward_passes = low_extra + high_extra
        else:  # pragma: no cover - protected by dispatch
            raise ValueError(f"Unsupported simple panda decoder {decoder_name!r}.")

        selected_scores = step_info["high_scores"] if choose_high else step_info["low_scores"]
        trace_row = {
            "step": None,
            "selected_layer": int(step_info["selected_layer"]),
            "divergence": float(step_info["divergence"]),
            "margin": float(step_info["margin"]),
            "instability": float(step_info["instability"]),
            "alpha": float(self.panda_high_alpha if choose_high else self.panda_low_alpha),
            "ablation_mode": decoder_name,
            "risk_triggered": float(step_info["panda_disagreement"]),
            "risk_score": float(step_info["panda_divergence"]),
            "jsd_current": float(step_info["jsd_current"]),
            "selection_margin": (
                float(high_total - low_total) if (low_total is not None and high_total is not None) else None
            ),
            "selection_score": float(selected_total),
            "fallback_used": float(disagreement and not choose_high),
            "baseline_margin": float(step_info["margin"]),
            "panda_divergence": float(step_info["panda_divergence"]),
            "panda_safe_confidence": float(step_info["panda_safe_confidence"]),
            "panda_truth_confidence": float(step_info["panda_truth_confidence"]),
            "panda_token_mismatch": float(step_info["panda_token_mismatch"]),
            "panda_disagreement": float(step_info["panda_disagreement"]),
            "panda_selected_truth": float(choose_high),
            "panda_arbitration_active": float(disagreement),
            "simple_panda_horizon": 1 if decoder_name == "simple_panda_h1" else 2,
        }
        return selected_scores, trace_row, extra_forward_passes

