"""Core evaluator implementation for PAnDa, TBASCo, and DoLa variants."""

import importlib.util
import math
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DynDoLaConfig
from .utils import get_decoder_label, get_decoder_names, parse_bucket_spec

torch.set_grad_enabled(False)


def get_base_model(model):
    """Return the decoder backbone that exposes the final normalization module."""
    for attr_name in ("model", "transformer", "backbone", "base_model"):
        candidate = getattr(model, attr_name, None)
        if candidate is not None and candidate is not model:
            return candidate
    return model


def apply_final_norm(base_model, hidden_state):
    """Apply the backbone's final normalization before projecting hidden states."""
    norm_modules = [
        getattr(base_model, "norm", None),
        getattr(base_model, "ln_f", None),
        getattr(base_model, "final_layer_norm", None),
        getattr(base_model, "final_layernorm", None),
    ]
    decoder = getattr(base_model, "decoder", None)
    if decoder is not None:
        norm_modules.extend(
            [
                getattr(decoder, "norm", None),
                getattr(decoder, "final_layer_norm", None),
                getattr(decoder, "final_layernorm", None),
            ]
        )
    for norm_module in norm_modules:
        if norm_module is not None:
            return norm_module(hidden_state)
    return hidden_state


def make_runtime_summary(latency_seconds, decoder_steps, forward_passes=None, generated_tokens=None):
    latency_seconds = float(latency_seconds)
    decoder_steps = int(decoder_steps)
    forward_passes = decoder_steps if forward_passes is None else int(forward_passes)
    generated_tokens = decoder_steps if generated_tokens is None else int(generated_tokens)
    latency_per_step_ms = (1000.0 * latency_seconds / decoder_steps) if decoder_steps > 0 else None
    steps_per_second = (decoder_steps / latency_seconds) if latency_seconds > 0.0 else None
    latency_per_forward_ms = (1000.0 * latency_seconds / forward_passes) if forward_passes > 0 else None
    tokens_per_forward = (generated_tokens / forward_passes) if forward_passes > 0 else None
    return {
        "latency_seconds": latency_seconds,
        "decoder_steps": decoder_steps,
        "forward_passes": forward_passes,
        "generated_tokens": generated_tokens,
        "latency_per_step_ms": latency_per_step_ms,
        "latency_per_forward_ms": latency_per_forward_ms,
        "steps_per_second": steps_per_second,
        "tokens_per_forward": tokens_per_forward,
        "factual_speedup": tokens_per_forward,
    }


def merge_runtime_summaries(runtime_summaries):
    total_latency = 0.0
    total_steps = 0
    total_forward_passes = 0
    total_generated_tokens = 0
    for runtime_summary in runtime_summaries:
        if not runtime_summary:
            continue
        total_latency += float(runtime_summary.get("latency_seconds") or 0.0)
        total_steps += int(runtime_summary.get("decoder_steps") or 0)
        total_forward_passes += int(runtime_summary.get("forward_passes") or 0)
        total_generated_tokens += int(runtime_summary.get("generated_tokens") or 0)
    return make_runtime_summary(
        total_latency,
        total_steps,
        forward_passes=total_forward_passes,
        generated_tokens=total_generated_tokens,
    )


class Stage4Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.use_chat_template = not args.no_chat_template
        self.decoder_names = get_decoder_names(args)
        self.decoder_labels = {name: get_decoder_label(name) for name in self.decoder_names}
        self.fixed_alpha_value = float(args.fixed_alpha_value)
        self.jacobi_window_size = int(args.jacobi_window_size)
        self.jacobi_max_iters = int(args.jacobi_max_iters)
        self.panda_divergence_threshold = float(args.panda_divergence_threshold)
        self.panda_truth_bias = float(args.panda_truth_bias)
        self.panda_early_agreement_shortcut = bool(args.panda_early_agreement_shortcut)
        self.alpacaeval_max_new_tokens = int(args.alpacaeval_max_new_tokens)
        self.halueval_root = args.halueval_root
        self.halueval_tasks = tuple(
            part.strip() for part in str(args.halueval_tasks).split(",") if part.strip()
        )
        self.tbasco_low = float(args.tbasco_low)
        self.tbasco_high = float(args.tbasco_high)
        self.dola_relative_top = float(args.dola_relative_top)
        self.dola_relative_top_value = float(args.dola_relative_top_value)
        self.global_bucket_override = parse_bucket_spec(args.shallow_bucket)
        if self.panda_divergence_threshold < 0.0:
            raise ValueError("--panda-divergence-threshold must be >= 0")
        if self.tbasco_low < 0.0:
            raise ValueError("--tbasco-low must be >= 0")
        if self.tbasco_high < 0.0:
            raise ValueError("--tbasco-high must be >= 0")
        if self.tbasco_low > self.tbasco_high:
            raise ValueError("--tbasco-low must be <= --tbasco-high")
        if self.jacobi_window_size < 1:
            raise ValueError("--jacobi-window-size must be >= 1")
        if self.jacobi_max_iters < 1:
            raise ValueError("--jacobi-max-iters must be >= 1")
        if self.alpacaeval_max_new_tokens < 1:
            raise ValueError("--alpacaeval-max-new-tokens must be >= 1")
        if args.include_halueval and not self.halueval_root:
            raise ValueError("--halueval-root is required when --include-halueval is enabled")
        if args.include_halueval and not self.halueval_tasks:
            raise ValueError("--halueval-tasks must contain at least one task when --include-halueval is enabled")

        print(
            {
                "model_name": args.model_name,
                "model_dtype": str(self.dtype),
                "mode": args.mode,
                "decoders": self.decoder_labels,
                "fixed_alpha_value": self.fixed_alpha_value,
                "jacobi_config": {
                    "window_size": self.jacobi_window_size,
                    "max_iters": self.jacobi_max_iters,
                    "init_strategy": "repeat_last",
                    "commit_strategy": "stable_prefix_then_fallback_1",
                },
                "panda_config": {
                    "divergence_threshold": self.panda_divergence_threshold,
                    "truth_bias": self.panda_truth_bias,
                    "low_alpha": self.tbasco_low,
                    "high_alpha": self.tbasco_high,
                    "local_score": "top1_confidence",
                    "early_agreement_shortcut": self.panda_early_agreement_shortcut,
                },
                "alpacaeval_config": {
                    "include_alpacaeval": args.include_alpacaeval,
                    "max_new_tokens": self.alpacaeval_max_new_tokens,
                    "official_scorer_installed": importlib.util.find_spec("alpaca_eval") is not None,
                },
                "halueval_config": {
                    "include_halueval": args.include_halueval,
                    "root": self.halueval_root,
                    "tasks": self.halueval_tasks,
                },
                "strict_eval": args.strict_eval,
                "dola_relative_top": self.dola_relative_top,
                "dola_relative_top_value": self.dola_relative_top_value,
                "tbasco": {
                    "low": self.tbasco_low,
                    "high": self.tbasco_high,
                },
            }
        )

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            token=self.hf_token,
            local_files_only=args.local_files_only,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            token=self.hf_token,
            local_files_only=args.local_files_only,
        )
        self.model.eval()
        self.model_input_device = next(self.model.parameters()).device
        self._confidence_allowed_token_ids = None
        self._confidence_first_token_ids = None

        num_layers = getattr(self.model.config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("Could not infer num_hidden_layers from model.config.")
        default_bucket = self.global_bucket_override
        if default_bucket is None:
            default_bucket = list(range(0, max(1, num_layers // 4), 2))
            if not default_bucket:
                default_bucket = [0]
        default_bucket = [idx for idx in default_bucket if 0 <= idx < num_layers]
        if not default_bucket:
            raise ValueError("No valid shallow bucket indices remain after filtering against model depth.")
        self.mature_layer_index = num_layers - 1
        self.default_bucket = list(default_bucket)
        self.cfg = DynDoLaConfig(
            shallow_bucket=list(self.default_bucket),
            jacobi_window_size=self.jacobi_window_size,
            jacobi_max_iters=self.jacobi_max_iters,
            panda_divergence_threshold=self.panda_divergence_threshold,
            panda_truth_bias=self.panda_truth_bias,
            panda_early_agreement_shortcut=self.panda_early_agreement_shortcut,
        )

        print(
            {
                "model_input_device": str(self.model_input_device),
                "num_layers": num_layers,
                "default_shallow_bucket": self.default_bucket,
                "dola_mature_layer": self.mature_layer_index,
            }
        )

    def init_panda_state(self):
        return {"block_index": 0}

    @staticmethod
    def common_prefix_length(left_tokens, right_tokens):
        left_list = left_tokens[0].tolist()
        right_list = right_tokens[0].tolist()
        prefix_len = 0
        for left_token, right_token in zip(left_list, right_list):
            if int(left_token) != int(right_token):
                break
            prefix_len += 1
        return prefix_len

    @staticmethod
    def repeat_last_token_buffer(generated, window_size):
        return generated[:, -1:].repeat(1, window_size)

    @staticmethod
    def logits_entropy(logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return float((-(probs * log_probs).sum(dim=-1)).item())

    @staticmethod
    def top1_confidence(logits):
        probs = F.softmax(logits, dim=-1)
        return float(torch.max(probs, dim=-1).values.item())

    def select_dynamic_layer(self, step, selected_layer, layer_logits, p_final):
        if step % self.cfg.update_every != 0:
            return selected_layer
        best_score = -float("inf")
        for candidate_idx in self.cfg.shallow_bucket:
            if candidate_idx >= len(layer_logits):
                continue
            candidate_logits = layer_logits[candidate_idx]
            candidate_probs = F.softmax(candidate_logits / self.cfg.tau, dim=-1)
            score = self.js_divergence(p_final, candidate_probs)
            if score > best_score:
                best_score = score
                selected_layer = candidate_idx
        return selected_layer

    def compute_instability_terms(self, final_logits, shallow_logits, p_final):
        p_shallow = F.softmax(shallow_logits / self.cfg.tau, dim=-1)
        divergence = self.kl_divergence(p_final, p_shallow)
        margin = self.top1_top2_margin(final_logits)
        instability = divergence - margin
        return divergence, margin, instability

    def get_fixed_alpha_for_decoder(self, decoder_name):
        if decoder_name == "tbasco_low":
            return self.tbasco_low
        if decoder_name == "tbasco_high":
            return self.tbasco_high
        return self.fixed_alpha_value

    def clamp_fixed_alpha(self, decoder_name):
        requested_alpha = self.get_fixed_alpha_for_decoder(decoder_name)
        if decoder_name == "fixed_alpha_dola":
            return max(self.cfg.alpha_min, min(self.cfg.alpha_max, requested_alpha))
        return max(self.cfg.alpha_min, requested_alpha)

    def compute_fixed_alpha_step(self, decoder_name, final_logits, layer_logits, decoder_state):
        if decoder_state is None:
            decoder_state = self.init_fixed_alpha_state()

        step = decoder_state["step"]
        selected_layer = decoder_state["selected_layer"]
        p_final = F.softmax(final_logits / self.cfg.tau, dim=-1)

        selected_layer = self.select_dynamic_layer(step, selected_layer, layer_logits, p_final)

        shallow_logits = layer_logits[selected_layer]
        divergence, margin, instability = self.compute_instability_terms(
            final_logits,
            shallow_logits,
            p_final,
        )
        alpha = self.clamp_fixed_alpha(decoder_name)
        logits = final_logits - alpha * shallow_logits

        trace_row = {
            "step": step,
            "selected_layer": int(selected_layer),
            "divergence": float(divergence),
            "margin": float(margin),
            "instability": float(instability),
            "alpha": float(alpha),
            "ablation_mode": decoder_name,
        }
        next_state = {"selected_layer": selected_layer, "step": step + 1}
        return logits, next_state, trace_row

    def prepare_prompt(self, prompt, add_generation_prompt=True):
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            if isinstance(prompt, (list, tuple)):
                messages = list(prompt)
            else:
                messages = [{"role": "user", "content": prompt}]
            tokenized_output = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
            )
            input_ids = tokenized_output if isinstance(tokenized_output, torch.Tensor) else tokenized_output.input_ids
        else:
            if isinstance(prompt, (list, tuple)):
                prompt = "\n\n".join(
                    str(message.get("content", "")).strip()
                    for message in prompt
                    if str(message.get("content", "")).strip()
                )
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return input_ids.to(self.model_input_device)

    def forward_with_layer_logits(self, input_ids):
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        base_model = get_base_model(self.model)
        lm_head = self.model.get_output_embeddings()
        layer_logits = []
        for hidden_state in outputs.hidden_states[1:]:
            normalized = apply_final_norm(base_model, hidden_state)
            logits = lm_head(normalized[:, -1, :]).float()
            layer_logits.append(logits)
        final_logits = outputs.logits[:, -1, :].float()
        return layer_logits, final_logits

    def forward_with_window_layer_logits(self, input_ids, window_size):
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        base_model = get_base_model(self.model)
        lm_head = self.model.get_output_embeddings()
        window_slice = slice(-(window_size + 1), -1)
        layer_logits = []
        for hidden_state in outputs.hidden_states[1:]:
            normalized = apply_final_norm(base_model, hidden_state)
            logits = lm_head(normalized[:, window_slice, :]).float()
            layer_logits.append(logits)
        final_logits = outputs.logits[:, window_slice, :].float()
        return layer_logits, final_logits

    @staticmethod
    def top1_top2_margin(logits):
        top2 = torch.topk(logits, k=2, dim=-1).values[0]
        return (top2[0] - top2[1]).item()

    @staticmethod
    def js_divergence(p, q, eps=1e-8):
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)), dim=-1)
        kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)), dim=-1)
        return (0.5 * (kl_pm + kl_qm)).item()

    @staticmethod
    def kl_divergence(p, q, eps=1e-8):
        return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=-1).item()

    def select_dynamic_layers_for_window(self, final_logits, layer_logits):
        window_size = int(final_logits.shape[1])
        selected_layers = []
        jsd_scores = []
        final_probs = F.softmax(final_logits / self.cfg.tau, dim=-1)
        for position_idx in range(window_size):
            p_final = final_probs[:, position_idx, :]
            best_score = -float("inf")
            selected_layer = self.cfg.shallow_bucket[0]
            for candidate_idx in self.cfg.shallow_bucket:
                if candidate_idx >= len(layer_logits):
                    continue
                candidate_logits = layer_logits[candidate_idx][:, position_idx, :]
                candidate_probs = F.softmax(candidate_logits / self.cfg.tau, dim=-1)
                score = self.js_divergence(p_final, candidate_probs)
                if score > best_score:
                    best_score = score
                    selected_layer = candidate_idx
            selected_layers.append(int(selected_layer))
            jsd_scores.append(float(best_score))
        return selected_layers, jsd_scores

    def run_panda_block(self, generated, window_size):
        window_size = int(window_size)
        low_alpha = self.clamp_fixed_alpha("tbasco_low")
        high_alpha = self.clamp_fixed_alpha("tbasco_high")
        buffer = self.repeat_last_token_buffer(generated, window_size)
        previous_buffer = buffer.clone()
        final_rows = []
        first_scores = None
        converged = False
        passes_used = 0

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
                low_scores = final_logits_pos - low_alpha * shallow_logits
                high_scores = final_logits_pos - high_alpha * shallow_logits
                low_token = torch.argmax(low_scores, dim=-1)
                high_token = torch.argmax(high_scores, dim=-1)
                safe_confidence = self.top1_confidence(low_scores)
                truth_confidence = self.top1_confidence(high_scores)
                token_mismatch = int(int(low_token.item()) != int(high_token.item()))
                if self.panda_early_agreement_shortcut and not token_mismatch:
                    regime_jsd = 0.0
                    disagreement = 0
                else:
                    low_probs = F.softmax(low_scores / self.cfg.tau, dim=-1)
                    high_probs = F.softmax(high_scores / self.cfg.tau, dim=-1)
                    regime_jsd = self.js_divergence(low_probs, high_probs)
                    disagreement = int(token_mismatch and regime_jsd >= float(self.panda_divergence_threshold))
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
                        "low_scores": low_scores,
                        "high_scores": high_scores,
                        "low_token": low_token,
                        "high_token": high_token,
                        "panda_token_mismatch": float(token_mismatch),
                        "panda_divergence": float(regime_jsd),
                        "panda_safe_confidence": float(safe_confidence),
                        "panda_truth_confidence": float(truth_confidence),
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
                use_truth = arbitration_active and (
                    float(row["panda_truth_confidence"])
                    > float(row["panda_safe_confidence"]) - float(self.panda_truth_bias)
                )
                selected_scores = row["high_scores"] if use_truth else row["low_scores"]
                selected_token = row["high_token"] if use_truth else row["low_token"]
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
                        "alpha": float(high_alpha if use_truth else low_alpha),
                        "ablation_mode": "panda",
                        "risk_triggered": float(row["panda_disagreement"]),
                        "risk_score": float(row["panda_divergence"]),
                        "jsd_current": float(row["jsd_current"]),
                        "selection_margin": float(
                            float(row["panda_truth_confidence"]) - float(row["panda_safe_confidence"])
                        ),
                        "selection_score": float(
                            row["panda_truth_confidence"] if use_truth else row["panda_safe_confidence"]
                        ),
                        "fallback_used": float(not use_truth),
                        "baseline_margin": float(row["margin"]),
                        "jacobi_position": int(position_idx),
                        "jacobi_window_size": int(window_size),
                        "jacobi_pass_index": int(iteration_idx),
                        "panda_divergence": float(row["panda_divergence"]),
                        "panda_safe_confidence": float(row["panda_safe_confidence"]),
                        "panda_truth_confidence": float(row["panda_truth_confidence"]),
                        "panda_token_mismatch": float(row["panda_token_mismatch"]),
                        "panda_disagreement": float(row["panda_disagreement"]),
                        "panda_selected_truth": float(use_truth),
                        "panda_arbitration_active": float(arbitration_active),
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

        return {
            "buffer": buffer,
            "first_scores": first_scores,
            "position_rows": final_rows,
            "forward_passes": int(passes_used),
            "converged": bool(converged),
            "stable_prefix_len": int(stable_prefix_len),
            "commit_len": int(commit_len),
        }

    @staticmethod
    def get_relative_top_filter(scores, relative_top=0.1, min_tokens_to_keep=1):
        scores_normalized = scores.log_softmax(dim=-1)
        sorted_logits, _ = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + math.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    @staticmethod
    def official_dola_js_divergence(mature_logits, premature_logits):
        softmax_mature_layer = F.softmax(mature_logits, dim=-1)
        softmax_premature_layer = F.softmax(premature_logits, dim=-1)
        average_distribution = 0.5 * (softmax_mature_layer + softmax_premature_layer)
        log_softmax_mature_layer = F.log_softmax(mature_logits, dim=-1)
        log_softmax_premature_layer = F.log_softmax(premature_logits, dim=-1)
        kl1 = F.kl_div(log_softmax_mature_layer, average_distribution, reduction="none").mean(-1)
        kl2 = F.kl_div(log_softmax_premature_layer, average_distribution, reduction="none").mean(-1)
        return float((0.5 * (kl1 + kl2)).mean().item())

    def build_official_dola_scores(self, final_logits, layer_logits):
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
            raise ValueError("DoLa could not score any candidate layers in the current shallow bucket.")

        best_candidate = max(candidate_metrics, key=lambda row: row["jsd_current"])
        selected_layer = int(best_candidate["layer"])
        premature_logits = layer_logits[selected_layer]
        mature_log_probs = F.log_softmax(final_logits, dim=-1)
        premature_log_probs = F.log_softmax(premature_logits, dim=-1)
        contrast_scores = mature_log_probs - premature_log_probs
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
            "alpha": None,
            "ablation_mode": "official_dola",
            "jsd_current": float(best_candidate["jsd_current"]),
            "selection_score": float(best_candidate["jsd_current"]),
        }
        return contrast_scores, trace_row

    def decode_token(self, token_id):
        return self.tokenizer.decode([int(token_id)], skip_special_tokens=False)

    def decode_continuation(self, full_sequence, prompt_length):
        continuation = full_sequence[0, prompt_length:]
        return self.tokenizer.decode(continuation, skip_special_tokens=True).strip()

    @staticmethod
    def synchronize_cuda():
        if not torch.cuda.is_available():
            return
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_idx)

    def init_fixed_alpha_state(self):
        return {"selected_layer": self.cfg.shallow_bucket[0], "step": 0}

    def decoder_step_logits(self, decoder_name, generated, decoder_state=None):
        if decoder_name == "greedy":
            logits = self.model(input_ids=generated, use_cache=False).logits[:, -1, :].float()
            return logits, decoder_state, None, False

        layer_logits, final_logits = self.forward_with_layer_logits(generated)

        if decoder_name == "dola":
            contrast_scores, trace_row = self.build_official_dola_scores(final_logits, layer_logits)
            return contrast_scores, decoder_state, trace_row, True

        if decoder_name not in {"fixed_alpha_dola", "tbasco_low", "tbasco_high"}:
            raise ValueError(f"Unknown decoder: {decoder_name}")

        logits, next_state, trace_row = self.compute_fixed_alpha_step(
            decoder_name,
            final_logits,
            layer_logits,
            decoder_state,
        )
        return logits, next_state, trace_row, False

    def generate_with_panda_decoder(self, prompt, max_new_tokens=96, stop_on_eos=True):
        generated = self.prepare_prompt(prompt)
        prompt_length = generated.shape[1]
        eos_token_id = self.tokenizer.eos_token_id
        trace = []
        generated_steps = 0
        forward_passes = 0
        panda_state = self.init_panda_state()

        self.synchronize_cuda()
        start_time = time.perf_counter()
        while generated_steps < max_new_tokens:
            remaining_tokens = max_new_tokens - generated_steps
            block_window_size = min(self.jacobi_window_size, remaining_tokens)
            block_result = self.run_panda_block(generated, block_window_size)
            forward_passes += int(block_result["forward_passes"])

            commit_len = min(int(block_result["commit_len"]), remaining_tokens)
            commit_tokens = block_result["buffer"][:, :commit_len]
            if stop_on_eos and eos_token_id is not None:
                eos_positions = (commit_tokens[0] == eos_token_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    commit_len = int(eos_positions[0].item()) + 1
                    commit_tokens = commit_tokens[:, :commit_len]

            for position_idx in range(commit_len):
                row = dict(block_result["position_rows"][position_idx])
                row["step"] = len(trace)
                row["token_id"] = int(commit_tokens[0, position_idx].item())
                row["token_text"] = self.decode_token(commit_tokens[0, position_idx].item())
                row["jacobi_block_index"] = int(panda_state["block_index"])
                trace.append(row)

            generated = torch.cat([generated, commit_tokens.to(generated.device)], dim=-1)
            generated_steps += commit_len
            panda_state["block_index"] += 1

            if stop_on_eos and eos_token_id is not None and commit_tokens[0, -1].item() == eos_token_id:
                break

        self.synchronize_cuda()
        elapsed = time.perf_counter() - start_time

        return self.decode_continuation(generated, prompt_length), trace, make_runtime_summary(
            elapsed,
            generated_steps,
            forward_passes=forward_passes,
            generated_tokens=generated_steps,
        )

    def generate_with_decoder(self, prompt, decoder_name, max_new_tokens=96, stop_on_eos=True):
        if decoder_name == "tbasco":
            from .evaluation import generate_with_tbasco_decoder

            return generate_with_tbasco_decoder(
                self,
                prompt,
                max_new_tokens=max_new_tokens,
                stop_on_eos=stop_on_eos,
            )
        if decoder_name == "panda":
            return self.generate_with_panda_decoder(prompt, max_new_tokens=max_new_tokens, stop_on_eos=stop_on_eos)
        if decoder_name not in {"greedy", "dola", "fixed_alpha_dola", "tbasco_low", "tbasco_high"}:
            raise ValueError(f"Unknown decoder: {decoder_name}")

        generated = self.prepare_prompt(prompt)
        prompt_length = generated.shape[1]
        eos_token_id = self.tokenizer.eos_token_id
        trace = []
        generated_steps = 0
        forward_passes = 0
        decoder_state = (
            self.init_fixed_alpha_state()
            if decoder_name in {"fixed_alpha_dola", "tbasco_low", "tbasco_high"}
            else None
        )

        self.synchronize_cuda()
        start_time = time.perf_counter()
        for _ in range(max_new_tokens):
            scores, decoder_state, trace_row, _ = self.decoder_step_logits(decoder_name, generated, decoder_state)
            extra_forward_passes = 0 if trace_row is None else int(trace_row.get("extra_forward_passes") or 0)
            forward_passes += 1 + extra_forward_passes
            next_token = torch.argmax(scores, dim=-1, keepdim=True)
            generated_steps += 1
            if trace_row is not None:
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

    def candidate_to_ids(self, choice_text):
        if not choice_text.startswith((" ", "\n")):
            choice_text = " " + choice_text
        ids = self.tokenizer(choice_text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
        if not ids:
            raise ValueError(f"Empty tokenization for choice: {choice_text!r}")
        return ids

    def score_candidate_with_decoder(self, prompt, decoder_name, choice_text):
        if decoder_name == "panda":
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
                block_result = self.run_panda_block(generated, block_window_size)
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

        if decoder_name == "tbasco":
            raise ValueError("Use TBASCo-specific choice scoring helpers instead of score_candidate_with_decoder().")
        if decoder_name not in {"greedy", "dola", "fixed_alpha_dola", "tbasco_low", "tbasco_high"}:
            raise ValueError(f"Unknown decoder: {decoder_name}")

        generated = self.prepare_prompt(prompt)
        decoder_state = (
            self.init_fixed_alpha_state()
            if decoder_name in {"fixed_alpha_dola", "tbasco_low", "tbasco_high"}
            else None
        )
        total_logprob = 0.0
        token_ids = self.candidate_to_ids(choice_text)
        trace = []
        forward_passes = 0

        self.synchronize_cuda()
        start_time = time.perf_counter()
        for token_id in token_ids:
            scores, decoder_state, trace_row, scores_are_logprobs = self.decoder_step_logits(
                decoder_name, generated, decoder_state
            )
            extra_forward_passes = 0 if trace_row is None else int(trace_row.get("extra_forward_passes") or 0)
            forward_passes += 1 + extra_forward_passes
            if scores_are_logprobs:
                total_logprob += float(scores[0, token_id].item())
            else:
                logprobs = torch.log_softmax(scores, dim=-1)
                total_logprob += float(logprobs[0, token_id].item())
            next_token = torch.tensor([[token_id]], device=generated.device)
            if trace_row is not None:
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


def score_choices_with_decoder(evaluator, prompt, choices, decoder_name):
    from .evaluation import score_choices_with_decoder as impl

    return impl(evaluator, prompt, choices, decoder_name)


def query_pairwise_candidate_preference(evaluator, prompt, candidate_a, candidate_b):
    from .evaluation import query_pairwise_candidate_preference as impl

    return impl(evaluator, prompt, candidate_a, candidate_b)


__all__ = [
    "Stage4Evaluator",
    "apply_final_norm",
    "get_base_model",
    "make_runtime_summary",
    "merge_runtime_summaries",
    "query_pairwise_candidate_preference",
    "score_choices_with_decoder",
]
