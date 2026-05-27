"""Shared decoder mixins for greedy, fixed-alpha, and DoLa decoding."""

import math
import time

import torch
import torch.nn.functional as F

from ..core import apply_final_norm, get_base_model, make_runtime_summary


class BaseDecoderMixin:
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
        # Reuse the shallow layer whose distribution is currently farthest from the final layer.
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
        # Hidden states need the model's final norm before they are comparable through the LM head.
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


class FixedAlphaDecoderMixin:
    def get_fixed_alpha_for_decoder(self, decoder_name):
        return float(self.fixed_alpha_decoders[decoder_name])

    def clamp_fixed_alpha(self, decoder_name):
        requested_alpha = self.get_fixed_alpha_for_decoder(decoder_name)
        return max(self.cfg.alpha_min, min(self.cfg.alpha_max, requested_alpha))

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


class DolaDecoderMixin:
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
        # Official DoLa contrasts mature and premature log-prob views, not raw logits.
        contrast_scores = mature_log_probs - premature_log_probs
        contrast_scores = F.log_softmax(contrast_scores, dim=-1)
        if self.dola_relative_top > 0.0:
            # Preserve only the mature layer's plausible token set before decoding.
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


class DecoderLoopMixin:
    def decoder_step_logits(self, decoder_name, generated, decoder_state=None):
        if decoder_name == "greedy":
            logits = self.model(input_ids=generated, use_cache=False).logits[:, -1, :].float()
            return logits, decoder_state, None, False

        layer_logits, final_logits = self.forward_with_layer_logits(generated)

        if decoder_name == "dola":
            contrast_scores, trace_row = self.build_official_dola_scores(final_logits, layer_logits)
            return contrast_scores, decoder_state, trace_row, True

        if decoder_name not in self.fixed_alpha_decoder_names:
            raise ValueError(f"Unknown decoder: {decoder_name}")

        logits, next_state, trace_row = self.compute_fixed_alpha_step(
            decoder_name,
            final_logits,
            layer_logits,
            decoder_state,
        )
        return logits, next_state, trace_row, False

    def generate_with_decoder(self, prompt, decoder_name, max_new_tokens=96, stop_on_eos=True):
        if decoder_name == "panda":
            return self.generate_with_panda_decoder(prompt, max_new_tokens=max_new_tokens, stop_on_eos=stop_on_eos)
        if decoder_name not in ("greedy", "dola", *self.fixed_alpha_decoder_names):
            raise ValueError(f"Unknown decoder: {decoder_name}")

        generated = self.prepare_prompt(prompt)
        prompt_length = generated.shape[1]
        eos_token_id = self.tokenizer.eos_token_id
        trace = []
        generated_steps = 0
        forward_passes = 0
        decoder_state = (
            self.init_fixed_alpha_state()
            if decoder_name in self.fixed_alpha_decoder_names
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
                # Teacher-forced scoring only uses the first position's scores before appending the gold token.
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

        if decoder_name not in ("greedy", "dola", *self.fixed_alpha_decoder_names):
            raise ValueError(f"Unknown decoder: {decoder_name}")

        generated = self.prepare_prompt(prompt)
        decoder_state = (
            self.init_fixed_alpha_state()
            if decoder_name in self.fixed_alpha_decoder_names
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
