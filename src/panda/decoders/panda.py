"""PAnDa block-refinement decoder helpers."""

import time

import torch
import torch.nn.functional as F

from ..core import make_runtime_summary


class PandaDecoderMixin:
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
        low_alpha = self.clamp_fixed_alpha("fixed_alpha_dola_low")
        high_alpha = self.clamp_fixed_alpha("fixed_alpha_dola_high")
        # Start each speculative block from a cheap placeholder, then refine all positions in parallel.
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
                    # Arbitration only starts from the first meaningful low/high split onward.
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
                # Prefer the high-alpha view only when disagreement is active and the truth view is not weaker.
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

        # Commit only the stable prefix so unstable suffix tokens can be reconsidered next block.
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

            # A block may speculate several tokens, but only the committed prefix is appended.
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


def run_panda_block(evaluator, generated, window_size):
    return evaluator.run_panda_block(generated, window_size)


def generate_with_panda_decoder(evaluator, prompt, max_new_tokens=96, stop_on_eos=True):
    return evaluator.generate_with_panda_decoder(
        prompt,
        max_new_tokens=max_new_tokens,
        stop_on_eos=stop_on_eos,
    )
