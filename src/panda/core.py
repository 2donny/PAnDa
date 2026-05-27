"""Shared low-level helpers for model access and runtime accounting."""


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
