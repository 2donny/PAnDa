"""Configuration classes and constants for PAnDa evaluation."""

from dataclasses import dataclass


CANONICAL_COMPARISON_PRESETS = {
    "panda": "panda",
}


FIXED_ALPHA_DECODER_ALPHAS = {
    "fixed_alpha_dola_low": 0.1,
    "fixed_alpha_dola": 0.5,
    "fixed_alpha_dola_high": 0.95,
}

DEFAULT_DECODER_NAMES = (
    "greedy",
    "dola",
    "fixed_alpha_dola_low",
    "fixed_alpha_dola",
    "fixed_alpha_dola_high",
    "panda",
)

PANDA_SAFE_ALPHA = FIXED_ALPHA_DECODER_ALPHAS["fixed_alpha_dola_low"]
PANDA_TRUTH_ALPHA = FIXED_ALPHA_DECODER_ALPHAS["fixed_alpha_dola_high"]


DECODER_LABELS = {
    "greedy": "greedy",
    "dola": "dola",
    "fixed_alpha_dola_low": "fixed alpha dola (0.1)",
    "fixed_alpha_dola": "fixed alpha dola (0.5)",
    "fixed_alpha_dola_high": "fixed alpha dola (0.95)",
    "panda": "panda",
}


@dataclass
class DynDoLaConfig:
    """Configuration for PAnDa and DoLa decoder variants."""

    shallow_bucket: list
    tau: float = 0.5
    alpha_min: float = 0.1
    alpha_max: float = 0.95
    update_every: int = 4
    jacobi_window_size: int = 4
    jacobi_max_iters: int = 2
    panda_divergence_threshold: float = 0.05
    panda_truth_bias: float = 0.02
    panda_early_agreement_shortcut: bool = False
