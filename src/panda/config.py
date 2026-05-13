"""Configuration classes and constants for PAnDa evaluation."""

from dataclasses import dataclass

# Dataset sources
DEFAULT_STRATEGYQA_DATASET = "tasksource/bigbench"
DEFAULT_STRATEGYQA_CONFIG = "strategyqa"
DEFAULT_STRATEGYQA_SPLIT = "validation"
DEFAULT_STRATEGYQA_SOURCE = (
    f"{DEFAULT_STRATEGYQA_DATASET}/{DEFAULT_STRATEGYQA_CONFIG}:{DEFAULT_STRATEGYQA_SPLIT}"
)
DEFAULT_ALPACAEVAL_DATASET = "tatsu-lab/alpaca_eval"
DEFAULT_ALPACAEVAL_CONFIG = "alpaca_eval_gpt4_baseline"
DEFAULT_ALPACAEVAL_SPLIT = "eval"
DEFAULT_ALPACAEVAL_SOURCE = (
    f"{DEFAULT_ALPACAEVAL_DATASET}/{DEFAULT_ALPACAEVAL_CONFIG}:{DEFAULT_ALPACAEVAL_SPLIT}"
)

# Preset mappings
CANONICAL_COMPARISON_PRESETS = {
    "panda": "panda",
    "tbasco": "tbasco",
}

# Decoder labels for display
DECODER_LABELS = {
    "greedy": "greedy",
    "dola": "dola",
    "fixed_alpha_dola": "fixed alpha dola",
    "tbasco": "tbasco",
    "panda": "panda",
    "tbasco_low": "tbasco low",
    "tbasco_high": "tbasco high",
}


@dataclass
class DynDoLaConfig:
    """Configuration for PAnDa and DoLa decoder variants."""
    shallow_bucket: list
    tau: float = 0.5
    alpha_min: float = 0.1
    alpha_max: float = 0.8
    update_every: int = 4
    jacobi_window_size: int = 4
    jacobi_max_iters: int = 2
    panda_divergence_threshold: float = 0.05
    panda_truth_bias: float = 0.02
    panda_early_agreement_shortcut: bool = False
