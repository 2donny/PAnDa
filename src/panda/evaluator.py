"""Core evaluator implementation for PAnDa and DoLa variants."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    DynDoLaConfig,
    FIXED_ALPHA_DECODER_ALPHAS,
    PANDA_SAFE_ALPHA,
    PANDA_TRUTH_ALPHA,
)
from .core import apply_final_norm, get_base_model, make_runtime_summary, merge_runtime_summaries
from .decoders import (
    BaseDecoderMixin,
    DecoderLoopMixin,
    DolaDecoderMixin,
    FixedAlphaDecoderMixin,
    PandaDecoderMixin,
)
from .utils import get_decoder_label, get_decoder_names, parse_bucket_spec

torch.set_grad_enabled(False)


class Stage4Evaluator(
    BaseDecoderMixin,
    FixedAlphaDecoderMixin,
    DolaDecoderMixin,
    PandaDecoderMixin,
    DecoderLoopMixin,
):
    """Shared runtime state plus decoder mixins for the public evaluation CLI."""

    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.use_chat_template = not args.no_chat_template

        self.decoder_names = get_decoder_names(args)
        self.decoder_labels = {name: get_decoder_label(name) for name in self.decoder_names}
        self.fixed_alpha_decoders = dict(FIXED_ALPHA_DECODER_ALPHAS)
        self.fixed_alpha_decoder_names = tuple(self.fixed_alpha_decoders)
        self.panda_low_alpha = float(PANDA_SAFE_ALPHA)
        self.panda_high_alpha = float(PANDA_TRUTH_ALPHA)
        self.jacobi_window_size = int(args.jacobi_window_size)
        self.jacobi_max_iters = int(args.jacobi_max_iters)
        self.panda_divergence_threshold = float(args.panda_divergence_threshold)
        self.panda_truth_bias = float(args.panda_truth_bias)
        self.panda_early_agreement_shortcut = bool(args.panda_early_agreement_shortcut)
        self.dola_relative_top = float(args.dola_relative_top)
        self.dola_relative_top_value = float(args.dola_relative_top_value)
        self.global_bucket_override = parse_bucket_spec(args.shallow_bucket)

        self._validate_args()
        self._print_run_config()

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model_input_device = next(self.model.parameters()).device
        self._confidence_allowed_token_ids = None
        self._confidence_first_token_ids = None

        self._configure_layers()

    def _validate_args(self):
        if self.panda_divergence_threshold < 0.0:
            raise ValueError("--panda-divergence-threshold must be >= 0")
        if self.jacobi_window_size < 1:
            raise ValueError("--jacobi-window-size must be >= 1")
        if self.jacobi_max_iters < 1:
            raise ValueError("--jacobi-max-iters must be >= 1")

    def _print_run_config(self):
        print(
            {
                "model_name": self.args.model_name,
                "model_dtype": str(self.dtype),
                "mode": self.args.mode,
                "decoders": self.decoder_labels,
                "fixed_alpha_decoders": self.fixed_alpha_decoders,
                "jacobi_config": {
                    "window_size": self.jacobi_window_size,
                    "max_iters": self.jacobi_max_iters,
                    "init_strategy": "repeat_last",
                    "commit_strategy": "stable_prefix_then_fallback_1",
                },
                "panda_config": {
                    "divergence_threshold": self.panda_divergence_threshold,
                    "truth_bias": self.panda_truth_bias,
                    "low_alpha": self.panda_low_alpha,
                    "high_alpha": self.panda_high_alpha,
                    "local_score": "top1_confidence",
                    "early_agreement_shortcut": self.panda_early_agreement_shortcut,
                },
                "strict_eval": self.args.strict_eval,
                "dola_relative_top": self.dola_relative_top,
                "dola_relative_top_value": self.dola_relative_top_value,
                "low_high_regimes": {
                    "low_alpha": self.panda_low_alpha,
                    "high_alpha": self.panda_high_alpha,
                },
            }
        )

    def _load_tokenizer(self):
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            token=self.hf_token,
            local_files_only=self.args.local_files_only,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self):
        print("Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            token=self.hf_token,
            local_files_only=self.args.local_files_only,
        )
        model.eval()
        return model

    def _configure_layers(self):
        num_layers = getattr(self.model.config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("Could not infer num_hidden_layers from model.config.")

        default_bucket = self._resolve_default_bucket(num_layers)
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

    def _resolve_default_bucket(self, num_layers):
        default_bucket = self.global_bucket_override
        if default_bucket is None:
            default_bucket = list(range(0, max(1, num_layers // 4), 2))
            if not default_bucket:
                default_bucket = [0]
        default_bucket = [idx for idx in default_bucket if 0 <= idx < num_layers]
        if not default_bucket:
            raise ValueError("No valid shallow bucket indices remain after filtering against model depth.")
        return default_bucket


def score_choices_with_decoder(evaluator, prompt, choices, decoder_name):
    from .evaluation import score_choices_with_decoder as impl

    return impl(evaluator, prompt, choices, decoder_name)


def query_pairwise_candidate_preference(evaluator, prompt, candidate_a, candidate_b):
    from . import evaluation as evaluation_module

    if not hasattr(evaluation_module, "query_pairwise_candidate_preference"):
        raise NotImplementedError("query_pairwise_candidate_preference is not implemented in panda.evaluation.")
    return evaluation_module.query_pairwise_candidate_preference(
        evaluator,
        prompt,
        candidate_a,
        candidate_b,
    )


__all__ = [
    "Stage4Evaluator",
    "apply_final_norm",
    "get_base_model",
    "make_runtime_summary",
    "merge_runtime_summaries",
    "query_pairwise_candidate_preference",
    "score_choices_with_decoder",
]
