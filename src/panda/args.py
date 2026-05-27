"""Argument parsing and preset application for PAnDa evaluation."""

import argparse

from .utils import normalize_comparison_preset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="PAnDa evaluation driver.")

    # Model and evaluation setup
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", choices=("sanity", "subset", "full"), default="sanity")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--results-dir", default="results/dev/run")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--strict-eval", action="store_true")

    # Dataset limits
    parser.add_argument("--truthfulqa-limit", type=str, default=None)

    # Comparison preset
    parser.add_argument(
        "--comparison-preset",
        default=None,
        metavar="{panda}",
        help="Apply the public PAnDa comparison preset.",
    )

    # Decoder configuration
    parser.add_argument(
        "--shallow-bucket",
        type=str,
        default=None,
        help="Comma-separated global shallow bucket override, e.g. 0,2,4,6,8",
    )
    parser.add_argument(
        "--jacobi-window-size",
        type=int,
        default=4,
        help="Speculative window size for PAnDa block refinement.",
    )
    parser.add_argument(
        "--jacobi-max-iters",
        type=int,
        default=2,
        help="Maximum refinement passes per speculative block.",
    )
    parser.add_argument(
        "--panda-divergence-threshold",
        type=float,
        default=0.05,
        help="Minimum low/high regime JSD required before PAnDa treats a block position as a local disagreement.",
    )
    parser.add_argument(
        "--panda-truth-bias",
        type=float,
        default=0.02,
        help="Truth-bias margin used by PAnDa when deciding whether to keep the high-alpha local update.",
    )
    parser.add_argument(
        "--panda-early-agreement-shortcut",
        action="store_true",
        help="Commit the leading PAnDa block prefix when low/high regimes do not produce a genuine conflict.",
    )
    parser.add_argument(
        "--dola-relative-top",
        type=float,
        default=0.1,
        help="Relative-top filtering value used by the official DoLa baseline.",
    )
    parser.add_argument(
        "--dola-relative-top-value",
        type=float,
        default=-1000.0,
        help="Replacement log-score assigned to tokens masked by DoLa relative-top filtering.",
    )

    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def apply_comparison_preset(args):
    """Apply preset configuration to args."""
    args.comparison_preset = normalize_comparison_preset(args.comparison_preset)

    if args.comparison_preset == "panda":
        if args.model_name == "Qwen/Qwen2.5-3B-Instruct":
            args.model_name = "HINT-lab/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration"
        args.strict_eval = True

    return args
