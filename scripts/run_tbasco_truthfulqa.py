#!/usr/bin/env python3
"""Run the TBASCo TruthfulQA preset."""

from pathlib import Path

from _quick_run import run_preset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "dev" / "tbasco_truthfulqa_sanity10"


if __name__ == "__main__":
    run_preset(
        preset_name="tbasco",
        default_results_dir=DEFAULT_RESULTS_DIR,
        description="TBASCo TruthfulQA quick run",
    )
