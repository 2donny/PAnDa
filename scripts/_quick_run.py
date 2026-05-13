#!/usr/bin/env python3
"""Shared quick-run launcher for public benchmark presets."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
INSTALL_CMD = [sys.executable, "-m", "pip", "install", "-e", str(REPO_ROOT)]


def _parse_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help=(
            "Install project dependencies into the current Python environment before running. "
            "Useful for a first-time quick run."
        ),
    )
    return parser.parse_known_args(argv)


def _ensure_repo_root():
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(SRC_ROOT))


def _bootstrap_environment():
    print({"quick_run": "bootstrap", "repo_root": str(REPO_ROOT), "command": " ".join(INSTALL_CMD)})
    subprocess.check_call(INSTALL_CMD)


def _load_entrypoint():
    try:
        return importlib.import_module("panda.entrypoint")
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "unknown"
        install_hint = " ".join(INSTALL_CMD)
        raise SystemExit(
            "\n".join(
                [
                    f"Missing Python dependency: {missing_module}",
                    f"Run this once, then rerun the script:\n  {install_hint}",
                    "Or use the script's bootstrap mode:\n"
                    f"  {Path(sys.argv[0]).name} --bootstrap",
                ]
            )
        ) from exc


def run_preset(*, preset_name, default_results_dir, description, argv=None):
    wrapper_args, passthrough_args = _parse_wrapper_args(sys.argv[1:] if argv is None else argv)
    _ensure_repo_root()

    if wrapper_args.bootstrap:
        _bootstrap_environment()

    entrypoint = _load_entrypoint()
    script_path = Path(sys.argv[0]).resolve()
    default_args = [
        "--comparison-preset",
        preset_name,
        "--mode",
        "sanity",
        "--save-results",
        "--results-dir",
        str(default_results_dir),
    ]

    print(
        {
            "quick_run": description,
            "script": str(script_path),
            "repo_root": str(REPO_ROOT),
            "preset": preset_name,
            "results_dir": str(default_results_dir),
        }
    )

    sys.argv = [sys.argv[0], *default_args, *passthrough_args]
    entrypoint.main()

