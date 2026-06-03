"""Small import-time compatibility shims for fragile local environments."""

from __future__ import annotations

import importlib.util


_BLOCKED_OPTIONAL_PACKAGES = {
    "keras",
    "scipy",
    "sklearn",
    "tensorflow",
    "tf_keras",
}


def suppress_problematic_optional_dependency_detection():
    """Hide optional stacks that crash this repo's local Transformers imports.

    Transformers uses ``importlib.util.find_spec`` to decide whether optional
    dependencies such as scikit-learn, SciPy, and TensorFlow are available.
    In this environment those packages are installed, but some are compiled
    against an older NumPy ABI and crash during import. We only need the causal
    LM path here, so hiding these optional packages is the safest narrow fix.
    """

    current_find_spec = importlib.util.find_spec
    if getattr(current_find_spec, "_panda_optional_dep_patch", False):
        return

    def patched_find_spec(name, package=None):
        root_name = name.split(".", 1)[0]
        if root_name in _BLOCKED_OPTIONAL_PACKAGES:
            return None
        return current_find_spec(name, package)

    patched_find_spec._panda_optional_dep_patch = True
    importlib.util.find_spec = patched_find_spec
