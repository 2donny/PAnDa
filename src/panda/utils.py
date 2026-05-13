"""Utility functions for text normalization and parsing."""

import re
import math

from .config import CANONICAL_COMPARISON_PRESETS, DECODER_LABELS


def normalize_text(text):
    """Normalize text for comparison."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def normalize_comparison_preset(preset_name):
    """Validate and normalize a public comparison preset name."""
    if preset_name is None:
        return None
    if preset_name in CANONICAL_COMPARISON_PRESETS:
        return preset_name
    accepted_names = tuple(CANONICAL_COMPARISON_PRESETS)
    raise ValueError(
        f"Unknown comparison preset {preset_name!r}. Supported values are: {', '.join(accepted_names)}."
    )


def parse_bucket_spec(spec):
    """Parse comma-separated bucket indices."""
    if spec is None:
        return None
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError(f"Invalid empty bucket spec: {spec!r}")
    return values


def parse_int_grid(spec):
    """Parse a comma-separated integer grid."""
    if spec is None:
        return None
    values = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError(f"Invalid empty integer grid: {spec!r}")
    return values


def parse_weight_grid(spec):
    """Parse weight triples like ``1,0.5,0.25;0.75,0.5,0.5``."""
    if spec is None:
        return None
    triples = []
    for group in str(spec).split(";"):
        group = group.strip()
        if not group:
            continue
        parts = [part.strip() for part in group.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid weight triple {group!r}; expected three comma-separated floats."
            )
        triples.append(tuple(float(part) for part in parts))
    if not triples:
        raise ValueError(f"Invalid empty weight grid: {spec!r}")
    return triples


def sample_candidate_rows(candidates, limit, rng):
    """Sample rows from candidate list."""
    population_size = len(candidates)
    if population_size == 0:
        return [], {
            "usable_row_count": 0,
            "selected_row_count": 0,
            "sampling_mode": "empty",
            "selected_source_indices": [],
        }

    if limit is None or limit >= population_size:
        selected_positions = list(range(population_size))
        sampling_mode = "all_usable_rows"
    else:
        selected_positions = sorted(rng.sample(range(population_size), int(limit)))
        sampling_mode = "seeded_random_subset"

    selected_rows = []
    selected_source_indices = []
    for position in selected_positions:
        row = dict(candidates[position])
        selected_source_indices.append(int(row.pop("source_idx")))
        selected_rows.append(row)

    manifest = {
        "usable_row_count": population_size,
        "selected_row_count": len(selected_rows),
        "sampling_mode": sampling_mode,
    }
    if sampling_mode == "seeded_random_subset":
        manifest["selected_source_indices"] = selected_source_indices
    return selected_rows, manifest


def make_sampling_rng(seed, dataset_name):
    """Create a seeded RNG for dataset sampling."""
    import random
    return random.Random(f"{int(seed)}:{dataset_name}")


def canonicalize_number_text(text):
    """Extract final number from text."""
    text = str(text).strip().replace(",", "")
    final_answer_matches = re.findall(
        r"final\s+answer\s*:\s*([-+]?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if final_answer_matches:
        return final_answer_matches[-1].rstrip(". ")
    if text.lower().startswith("final answer:"):
        text = text.split(":", 1)[1].strip()
    text = text.rstrip(". ")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return text
    return None


def canonicalize_yes_no_label(value):
    """Normalize yes/no labels to standard form."""
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return canonicalize_yes_no_label(value[0])
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return "yes" if value == 1 else "no" if value == 0 else None

    text = str(value).strip().lower()
    text = re.sub(r"^[\"'\[\(\s]+|[\"'\]\)\s]+$", "", text)
    text = re.sub(r"[.?!]+$", "", text).strip()
    label_map = {
        "yes": "yes",
        "y": "yes",
        "true": "yes",
        "1": "yes",
        "no": "no",
        "n": "no",
        "false": "no",
        "0": "no",
    }
    return label_map.get(text)


def mean_or_none(values):
    """Calculate mean of values, ignoring None."""
    import statistics
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return statistics.mean(numeric_values)


def softmax_over_scores(scores):
    """Compute softmax probabilities from scores."""
    max_score = max(scores)
    shifted = [math.exp(score - max_score) for score in scores]
    denom = sum(shifted)
    return [value / denom for value in shifted]


def resolve_limit(raw_value, mode, default_value):
    """Resolve dataset limit based on mode."""
    if raw_value is not None:
        if raw_value.lower() in {"none", "all", "full"}:
            return None
        return int(raw_value)
    if mode == "sanity":
        return default_value
    if mode == "subset":
        return 20
    return None


def get_decoder_label(decoder_name):
    """Get display label for decoder."""
    return DECODER_LABELS.get(decoder_name, decoder_name)


def get_decoder_names(args=None):
    """Get list of decoders to evaluate."""
    del args
    return ("greedy", "dola", "fixed_alpha_dola", "tbasco", "panda")
