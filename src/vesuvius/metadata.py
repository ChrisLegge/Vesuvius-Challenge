"""Validation helpers for tracked model metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_METADATA_FIELDS = {
    "model",
    "seed",
    "fold",
    "patch_size",
    "best_val_score",
    "best_val_breakdown",
    "model_role",
    "tl",
    "th",
    "temperature",
}


def validate_model_metadata(path: str | Path) -> dict[str, Any]:
    """Return metadata if it has the fields expected by the docs."""

    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_METADATA_FIELDS.difference(metadata))
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")

    patch_size = metadata["patch_size"]
    if len(patch_size) != 3 or any(int(axis) <= 0 for axis in patch_size):
        raise ValueError("patch_size must contain three positive axes")

    score = float(metadata["best_val_score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError("best_val_score should be a normalized proxy score")

    low = float(metadata["tl"])
    high = float(metadata["th"])
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("hysteresis thresholds must satisfy 0 <= tl < th <= 1")

    return metadata
