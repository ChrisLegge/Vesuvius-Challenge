"""Utilities for the Vesuvius surface detection research repository."""

# Core imports — scipy/numpy only, no PyTorch required.
from .config import InferenceConfig, load_inference_config
from .metadata import REQUIRED_METADATA_FIELDS, validate_model_metadata
from .metrics import composite_score, surface_dice_tau, voi_score, voi_split_merge
from .postprocess import hysteresis_components

__all__ = [
    "InferenceConfig",
    "REQUIRED_METADATA_FIELDS",
    "composite_score",
    "hysteresis_components",
    "load_inference_config",
    "surface_dice_tau",
    "validate_model_metadata",
    "voi_score",
    "voi_split_merge",
]

# PyTorch-dependent imports — model architecture and loss functions.
# Guarded so the core package remains importable in scipy-only environments
# (e.g. the CI core-tests job, analysis scripts, metric evaluation).
try:
    from .model import ResidualUNet3D, build_model
    from .losses import (
        ANTI_MERGE_WEIGHTS,
        GENERALIST_WEIGHTS,
        SURFACE_SPECIALIST_WEIGHTS,
        PhaseGatedLoss,
        PhaseWeights,
        loss_bnd,
        loss_cldice,
        loss_compreg,
        loss_gapneg,
        loss_msr,
        loss_sdf,
        loss_surfdist,
        loss_topoguide,
        loss_tversky,
    )
    __all__ += [
        "ANTI_MERGE_WEIGHTS",
        "GENERALIST_WEIGHTS",
        "SURFACE_SPECIALIST_WEIGHTS",
        "PhaseGatedLoss",
        "PhaseWeights",
        "ResidualUNet3D",
        "build_model",
        "loss_bnd",
        "loss_cldice",
        "loss_compreg",
        "loss_gapneg",
        "loss_msr",
        "loss_sdf",
        "loss_surfdist",
        "loss_topoguide",
        "loss_tversky",
    ]
except ModuleNotFoundError:
    # PyTorch not installed — model and loss functions unavailable.
    # Install with: pip install torch
    pass
