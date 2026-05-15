"""Configuration helpers for lightweight repository validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class InferenceConfig:
    """Small typed view over the public inference config summary."""

    active_models: tuple[str, ...]
    weights: tuple[float, ...]
    method: str
    temperature: float
    max_seconds_per_volume: int
    tta_enabled: bool
    degradation_levels: int

    @property
    def weight_sum(self) -> float:
        return sum(self.weights)

    def validate(self) -> None:
        if len(self.active_models) != len(self.weights):
            raise ValueError("active model count must match ensemble weight count")
        if not 0.999 <= self.weight_sum <= 1.001:
            raise ValueError(f"ensemble weights must sum to 1.0, got {self.weight_sum:.4f}")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.max_seconds_per_volume <= 0:
            raise ValueError("runtime budget must be positive")
        if self.degradation_levels < 1:
            raise ValueError("at least one degradation level is required")


def load_inference_config(path: str | Path) -> InferenceConfig:
    """Load and validate the public inference configuration summary."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ensemble = data["ensemble"]
    runtime = data["runtime"]
    config = InferenceConfig(
        active_models=tuple(ensemble["active_models"]),
        weights=tuple(float(weight) for weight in ensemble["weights"]),
        method=str(ensemble["method"]),
        temperature=float(ensemble["temperature"]),
        max_seconds_per_volume=int(runtime["max_seconds_per_volume"]),
        tta_enabled=bool(runtime["tta_enabled"]),
        degradation_levels=int(runtime["degradation_levels"]),
    )
    config.validate()
    return config
