from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: str


FEATURE_REGISTRY = {
    "expected_move": FeatureSpec(name="expected_move", version="1.0.0"),
    "gamma_flip": FeatureSpec(name="gamma_flip", version="1.0.0"),
}
