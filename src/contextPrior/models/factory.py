from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from ..config import MotifInferenceConfig
from .heads import ProteinMotifClassifier


@dataclass
class BuiltModel:
    model: Any
    family: str
    input_kind: str


def build_motif_config(config: dict[str, Any]) -> MotifInferenceConfig:
    motif_cfg = dict(config["model"]["motif"])
    motif_cfg.update(config["model"]["input"])
    valid_keys = {field.name for field in fields(MotifInferenceConfig)}
    unknown_keys = set(motif_cfg) - valid_keys
    if unknown_keys:
        raise ValueError(f"Unsupported motif config keys: {sorted(unknown_keys)}")
    return MotifInferenceConfig(**{key: value for key, value in motif_cfg.items() if key in valid_keys})


def build_model(config: dict[str, Any], *, num_labels: int, task_type: str) -> BuiltModel:
    method_name = str(config["experiment"]["method"])
    if method_name != "ours":
        raise ValueError(
            "This public release only includes the ContextPrior model. "
            f"Unsupported experiment method: {method_name!r}"
        )

    motif_cfg = build_motif_config(config)
    return BuiltModel(
        model=ProteinMotifClassifier(motif_cfg, num_labels),
        family="ours",
        input_kind="motif",
    )
