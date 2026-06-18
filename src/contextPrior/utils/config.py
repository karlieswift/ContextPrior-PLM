from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: dict, update: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config_stack(config_paths: list[str | Path]) -> dict:
    config: dict = {}
    for path in config_paths:
        config = deep_merge(config, load_yaml(path))
    return config
