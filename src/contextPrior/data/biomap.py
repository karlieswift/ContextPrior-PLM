"""
@Env: /anaconda3/python3.11
@Time: 2026/3/18-9:22
@Auth: karlieswift
@File:
@Desc:
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..tasks.registry import resolve_task_spec
from .splits import make_validation_split, subsample_indices

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - import is environment dependent
    load_dataset = None
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None


SEQUENCE_COLUMN_CANDIDATES = ["primary", "sequence", "seq", "Sequence", "protein_sequence"]
LABEL_COLUMN_CANDIDATES = ["label", "labels", "target", "targets"]


@dataclass
class LoadedTask:
    task_name: str
    task_type: str
    num_labels: int
    primary_metric: str
    higher_is_better: bool
    train_sequences: list[str]
    train_labels: list[Any]
    val_sequences: list[str]
    val_labels: list[Any]
    test_sequences: list[str]
    test_labels: list[Any]
    split_mode: str


def load_biomap_task(task_name: str, *, val_fraction: float, seed: int, train_fraction: float = 1.0) -> LoadedTask:
    if load_dataset is None:
        raise ImportError(
            "datasets is required to load BioMap tasks."
        ) from _DATASETS_IMPORT_ERROR

    dataset_name = f"biomap-research/{task_name}"
    ds = load_dataset(dataset_name,download_mode="reuse_dataset_if_exists")
    if "train" not in ds or "test" not in ds:
        raise RuntimeError(f"{task_name} must expose at least train/test splits, got {list(ds.keys())}")

    train_split = ds["train"]
    test_split = ds["test"]
    seq_key = _choose_column(train_split.column_names, SEQUENCE_COLUMN_CANDIDATES, "sequence")
    label_key = _choose_column(train_split.column_names, LABEL_COLUMN_CANDIDATES, "label")

    train_sequences_all = list(train_split[seq_key])
    train_labels_all = list(train_split[label_key])
    spec = resolve_task_spec(task_name, train_labels_all)

    if "validation" in ds:
        val_split = ds["validation"]
        val_sequences = list(val_split[seq_key])
        val_labels = list(val_split[label_key])
        split_mode = "official_validation"
        train_indices = np.arange(len(train_sequences_all))
    elif "valid" in ds:
        val_split = ds["valid"]
        val_sequences = list(val_split[seq_key])
        val_labels = list(val_split[label_key])
        split_mode = "official_valid"
        train_indices = np.arange(len(train_sequences_all))
    elif "val" in ds:
        val_split = ds["val"]
        val_sequences = list(val_split[seq_key])
        val_labels = list(val_split[label_key])
        split_mode = "official_val"
        train_indices = np.arange(len(train_sequences_all))
    else:
        train_indices, val_indices = make_validation_split(
            train_labels_all,
            task_type=spec["task_type"],
            val_fraction=val_fraction,
            seed=seed,
        )
        val_sequences = [train_sequences_all[i] for i in val_indices]
        val_labels = [train_labels_all[i] for i in val_indices]
        split_mode = f"train_split_{val_fraction:.2f}"

    if "validation" in ds or "valid" in ds or "val" in ds:
        train_indices = np.arange(len(train_sequences_all))

    train_indices = subsample_indices(train_indices, fraction=train_fraction, seed=seed)
    train_sequences = [train_sequences_all[i] for i in train_indices]
    train_labels = [train_labels_all[i] for i in train_indices]
    test_sequences = list(test_split[seq_key])
    test_labels = list(test_split[label_key])
    if spec["task_type"] == "classification":
        train_labels, val_labels, test_labels, num_labels = _normalize_classification_labels(
            train_labels,
            val_labels,
            test_labels,
        )
        spec["num_labels"] = num_labels

    return LoadedTask(
        task_name=task_name,
        task_type=spec["task_type"],
        num_labels=spec["num_labels"],
        primary_metric=spec["primary_metric"],
        higher_is_better=spec["higher_is_better"],
        train_sequences=train_sequences,
        train_labels=train_labels,
        val_sequences=val_sequences,
        val_labels=val_labels,
        test_sequences=test_sequences,
        test_labels=test_labels,
        split_mode=split_mode,
    )


def _choose_column(columns: list[str], candidates: list[str], kind: str) -> str:
    for name in candidates:
        if name in columns:
            return name
    lowered = {col.lower(): col for col in columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise KeyError(f"Could not infer {kind} column from {columns}")


def _normalize_classification_labels(
    train_labels: list[Any],
    val_labels: list[Any],
    test_labels: list[Any],
) -> tuple[list[int], list[int], list[int], int]:
    """Map BioMap classification labels to contiguous class ids.

    Some BioMap tasks use many classes (for example fold prediction). Treating
    them as regression is wrong, and CrossEntropyLoss also requires class ids in
    [0, num_labels). This helper keeps numeric 0/1 labels simple while making
    string or sparse numeric labels safe.
    """

    all_labels = list(train_labels) + list(val_labels) + list(test_labels)
    unique = sorted(set(all_labels), key=lambda x: (str(type(x)), str(x)))
    mapping = {label: idx for idx, label in enumerate(unique)}
    return (
        [mapping[label] for label in train_labels],
        [mapping[label] for label in val_labels],
        [mapping[label] for label in test_labels],
        len(unique),
    )
