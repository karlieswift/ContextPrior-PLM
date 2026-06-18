from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


def make_validation_split(
    labels: list[Any],
    *,
    task_type: str,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(labels)
    indices = np.arange(n)
    if val_fraction <= 0 or val_fraction >= 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    if task_type == "classification":
        try:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_fraction,
                random_state=seed,
                stratify=np.asarray(labels),
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_fraction,
                random_state=seed,
                stratify=None,
            )
        return np.sort(train_idx), np.sort(val_idx)

    if task_type == "multilabel":
        stratify = np.asarray(labels).sum(axis=1)
        try:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_fraction,
                random_state=seed,
                stratify=stratify,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_fraction,
                random_state=seed,
                stratify=None,
            )
        return np.sort(train_idx), np.sort(val_idx)

    y = np.asarray(labels, dtype=np.float32)
    bins = _regression_bins(y)
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            stratify=bins,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            stratify=None,
        )
    return np.sort(train_idx), np.sort(val_idx)


def subsample_indices(indices: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.sort(indices)
    if fraction <= 0.0:
        raise ValueError(f"fraction must be > 0, got {fraction}")
    rng = np.random.default_rng(seed)
    count = max(1, int(round(len(indices) * fraction)))
    chosen = rng.choice(indices, size=count, replace=False)
    return np.sort(chosen)


def _regression_bins(y: np.ndarray, num_bins: int = 10) -> np.ndarray:
    if y.size < num_bins:
        return np.zeros_like(y, dtype=np.int64)
    quantiles = np.quantile(y, np.linspace(0.0, 1.0, num_bins + 1))
    quantiles[0] -= 1e-8
    quantiles[-1] += 1e-8
    bins = np.digitize(y, quantiles[1:-1], right=False)
    return bins.astype(np.int64)
