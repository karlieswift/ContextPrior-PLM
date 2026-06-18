from __future__ import annotations

from typing import Any

import numpy as np


def _safe_metric(fn, default: float = float("nan"), **kwargs) -> float:
    try:
        value = fn(**kwargs)
    except Exception:
        return default
    try:
        return float(value)
    except Exception:
        return default


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    metrics: dict[str, float] = {}
    if task_type == "classification":
        pred_class = np.argmax(y_pred, axis=1)
        metrics["acc"] = _safe_metric(accuracy_score, y_true=y_true, y_pred=pred_class)
        metrics["f1_macro"] = _safe_metric(f1_score, y_true=y_true, y_pred=pred_class, average="macro")
        metrics["f1_micro"] = _safe_metric(f1_score, y_true=y_true, y_pred=pred_class, average="micro")
        metrics["mcc"] = _safe_metric(matthews_corrcoef, y_true=y_true, y_pred=pred_class)
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        if y_pred.shape[1] == 2:
            probs = _softmax(y_pred)[:, 1]
            if len(np.unique(y_true)) >= 2:
                metrics["roc_auc"] = _safe_metric(roc_auc_score, y_true=y_true, y_score=probs)
                metrics["pr_auc"] = _safe_metric(average_precision_score, y_true=y_true, y_score=probs)
        elif y_pred.shape[1] > 2 and len(np.unique(y_true)) >= 2:
            probs = _softmax(y_pred)
            labels = np.arange(y_pred.shape[1])
            metrics["roc_auc"] = _multiclass_roc_auc(y_true, probs, labels)
            metrics["pr_auc"] = _multiclass_pr_auc(y_true, probs, labels)
        return metrics

    if task_type == "multilabel":
        probs = _sigmoid(y_pred)
        pred = (probs >= 0.5).astype(np.int64)
        metrics["f1_macro"] = _safe_metric(f1_score, y_true=y_true, y_pred=pred, average="macro", zero_division=0)
        metrics["f1_micro"] = _safe_metric(f1_score, y_true=y_true, y_pred=pred, average="micro", zero_division=0)
        metrics["roc_auc_macro"] = _safe_metric(roc_auc_score, y_true=y_true, y_score=probs, average="macro")
        metrics["pr_auc_macro"] = _safe_metric(average_precision_score, y_true=y_true, y_score=probs, average="macro")
        return metrics

    pred = y_pred.reshape(-1)
    true = y_true.reshape(-1)
    metrics["mae"] = _safe_metric(mean_absolute_error, y_true=true, y_pred=pred)
    metrics["rmse"] = float(np.sqrt(mean_squared_error(true, pred)))
    metrics["r2"] = _safe_metric(r2_score, y_true=true, y_pred=pred)
    metrics["pearson"] = _pearson(true, pred)
    metrics["spearman"] = _spearman(true, pred)
    return metrics


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _multiclass_roc_auc(y_true: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(
            roc_auc_score(
                y_true,
                probs,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        return float("nan")


def _multiclass_pr_auc(y_true: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> float:
    try:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import average_precision_score

        y_bin = label_binarize(y_true, classes=labels)
        if y_bin.shape != probs.shape:
            return float("nan")
        return float(average_precision_score(y_bin, probs, average="macro"))
    except Exception:
        return float("nan")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        value = spearmanr(x, y).correlation
    except Exception:
        return float("nan")
    return float(value)


def is_better(metric_name: str, higher_is_better: bool, current: dict[str, float], best_value: float | None) -> bool:
    if metric_name not in current:
        finite_values = {k: v for k, v in current.items() if _is_finite(v)}
        if finite_values:
            fallback = next(iter(finite_values))
            print(
                f"[metrics] warning: primary metric '{metric_name}' missing; "
                f"falling back to '{fallback}' for checkpoint selection.",
                flush=True,
            )
            value = float(finite_values[fallback])
        else:
            value = float("nan")
    else:
        value = float(current[metric_name])
    if best_value is None:
        return True
    if not _is_finite(value):
        return False
    if not _is_finite(best_value):
        return True
    return value > best_value if higher_is_better else value < best_value


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def detach_to_numpy(value: Any) -> np.ndarray:
    try:
        return value.detach().cpu().numpy()
    except Exception:
        return np.asarray(value)
