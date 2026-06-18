from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..tasks.metrics import compute_metrics, detach_to_numpy, is_better
from .logger import ExperimentLogger

def print_model_parameters(model):
    # print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("=" * 50)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable Parameters: {non_trainable_params:,}")
    print("=" * 50)

    print("=" * 50)
    print("Frozen (non-trainable) parameters:")
    for name, p in model.named_parameters():
        if not p.requires_grad:
            print(f"{name:60s} shape={tuple(p.shape)}")
    print("=" * 50)


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("=" * 50)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable Parameters: {non_trainable_params:,}")
    print("=" * 50)


@dataclass
class TrainingResult:
    best_epoch: int
    best_metric_value: float
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    checkpoint_path: str


def build_loss_fn(task_type: str) -> nn.Module:
    if task_type == "classification":
        return nn.CrossEntropyLoss()
    if task_type == "multilabel":
        return nn.BCEWithLogitsLoss()
    return nn.MSELoss()
def train_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    task_type: str,
    task_name: str,
    primary_metric: str,
    higher_is_better: bool,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    run_dir: Path,
    logger: ExperimentLogger,
    config: dict[str, Any],
) -> TrainingResult:

    model.to(device)
    print_model_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = build_loss_fn(task_type)

    best_metric_value: float | None = None
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    best_ckpt = run_dir / "checkpoint_best.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            task_type=task_type,
            task_name=task_name,
            max_grad_norm=max_grad_norm,
            training=True,
            epoch=epoch,
            metric_update_every=int(config.get("train", {}).get("train_metric_update_every", 10)),
            collect_epoch_metrics=bool(config.get("train", {}).get("collect_train_metrics", True)),
        )

        val_metrics, _ = evaluate_model(model, val_loader, loss_fn, device, task_type, task_name)
        test_metrics_epoch: dict[str, float] = {}
        if bool(config.get("train", {}).get("eval_test_each_epoch", False)):
            test_metrics_epoch, _ = evaluate_model(model, test_loader, loss_fn, device, task_type, task_name)
        payload = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            **{f"test/{k}": v for k, v in test_metrics_epoch.items()},
        }
        print(payload)
        logger.log_epoch(payload)

        if is_better(primary_metric, higher_is_better, val_metrics, best_metric_value):
            best_metric_value = float(val_metrics[primary_metric])
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "config": config,
                    "task_name": task_name,
                    "task_type": task_type,
                    "primary_metric": primary_metric,
                    "higher_is_better": higher_is_better,
                },
                best_ckpt,
            )

    if not best_ckpt.exists():
        raise RuntimeError(f"No checkpoint was saved for {task_name}; check validation metric '{primary_metric}'.")
    checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    test_metrics, test_outputs = evaluate_model(model, test_loader, loss_fn, device, task_type, task_name)
    payload = {**{f"test/{k}": v for k, v in test_metrics.items()}}
    print(payload)
    logger.log_epoch({"epoch": best_epoch, **payload})

    summary = {
        "best_epoch": best_epoch,
        "primary_metric": primary_metric,
        "best_metric_value": best_metric_value,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_best": str(best_ckpt),
    }
    logger.write_summary(summary)
    logger.write_plot_data(test_outputs)
    _write_prediction_csv(
        run_dir / "predictions_test.csv",
        test_outputs["y_true"],
        test_outputs["y_pred"],
        metadata=test_outputs.get("metadata"),
    )

    return TrainingResult(
        best_epoch=best_epoch,
        best_metric_value=float(best_metric_value),
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        checkpoint_path=str(best_ckpt),
    )


def train_model1(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    task_type: str,
    task_name: str,
    primary_metric: str,
    higher_is_better: bool,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    run_dir: Path,
    logger: ExperimentLogger,
    config: dict[str, Any],
) -> TrainingResult:
    model.to(device)
    print_model_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = build_loss_fn(task_type)

    best_metric_value: float | None = None
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    best_ckpt = run_dir / "checkpoint_best.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            task_type=task_type,
            task_name=task_name,
            max_grad_norm=max_grad_norm,
            training=True,
            epoch=epoch,
            metric_update_every=int(config.get("train", {}).get("train_metric_update_every", 10)),
            collect_epoch_metrics=bool(config.get("train", {}).get("collect_train_metrics", True)),
        )
        val_metrics, _ = evaluate_model(model, val_loader, loss_fn, device, task_type,task_name)
        test_metrics_epoch: dict[str, float] = {}
        if bool(config.get("train", {}).get("eval_test_each_epoch", False)):
            test_metrics_epoch, _ = evaluate_model(model, test_loader, loss_fn, device, task_type, task_name)
        payload = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            **{f"test/{k}": v for k, v in test_metrics_epoch.items()},
        }
        print(payload)
        logger.log_epoch(payload)
        if is_better(primary_metric, higher_is_better, val_metrics, best_metric_value):
            best_metric_value = float(val_metrics[primary_metric])
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "config": config,
                },
                best_ckpt,
            )

    checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    test_metrics, test_outputs = evaluate_model(model, test_loader, loss_fn, device, task_type,task_name)
    payload = {
        **{f"test/{k}": v for k, v in test_metrics.items()},
    }
    print(payload)
    summary = {
        "best_epoch": best_epoch,
        "primary_metric": primary_metric,
        "best_metric_value": best_metric_value,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_best": str(best_ckpt),
    }
    logger.write_summary(summary)
    logger.write_plot_data(test_outputs)
    _write_prediction_csv(run_dir / "predictions_test.csv", test_outputs["y_true"], test_outputs["y_pred"])
    return TrainingResult(
        best_epoch=best_epoch,
        best_metric_value=float(best_metric_value),
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        checkpoint_path=str(best_ckpt),
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    task_type: str,
    task_name: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()
    total_loss = torch.zeros((), device=device)
    num_batches = 0
    ys_true = []
    ys_pred = []
    metadata_rows: list[Any] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval {} {}".format(task_type, task_name), leave=True):
            batch = _move_batch_to_device(batch, device)
            metadata = batch.pop("metadata", None)
            y = batch.pop("labels")
            model_inputs = _filter_model_inputs(batch)
            out = model(**model_inputs)
            logits = out["logits"]
            loss = _compute_loss(loss_fn, logits, y, task_type)
            total_loss = total_loss + loss.detach()
            num_batches += 1
            ys_true.append(detach_to_numpy(y))
            ys_pred.append(detach_to_numpy(logits))
            if metadata is not None:
                metadata_rows.extend(metadata)

    y_true = np.concatenate(ys_true, axis=0)
    y_pred = np.concatenate(ys_pred, axis=0)
    metrics = compute_metrics(y_true, y_pred, task_type)
    metrics["loss"] = float((total_loss / max(num_batches, 1)).detach().cpu())
    outputs = {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "metrics": metrics,
    }
    if metadata_rows:
        outputs["metadata"] = metadata_rows
    return metrics, outputs


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    task_type: str,
    task_name: str,
    max_grad_norm: float,
    training: bool,
    epoch: int,
    metric_update_every: int = 10,
    collect_epoch_metrics: bool = True,
) -> dict[str, float]:

    model.train(training)
    running_loss = torch.zeros((), device=device)
    total_items = 0


    abs_err_sum = 0.0
    sq_err_sum = 0.0
    sum_true = 0.0
    sum_pred = 0.0
    sum_true_sq = 0.0
    sum_pred_sq = 0.0
    sum_true_pred = 0.0
    correct = 0

    ys_true = []
    ys_pred = []

    desc = f"epoch: {epoch} {task_type} {task_name} {'train' if training else 'eval'}"
    progress = tqdm(loader, desc=desc, leave=True)
    metric_update_every = max(int(metric_update_every or 1), 1)

    for step, batch in enumerate(progress, start=1):
        batch = _move_batch_to_device(batch, device)
        batch.pop("metadata", None)
        y = batch.pop("labels")
        out = model(**_filter_model_inputs(batch))
        logits = out["logits"]
        loss = out.get("loss")
        if loss is None:
            loss = _compute_loss(loss_fn, logits, y, task_type)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        batch_size = int(y.shape[0]) if hasattr(y, "shape") and len(y.shape) > 0 else 1
        running_loss = running_loss + loss.detach()
        total_items += batch_size

        with torch.no_grad():
            if task_type == "regression":
                pred = logits.detach().float().reshape(-1)
                true = y.detach().float().reshape(-1)
                n = int(true.numel())
                if n > 0:
                    diff = pred - true
                    abs_err_sum += float(diff.abs().sum().detach().cpu())
                    sq_err_sum += float((diff * diff).sum().detach().cpu())
                    sum_true += float(true.sum().detach().cpu())
                    sum_pred += float(pred.sum().detach().cpu())
                    sum_true_sq += float((true * true).sum().detach().cpu())
                    sum_pred_sq += float((pred * pred).sum().detach().cpu())
                    sum_true_pred += float((true * pred).sum().detach().cpu())
                    total_items += n - batch_size
            elif task_type == "classification":
                pred_class = torch.argmax(logits.detach(), dim=1)
                true_class = y.detach().long().reshape(-1)
                correct += int((pred_class.reshape(-1) == true_class).sum().detach().cpu())
            elif task_type == "multilabel":
                # Exact multilabel F1/AUC are computed at epoch end.  The live bar
                # still reports loss so it remains cheap.
                pass

            if collect_epoch_metrics:
                ys_true.append(detach_to_numpy(y))
                ys_pred.append(detach_to_numpy(logits))

        if step == 1 or step % metric_update_every == 0 or step == len(loader):
            postfix: dict[str, float] = {"loss": float((running_loss / step).detach().cpu())}
            if task_type == "regression" and total_items > 0:
                postfix["mae"] = abs_err_sum / total_items
                postfix["rmse"] = float(np.sqrt(sq_err_sum / total_items))
                pearson = _online_pearson(
                    total_items,
                    sum_true,
                    sum_pred,
                    sum_true_sq,
                    sum_pred_sq,
                    sum_true_pred,
                )
                if not np.isnan(pearson):
                    postfix["pearson"] = pearson
            elif task_type == "classification" and total_items > 0:
                postfix["acc"] = correct / total_items
            progress.set_postfix({k: f"{v:.4f}" for k, v in postfix.items()})

    avg_loss = float((running_loss / max(len(loader), 1)).detach().cpu())
    metrics: dict[str, float] = {"loss": avg_loss}

    if collect_epoch_metrics and ys_true and ys_pred:
        y_true_np = np.concatenate(ys_true, axis=0)
        y_pred_np = np.concatenate(ys_pred, axis=0)
        epoch_metrics = compute_metrics(y_true_np, y_pred_np, task_type)
        metrics.update(epoch_metrics)
    else:
        if task_type == "regression" and total_items > 0:
            metrics["mae"] = abs_err_sum / total_items
            metrics["rmse"] = float(np.sqrt(sq_err_sum / total_items))
            metrics["pearson"] = _online_pearson(
                total_items,
                sum_true,
                sum_pred,
                sum_true_sq,
                sum_pred_sq,
                sum_true_pred,
            )
        elif task_type == "classification" and total_items > 0:
            metrics["acc"] = correct / total_items

    return metrics


def _online_pearson(
    n: int,
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_y2: float,
    sum_xy: float,
) -> float:
    if n < 2:
        return float("nan")
    numerator = n * sum_xy - sum_x * sum_y
    denom_x = n * sum_x2 - sum_x * sum_x
    denom_y = n * sum_y2 - sum_y * sum_y
    denominator = float(np.sqrt(max(denom_x, 0.0) * max(denom_y, 0.0)))
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _compute_loss(loss_fn: nn.Module, logits: torch.Tensor, y: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "classification":
        return loss_fn(logits, y)
    if task_type == "multilabel":
        return loss_fn(logits, y)
    return loss_fn(logits.squeeze(-1), y)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


_MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "pooling_mask",
    "x",
    "residue_mask",
    "motif_mask",
    "seq_input_ids",
    "seq_attention_mask",
    "motif_residue_index",
    "motif_nv_features",
}


def _filter_model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    """Remove metadata fields before forwarding a batch through a model."""
    return {k: v for k, v in batch.items() if k in _MODEL_INPUT_KEYS}


def _write_prediction_csv(path: Path, y_true: list[Any], y_pred: list[Any], metadata: list[Any] | None = None) -> None:
    def _normalize_row(row: Any) -> list[Any]:
        if isinstance(row, list):
            return row
        return [row]

    true_rows = [_normalize_row(row) for row in y_true]
    pred_rows = [_normalize_row(row) for row in y_pred]
    true_dim = max((len(row) for row in true_rows), default=1)
    pred_dim = max((len(row) for row in pred_rows), default=1)

    meta_keys: list[str] = []
    if metadata:
        for row in metadata:
            if isinstance(row, dict):
                for key in row.keys():
                    if key not in meta_keys:
                        meta_keys.append(key)

    header = meta_keys + [f"y_true_{i}" for i in range(true_dim)] + [f"y_pred_{i}" for i in range(pred_dim)]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, (true_row, pred_row) in enumerate(zip(true_rows, pred_rows)):
            meta_values: list[Any] = []
            if metadata:
                row_meta = metadata[idx] if idx < len(metadata) and isinstance(metadata[idx], dict) else {}
                meta_values = [row_meta.get(k, "") for k in meta_keys]
            writer.writerow(
                meta_values +
                true_row + [""] * (true_dim - len(true_row)) +
                pred_row + [""] * (pred_dim - len(pred_row))
            )
