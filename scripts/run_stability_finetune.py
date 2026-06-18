#!/usr/bin/env python
"""
@Env: /anaconda3/python3.11
@Time: 2026/3/18-9:20
@Auth: karlieswift
@File: run_stability_finetune.py
@Desc:
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stability_common import build_stability_loaders  # noqa: E402
from contextPrior.data.stability import load_stability_task_from_config  # noqa: E402
from contextPrior.models import build_model, summarize_parameter_groups  # noqa: E402
from contextPrior.training.logger import ExperimentLogger  # noqa: E402
from contextPrior.training.trainer import train_model  # noqa: E402
from contextPrior.utils import load_config_stack, make_run_dir, set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train/evaluate motif graph models on external stability datasets.")
    p.add_argument("--config", action="append", required=True, help="YAML config files loaded in order.")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config_stack(args.config)
    seed = int(config["experiment"].get("seed", 63))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = load_stability_task_from_config(config, seed=seed)
    num_labels = 1
    built = build_model(config, num_labels=num_labels, task_type=task.task_type)

    output_root = Path(config["output"]["root_dir"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    run_dir = make_run_dir(
        output_root,
        task_name=task.task_name,
        method=config["experiment"]["method"],
        backbone_name=config["experiment"]["backbone_name"],
        seed=seed,
    )
    logger = ExperimentLogger(
        run_dir,
        use_wandb=bool(config["logging"].get("use_wandb", False)),
        wandb_project=str(config["logging"].get("wandb_project", "esm-motif-stability")),
        run_name=f"{task.task_name}-{config['experiment']['method']}-{config['experiment']['backbone_name']}",
        config=config,
    )
    loaders = build_stability_loaders(config, built.input_kind, task, seed=seed)
    print(
        f"Dataset sizes for {task.task_name}: "
        f"train={len(task.train_sequences)}, val={len(task.val_sequences)}, test={len(task.test_sequences)}"
    )
    param_report = summarize_parameter_groups(built.model)
    (run_dir / "param_report.json").write_text(json.dumps(param_report, indent=2), encoding="utf-8")

    result = train_model(
        model=built.model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        task_type=task.task_type,
        task_name=task.task_name,
        primary_metric=task.primary_metric,
        higher_is_better=task.higher_is_better,
        device=device,
        epochs=int(config["train"].get("epochs", 20)),
        lr=float(config["train"].get("lr", 2e-5)),
        weight_decay=float(config["train"].get("weight_decay", 1e-2)),
        max_grad_norm=float(config["train"].get("max_grad_norm", 1.0)),
        run_dir=run_dir,
        logger=logger,
        config=config,
    )
    logger.close()
    row = {
        "task_name": task.task_name,
        "dataset_name": task.dataset_name,
        "task_type": task.task_type,
        "split_mode": task.split_mode,
        "method": config["experiment"]["method"],
        "backbone_name": config["experiment"]["backbone_name"],
        "scale_name": config["experiment"].get("scale_name"),
        "model_variant": config["experiment"].get("model_variant", "contextprior"),
        "seed": seed,
        "primary_metric": task.primary_metric,
        "best_epoch": result.best_epoch,
        "best_metric_value": result.best_metric_value,
        "checkpoint_path": result.checkpoint_path,
        "label_column": task.label_column,
        "sequence_column": task.sequence_column,
        **param_report,
        **{f"val/{k}": v for k, v in result.val_metrics.items()},
        **{f"test/{k}": v for k, v in result.test_metrics.items()},
    }
    (run_dir / "table_row.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    latest = output_root / "latest_stability_run_summary.json"
    latest.write_text(json.dumps([row], indent=2), encoding="utf-8")
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()

