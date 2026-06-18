"""
@Env: /anaconda3/python3.11
@Time: 2026/3/18-9:22
@Auth: karlieswift
@File: run_finetune.py
@Desc:
"""


from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contextPrior.data import (  # noqa: E402
    MotifTokenizer,
    MotifWindowConfig,
    ProteinTokenizer,
    SequenceLabelDataset,
    build_motif_collator,
    build_sequence_collator,
    load_biomap_task,
    load_esm_tokenizer,
)
from contextPrior.models import build_model, summarize_parameter_groups  # noqa: E402
from contextPrior.training.logger import ExperimentLogger  # noqa: E402
from contextPrior.training.trainer import train_model  # noqa: E402
from contextPrior.utils import (  # noqa: E402
    build_seeded_generator,
    load_config_stack,
    make_run_dir,
    seed_worker,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Config file paths loaded in order. Later files override earlier files.",
    )
    return parser.parse_args()

def print_run_graph_debug(config: dict) -> None:
    motif_cfg = config.get("model", {}).get("motif", {})
    exp_cfg = config.get("experiment", {})

    lines = [
        f"seed: {exp_cfg.get('seed')}",
        f"motif_len: {motif_cfg.get('motif_len')}",
        f"max_motifs: {motif_cfg.get('max_motifs')}",
        f"graph topology: local_window={motif_cfg.get('graph_local_window')}, topk_nv={motif_cfg.get('graph_topk_nv')}",
        "graph gate descriptor: cosine + local_indicator + nv_similarity",
    ]

    print("\n".join(lines), flush=True)
def main():
    args = parse_args()
    config = load_config_stack(args.config)
    print_run_graph_debug(config)
    base_seed = int(config["experiment"]["seed"])
    set_seed(base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(config["output"]["root_dir"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    all_rows = []
    for task_name in config["experiment"]["tasks"]:
        set_seed(base_seed)
        task = load_biomap_task(
            task_name,
            val_fraction=float(config["data"]["val_fraction"]),
            seed=base_seed,
            train_fraction=float(config["data"].get("train_fraction", 1.0)),
        )
        num_labels = 1 if task.task_type == "regression" else task.num_labels
        built = build_model(config, num_labels=num_labels, task_type=task.task_type)

        run_dir = make_run_dir(
            output_root,
            task_name=task.task_name,
            method=config["experiment"]["method"],
            backbone_name=config["experiment"]["backbone_name"],
            seed=int(config["experiment"]["seed"]),
        )
        logger = ExperimentLogger(
            run_dir,
            use_wandb=bool(config["logging"]["use_wandb"]),
            wandb_project=str(config["logging"]["wandb_project"]),
            run_name=f"{task.task_name}-{config['experiment']['method']}-{config['experiment']['backbone_name']}",
            config=config,
        )

        loaders = build_loaders(config, built.input_kind, task, seed=base_seed)
        param_report = summarize_parameter_groups(built.model)
        (run_dir / "param_report.json").write_text(json.dumps(param_report, indent=2), encoding="utf-8")

        result = train_model(
            model=built.model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            test_loader=loaders["test"],
            task_type=task.task_type,
            task_name=task_name,
            primary_metric=task.primary_metric,
            higher_is_better=task.higher_is_better,
            device=device,
            epochs=int(config["train"]["epochs"]),
            lr=float(config["train"]["lr"]),
            weight_decay=float(config["train"]["weight_decay"]),
            max_grad_norm=float(config["train"]["max_grad_norm"]),
            run_dir=run_dir,
            logger=logger,
            config=config,
        )
        logger.close()

        row = {
            "task_name": task.task_name,
            "task_type": task.task_type,
            "split_mode": task.split_mode,
            "method": config["experiment"]["method"],
            "backbone_name": config["experiment"]["backbone_name"],
            "scale_name": config["experiment"]["scale_name"],
            "seed": int(config["experiment"]["seed"]),
            "primary_metric": task.primary_metric,
            "best_epoch": result.best_epoch,
            "best_metric_value": result.best_metric_value,
            "checkpoint_path": result.checkpoint_path,
            **param_report,
            **{f"val/{k}": v for k, v in result.val_metrics.items()},
            **{f"test/{k}": v for k, v in result.test_metrics.items()},
        }
        (run_dir / "table_row.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        all_rows.append(row)

    summary_path = Path(config["output"]["root_dir"]) / "latest_run_summary.json"
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")


def build_loaders(config: dict, input_kind: str, task, *, seed: int):
    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))

    train_ds = SequenceLabelDataset(task.train_sequences, task.train_labels, task.task_type)
    val_ds = SequenceLabelDataset(task.val_sequences, task.val_labels, task.task_type)
    test_ds = SequenceLabelDataset(task.test_sequences, task.test_labels, task.task_type)

    if input_kind == "sequence":
        tokenizer = load_esm_tokenizer(
            config["model"]["input"].get("plm_model_name_or_path", config["model"]["input"]["esm_model_name_or_path"]),
            bool(config["model"]["input"].get("esm_local_files_only", False)),
            bool(config["model"]["input"].get("plm_trust_remote_code", False)),
            config["model"]["input"].get("plm_tokenizer_use_fast"),
        )
        collate_fn = build_sequence_collator(
            tokenizer,
            int(config["data"]["esm_max_seq_len"]),
            str(config["model"]["input"].get("plm_sequence_format", "spaced_aa")),
        )
    else:
        protein_tokenizer = ProteinTokenizer()
        motif_cfg = MotifWindowConfig(
            motif_len=int(config["data"]["motif_len"]),
            motif_stride=int(config["data"]["motif_stride"]),
            max_motifs=int(config["data"]["max_motifs"]),
        )
        motif_tokenizer = MotifTokenizer(protein_tokenizer, motif_cfg)
        use_esm_residue = config["model"]["input"]["input_feature_source"] == "esm_residue"
        esm_tokenizer = None
        if use_esm_residue:
            esm_tokenizer = load_esm_tokenizer(
                config["model"]["input"]["esm_model_name_or_path"],
                bool(config["model"]["input"].get("esm_local_files_only", False)),
                bool(config["model"]["input"].get("plm_trust_remote_code", False)),
                config["model"]["input"].get("plm_tokenizer_use_fast"),
            )
        collate_fn = build_motif_collator(
            protein_tokenizer=protein_tokenizer,
            motif_tokenizer=motif_tokenizer,
            esm_tokenizer=esm_tokenizer,
            max_seq_len=int(config["data"]["esm_max_seq_len"]),
            use_esm_residue=use_esm_residue,
            use_graph_nv=bool(
                config["experiment"].get("method") == "ours"
                and int(config["model"]["motif"].get("graph_topk_nv", 2)) > 0
            ),
        )

    common_loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
    }

    return {
        "train": DataLoader(
            train_ds,
            shuffle=True,
            generator=build_seeded_generator(seed),
            **common_loader_kwargs,
        ),
        "val": DataLoader(
            val_ds,
            shuffle=False,
            generator=build_seeded_generator(seed + 1),
            **common_loader_kwargs,
        ),
        "test": DataLoader(
            test_ds,
            shuffle=False,
            generator=build_seeded_generator(seed + 2),
            **common_loader_kwargs,
        ),
    }


if __name__ == "__main__":
    main()

