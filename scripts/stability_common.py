"""
@Env: /anaconda3/python3.11
@Time: 2026/3/18-9:22
@Auth: karlieswift
@File: stability_common.py
@Desc:
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
    build_motif_collator,
    build_sequence_collator,
    load_esm_tokenizer,
)
from contextPrior.data.stability import (  # noqa: E402
    SequenceLabelMetadataDataset,
    wrap_collator_with_metadata,
)
from contextPrior.utils import build_seeded_generator, seed_worker  # noqa: E402


def build_stability_loaders(config: dict[str, Any], input_kind: str, task, *, seed: int):
    batch_size = int(config["data"].get("batch_size", 2))
    num_workers = int(config["data"].get("num_workers", 0))

    train_ds = SequenceLabelMetadataDataset(task.train_sequences, task.train_labels, task.train_metadata, task.task_type)
    val_ds = SequenceLabelMetadataDataset(task.val_sequences, task.val_labels, task.val_metadata, task.task_type)
    test_ds = SequenceLabelMetadataDataset(task.test_sequences, task.test_labels, task.test_metadata, task.task_type)

    if input_kind == "sequence":
        tokenizer = load_esm_tokenizer(
            config["model"]["input"].get("plm_model_name_or_path", config["model"]["input"]["esm_model_name_or_path"]),
            bool(config["model"]["input"].get("esm_local_files_only", False)),
            bool(config["model"]["input"].get("plm_trust_remote_code", False)),
            config["model"]["input"].get("plm_tokenizer_use_fast"),
        )
        base_collate = build_sequence_collator(
            tokenizer,
            int(config["data"].get("esm_max_seq_len", 1022)),
            str(config["model"]["input"].get("plm_sequence_format", "spaced_aa")),
        )
    else:
        protein_tokenizer = ProteinTokenizer()
        motif_cfg = MotifWindowConfig(
            motif_len=int(config["data"].get("motif_len", config["model"]["motif"].get("motif_len", 40))),
            motif_stride=int(config["data"].get("motif_stride", config["model"]["motif"].get("motif_stride", 20))),
            max_motifs=int(config["data"].get("max_motifs", config["model"]["motif"].get("max_motifs", 24))),
        )
        motif_tokenizer = MotifTokenizer(protein_tokenizer, motif_cfg)
        use_esm_residue = config["model"]["input"].get("input_feature_source") == "esm_residue"
        esm_tokenizer = None
        if use_esm_residue:
            esm_tokenizer = load_esm_tokenizer(
                config["model"]["input"]["esm_model_name_or_path"],
                bool(config["model"]["input"].get("esm_local_files_only", False)),
                bool(config["model"]["input"].get("plm_trust_remote_code", False)),
                config["model"]["input"].get("plm_tokenizer_use_fast"),
            )
        base_collate = build_motif_collator(
            protein_tokenizer=protein_tokenizer,
            motif_tokenizer=motif_tokenizer,
            esm_tokenizer=esm_tokenizer,
            max_seq_len=int(config["data"].get("esm_max_seq_len", 1022)),
            use_esm_residue=use_esm_residue,
            use_graph_nv=bool(
                config["experiment"].get("method") == "ours"
                and int(config["model"]["motif"].get("graph_topk_nv", 2)) > 0
            ),
        )
    collate_fn = wrap_collator_with_metadata(base_collate)

    common = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "worker_init_fn": seed_worker,
    }
    return {
        "train": DataLoader(train_ds, shuffle=True, generator=build_seeded_generator(seed), **common),
        "val": DataLoader(val_ds, shuffle=False, generator=build_seeded_generator(seed + 1), **common),
        "test": DataLoader(test_ds, shuffle=False, generator=build_seeded_generator(seed + 2), **common),
    }


def load_checkpoint_config(checkpoint_path: str | Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain config")
    return ckpt["config"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

