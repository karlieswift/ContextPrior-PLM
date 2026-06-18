"""
@Env: /anaconda3/python3.11
@Time: 2026/3/18-9:22
@Auth: karlieswift
@File:
@Desc:
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .motif_graph_features import build_motif_nv_feature_tensors
from .tokenization import MotifTokenizer, normalize_sequence_for_esm, normalize_sequence_for_motif, normalize_sequence_for_plm

try:
    from transformers import AutoTokenizer, EsmTokenizer
except ImportError as exc:  # pragma: no cover - import is environment dependent
    AutoTokenizer = None
    EsmTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


class SequenceLabelDataset(Dataset):
    def __init__(self, sequences: list[str], labels: list[Any], task_type: str):
        self.sequences = sequences
        self.labels = labels
        self.task_type = task_type

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        if self.task_type == "classification":
            label = torch.tensor(int(label), dtype=torch.long)
        elif self.task_type == "multilabel":
            label = torch.tensor(np.asarray(label), dtype=torch.float32)
        else:
            label = torch.tensor(float(label), dtype=torch.float32)
        return self.sequences[idx], label


def load_esm_tokenizer(
    model_name_or_path: str,
    local_files_only: bool,
    trust_remote_code: bool = False,
    use_fast: bool | None = None,
):
    if EsmTokenizer is None and AutoTokenizer is None:
        raise ImportError(
            "transformers with AutoTokenizer/EsmTokenizer is required to run sequence PLM experiments."
        ) from _TRANSFORMERS_IMPORT_ERROR
    if AutoTokenizer is not None:
        kwargs = {
            "local_files_only": bool(local_files_only),
            "trust_remote_code": bool(trust_remote_code),
            "do_lower_case": False,
        }
        if use_fast is not None:
            kwargs["use_fast"] = bool(use_fast)
        try:
            return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        except Exception:
            # Slow tokenization is a stable fallback for local/offline
            # transformer tokenizer setups.
            if kwargs.get("use_fast") is False:
                raise
            kwargs["use_fast"] = False
            return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    return EsmTokenizer.from_pretrained(model_name_or_path, local_files_only=bool(local_files_only))


def build_sequence_collator(tokenizer, max_seq_len: int, sequence_format: str = "spaced_aa"):
    def collate(batch):
        seqs, labels = zip(*batch)
        # Keep ESM-compatible spaced amino-acid formatting by default.
        if str(sequence_format or "spaced_aa") == "esm":
            seqs = [normalize_sequence_for_esm(seq, max_seq_len) for seq in seqs]
        else:
            seqs = [normalize_sequence_for_plm(seq, max_seq_len, sequence_format) for seq in seqs]
        encoded = tokenizer(
            list(seqs),
            padding=True,
            truncation=True,
            max_length=max_seq_len + 2,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        attention_mask = encoded["attention_mask"]
        special_tokens_mask = encoded.pop("special_tokens_mask", torch.zeros_like(attention_mask))
        pooling_mask = attention_mask.to(torch.bool) & ~special_tokens_mask.to(torch.bool)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": attention_mask,
            "pooling_mask": pooling_mask,
            "labels": torch.stack(labels, 0),
        }

    return collate


def build_motif_collator(
    *,
    protein_tokenizer,
    motif_tokenizer: MotifTokenizer,
    esm_tokenizer,
    max_seq_len: int,
    use_esm_residue: bool,
    use_graph_nv: bool = True,
):
    def collate(batch):
        batch_size = len(batch)
        motif_cfg = motif_tokenizer.cfg
        x = torch.full(
            (batch_size, motif_cfg.max_motifs, motif_cfg.motif_len),
            fill_value=protein_tokenizer.pad_token_id,
            dtype=torch.long,
        )
        residue_mask = torch.zeros((batch_size, motif_cfg.max_motifs, motif_cfg.motif_len), dtype=torch.bool)
        motif_mask = torch.zeros((batch_size, motif_cfg.max_motifs), dtype=torch.bool)
        motif_residue_index = torch.full((batch_size, motif_cfg.max_motifs, motif_cfg.motif_len), fill_value=-1, dtype=torch.long)
        labels = []
        esm_sequences: list[str] = []
        motif_sequences = [["" for _ in range(motif_cfg.max_motifs)] for _ in range(batch_size)]

        for i, (seq, label) in enumerate(batch):
            seq_clean = normalize_sequence_for_motif(seq)
            if use_esm_residue:
                seq_clean = seq_clean[:max_seq_len]
            motifs, positions = motif_tokenizer.split_sequence_with_positions(seq_clean)
            for t, (motif_ids, motif_pos) in enumerate(zip(motifs, positions)):
                if t >= motif_cfg.max_motifs:
                    break
                length = min(len(motif_ids), motif_cfg.motif_len)
                if length == 0:
                    continue
                x[i, t, :length] = torch.tensor(motif_ids[:length], dtype=torch.long)
                motif_residue_index[i, t, :length] = torch.tensor(motif_pos[:length], dtype=torch.long)
                residue_mask[i, t, :length] = True
                motif_mask[i, t] = True
                start = int(motif_pos[0]) if motif_pos else 0
                motif_sequences[i][t] = seq_clean[start : start + length]
            if use_esm_residue:
                esm_sequences.append(normalize_sequence_for_esm(seq_clean, max_seq_len))
            labels.append(label)

        batch_dict = {
            "x": x,
            "residue_mask": residue_mask,
            "motif_mask": motif_mask,
            "labels": torch.stack(labels, 0),
        }
        batch_dict.update(
            build_motif_nv_feature_tensors(
                motif_sequences,
                enabled=bool(use_graph_nv),
            )
        )
        if use_esm_residue:
            encoded = esm_tokenizer(
                esm_sequences,
                padding=True,
                truncation=True,
                max_length=max_seq_len + 2,
                return_tensors="pt",
            )
            batch_dict["seq_input_ids"] = encoded["input_ids"]
            batch_dict["seq_attention_mask"] = encoded["attention_mask"]
            batch_dict["motif_residue_index"] = motif_residue_index
        return batch_dict

    return collate
