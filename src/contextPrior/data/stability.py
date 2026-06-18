from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .biomap import LoadedTask
from .mutations import infer_mutations_from_pair, parse_mutation_string
from .splits import make_validation_split, subsample_indices

try:
    from datasets import DatasetDict, load_dataset, load_from_disk
except ImportError as exc:  # pragma: no cover
    DatasetDict = None
    load_dataset = None
    load_from_disk = None
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None


SEQUENCE_COLUMNS = [
    "mutant_sequence", "mutated_sequence", "sequence", "seq", "aa_seq", "primary", "protein_sequence", "protein",
]
WT_SEQUENCE_COLUMNS = [
    "wt_sequence", "wildtype_sequence", "wild_type_sequence", "WT_sequence", "WT_seq", "aa_seq_wt", "target_seq",
]
LABEL_COLUMNS = [
    "ddG_ML", "ddG", "delta_delta_g", "dG_ML", "dG", "stability", "label", "target", "score",
]
MUTATION_COLUMNS = ["mutation", "mutations", "mut_type", "variant", "variant_name", "mutant"]
PROTEIN_ID_COLUMNS = ["WT_name", "protein_name", "protein_id", "uid", "name", "target", "pdb_id"]


@dataclass
class StabilityTask(LoadedTask):
    train_metadata: list[dict[str, Any]] | None = None
    val_metadata: list[dict[str, Any]] | None = None
    test_metadata: list[dict[str, Any]] | None = None
    label_column: str | None = None
    sequence_column: str | None = None
    dataset_name: str | None = None


class SequenceLabelMetadataDataset:
    def __init__(self, sequences: list[str], labels: list[Any], metadata: list[dict[str, Any]] | None, task_type: str):
        self.sequences = sequences
        self.labels = labels
        self.metadata = metadata or [{} for _ in sequences]
        self.task_type = task_type

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        import torch

        label = self.labels[idx]
        if self.task_type == "classification":
            label = torch.tensor(int(label), dtype=torch.long)
        elif self.task_type == "multilabel":
            label = torch.tensor(np.asarray(label), dtype=torch.float32)
        else:
            label = torch.tensor(float(label), dtype=torch.float32)
        return self.sequences[idx], label, self.metadata[idx]


def wrap_collator_with_metadata(base_collator):
    """Adapt an existing (sequence, label) collator to accept (sequence, label, metadata)."""
    def collate(batch):
        seq_label = [(seq, label) for seq, label, _meta in batch]
        metadata = [dict(meta) for _seq, _label, meta in batch]
        out = base_collator(seq_label)
        out["metadata"] = metadata
        return out
    return collate


def load_stability_task_from_config(config: dict[str, Any], *, seed: int) -> StabilityTask:
    data_cfg = config.get("stability_data", config.get("data", {}))
    global_data_cfg = config.get("data", {})
    source = str(data_cfg.get("source", "saprothub_meta_stability"))


    val_sample_fraction = float(data_cfg.get("val_sample_fraction", global_data_cfg.get("val_sample_fraction", 1.0)))
    test_sample_fraction = float(data_cfg.get("test_sample_fraction", global_data_cfg.get("test_sample_fraction", 1.0)))
    max_train_examples = data_cfg.get("max_train_examples", global_data_cfg.get("max_train_examples"))
    max_val_examples = data_cfg.get("max_val_examples", global_data_cfg.get("max_val_examples"))
    max_test_examples = data_cfg.get("max_test_examples", global_data_cfg.get("max_test_examples"))
    if source in {"saprothub_meta_stability", "saprothub"}:
        return load_hf_stability_task(
            dataset_name=str(data_cfg.get("dataset_name", "SaProtHub/Dataset-Meta-scale-protein-stability")),
            subset=data_cfg.get("subset"),
            data_dir=data_cfg.get("data_dir"),
            task_name=str(data_cfg.get("task_name", "saprothub_meta_stability")),
            label_column=data_cfg.get("label_column"),
            sequence_column=data_cfg.get("sequence_column"),
            val_fraction=float(data_cfg.get("val_fraction", config.get("data", {}).get("val_fraction", 0.05))),
            train_fraction=float(data_cfg.get("train_fraction", config.get("data", {}).get("train_fraction", 1.0))),
            val_sample_fraction=val_sample_fraction,
            test_sample_fraction=test_sample_fraction,
            max_train_examples=max_train_examples,
            max_val_examples=max_val_examples,
            max_test_examples=max_test_examples,
            seed=seed,
            local_path=data_cfg.get("local_path"),
        )
    if source in {"megascale", "megascale_dataset3_single"}:
        return load_hf_stability_task(
            dataset_name=str(data_cfg.get("dataset_name", "RosettaCommons/MegaScale")),
            subset=data_cfg.get("subset", "dataset3_single"),
            data_dir=data_cfg.get("data_dir", "dataset3_single"),
            task_name=str(data_cfg.get("task_name", "megascale_dataset3_single")),
            label_column=data_cfg.get("label_column", "ddG_ML"),
            sequence_column=data_cfg.get("sequence_column", "aa_seq"),
            val_fraction=float(data_cfg.get("val_fraction", config.get("data", {}).get("val_fraction", 0.05))),
            train_fraction=float(data_cfg.get("train_fraction", config.get("data", {}).get("train_fraction", 1.0))),
            val_sample_fraction=val_sample_fraction,
            test_sample_fraction=test_sample_fraction,
            max_train_examples=max_train_examples,
            max_val_examples=max_val_examples,
            max_test_examples=max_test_examples,
            seed=seed,
            local_path=data_cfg.get("local_path"),
        )
    if source in {"csv", "local_csv"}:
        return load_csv_stability_task(
            path=Path(data_cfg["path"]),
            task_name=str(data_cfg.get("task_name", "csv_stability")),
            sequence_column=data_cfg.get("sequence_column"),
            label_column=data_cfg.get("label_column"),
            split_column=data_cfg.get("split_column"),
            val_fraction=float(data_cfg.get("val_fraction", config.get("data", {}).get("val_fraction", 0.05))),
            train_fraction=float(data_cfg.get("train_fraction", config.get("data", {}).get("train_fraction", 1.0))),
            val_sample_fraction=val_sample_fraction,
            test_sample_fraction=test_sample_fraction,
            max_train_examples=max_train_examples,
            max_val_examples=max_val_examples,
            max_test_examples=max_test_examples,
            seed=seed,
        )
    raise ValueError(f"Unsupported stability_data.source={source}")


def load_hf_stability_task(
    *,
    dataset_name: str,
    subset: str | None,
    data_dir: str | None,
    task_name: str,
    label_column: str | None,
    sequence_column: str | None,
    val_fraction: float,
    train_fraction: float,
    val_sample_fraction: float = 1.0,
    test_sample_fraction: float = 1.0,
    max_train_examples: int | None = None,
    max_val_examples: int | None = None,
    max_test_examples: int | None = None,
    seed: int = 63,
    local_path: str | None = None,
) -> StabilityTask:
    if load_dataset is None:
        raise ImportError("datasets is required for stability datasets") from _DATASETS_IMPORT_ERROR
    if local_path:
        ds = load_from_disk(local_path)
    else:
        kwargs: dict[str, Any] = {}
        if subset:
            kwargs["name"] = subset
        if data_dir:
            kwargs["data_dir"] = data_dir
        ds = load_dataset(dataset_name, **kwargs)
    return _datasetdict_to_task(
        ds,
        task_name=task_name,
        dataset_name=dataset_name,
        label_column=label_column,
        sequence_column=sequence_column,
        val_fraction=val_fraction,
        train_fraction=train_fraction,
        val_sample_fraction=val_sample_fraction,
        test_sample_fraction=test_sample_fraction,
        max_train_examples=max_train_examples,
        max_val_examples=max_val_examples,
        max_test_examples=max_test_examples,
        seed=seed,
    )


def load_csv_stability_task(
    *,
    path: Path,
    task_name: str,
    sequence_column: str | None,
    label_column: str | None,
    split_column: str | None,
    val_fraction: float,
    train_fraction: float,
    val_sample_fraction: float = 1.0,
    test_sample_fraction: float = 1.0,
    max_train_examples: int | None = None,
    max_val_examples: int | None = None,
    max_test_examples: int | None = None,
    seed: int = 63,
) -> StabilityTask:
    import pandas as pd

    df = pd.read_csv(path)
    seq_col = sequence_column or _choose_column(list(df.columns), SEQUENCE_COLUMNS, "sequence")
    label_col = label_column or _choose_column(list(df.columns), LABEL_COLUMNS, "label")
    split_col = split_column if split_column in df.columns else None
    rows = df.to_dict("records")

    if split_col:
        train_rows = [r for r in rows if str(r.get(split_col, "")).lower() in {"train", "training"}]
        val_rows = [r for r in rows if str(r.get(split_col, "")).lower() in {"valid", "validation", "val"}]
        test_rows = [r for r in rows if str(r.get(split_col, "")).lower() in {"test", "testing"}]
        if not val_rows:
            train_rows, val_rows = _split_rows(train_rows, label_col, val_fraction, seed)
    else:
        train_rows, val_rows, test_rows = _random_train_val_test(rows, label_col, val_fraction, seed)

    train_rows = _limit_rows(_subsample_rows(train_rows, train_fraction, seed), max_train_examples, seed + 11)
    val_rows = _limit_rows(_subsample_rows(val_rows, val_sample_fraction, seed + 1), max_val_examples, seed + 12)
    test_rows = _limit_rows(_subsample_rows(test_rows, test_sample_fraction, seed + 2), max_test_examples, seed + 13)
    return _rows_to_task(
        train_rows, val_rows, test_rows,
        task_name=task_name,
        dataset_name=str(path),
        sequence_column=seq_col,
        label_column=label_col,
        split_mode="csv_split",
    )


def _datasetdict_to_task(
    ds,
    *,
    task_name: str,
    dataset_name: str,
    label_column: str | None,
    sequence_column: str | None,
    val_fraction: float,
    train_fraction: float,
    val_sample_fraction: float = 1.0,
    test_sample_fraction: float = 1.0,
    max_train_examples: int | None = None,
    max_val_examples: int | None = None,
    max_test_examples: int | None = None,
    seed: int = 63,
) -> StabilityTask:
    if not isinstance(ds, dict):
        raise RuntimeError(f"Expected DatasetDict-like object, got {type(ds)}")
    split_names = list(ds.keys())
    # Some HF datasets ship a single split with split_name column.
    if "train" in ds:
        train = ds["train"]
    else:
        train = ds[split_names[0]]
    seq_col = sequence_column or _choose_column(train.column_names, SEQUENCE_COLUMNS, "sequence")
    label_col = label_column or _choose_column(train.column_names, LABEL_COLUMNS, "label")

    if "train" in ds and ("test" in ds or "validation" in ds or "valid" in ds or "val" in ds):
        train_rows_all = [dict(r) for r in ds["train"]]
        if "validation" in ds:
            val_rows = [dict(r) for r in ds["validation"]]
            split_mode = "official_validation"
        elif "valid" in ds:
            val_rows = [dict(r) for r in ds["valid"]]
            split_mode = "official_valid"
        elif "val" in ds:
            val_rows = [dict(r) for r in ds["val"]]
            split_mode = "official_val"
        else:
            train_rows_all, val_rows = _split_rows(train_rows_all, label_col, val_fraction, seed)
            split_mode = f"train_split_{val_fraction:.2f}"
        test_rows = [dict(r) for r in ds["test"]] if "test" in ds else val_rows
    else:
        rows = [dict(r) for r in train]
        split_col = _find_split_column(rows)
        if split_col:
            train_rows_all = [r for r in rows if str(r.get(split_col, "")).lower() in {"train", "training"}]
            val_rows = [r for r in rows if str(r.get(split_col, "")).lower() in {"valid", "validation", "val"}]
            test_rows = [r for r in rows if str(r.get(split_col, "")).lower() in {"test", "testing"}]
            if not val_rows:
                train_rows_all, val_rows = _split_rows(train_rows_all, label_col, val_fraction, seed)
            split_mode = f"column_{split_col}"
        else:
            train_rows_all, val_rows, test_rows = _random_train_val_test(rows, label_col, val_fraction, seed)
            split_mode = f"random_train_val_test_{val_fraction:.2f}"

    train_rows = _limit_rows(_subsample_rows(train_rows_all, train_fraction, seed), max_train_examples, seed + 11)
    val_rows = _limit_rows(_subsample_rows(val_rows, val_sample_fraction, seed + 1), max_val_examples, seed + 12)
    test_rows = _limit_rows(_subsample_rows(test_rows, test_sample_fraction, seed + 2), max_test_examples, seed + 13)
    return _rows_to_task(
        train_rows, val_rows, test_rows,
        task_name=task_name,
        dataset_name=dataset_name,
        sequence_column=seq_col,
        label_column=label_col,
        split_mode=split_mode,
    )


def _rows_to_task(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    task_name: str,
    dataset_name: str,
    sequence_column: str,
    label_column: str,
    split_mode: str,
) -> StabilityTask:
    def clean(rows: list[dict[str, Any]]):
        seqs: list[str] = []
        labels: list[float] = []
        meta: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            seq = row.get(sequence_column)
            label = row.get(label_column)
            if seq is None or label is None:
                continue
            try:
                y = float(label)
            except Exception:
                continue
            if not np.isfinite(y):
                continue
            seq = str(seq).strip().upper()
            if not seq:
                continue
            seqs.append(seq)
            labels.append(y)
            meta.append(_build_metadata(row, sequence_column=sequence_column, label_column=label_column, row_index=i))
        return seqs, labels, meta

    train_sequences, train_labels, train_meta = clean(train_rows)
    val_sequences, val_labels, val_meta = clean(val_rows)
    test_sequences, test_labels, test_meta = clean(test_rows)
    return StabilityTask(
        task_name=task_name,
        task_type="regression",
        num_labels=1,
        primary_metric="spearman",
        higher_is_better=True,
        train_sequences=train_sequences,
        train_labels=train_labels,
        val_sequences=val_sequences,
        val_labels=val_labels,
        test_sequences=test_sequences,
        test_labels=test_labels,
        split_mode=split_mode,
        train_metadata=train_meta,
        val_metadata=val_meta,
        test_metadata=test_meta,
        label_column=label_column,
        sequence_column=sequence_column,
        dataset_name=dataset_name,
    )


def _build_metadata(row: dict[str, Any], *, sequence_column: str, label_column: str, row_index: int) -> dict[str, Any]:
    meta: dict[str, Any] = {"row_index": row_index}
    for col in PROTEIN_ID_COLUMNS + MUTATION_COLUMNS + WT_SEQUENCE_COLUMNS:
        if col in row and row[col] is not None:
            meta[col] = row[col]
    meta["sequence"] = row.get(sequence_column, "")
    meta["label_raw"] = row.get(label_column, "")
    mut_col = _first_existing(row, MUTATION_COLUMNS)
    if mut_col:
        meta["mutation"] = row.get(mut_col)
    wt_col = _first_existing(row, WT_SEQUENCE_COLUMNS)
    if wt_col:
        meta["wt_sequence"] = row.get(wt_col)
    if "mutation" not in meta and meta.get("wt_sequence"):
        muts = infer_mutations_from_pair(str(meta["wt_sequence"]), str(meta["sequence"]))
        if muts:
            meta["mutation"] = "/".join(m.label() for m in muts)
    meta["mutation_count"] = len(parse_mutation_string(str(meta.get("mutation", ""))))
    return meta


def _first_existing(row: dict[str, Any], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in row and row[c] is not None:
            return c
    return None


def _choose_column(columns: list[str], candidates: list[str], kind: str) -> str:
    for name in candidates:
        if name in columns:
            return name
    lowered = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise KeyError(f"Could not infer {kind} column from {columns}")


def _find_split_column(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    for col in ["split", "split_name", "stage", "set", "partition"]:
        if col in rows[0]:
            values = {str(r.get(col, "")).lower() for r in rows[: min(len(rows), 1000)]}
            if values & {"train", "valid", "validation", "val", "test"}:
                return col
    return None


def _split_rows(rows: list[dict[str, Any]], label_col: str, val_fraction: float, seed: int):
    labels = [r.get(label_col, 0.0) for r in rows]
    train_idx, val_idx = make_validation_split(labels, task_type="regression", val_fraction=val_fraction, seed=seed)
    return [rows[i] for i in train_idx], [rows[i] for i in val_idx]


def _random_train_val_test(rows: list[dict[str, Any]], label_col: str, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(idx) * val_fraction)))
    n_val = max(1, int(round(len(idx) * val_fraction)))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return [rows[i] for i in train_idx], [rows[i] for i in val_idx], [rows[i] for i in test_idx]


def _limit_rows(rows: list[dict[str, Any]], max_examples: int | None, seed: int):
    if max_examples is None:
        return rows
    try:
        max_examples_int = int(max_examples)
    except Exception:
        return rows
    if max_examples_int <= 0 or len(rows) <= max_examples_int:
        return rows
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    keep = sorted(idx[:max_examples_int].tolist())
    return [rows[i] for i in keep]


def _subsample_rows(rows: list[dict[str, Any]], fraction: float, seed: int):
    if fraction >= 1.0:
        return rows
    idx = subsample_indices(np.arange(len(rows)), fraction=fraction, seed=seed)
    return [rows[i] for i in idx]
