from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BIOMAP_SEQUENCE_TASKS = [
    "antibiotic_resistance",
    "fold_prediction",
    "cloning_clf",
    "enzyme_catalytic_efficiency",
    "fitness_prediction",
    "fluorescence_prediction",
    "localization_prediction",
    "material_production",
    "metal_ion_binding",
    "optimal_ph",
    "optimal_temperature",
    "temperature_stability",
    "tcr_pmhc_affinity",
    "stability_prediction",
    "solubility_prediction",
    "peptide_HLA_MHC_affinity",
]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    source: str = "biomap-research"
    task_type: str = "auto"
    num_labels: int | None = None
    primary_metric: str | None = None
    higher_is_better: bool | None = None


TASK_REGISTRY: dict[str, TaskSpec] = {
    # Structure-level tasks from the BioMap table. contact_prediction and
    # secondary_structure_prediction require contact/per-residue heads and are not
    # part of the current sequence-level finetuning script.
    "fold_prediction": TaskSpec(
        name="fold_prediction",
        task_type="classification",
        primary_metric="acc",
        higher_is_better=True,
    ),
    # Development/property tasks.
    "solubility_prediction": TaskSpec(
        name="solubility_prediction",
        task_type="classification",
        primary_metric="acc",
        higher_is_better=True,
    ),
    "stability_prediction": TaskSpec(
        name="stability_prediction",
        task_type="regression",
        num_labels=1,
        primary_metric="spearman",
        higher_is_better=True,
    ),
    "temperature_stability": TaskSpec(
        name="temperature_stability",
        task_type="classification",
        primary_metric="mcc",
        higher_is_better=True,
    ),
    "optimal_temperature": TaskSpec(
        name="optimal_temperature",
        task_type="regression",
        num_labels=1,
        primary_metric="spearman",
        higher_is_better=True,
    ),
    "optimal_ph": TaskSpec(
        name="optimal_ph",
        task_type="classification",
        primary_metric="roc_auc",
        higher_is_better=True,
    ),
    "cloning_clf": TaskSpec(
        name="cloning_clf",
        task_type="classification",
        primary_metric="roc_auc",
        higher_is_better=True,
    ),
    "material_production": TaskSpec(
        name="material_production",
        task_type="classification",
        primary_metric="roc_auc",
        higher_is_better=True,
    ),
    # Interaction tasks.
    "metal_ion_binding": TaskSpec(
        name="metal_ion_binding",
        task_type="classification",
        primary_metric="acc",
        higher_is_better=True,
    ),
    "peptide_HLA_MHC_affinity": TaskSpec(
        name="peptide_HLA_MHC_affinity",
        task_type="classification",
        primary_metric="roc_auc",
        higher_is_better=True,
    ),
    "tcr_pmhc_affinity": TaskSpec(
        name="tcr_pmhc_affinity",
        task_type="classification",
        primary_metric="roc_auc",
        higher_is_better=True,
    ),
    # Functional tasks.
    "antibiotic_resistance": TaskSpec(
        name="antibiotic_resistance",
        task_type="classification",
        primary_metric="acc",
        higher_is_better=True,
    ),
    "fluorescence_prediction": TaskSpec(
        name="fluorescence_prediction",
        task_type="regression",
        num_labels=1,
        primary_metric="spearman",
        higher_is_better=True,
    ),
    "fitness_prediction": TaskSpec(
        name="fitness_prediction",
        task_type="regression",
        num_labels=1,
        primary_metric="spearman",
        higher_is_better=True,
    ),
    "localization_prediction": TaskSpec(
        name="localization_prediction",
        task_type="classification",
        primary_metric="acc",
        higher_is_better=True,
    ),
    "enzyme_catalytic_efficiency": TaskSpec(
        name="enzyme_catalytic_efficiency",
        task_type="regression",
        num_labels=1,
        primary_metric="pearson",
        higher_is_better=True,
    ),
}


def infer_task_type(labels: list[Any]) -> tuple[str, int]:
    first = labels[0]
    if isinstance(first, list):
        if first and isinstance(first[0], list):
            return "contact", 0
        return "multilabel", len(first)

    try:
        import numpy as np

        y_float = np.asarray(labels, dtype=np.float32)
        if len(np.unique(y_float)) > 50:
            return "regression", 1
        return "classification", int(len(np.unique(y_float)))
    except Exception:
        return "classification", len(set(labels))


def default_primary_metric(task_type: str, num_labels: int) -> tuple[str, bool]:
    if task_type == "classification":
        metric = "mcc" if num_labels <= 10 else "f1_macro"
        return metric, True
    if task_type == "multilabel":
        return "f1_macro", True
    return "rmse", False


def infer_num_labels_for_forced_type(labels: list[Any], task_type: str, explicit_num_labels: int | None) -> int:
    if explicit_num_labels is not None:
        return int(explicit_num_labels)
    if task_type == "regression":
        return 1
    if task_type == "multilabel":
        first = labels[0]
        return len(first) if isinstance(first, list) else 1
    if task_type == "classification":
        try:
            import numpy as np

            return int(len(np.unique(np.asarray(labels))))
        except Exception:
            return int(len(set(labels)))
    if task_type == "contact":
        return 0
    raise ValueError(f"Unsupported task_type: {task_type}")


def resolve_task_spec(task_name: str, labels: list[Any]) -> dict[str, Any]:
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unsupported BioMap task: {task_name}")
    base = TASK_REGISTRY[task_name]
    if base.task_type == "auto":
        task_type, num_labels = infer_task_type(labels)
    else:
        task_type = base.task_type
        num_labels = infer_num_labels_for_forced_type(labels, task_type, base.num_labels)
    primary_metric, higher_is_better = default_primary_metric(task_type, num_labels)
    if base.primary_metric is not None:
        primary_metric = base.primary_metric
    if base.higher_is_better is not None:
        higher_is_better = base.higher_is_better
    return {
        "name": task_name,
        "source": base.source,
        "task_type": task_type,
        "num_labels": num_labels,
        "primary_metric": primary_metric,
        "higher_is_better": higher_is_better,
    }
