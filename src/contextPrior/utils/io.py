from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_run_dir(root: str | Path, task_name: str, method: str, backbone_name: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / task_name / method / backbone_name / f"seed_{seed}" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path
