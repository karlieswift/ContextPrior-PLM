from __future__ import annotations

import json
from pathlib import Path


class ExperimentLogger:
    def __init__(self, run_dir: Path, *, use_wandb: bool, wandb_project: str, run_name: str, config: dict):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics_epoch.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.plot_data_path = self.run_dir / "plot_data.json"
        self.config_path = self.run_dir / "config.resolved.json"
        self._write_json(self.config_path, config)

        self.wandb_run = None
        if use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(project=wandb_project, name=run_name, config=config)
            except Exception:
                self.wandb_run = None

    def log_epoch(self, payload: dict):
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log(payload)

    def write_summary(self, payload: dict):
        self._write_json(self.summary_path, payload)
        if self.wandb_run is not None:
            self.wandb_run.summary.update(payload)

    def write_plot_data(self, payload: dict):
        self._write_json(self.plot_data_path, payload)

    def close(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()

    @staticmethod
    def _write_json(path: Path, payload: dict):
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
