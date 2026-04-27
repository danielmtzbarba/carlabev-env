import os
import sys
import logging
import json

from datetime import datetime
from rich.console import Console

class BaseLogger:
    """Common interface for all logging modules (sim + training)."""

    def __init__(self, exp_name="default", log_dir="runs", enabled=True):
        self.enabled = enabled
        self.console = Console()
        if not enabled:
            self.writer = None
            return

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Basic stdout + file setup
        self.logger = logging.getLogger(exp_name)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(self.log_dir, "train.log"))
        stdout_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stdout_handler)

        self.msg(f"[Logger initialized at] {self.log_dir}")

    # --- common utilities ---
    def msg(self, text):
        if self.enabled:
            self.logger.info(text)

    def _to_serializable(self, value):
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, dict):
            return {str(k): self._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        return value

    def _extract_indexed_record(self, info, idx):
        record = {}
        for key, value in info.items():
            if isinstance(value, (list, tuple)):
                record[key] = self._to_serializable(value[idx])
            elif hasattr(value, "__getitem__") and not isinstance(value, (str, bytes, dict)):
                try:
                    record[key] = self._to_serializable(value[idx])
                except Exception:
                    record[key] = self._to_serializable(value)
            else:
                record[key] = self._to_serializable(value)
        return record

    def close(self):
        if self.writer:
            self.writer.close()

    def log_step(self, data, step=None):
        if not self.enabled:
            return
        for k, v in data.items():
            if isinstance(v, (int, float)):
                self.log_scalar(f"sim/{k}", v, step)

    def log_episode(self, info, idx):
        if not self.enabled:
            return

        data = self._extract_indexed_record(info, idx)
        # Console output
        scenario_bits = []
        if data.get("scenario_preset_id"):
            scenario_bits.append(f"preset: {data['scenario_preset_id']}")
        elif data.get("scene"):
            scenario_bits.append(f"scene: {data['scene']}")
        if data.get("level") is not None:
            scenario_bits.append(f"level: {data['level']}")

        self.console.print(
            f"Ep {data['episode']} | Return: [green]{data['return']:.2f}[/green] | "
            f"len_route: {int(data['len_ego_route'])} | num_veh: {data['num_vehicles']} | "
            f"len_steps: {data['length']} | cause: {data['termination']} | "
            + (" | ".join(scenario_bits) if scenario_bits else "")
        )

        json_path = os.path.join(self.log_dir, "episodes.jsonl")
        with open(json_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(data) + "\n")

        # Write to TensorBoard
        for k, v in data.items():
            if isinstance(v, (int, float)):
                if hasattr(self, "log_scalar"):
                    self.log_scalar(f"sim/{k}", v, data["episode"])

    def log_learning(self, step, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self.log_scalar(f"train/{k}", v, step)

    def log_evaluation(self, results_dict, global_step=None):
        if not self.enabled:
            return
        table = Table(
            title=f"Evaluation @ step {global_step}", header_style="bold cyan"
        )
        table.add_column("Metric")
        table.add_column("Value", justify="right")

        for k, v in results_dict.items():
            table.add_row(k, f"{v:.3f}" if isinstance(v, float) else str(v))
            self.log_scalar(f"eval/{k}", v, global_step)
        self.console.print(table)
