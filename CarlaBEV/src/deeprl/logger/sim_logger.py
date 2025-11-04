import json
from rich.table import Table
from .base_logger import BaseLogger


class SimLogger(BaseLogger):
    """Logs simulator-level episode and step metrics."""

    def log_step(self, data, step=None):
        if not self.enabled:
            return
        for k, v in data.items():
            if isinstance(v, (int, float)):
                self.log_scalar(f"sim/{k}", v, step)

    def log_episode(self, info):
        if not self.enabled:
            return

        # Console output
        self.console.print(
            f"Ep {info["episode"]} | Return: [green]{info["return"]:.2f}[/green] | "
            f"Len: {info["length"]} | Cause: {info["termination"]}"
        )

        # Append JSON
        json_path = f"{self.log_dir}/episodes.jsonl"
        with open(json_path, "a") as f:
            f.write(json.dumps(info) + "\n")

        # Write to TensorBoard
        for k, v in info.items():
            if isinstance(v, (int, float)):
                self.log_scalar(f"sim/{k}", v, info["episode"])
