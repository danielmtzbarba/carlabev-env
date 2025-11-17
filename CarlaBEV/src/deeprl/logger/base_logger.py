import os
import sys
import logging

from datetime import datetime
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter


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
        self.writer = SummaryWriter(self.log_dir)

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

    def log_scalar(self, tag, value, step=None):
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)

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

        data = info
        # Console output
        #
        self.console.print(
            f"Ep {data["episode"][idx]} | Return: [green]{data["return"][idx]:.2f}[/green] | "
            f"len_route: {int(data["len_ego_route"][idx])} | num_veh: {data["num_vehicles"][idx]} | "
            f"len_steps: {data["length"][idx]} | cause: {data["termination"][idx]} | "
        )

        # Write to TensorBoard
        for k, v in info.items():
            if isinstance(v, (int, float)):
                self.log_scalar(f"sim/{k}", v, info["episode"])

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
