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
        self.log_dir = os.path.join(log_dir, exp_name + "_" + ts)
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
