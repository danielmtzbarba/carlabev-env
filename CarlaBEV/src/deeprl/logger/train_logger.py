from rich.table import Table
from .base_logger import BaseLogger


class TrainLogger(BaseLogger):
    """Logs training losses, PPO metrics, and evaluation results."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(exp_name=config.exp_name, *args, **kwargs)
        # Hyperparams summary
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join([f"|{k}|{v}|" for k, v in vars(config).items()]),
        )

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
