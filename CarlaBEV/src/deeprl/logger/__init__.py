import os
from datetime import datetime
from .base_logger import BaseLogger
from .sim_logger import SimLogger
from .train_logger import TrainLogger


def create_loggers(cfg):
    """
    Create unified SimLogger + TrainLogger that share the same SummaryWriter
    and timestamped experiment directory.

    Parameters
    ----------
    cfg : Namespace or config object
        Must contain at least:
            - cfg.exp_name
            - cfg.logging.enabled
            - cfg.logging.dir (optional, default 'runs')

    Returns
    -------
    sim_logger : SimLogger
    train_logger : TrainLogger
    """

    # Determine experiment folder
    log_root = getattr(cfg.logging, "dir", "runs")
    exp_name = getattr(cfg, "exp_name", "default")
    exp_dir = os.path.join(log_root, f"{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)

    # Initialize base logger first (shared writer + folder)
    base_logger = BaseLogger(
        exp_name=exp_name, log_dir=log_root, enabled=cfg.logging.enabled
    )
    base_logger.log_dir = exp_dir  # ensure correct directory name

    # Share writer and log_dir with child loggers
    sim_logger = SimLogger(exp_name=exp_name, enabled=cfg.logging.enabled)
    sim_logger.writer = base_logger.writer
    sim_logger.log_dir = exp_dir

    train_logger = TrainLogger(config=cfg, enabled=cfg.logging.enabled)
    train_logger.writer = base_logger.writer
    train_logger.log_dir = exp_dir

    # Optional: write simple hyperparameter file
    with open(os.path.join(exp_dir, "hyperparams.txt"), "w") as f:
        for k, v in vars(cfg).items():
            f.write(f"{k}: {v}\n")

    base_logger.msg(f"[LogManager] Unified loggers created at: {exp_dir}")

    return sim_logger, train_logger
