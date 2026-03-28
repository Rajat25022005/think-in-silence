import logging
import sys
from typing import Dict, Optional


def get_logger(name: str = "think-in-silence") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class MetricLogger:
    def __init__(self, use_wandb: bool = False, project: str = "think-in-silence",
                 run_name: Optional[str] = None, config: Optional[dict] = None):
        self.use_wandb = use_wandb
        self._run = None

        if use_wandb:
            try:
                import wandb
                self._run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config or {},
                    resume="allow"
                )
            except Exception as e:
                get_logger().warning(f"WandB init failed: {e}. Logging locally only.")
                self.use_wandb = False

    def log(self, metrics: Dict, step: int):
        if self.use_wandb and self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self):
        if self.use_wandb and self._run is not None:
            self._run.finish()
