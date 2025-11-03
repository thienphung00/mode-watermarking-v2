"""
Unified logging facade for experiment tracking.

Supports multiple backends: console, TensorBoard, Weights & Biases, MLflow.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class LogBackend(Enum):
    """Supported logging backends."""

    CONSOLE = "console"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    MLFLOW = "mlflow"


@dataclass
class LoggerConfig:
    """Configuration for experiment logger."""

    backend: str
    project: Optional[str] = None
    run_name: Optional[str] = None
    log_dir: str = "outputs/experiments"
    tags: Optional[List[str]] = None

    def __post_init__(self):
        """Validate backend."""
        valid_backends = [b.value for b in LogBackend]
        if self.backend not in valid_backends:
            raise ValueError(
                f"Invalid backend: {self.backend}. "
                f"Must be one of {valid_backends}"
            )


class ExperimentLogger:
    """Unified logger interface for multiple backends."""

    def __init__(self, config: LoggerConfig):
        """
        Initialize experiment logger.

        Args:
            config: Logger configuration
        """
        self.config = config
        self.backend = LogBackend(config.backend)
        self.step = 0

        # Initialize Python logging
        self._setup_python_logging()

        # Initialize backend-specific loggers
        self._backend_logger = None
        if self.backend == LogBackend.TENSORBOARD:
            self._init_tensorboard()
        elif self.backend == LogBackend.WANDB:
            self._init_wandb()
        elif self.backend == LogBackend.MLFLOW:
            self._init_mlflow()

    def _setup_python_logging(self):
        """Setup standard Python logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self._logger = logging.getLogger(__name__)

    def _init_tensorboard(self):
        """Initialize TensorBoard logger."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_path = Path(self.config.log_dir) / (
                self.config.run_name or "tensorboard"
            )
            log_path.mkdir(parents=True, exist_ok=True)
            self._backend_logger = SummaryWriter(str(log_path))
            self._logger.info(f"TensorBoard logging to {log_path}")
        except ImportError:
            self._logger.warning(
                "TensorBoard not available. Falling back to console."
            )
            self._backend_logger = None

    def _init_wandb(self):
        """Initialize Weights & Biases logger."""
        try:
            import wandb

            wandb.init(
                project=self.config.project or "mode-watermarking",
                name=self.config.run_name,
                tags=self.config.tags or [],
                dir=self.config.log_dir,
            )
            self._backend_logger = wandb
            self._logger.info("W&B logging initialized")
        except ImportError:
            self._logger.warning(
                "wandb not available. Falling back to console."
            )
            self._backend_logger = None

    def _init_mlflow(self):
        """Initialize MLflow logger."""
        try:
            import mlflow

            mlflow.set_experiment(
                self.config.project or "mode-watermarking"
            )
            if self.config.run_name:
                mlflow.start_run(run_name=self.config.run_name)
            else:
                mlflow.start_run()
            self._backend_logger = mlflow
            self._logger.info("MLflow logging initialized")
        except ImportError:
            self._logger.warning(
                "mlflow not available. Falling back to console."
            )
            self._backend_logger = None

    def info(self, msg: str, **kwargs) -> None:
        """Log info message."""
        self._logger.info(msg, **kwargs)

    def debug(self, msg: str, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        """Log error message."""
        self._logger.error(msg, **kwargs)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics dictionary.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number (uses internal counter if None)
        """
        if step is None:
            step = self.step

        # Console logging
        metrics_str = ", ".join(
            [f"{k}={v:.4f}" for k, v in metrics.items()]
        )
        self._logger.info(f"Step {step}: {metrics_str}")

        # Backend-specific logging
        if self._backend_logger is None:
            return

        if self.backend == LogBackend.TENSORBOARD:
            for key, value in metrics.items():
                self._backend_logger.add_scalar(key, value, step)

        elif self.backend == LogBackend.WANDB:
            self._backend_logger.log(metrics, step=step)

        elif self.backend == LogBackend.MLFLOW:
            self._backend_logger.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """
        Log file artifact.

        Args:
            path: Path to artifact file
            name: Optional artifact name
        """
        if not Path(path).exists():
            self._logger.warning(f"Artifact not found: {path}")
            return

        if self._backend_logger is None:
            return

        if self.backend == LogBackend.WANDB:
            self._backend_logger.log_artifact(path, name=name)

        elif self.backend == LogBackend.MLFLOW:
            self._backend_logger.log_artifact(path, artifact_path=name)

    def set_step(self, step: int) -> None:
        """
        Set current step number.

        Args:
            step: Step number
        """
        self.step = step

    def finalize(self, status: str = "completed") -> None:
        """
        Finalize logging and cleanup.

        Args:
            status: Final status (completed, failed, etc.)
        """
        self._logger.info(f"Experiment {status}")

        if self._backend_logger is None:
            return

        if self.backend == LogBackend.TENSORBOARD:
            self._backend_logger.close()

        elif self.backend == LogBackend.WANDB:
            self._backend_logger.finish()

        elif self.backend == LogBackend.MLFLOW:
            self._backend_logger.end_run()


def make_logger_from_yaml(yaml_path: str) -> ExperimentLogger:
    """
    Create logger from YAML configuration file.

    Args:
        yaml_path: Path to YAML config file

    Returns:
        Configured ExperimentLogger
    """
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Extract logging section
    logging_dict = config_dict.get("logging", {})
    config = LoggerConfig(**logging_dict)

    return ExperimentLogger(config)
