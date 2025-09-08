"""
Structured logging configuration for the content pipeline.

Provides centralized logging setup with structured output,
file rotation, and performance monitoring.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

import structlog
from structlog import configure, get_logger
from structlog.processors import JSONRenderer, TimeStamper, add_log_level

from .config import config


class PipelineLoggerSetup:
    """Centralized logging configuration for the pipeline."""

    def __init__(self):
        self.log_config = config.logging
        self.log_dir = Path(config.project_root) / "logs"
        self.log_dir.mkdir(exist_ok=True)

    def setup_logging(self) -> None:
        """Configure structured logging for the entire pipeline."""

        # Configure standard library logging
        self._setup_stdlib_logging()

        # Configure structlog
        self._setup_structlog()

        # Setup custom handlers
        self._setup_custom_handlers()

    def _setup_stdlib_logging(self) -> None:
        """Configure standard library logging."""

        # Create formatters
        if self.log_config.format == "json":
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s", '
                '"module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}'
            )
        else:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, self.log_config.level))

        # File handler with rotation
        if self.log_config.file_path:
            file_path = Path(self.log_config.file_path)
        else:
            file_path = self.log_dir / "pipeline.log"

        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=self.log_config.max_file_size,
            backupCount=self.log_config.backup_count,
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # Always debug level for files

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        # Set specific logger levels
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    def _setup_structlog(self) -> None:
        """Configure structlog for structured logging."""

        processors = [
            structlog.contextvars.merge_contextvars,
            add_log_level,
            TimeStamper(fmt="ISO"),
        ]

        if self.log_config.format == "json":
            processors.append(JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def _setup_custom_handlers(self) -> None:
        """Setup custom handlers for specific logging needs."""

        # Error handler - separate file for errors only
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log", maxBytes=50 * 1024 * 1024, backupCount=10  # 50MB
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
            )
        )

        # Performance handler - for timing and metrics
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "performance.log", maxBytes=100 * 1024 * 1024, backupCount=5  # 100MB
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

        # Add handlers to specific loggers
        logging.getLogger("pipeline.errors").addHandler(error_handler)
        logging.getLogger("pipeline.performance").addHandler(perf_handler)

    def get_logger(self, name: str) -> structlog.BoundLogger:
        """Get a structured logger for the given name."""
        return get_logger(name)


class PerformanceLogger:
    """Performance monitoring and logging."""

    def __init__(self, name: str):
        self.logger = logging.getLogger("pipeline.performance")
        self.name = name
        self.start_time = None
        self.metrics = {}

    def start_operation(self, operation: str) -> None:
        """Start timing an operation."""
        import time

        self.start_time = time.time()
        self.logger.info(f"Started {operation} in {self.name}")

    def end_operation(self, operation: str, **metrics) -> None:
        """End timing an operation and log results."""
        import time

        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.info(
                f"Completed {operation} in {self.name}: "
                f"duration={duration:.3f}s {' '.join(f'{k}={v}' for k, v in metrics.items())}"
            )
            self.start_time = None

    def log_metrics(self, **metrics) -> None:
        """Log performance metrics."""
        self.logger.info(
            f"Metrics for {self.name}: " f"{' '.join(f'{k}={v}' for k, v in metrics.items())}"
        )


class ErrorTracker:
    """Error tracking and reporting."""

    def __init__(self):
        self.logger = logging.getLogger("pipeline.errors")
        self.error_counts = {}
        self.recent_errors = []

    def track_error(self, error: Exception, context: dict = None) -> None:
        """Track and log an error with context."""
        error_type = type(error).__name__
        error_msg = str(error)

        # Count errors by type
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Store recent error
        error_info = {
            "type": error_type,
            "message": error_msg,
            "context": context or {},
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        }
        self.recent_errors.append(error_info)

        # Keep only recent errors
        if len(self.recent_errors) > 100:
            self.recent_errors = self.recent_errors[-100:]

        # Log the error
        self.logger.error(
            f"Error in pipeline: {error_type}: {error_msg}",
            extra={
                "error_type": error_type,
                "error_message": error_msg,
                "context": context,
            },
            exc_info=True,
        )

    def get_error_summary(self) -> dict:
        """Get summary of tracked errors."""
        return {
            "error_counts": self.error_counts.copy(),
            "recent_errors": self.recent_errors[-10:],  # Last 10 errors
            "total_errors": sum(self.error_counts.values()),
        }


# Global instances
logger_setup = PipelineLoggerSetup()
error_tracker = ErrorTracker()


def setup_pipeline_logging() -> None:
    """Setup logging for the entire pipeline."""
    logger_setup.setup_logging()


def get_pipeline_logger(name: str) -> structlog.BoundLogger:
    """Get a pipeline logger with the given name."""
    return logger_setup.get_logger(name)


def get_performance_logger(name: str) -> PerformanceLogger:
    """Get a performance logger for the given component."""
    return PerformanceLogger(name)


def track_pipeline_error(error: Exception, context: dict = None) -> None:
    """Track a pipeline error."""
    error_tracker.track_error(error, context)


def get_error_summary() -> dict:
    """Get error summary for reporting."""
    return error_tracker.get_error_summary()
