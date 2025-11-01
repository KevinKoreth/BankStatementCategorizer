"""
Custom Logger Module

A flexible logging system that supports console and file logging with colored output,
different log levels, and easy integration across multiple modules.

Usage:
    from logger import get_logger

    logger = get_logger(__name__)
    logger.info("Application started")
    logger.error("An error occurred")
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Enumeration of log levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console logging."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
        "BOLD": "\033[1m",  # Bold
    }

    def __init__(self, fmt: str, use_colors: bool = True):
        super().__init__(fmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors if enabled."""
        if self.use_colors and sys.stdout.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[levelname]}{self.COLORS['BOLD']}"
                    f"{levelname}{self.COLORS['RESET']}"
                )
                record.msg = (
                    f"{self.COLORS[levelname]}{record.msg}{self.COLORS['RESET']}"
                )

        return super().format(record)


class LoggerConfig:
    """Configuration for logger setup."""

    def __init__(
        self,
        name: str = "app",
        level: LogLevel = LogLevel.INFO,
        log_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        console_output: bool = True,
        file_output: bool = False,
        use_colors: bool = True,
        format_string: Optional[str] = None,
        date_format: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
    ):
        self.name = name
        self.level = level
        self.log_dir = log_dir
        self.log_file = log_file
        self.console_output = console_output
        self.file_output = file_output
        self.use_colors = use_colors
        self.format_string = format_string or self._default_format()
        self.date_format = date_format or "%Y-%m-%d %H:%M:%S"
        self.max_file_size = max_file_size
        self.backup_count = backup_count

    @staticmethod
    def _default_format() -> str:
        """Return default log format string."""
        return "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    def get_log_file_path(self) -> Optional[Path]:
        """Get the full path for the log file."""
        if not self.file_output:
            return None

        log_dir = Path(self.log_dir) if self.log_dir else Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = (
            self.log_file or f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        return log_dir / log_file


class LoggerManager:
    """Manages logger instances to ensure single logger per module."""

    _loggers = {}
    _default_config = None

    @classmethod
    def set_default_config(cls, config: LoggerConfig) -> None:
        """Set the default configuration for all new loggers."""
        cls._default_config = config

    @classmethod
    def get_logger(
        cls, name: str, config: Optional[LoggerConfig] = None
    ) -> logging.Logger:
        """
        Get or create a logger instance.

        Args:
            name: Name of the logger (usually __name__ of the module)
            config: Optional custom configuration for this logger

        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]

        # Use provided config, default config, or create new default
        logger_config = config or cls._default_config or LoggerConfig(name=name)

        logger = cls._create_logger(name, logger_config)
        cls._loggers[name] = logger

        return logger

    @classmethod
    def _create_logger(cls, name: str, config: LoggerConfig) -> logging.Logger:
        """Create and configure a new logger instance."""
        logger = logging.getLogger(name)
        logger.setLevel(config.level.value)
        logger.handlers.clear()  # Remove any existing handlers

        # Add console handler
        if config.console_output:
            cls._add_console_handler(logger, config)

        # Add file handler
        if config.file_output:
            cls._add_file_handler(logger, config)

        return logger

    @classmethod
    def _add_console_handler(cls, logger: logging.Logger, config: LoggerConfig) -> None:
        """Add a console handler to the logger."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level.value)

        formatter = ColoredFormatter(config.format_string, use_colors=config.use_colors)
        formatter.datefmt = config.date_format
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    @classmethod
    def _add_file_handler(cls, logger: logging.Logger, config: LoggerConfig) -> None:
        """Add a file handler to the logger with rotation."""
        from logging.handlers import RotatingFileHandler

        log_file_path = config.get_log_file_path()
        if not log_file_path:
            return

        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
        )
        file_handler.setLevel(config.level.value)

        # File logs don't need colors
        formatter = logging.Formatter(config.format_string, datefmt=config.date_format)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    @classmethod
    def reset_loggers(cls) -> None:
        """Reset all loggers (useful for testing)."""
        for logger in cls._loggers.values():
            logger.handlers.clear()
        cls._loggers.clear()
        cls._default_config = None


# Convenience function for easy imports
def get_logger(
    name: str,
    level: LogLevel = LogLevel.INFO,
    console: bool = True,
    file: bool = False,
    log_dir: Optional[str] = None,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Get a logger instance with simple configuration.

    Args:
        name: Name of the logger (use __name__ for module-level logging)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Enable console output
        file: Enable file output
        log_dir: Directory for log files (default: "logs")
        use_colors: Use colored output in console

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
        >>> logger.error("An error occurred", exc_info=True)
    """
    config = LoggerConfig(
        name=name,
        level=level,
        console_output=console,
        file_output=file,
        log_dir=log_dir,
        use_colors=use_colors,
    )

    return LoggerManager.get_logger(name, config)


def setup_global_logging(
    level: LogLevel = LogLevel.INFO,
    log_dir: str = "logs",
    console: bool = True,
    file: bool = True,
    use_colors: bool = True,
) -> None:
    """
    Setup global logging configuration for all modules.

    This should be called once at application startup.
    All subsequent calls to get_logger() will use this configuration.

    Args:
        level: Default log level for all loggers
        log_dir: Directory for log files
        console: Enable console output by default
        file: Enable file output by default
        use_colors: Use colored output in console

    Example:
        >>> setup_global_logging(
        ...     level=LogLevel.DEBUG,
        ...     log_dir="logs",
        ...     file=True
        ... )
    """
    config = LoggerConfig(
        name="app",
        level=level,
        log_dir=log_dir,
        console_output=console,
        file_output=file,
        use_colors=use_colors,
    )

    LoggerManager.set_default_config(config)


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Simple usage
    print("=== Example 1: Simple Logger ===")
    logger1 = get_logger("example1")
    logger1.debug("This is a debug message")
    logger1.info("This is an info message")
    logger1.warning("This is a warning message")
    logger1.error("This is an error message")
    logger1.critical("This is a critical message")

    # Example 2: Logger with file output
    print("\n=== Example 2: Logger with File Output ===")
    logger2 = get_logger("example2", level=LogLevel.DEBUG, file=True)
    logger2.info("This message goes to console and file")
    logger2.debug("Debug message also logged")

    # Example 3: Global configuration
    print("\n=== Example 3: Global Configuration ===")
    setup_global_logging(level=LogLevel.DEBUG, log_dir="app_logs", file=True)

    logger3 = get_logger("module1")
    logger4 = get_logger("module2")

    logger3.info("Module 1 message")
    logger4.warning("Module 2 warning")

    # Example 4: Exception logging
    print("\n=== Example 4: Exception Logging ===")
    logger5 = get_logger("exception_example")
    try:
        result = 10 / 0
    except ZeroDivisionError:
        logger5.error("Division error occurred", exc_info=True)

    print("\nCheck the 'logs' and 'app_logs' directories for log files!")
