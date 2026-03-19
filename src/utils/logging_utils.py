# text2sql_baseline\src\utils\logging_utils.py
"""Утилиты логирования для Text-to-SQL пайплайна."""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Настроить логирование для всего приложения.

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_file: Путь к файлу логов (опционально)
        format_string: Формат сообщений

    Returns:
        Настроенный logger
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
        )

    # Создаем logger
    logger = logging.getLogger("text2sql")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Очищаем существующие handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # File handler (опционально)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Получить logger с именем."""
    return logging.getLogger(f"text2sql.{name}")


class PipelineLogger:
    """Логгер для отслеживания шагов пайплайна."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_step_start(self, step_name: str, **kwargs):
        """Логировать начало шага."""
        params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"▶️ START: {step_name} ({params})")

    def log_step_end(self, step_name: str, latency_ms: float, **kwargs):
        """Логировать завершение шага."""
        params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(
            f"✅ END: {step_name} ({latency_ms:.0f}ms) {params}"
        )

    def log_error(self, step_name: str, error: Exception):
        """Логировать ошибку."""
        self.logger.error(f"❌ ERROR: {step_name} - {error}", exc_info=True)

    def log_retry(self, attempt: int, max_retries: int, reason: str):
        """Логировать retry."""
        self.logger.warning(
            f"🔄 RETRY: {attempt}/{max_retries} - {reason}"
        )

    def log_confidence_history(self, history: list[dict]):
        """Логировать историю confidence."""
        history_str = ', '.join(
            f"{h['attempt']}: {h['confidence']:.2f}" for h in history
        )
        self.logger.info(f"📊 CONFIDENCE HISTORY: [{history_str}]")
