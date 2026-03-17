"""
Retry logic с exponential backoff для LLM запросов.

Оптимизации:
1. Exponential Backoff - увеличение задержки между попытками
2. Jitter - случайная вариация задержки
3. Retry on specific errors - только для определённых ошибок

Использование:
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def generate_sql(prompt: str) -> str:
        ...
"""
import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryError(Exception):
    """Ошибка после исчерпания попыток."""

    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Декоратор для retry с exponential backoff.

    Args:
        max_retries: Максимальное количество попыток.
        base_delay: Базовая задержка в секундах.
        max_delay: Максимальная задержка.
        exponential_base: База экспоненты.
        jitter: Добавлять случайную вариацию.
        retryable_exceptions: Типы ошибок для retry.

    Returns:
        Декоратор.

    Example:
        @retry_with_backoff(max_retries=3, base_delay=0.5)
        def llm_generate(prompt: str) -> str:
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise RetryError(
                            f"Failed after {max_retries + 1} attempts: {e}"
                        ) from e

                    # Расчёт задержки
                    delay = min(base_delay * (exponential_base**attempt), max_delay)

                    if jitter:
                        # Добавляем ±25% случайности
                        delay = delay * (0.75 + random.random() * 0.5)

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                        f"{func.__name__}: {e}. Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)

            # Должно быть unreachable, но на всякий случай
            raise RetryError(f"Failed after {max_retries + 1} attempts")

        return wrapper

    return decorator


def retry_on_error(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    error_condition: Optional[Callable[[Any], bool]] = None,
) -> T:
    """
    Retry функция с проверкой условия ошибки.

    Args:
        func: Функция для выполнения.
        max_retries: Максимум попыток.
        base_delay: Базовая задержка.
        error_condition: Функция проверки ошибки.

    Returns:
        Результат функции.

    Example:
        result = retry_on_error(
            lambda: llm.generate(prompt),
            max_retries=3,
            error_condition=lambda x: x is None or x == ""
        )
    """
    last_result = None

    for attempt in range(max_retries + 1):
        try:
            result = func()
            last_result = result

            # Проверка условия ошибки
            if error_condition and error_condition(result):
                raise ValueError(f"Error condition met: {result}")

            return result

        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Max retries exceeded: {e}")
                raise

            delay = base_delay * (2**attempt)
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.2f}s..."
            )
            time.sleep(delay)

    return last_result  # type: ignore


class RetryConfig:
    """Конфигурация retry логики."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Получить задержку для попытки."""
        delay = min(
            self.base_delay * (self.exponential_base**attempt), self.max_delay
        )

        if self.jitter:
            delay = delay * (0.75 + random.random() * 0.5)

        return delay


class RetryExecutor:
    """Исполнитель retry логики."""

    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        self.config = config or RetryConfig()
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "retry_calls": 0,
            "failed_calls": 0,
            "total_retries": 0,
        }

    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        should_retry: Optional[Callable[[Exception], bool]] = None,
        **kwargs: Any,
    ) -> T:
        """
        Выполнить функцию с retry.

        Args:
            func: Функция.
            *args: Аргументы функции.
            should_retry: Функция проверки необходимости retry.
            **kwargs: KW-аргументы функции.

        Returns:
            Результат функции.
        """
        self.stats["total_calls"] += 1

        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                self.stats["successful_calls"] += 1
                return result

            except Exception as e:
                last_exception = e

                # Проверка необходимости retry
                if should_retry and not should_retry(e):
                    logger.debug(f"Not retrying: {e}")
                    self.stats["failed_calls"] += 1
                    raise

                if attempt == self.config.max_retries:
                    logger.error(f"Max retries exceeded: {e}")
                    self.stats["failed_calls"] += 1
                    raise

                # Retry
                delay = self.config.get_delay(attempt)
                self.stats["retry_calls"] += 1
                self.stats["total_retries"] += 1

                logger.warning(
                    f"Retry {attempt + 1}/{self.config.max_retries + 1} "
                    f"after {delay:.2f}s: {e}"
                )
                time.sleep(delay)

        raise RetryError(f"Failed after {self.config.max_retries + 1} attempts") from last_exception  # type: ignore

    def get_stats(self) -> dict:
        """Получить статистику."""
        total = self.stats["total_calls"] or 1
        return {
            **self.stats,
            "success_rate": round(
                self.stats["successful_calls"] / total * 100, 2
            ),
            "retry_rate": round(
                self.stats["total_retries"] / total * 100, 2
            ),
            "avg_retries_per_call": round(
                self.stats["total_retries"] / total, 2
            ),
        }


# Глобальный executor для LLM запросов
llm_retry_executor = RetryExecutor(
    RetryConfig(
        max_retries=3,
        base_delay=0.5,
        max_delay=10.0,
        jitter=True,
    )
)
