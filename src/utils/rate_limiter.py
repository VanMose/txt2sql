"""
Rate Limiter для LLM запросов.

Оптимизации:
1. Token Bucket - алгоритм ограничения запросов
2. Sliding Window - скользящее окно для статистики
3. Async support - асинхронная поддержка

Использование:
    rate_limiter = RateLimiter(requests_per_minute=60)

    @rate_limiter.limit()
    def generate(prompt: str) -> str:
        ...
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Deque, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RateLimitStats:
    """Статистика rate limiting."""

    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    total_wait_time_ms: float = 0
    window_start: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "allowed_requests": self.allowed_requests,
            "rejected_requests": self.rejected_requests,
            "rejection_rate": round(
                self.rejected_requests / max(self.total_requests, 1) * 100, 2
            ),
            "avg_wait_time_ms": round(
                self.total_wait_time_ms / max(self.allowed_requests, 1), 2
            ),
        }


class TokenBucket:
    """
    Token Bucket алгоритм для rate limiting.

    Attributes:
        capacity: Максимальное количество токенов.
        refill_rate: Токенов в секунду.
        tokens: Текущее количество токенов.
        last_refill: Время последнего пополнения.
    """

    def __init__(self, capacity: int, refill_rate: float) -> None:
        """
        Инициализировать Token Bucket.

        Args:
            capacity: Максимум токенов.
            refill_rate: Токенов в секунду.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Пополнить токены."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Запросить токены.

        Args:
            tokens: Количество токенов.
            blocking: Блокировать если нет токенов.
            timeout: Таймаут ожидания.

        Returns:
            True если токены получены.
        """
        start_time = time.time()

        while True:
            with self._lock:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                if not blocking:
                    return False

            # Проверка таймаута
            if timeout and (time.time() - start_time) >= timeout:
                return False

            # Ожидание
            time.sleep(0.1)

    def get_available_tokens(self) -> int:
        """Получить доступное количество токенов."""
        with self._lock:
            self._refill()
            return int(self.tokens)


class SlidingWindowCounter:
    """
    Sliding Window Counter для rate limiting.

    Подсчитывает запросы в скользящем окне.
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        """
        Инициализировать Sliding Window.

        Args:
            max_requests: Максимум запросов в окне.
            window_seconds: Размер окна в секундах.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Deque[float] = deque()
        self._lock = threading.Lock()

    def _cleanup(self) -> None:
        """Удалить старые запросы."""
        now = time.time()
        cutoff = now - self.window_seconds

        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Запросить разрешение на запрос.

        Args:
            blocking: Блокировать если лимит достигнут.
            timeout: Таймаут ожидания.

        Returns:
            True если разрешение получено.
        """
        start_time = time.time()

        while True:
            with self._lock:
                self._cleanup()

                if len(self.requests) < self.max_requests:
                    self.requests.append(time.time())
                    return True

                if not blocking:
                    return False

            # Проверка таймаута
            if timeout and (time.time() - start_time) >= timeout:
                return False

            # Ожидание
            time.sleep(0.1)

    def get_current_count(self) -> int:
        """Получить текущее количество запросов в окне."""
        with self._lock:
            self._cleanup()
            return len(self.requests)

    def get_remaining(self) -> int:
        """Получить оставшееся количество запросов."""
        with self._lock:
            self._cleanup()
            return max(0, self.max_requests - len(self.requests))


class RateLimiter:
    """
    Rate Limiter для LLM запросов.

    Комбинирует Token Bucket и Sliding Window.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_second: int = 10,
        burst_capacity: int = 10,
    ) -> None:
        """
        Инициализировать Rate Limiter.

        Args:
            requests_per_minute: Максимум запросов в минуту.
            requests_per_second: Максимум запросов в секунду.
            burst_capacity: Ёмкость burst.
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second

        # Token Bucket для burst
        self.token_bucket = TokenBucket(
            capacity=burst_capacity,
            refill_rate=requests_per_second,
        )

        # Sliding Window для минутного лимита
        self.sliding_window = SlidingWindowCounter(
            max_requests=requests_per_minute,
            window_seconds=60.0,
        )

        # Sliding Window для секундного лимита
        self.per_second_window = SlidingWindowCounter(
            max_requests=requests_per_second,
            window_seconds=1.0,
        )

        self.stats = RateLimitStats()
        self._lock = threading.Lock()

        logger.info(
            f"RateLimiter initialized: {requests_per_minute}/min, "
            f"{requests_per_second}/sec, burst={burst_capacity}"
        )

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Запросить разрешение на запрос.

        Args:
            blocking: Блокировать если лимит достигнут.
            timeout: Таймаут ожидания.

        Returns:
            True если разрешение получено.
        """
        with self._lock:
            self.stats.total_requests += 1

        start_time = time.time()

        # Проверка всех лимитов
        while True:
            # Token Bucket
            if not self.token_bucket.acquire(blocking=False):
                if not blocking:
                    with self._lock:
                        self.stats.rejected_requests += 1
                    return False

            # Per-second window
            if not self.per_second_window.acquire(blocking=False):
                if not blocking:
                    with self._lock:
                        self.stats.rejected_requests += 1
                    return False

            # Per-minute window
            if not self.sliding_window.acquire(blocking=False):
                if not blocking:
                    with self._lock:
                        self.stats.rejected_requests += 1
                    return False

            # Все лимиты пройдены
            with self._lock:
                self.stats.allowed_requests += 1
                wait_time = (time.time() - start_time) * 1000
                self.stats.total_wait_time_ms += wait_time

            return True

        # Проверка таймаута
        if timeout and (time.time() - start_time) >= timeout:
            with self._lock:
                self.stats.rejected_requests += 1
            return False

    def limit(
        self,
        blocking: bool = True,
        timeout: Optional[float] = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Декоратор для rate limiting.

        Args:
            blocking: Блокировать если лимит достигнут.
            timeout: Таймаут ожидания.

        Returns:
            Декоратор.

        Example:
            @rate_limiter.limit()
            def generate(prompt: str) -> str:
                ...
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                if not self.acquire(blocking=blocking, timeout=timeout):
                    raise RateLimitExceeded(
                        f"Rate limit exceeded for {func.__name__}"
                    )
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        return {
            **self.stats.to_dict(),
            "available_tokens": self.token_bucket.get_available_tokens(),
            "remaining_per_second": self.per_second_window.get_remaining(),
            "remaining_per_minute": self.sliding_window.get_remaining(),
        }

    def reset_stats(self) -> None:
        """Сбросить статистику."""
        self.stats = RateLimitStats()


class RateLimitExceeded(Exception):
    """Исключение при превышении лимита."""

    pass


# Глобальный rate limiter для LLM запросов
llm_rate_limiter = RateLimiter(
    requests_per_minute=60,
    requests_per_second=5,
    burst_capacity=10,
)
