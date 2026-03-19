# text2sql_baseline\src\utils\optimizations.py
"""Оптимизации: кэширование, батчинг, квантизация."""
import logging
import hashlib
import time
from typing import Any, Dict, List, Optional, Callable, TypeVar
from functools import wraps
from pathlib import Path
import json
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Кэширование
# =============================================================================

class LRUCache:
    """
    LRU (Least Recently Used) кэш с ограничением размера.

    Потокобезопасная реализация с блокировками.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Инициализировать LRU кэш.

        Args:
            max_size: Максимальный размер кэша.
            ttl_seconds: Время жизни записи в секундах.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """
        Получить значение из кэша.

        Args:
            key: Ключ.

        Returns:
            Значение или None.
        """
        with self._lock:
            if key not in self._cache:
                return None

            # Проверка TTL
            if time.time() - self._timestamps[key] > self.ttl_seconds:
                self._remove(key)
                return None

            # Обновление порядка доступа
            self._access_order.remove(key)
            self._access_order.append(key)

            logger.debug(f"Cache hit: {key}")
            return self._cache[key]

    def put(self, key: str, value: Any):
        """
        Положить значение в кэш.

        Args:
            key: Ключ.
            value: Значение.
        """
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self.max_size:
                # Удаление наименее используемого
                oldest = self._access_order.pop(0)
                self._remove(oldest)

            self._cache[key] = value
            self._access_order.append(key)
            self._timestamps[key] = time.time()

            logger.debug(f"Cache put: {key}")

    def _remove(self, key: str):
        """Удалить ключ из кэша."""
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]

    def clear(self):
        """Очистить кэш."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._timestamps.clear()
            logger.info("Cache cleared")

    def stats(self) -> Dict[str, int]:
        """Получить статистику кэша."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "keys": len(self._access_order),
        }


class DiskCache:
    """
    Кэш на диске для больших объектов.

    Использует JSON для сериализации.
    """

    def __init__(self, cache_dir: str, max_size_mb: float = 100.0):
        """
        Инициализировать дисковый кэш.

        Args:
            cache_dir: Директория для кэша.
            max_size_mb: Максимальный размер в МБ.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb

        logger.info(f"DiskCache initialized: {cache_dir}")

    def _get_key_path(self, key: str) -> Path:
        """Получить путь к файлу для ключа."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша."""
        path = self._get_key_path(key)

        if not path.exists():
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.debug(f"DiskCache hit: {key}")
            return data

        except Exception as e:
            logger.warning(f"DiskCache read error: {e}")
            return None

    def put(self, key: str, value: Any):
        """Положить значение в кэш."""
        path = self._get_key_path(key)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False)

            logger.debug(f"DiskCache put: {key}")

            # Проверка размера
            self._enforce_size_limit()

        except Exception as e:
            logger.error(f"DiskCache write error: {e}")

    def _enforce_size_limit(self):
        """Принудительное ограничение размера."""
        try:
            total_size = sum(
                f.stat().st_size for f in self.cache_dir.glob("*.json")
            )
            max_size_bytes = self.max_size_mb * 1024 * 1024

            if total_size > max_size_bytes:
                # Удаление старых файлов
                files = sorted(
                    self.cache_dir.glob("*.json"),
                    key=lambda f: f.stat().st_mtime
                )

                for f in files[: len(files) // 2]:
                    f.unlink()

                logger.info("DiskCache size limit enforced")

        except Exception as e:
            logger.warning(f"DiskCache size enforcement error: {e}")

    def clear(self):
        """Очистить кэш."""
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        logger.info("DiskCache cleared")


def cached(
    cache: LRUCache,
    key_fn: Optional[Callable[..., str]] = None,
):
    """
    Декоратор для кэширования результатов функции.

    Args:
        cache: Кэш для использования.
        key_fn: Функция для генерации ключа.

    Returns:
        Декоратор.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Генерация ключа
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__name__}:{args}:{kwargs}"

            # Проверка кэша
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result

            # Выполнение функции
            result = func(*args, **kwargs)

            # Сохранение в кэш
            cache.put(key, result)

            return result

        return wrapper
    return decorator


# =============================================================================
# Батчинг
# =============================================================================

def batched(iterable: List[T], batch_size: int) -> List[List[T]]:
    """
    Разбить итерируемый объект на батчи.

    Args:
        iterable: Итерируемый объект.
        batch_size: Размер батча.

    Returns:
        Список батчей.
    """
    batches = []

    for i in range(0, len(iterable), batch_size):
        batch = iterable[i:i + batch_size]
        batches.append(batch)

    return batches


def process_batched(
    items: List[T],
    processor: Callable[[List[T]], Any],
    batch_size: int = 32,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Any]:
    """
    Обработать элементы батчами.

    Args:
        items: Элементы для обработки.
        processor: Функция обработки батча.
        batch_size: Размер батча.
        progress_callback: Callback для прогресса (processed, total).

    Returns:
        Список результатов.
    """
    batches = batched(items, batch_size)
    results = []

    for i, batch in enumerate(batches):
        logger.debug(f"Processing batch {i + 1}/{len(batches)}")

        result = processor(batch)
        results.append(result)

        if progress_callback:
            progress_callback((i + 1) * batch_size, len(items))

    return results


# =============================================================================
# Квантизация
# =============================================================================

def get_quantization_config(
    use_4bit: bool = True,
    use_8bit: bool = False,
) -> Dict[str, Any]:
    """
    Получить конфигурацию квантизации для transformers.

    Args:
        use_4bit: Использовать 4-битную квантизацию.
        use_8bit: Использовать 8-битную квантизацию.

    Returns:
        Конфигурация для BitsAndBytes.
    """
    config = {}

    if use_4bit:
        config.update({
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True,
        })
    elif use_8bit:
        config.update({
            "load_in_8bit": True,
        })

    return config


def estimate_memory_usage(
    model_params: int,
    dtype: str = "float16",
    use_quantization: bool = False,
) -> float:
    """
    Оценить использование памяти моделью.

    Args:
        model_params: Количество параметров модели.
        dtype: Тип данных (float32, float16, int8, int4).
        use_quantization: Использовать квантизацию.

    Returns:
        Использование памяти в ГБ.
    """
    # Базовое использование (байты на параметр)
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "int8": 1,
        "int4": 0.5,
    }

    bpp = bytes_per_param.get(dtype, 2)

    if use_quantization:
        bpp = 0.5  # 4-bit

    memory_bytes = model_params * bpp
    memory_gb = memory_bytes / (1024 ** 3)

    # Добавляем overhead для активаций и кэша
    memory_gb *= 1.2

    return memory_gb


# =============================================================================
# Тайминг и профилирование
# =============================================================================

def timed(func: Callable[..., T]) -> Callable[..., T]:
    """
    Декоратор для замера времени выполнения.

    Args:
        func: Функция.

    Returns:
        Обёрнутая функция.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        logger.info(f"{func.__name__} executed in {elapsed * 1000:.0f}ms")

        return result

    return wrapper


class Timer:
    """Контекстный менеджер для замера времени."""

    def __init__(self, name: str = "Timer"):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        logger.debug(f"{self.name}: {self.elapsed * 1000:.0f}ms")


# =============================================================================
# Глобальные кэши
# =============================================================================

# Глобальный LRU кэш для embeddings
embedding_cache = LRUCache(max_size=5000, ttl_seconds=3600)

# Глобальный кэш для результатов поиска
search_cache = LRUCache(max_size=1000, ttl_seconds=300)

# Глобальный дисковый кэш для больших схем
schema_disk_cache = DiskCache(cache_dir="cache/schemas", max_size_mb=50.0)
