# src\services\result_cache.py
"""
Query Result Cache - Кэширование результатов SQL запросов.

Production features:
- TTL cache для авто-инвалидации
- LRU eviction policy
- Hash-based lookup
- Thread-safe

Example:
    from services.result_cache import ResultCache
    
    cache = ResultCache()
    cache.set(query, result, ttl=3600)
    result = cache.get(query)
"""
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


class ResultCache:
    """
    Кэш результатов SQL запросов.
    
    Attributes:
        max_size: Максимальное количество записей.
        default_ttl: TTL по умолчанию (секунды).
        _cache: OrderedDict для LRU eviction.
        _timestamps: Временные метки записей.
        _lock: Thread lock для безопасности.
    """
    
    def __init__(
        self,
        max_size: int = 5000,
        default_ttl: int = 3600,
    ) -> None:
        """
        Инициализировать кэш.
        
        Args:
            max_size: Максимальный размер кэша.
            default_ttl: TTL по умолчанию в секундах.
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Статистика
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"ResultCache initialized: max_size={max_size}, default_ttl={default_ttl}s")
    
    def _generate_key(self, query: str, **kwargs: Any) -> str:
        """
        Сгенерировать ключ кэша.
        
        Args:
            query: SQL запрос.
            **kwargs: Дополнительные параметры.
        
        Returns:
            MD5 хеш ключа.
        """
        key_data = {"query": query, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """
        Проверить истёк ли TTL у записи.
        
        Args:
            key: Ключ записи.
        
        Returns:
            True если запись истекла.
        """
        if key not in self._timestamps:
            return True
        
        timestamp = self._timestamps[key]
        age = time.time() - timestamp
        
        return age > self.default_ttl
    
    def _evict_if_needed(self) -> None:
        """Удалить старые записи если кэш переполнен."""
        while len(self._cache) >= self.max_size:
            # Удаляем oldest entry (LRU)
            oldest_key = next(iter(self._cache))
            self._remove(oldest_key)
            self.evictions += 1
    
    def _remove(self, key: str) -> None:
        """
        Удалить запись из кэша.
        
        Args:
            key: Ключ записи.
        """
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
    
    def get(self, query: str, **kwargs: Any) -> Optional[Any]:
        """
        Получить результат из кэша.
        
        Args:
            query: SQL запрос.
            **kwargs: Дополнительные параметры.
        
        Returns:
            Результат или None.
        """
        key = self._generate_key(query, **kwargs)
        
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                logger.debug(f"ResultCache MISS: {query[:50]}...")
                return None
            
            # Проверка TTL
            if self._is_expired(key):
                self._remove(key)
                self.misses += 1
                logger.debug(f"ResultCache EXPIRED: {query[:50]}...")
                return None
            
            # Update LRU order (move to end)
            self._cache.move_to_end(key)
            self.hits += 1
            
            hit_rate = self.hits / (self.hits + self.misses) * 100 if (self.hits + self.misses) > 0 else 0
            logger.debug(f"ResultCache HIT: {query[:50]}... (hit_rate={hit_rate:.1f}%)")
            
            return self._cache[key]
    
    def set(
        self,
        query: str,
        result: Any,
        ttl: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Сохранить результат в кэш.
        
        Args:
            query: SQL запрос.
            result: Результат выполнения.
            ttl: TTL в секундах (переопределение).
            **kwargs: Дополнительные параметры.
        """
        key = self._generate_key(query, **kwargs)
        
        with self._lock:
            # Evict old entries if needed
            self._evict_if_needed()
            
            # Сохранение записи
            self._cache[key] = result
            self._cache.move_to_end(key)
            
            # Установка TTL
            ttl = ttl if ttl is not None else self.default_ttl
            self._timestamps[key] = time.time()
            
            logger.debug(f"ResultCache SET: {query[:50]}... (ttl={ttl}s)")
    
    def delete(self, query: str, **kwargs: Any) -> bool:
        """
        Удалить запись из кэша.
        
        Args:
            query: SQL запрос.
            **kwargs: Дополнительные параметры.
        
        Returns:
            True если запись была удалена.
        """
        key = self._generate_key(query, **kwargs)
        
        with self._lock:
            if key in self._cache:
                self._remove(key)
                logger.debug(f"ResultCache DELETE: {query[:50]}...")
                return True
            return False
    
    def clear(self) -> None:
        """Очистить весь кэш."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            logger.info("ResultCache cleared")
    
    def cleanup_expired(self) -> int:
        """
        Очистить истёкшие записи.
        
        Returns:
            Количество удалённых записей.
        """
        removed = 0
        
        with self._lock:
            keys_to_remove = [
                key for key in self._cache
                if self._is_expired(key)
            ]
            
            for key in keys_to_remove:
                self._remove(key)
                removed += 1
        
        if removed > 0:
            logger.info(f"ResultCache cleanup: removed {removed} expired entries")
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику кэша.
        
        Returns:
            Статистика.
        """
        hit_rate = self.hits / (self.hits + self.misses) * 100 if (self.hits + self.misses) > 0 else 0
        
        return {
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(hit_rate, 2),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
        }
    
    def __len__(self) -> int:
        """Получить количество записей в кэше."""
        return len(self._cache)
    
    def __contains__(self, query: str) -> bool:
        """Проверить есть ли запрос в кэше (без проверки TTL)."""
        key = self._generate_key(query)
        return key in self._cache


# Глобальный кэш результатов
_global_result_cache: Optional[ResultCache] = None


def get_result_cache(
    max_size: Optional[int] = None,
    ttl: Optional[int] = None,
) -> ResultCache:
    """
    Получить глобальный кэш результатов.
    
    Args:
        max_size: Максимальный размер (переопределение).
        ttl: TTL в секундах (переопределение).
    
    Returns:
        ResultCache экземпляр.
    """
    global _global_result_cache
    
    if _global_result_cache is None:
        from src.config.settings import get_settings
        settings = get_settings()
        
        max_size = max_size or settings.result_cache_max_size
        ttl = ttl or settings.result_cache_ttl_seconds
        
        _global_result_cache = ResultCache(
            max_size=max_size,
            default_ttl=ttl,
        )
    
    return _global_result_cache
