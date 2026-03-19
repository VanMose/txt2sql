# src\llm\llm_cache.py
"""
LLM Cache - Кэширование ответов LLM с Semantic Search.

Кэширует:
- Промпт → Ответ LLM (exact match)
- Query + Schema → SQL candidates (semantic match)

Оптимизации:
1. Exact match cache - по хэшу промпта
2. Semantic cache - по cosine similarity эмбеддингов
3. Disk persistence - сохранение на диск

Использование:
    from llm.llm_cache import LLMCache, SemanticCache

    cache = LLMCache()
    semantic_cache = SemanticCache()
    
    result = cache.get_or_generate(prompt, generate_fn)
    result = semantic_cache.get_similar(query, threshold=0.95)
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LLMCache:
    """
    Кэш для LLM ответов (exact match).

    Attributes:
        cache_dir: Директория для кэша.
        cache_file: Путь к файлу кэша.
        cache: Данные кэша.
        max_size: Максимальный размер кэша.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_size: int = 10000
    ) -> None:
        """
        Инициализировать кэш.

        Args:
            cache_dir: Директория для кэша.
            max_size: Максимальное количество записей.
        """
        if cache_dir is None:
            from src.config.settings import get_settings
            settings = get_settings()
            cache_dir = Path(settings.base_dir) / "cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "llm_cache.json"
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

        self._load_cache()
        logger.info(f"LLMCache initialized: {len(self.cache)} entries")

    def _generate_key(self, prompt: str, **kwargs: Any) -> str:
        """
        Сгенерировать ключ кэша.

        Args:
            prompt: Промпт.
            **kwargs: Дополнительные параметры.

        Returns:
            MD5 хеш ключа.
        """
        key_data = {"prompt": prompt, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_cache(self) -> None:
        """Загрузить кэш из файла."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cache entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}

    def _save_cache(self) -> None:
        """Сохранить кэш в файл."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(self.cache)} cache entries")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def get(self, key: str) -> Optional[Any]:
        """
        Получить значение из кэша.

        Args:
            key: Ключ.

        Returns:
            Значение или None.
        """
        if key in self.cache:
            self.hits += 1
            logger.debug(f"Cache HIT: {key[:20]}...")
            return self.cache[key]

        self.misses += 1
        logger.debug(f"Cache MISS: {key[:20]}...")
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Сохранить значение в кэш.

        Args:
            key: Ключ.
            value: Значение.
        """
        if len(self.cache) >= self.max_size:
            # Удаляем половину старых записей
            keys_to_delete = list(self.cache.keys())[: self.max_size // 2]
            for k in keys_to_delete:
                del self.cache[k]
            logger.info(f"Cache trimmed: removed {len(keys_to_delete)} entries")

        self.cache[key] = value
        self._save_cache()

    def get_or_generate(
        self,
        prompt: str,
        generate_fn: Callable[[str], List[str]],
        **kwargs: Any
    ) -> List[str]:
        """
        Получить из кэша или сгенерировать.

        Args:
            prompt: Промпт.
            generate_fn: Функция генерации.
            **kwargs: Дополнительные параметры.

        Returns:
            Список ответов.
        """
        key = self._generate_key(prompt, **kwargs)

        cached = self.get(key)
        if cached is not None:
            return cached

        start_time = time.time()
        result = generate_fn(prompt)
        gen_time = time.time() - start_time

        self.set(key, result)

        hit_rate = self.hits / (self.hits + self.misses) * 100 if (self.hits + self.misses) > 0 else 0
        logger.info(f"Generated in {gen_time:.2f}s, cache hit rate: {hit_rate:.1f}%")

        return result

    def clear(self) -> None:
        """Очистить кэш."""
        self.cache = {}
        self._save_cache()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику кэша.

        Returns:
            Статистика.
        """
        hit_rate = self.hits / (self.hits + self.misses) * 100 if (self.hits + self.misses) > 0 else 0
        return {
            "entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "max_size": self.max_size,
        }


class SemanticCache:
    """
    Semantic Cache на основе cosine similarity.

    Кэширует запросы по семантической близости:
    - Если cosine_similarity(query, cached_query) > threshold → cache hit
    - Использует embedding модель для векторизации

    Attributes:
        cache_dir: Директория для кэша.
        threshold: Порог cosine similarity.
        max_size: Максимальный размер кэша.
        embeddings: Кэш эмбеддингов запросов.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        threshold: Optional[float] = None,
        max_size: int = 5000
    ) -> None:
        """
        Инициализировать semantic cache.

        Args:
            cache_dir: Директория для кэша.
            threshold: Порог cosine similarity.
            max_size: Максимальное количество записей.
        """
        from src.config.settings import get_settings
        settings = get_settings()
        
        if cache_dir is None:
            cache_dir = Path(settings.base_dir) / "cache"
        if threshold is None:
            threshold = settings.semantic_cache_threshold

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "semantic_cache.json"
        self.threshold = threshold
        self.max_size = max_size

        # Кэш: query → {response, embedding, timestamp}
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Embedding модель (ленивая загрузка)
        self._embedder = None
        
        # Статистика
        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0

        self._load_cache()
        logger.info(f"SemanticCache initialized: {len(self.cache)} entries, threshold={threshold}")

    @property
    def embedder(self):
        """Ленивая загрузка embedding модели."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                from src.config.settings import get_settings
                settings = get_settings()
                self._embedder = SentenceTransformer(settings.embedding_model)
                logger.info(f"SemanticCache: SentenceTransformer loaded '{settings.embedding_model}'")
            except ImportError:
                logger.warning("SemanticCache: sentence-transformers not available, using exact match only")
                self._embedder = None
        return self._embedder

    def _generate_key(self, text: str) -> str:
        """Сгенерировать ключ кэша."""
        return hashlib.md5(text.encode()).hexdigest()

    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Вычислить эмбеддинг текста.

        Args:
            text: Текст для эмбеддинга.

        Returns:
            Вектор эмбеддинга или None.
        """
        if self.embedder is None:
            return None

        key = self._generate_key(text)
        if key in self.embeddings:
            return self.embeddings[key]

        embedding = self.embedder.encode(text, normalize_embeddings=True)
        self.embeddings[key] = embedding
        return embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Вычислить cosine similarity между векторами.

        Args:
            a: Первый вектор.
            b: Второй вектор.

        Returns:
            Cosine similarity.
        """
        return float(np.dot(a, b))

    def _load_cache(self) -> None:
        """Загрузить кэш из файла."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.cache = data.get("cache", {})
                    # Эмбеддинги не загружаем, вычислим при необходимости
                logger.info(f"Loaded {len(self.cache)} semantic cache entries")
            except Exception as e:
                logger.warning(f"Failed to load semantic cache: {e}")
                self.cache = {}

    def _save_cache(self) -> None:
        """Сохранить кэш в файл (без эмбеддингов)."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({"cache": self.cache}, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(self.cache)} semantic cache entries")
        except Exception as e:
            logger.warning(f"Failed to save semantic cache: {e}")

    def get_similar(
        self,
        query: str,
        threshold: Optional[float] = None
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Найти семантически похожий запрос в кэше.

        Args:
            query: Запрос для поиска.
            threshold: Порог similarity (переопределение).

        Returns:
            Кортеж (cached_response, similarity) или (None, None).
        """
        if not self.cache:
            self.misses += 1
            return None, None

        threshold = threshold or self.threshold
        query_embedding = self._compute_embedding(query)

        if query_embedding is None:
            # Fallback на exact match
            key = self._generate_key(query)
            if key in self.cache:
                self.hits += 1
                return self.cache[key]["response"], 1.0
            self.misses += 1
            return None, None

        best_match = None
        best_similarity = 0.0

        for cached_query, data in self.cache.items():
            cached_embedding = self._compute_embedding(cached_query)
            if cached_embedding is None:
                continue

            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = data["response"]

        if best_match is not None:
            self.semantic_hits += 1
            self.hits += 1
            logger.debug(f"Semantic cache HIT: similarity={best_similarity:.4f}")
            return best_match, best_similarity

        self.misses += 1
        logger.debug(f"Semantic cache MISS")
        return None, None

    def set(self, query: str, response: Any) -> None:
        """
        Сохранить запрос и ответ в кэш.

        Args:
            query: Запрос.
            response: Ответ.
        """
        if len(self.cache) >= self.max_size:
            # Удаляем половину старых записей
            keys_to_delete = list(self.cache.keys())[: self.max_size // 2]
            for k in keys_to_delete:
                del self.cache[k]
                key = self._generate_key(k)
                if key in self.embeddings:
                    del self.embeddings[key]
            logger.info(f"SemanticCache trimmed: removed {len(keys_to_delete)} entries")

        self.cache[query] = {
            "response": response,
            "timestamp": time.time(),
        }

        # Вычисляем и кэшируем эмбеддинг
        self._compute_embedding(query)
        self._save_cache()

    def get_or_generate(
        self,
        query: str,
        generate_fn: Callable[[str], List[str]],
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Получить из semantic cache или сгенерировать.

        Args:
            query: Запрос.
            generate_fn: Функция генерации.
            threshold: Порог similarity.

        Returns:
            Список ответов.
        """
        # Проверка semantic cache
        cached_response, similarity = self.get_similar(query, threshold)
        if cached_response is not None:
            logger.info(f"Semantic cache hit: {similarity:.4f}")
            try:
                return json.loads(cached_response) if isinstance(cached_response, str) else cached_response
            except:
                return [cached_response]

        # Генерация нового ответа
        start_time = time.time()
        result = generate_fn(query)
        gen_time = time.time() - start_time

        # Сохранение в кэш
        self.set(query, json.dumps(result) if isinstance(result, list) else result)

        hit_rate = self.hits / (self.hits + self.misses) * 100 if (self.hits + self.misses) > 0 else 0
        logger.info(f"Generated in {gen_time:.2f}s, semantic cache hit rate: {hit_rate:.1f}%")

        return result

    def clear(self) -> None:
        """Очистить кэш."""
        self.cache.clear()
        self.embeddings.clear()
        self._save_cache()
        logger.info("SemanticCache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику кэша.

        Returns:
            Статистика.
        """
        hit_rate = self.hits / (self.hits + self.misses) * 100 if (self.hits + self.misses) > 0 else 0
        semantic_hit_rate = self.semantic_hits / (self.hits + self.misses) * 100 if (self.hits + self.misses) > 0 else 0
        return {
            "entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "semantic_hits": self.semantic_hits,
            "hit_rate": round(hit_rate, 2),
            "semantic_hit_rate": round(semantic_hit_rate, 2),
            "threshold": self.threshold,
            "max_size": self.max_size,
        }


# Глобальные кэши
_global_cache: Optional[LLMCache] = None
_global_semantic_cache: Optional[SemanticCache] = None


def get_llm_cache() -> LLMCache:
    """Получить глобальный LLM cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = LLMCache()
    return _global_cache


def get_semantic_cache(threshold: Optional[float] = None) -> SemanticCache:
    """Получить глобальный semantic cache."""
    global _global_semantic_cache
    if _global_semantic_cache is None:
        _global_semantic_cache = SemanticCache()
    if threshold is not None:
        _global_semantic_cache.threshold = threshold
    return _global_semantic_cache
