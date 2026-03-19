# src\config\settings.py
"""
Настройки проекта Text-to-SQL Pipeline.

Модуль предоставляет централизованное управление конфигурацией через Pydantic Settings.
Поддерживает загрузку из переменных окружения с префиксом TEXT2SQL_.

Example:
    >>> from config.settings import get_settings
    >>> settings = get_settings()
    >>> print(settings.llm_model)
    'Qwen2.5-Coder-3B'
"""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Базовая директория проекта (text2sql_baseline)
BASE_DIR: Path = Path(__file__).parent.parent.parent

# Явная загрузка .env перед созданием Settings
load_dotenv(BASE_DIR / ".env", override=True)
logger.debug(f"Loaded .env from {BASE_DIR / '.env'}")


class Settings(BaseSettings):
    """
    Конфигурация Text-to-SQL пайплайна.

    Attributes:
        base_dir: Базовая директория проекта.
        models_path: Путь к директории с моделями.
        data_path: Путь к директории с данными.
        logs_path: Путь к директории с логами.
        configs_path: Путь к директории с конфигурациями.
        llm_model: Модель для генерации SQL и оценки.
        use_local_model: Использовать локальную модель.
        gpu_memory_utilization: Утилизация памяти GPU.
        n_samples: Количество SQL кандидатов (self-consistency).
        temperature: Температура генерации.
        max_tokens: Максимальное количество токенов.
        top_k_tables: Количество таблиц для retrieval.
        embedding_model: Модель для эмбеддингов.
        use_local_embedding: Использовать локальную embedding модель.
        max_retries: Максимальное количество попыток.
        confidence_threshold: Порог уверенности.
        db_path: Путь к SQLite базе данных (single-DB режим).
        db_paths: Список путей к базам данных (multi-DB режим).
        auto_discover_dbs: Автоматическое обнаружение БД.
        top_k_dbs: Максимальное количество выбираемых БД.
        use_llm_ranking: Использовать LLM для ранжирования БД.
        qdrant_use_local: Использовать локальный Qdrant.
        qdrant_local_path: Путь для локального Qdrant.
        qdrant_url: URL удалённого Qdrant сервера.
        qdrant_api_key: API ключ Qdrant.
        qdrant_collection_name: Имя коллекции Qdrant.
        neo4j_uri: URI Neo4j сервера.
        neo4j_username: Имя пользователя Neo4j.
        neo4j_password: Пароль Neo4j.
        neo4j_database: Имя базы данных Neo4j.
        log_level: Уровень логирования.
        log_file: Путь к файлу логов.
        log_format: Формат логов.
    """

    # ===========================================
    # Paths
    # ===========================================

    base_dir: str = Field(default_factory=lambda: str(BASE_DIR))
    models_path: str = Field(default_factory=lambda: str(BASE_DIR / "llm_models"))
    data_path: str = Field(default_factory=lambda: str(BASE_DIR / "data"))
    logs_path: str = Field(default_factory=lambda: str(BASE_DIR / "logs"))
    configs_path: str = Field(default_factory=lambda: str(BASE_DIR / "configs"))

    # ===========================================
    # LLM Model Settings
    # ===========================================

    # Ollama Settings (primary)
    use_ollama: bool = Field(default=True, description="Использовать Ollama API")
    ollama_base_url: str = Field(default="http://localhost:11434", description="URL Ollama сервера")
    
    # Transformers Settings (fallback)
    llm_model: str = Field(default="qwen2.5-coder:1.5b", description="Модель для генерации SQL и judge")
    use_local_model: bool = Field(default=False)
    gpu_memory_utilization: float = Field(default=0.95, ge=0.1, le=1.0)

    # Generation Parameters
    n_samples: int = Field(default=2, ge=1, description="Количество SQL кандидатов (self-consistency)")
    temperature: float = Field(default=0.15, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1)

    # ===========================================
    # Retrieval Settings
    # ===========================================

    top_k_tables: int = Field(default=5, ge=1, description="Количество таблиц для retrieval")
    embedding_model: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2")
    use_local_embedding: bool = Field(default=True)
    
    # Reranking Settings
    use_reranking: bool = Field(default=True, description="Использовать cross-encoder reranking")
    reranker_model: str = Field(default="BAAI/bge-reranker-base", description="Модель для reranking")
    use_local_reranker: bool = Field(default=True, description="Использовать локальную reranker модель")
    reranker_top_k: int = Field(default=5, ge=1, description="Количество таблиц после reranking")

    # ===========================================
    # Retry & Confidence
    # ===========================================

    max_retries: int = Field(default=1, ge=0)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # ===========================================
    # Single-DB Settings (legacy)
    # ===========================================

    db_path: str = Field(default="db.sqlite")

    # ===========================================
    # Multi-DB Settings
    # ===========================================

    db_paths: List[str] = Field(default_factory=list)
    auto_discover_dbs: bool = Field(default=True)
    top_k_dbs: int = Field(default=2, ge=1)
    use_llm_ranking: bool = Field(default=True)

    # ===========================================
    # Optimization Settings (NEW)
    # ===========================================

    use_4bit_quantization: bool = Field(default=True, description="Использовать 4-битную квантизацию")
    use_torch_compile: bool = Field(default=True, description="Использовать torch.compile")
    use_flash_attention: bool = Field(default=False, description="Использовать Flash Attention 2")
    use_semantic_cache: bool = Field(default=True, description="Использовать semantic cache")
    semantic_cache_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Порог cosine similarity для semantic cache")
    use_parallel_execution: bool = Field(default=True, description="Использовать параллельное выполнение")
    use_prompt_cache: bool = Field(default=True, description="Использовать кэширование промптов")
    model_warmup: bool = Field(default=True, description="Прогрев модели при инициализации")
    early_exit_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Порог confidence для early exit")

    # ===========================================
    # Qdrant Vector DB Settings
    # ===========================================

    qdrant_use_local: bool = Field(default=True)
    qdrant_local_path: str = Field(default="qdrant_storage")
    qdrant_url: Optional[str] = Field(default=None)
    qdrant_api_key: Optional[str] = Field(default=None)
    qdrant_collection_name: str = Field(default="text2sql_schemas")

    # ===========================================
    # Neo4j Graph DB Settings
    # ===========================================

    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")
    neo4j_database: str = Field(default="neo4j")

    # ===========================================
    # Logging
    # ===========================================

    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/text2sql.log")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # ===========================================
    # Additional Agent Settings (NEW)
    # ===========================================

    # Agent temperatures
    refiner_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Температура для refiner агента")
    judge_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Температура для judge агента")
    validator_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Температура для validator агента")

    # Cache settings
    router_cache_max_size: int = Field(default=200, ge=1, description="Максимальный размер кэша router агента")
    reranker_cache_ttl: int = Field(default=300, ge=0, description="TTL кэша reranker в секундах")
    cache_ttl_seconds: int = Field(default=300, ge=0, description="TTL кэша по умолчанию в секундах")

    # SQL Guardrails
    sql_default_limit: int = Field(default=1000, ge=1, description="Лимит по умолчанию для SQL запросов")
    sql_max_limit: int = Field(default=10000, ge=1, description="Максимальный лимит для SQL запросов")

    # Retry settings
    llm_retry_max_retries: int = Field(default=3, ge=0, description="Максимум попыток для LLM retry")
    llm_retry_base_delay: float = Field(default=0.5, ge=0.0, description="Базовая задержка для LLM retry")
    llm_retry_max_delay: float = Field(default=10.0, ge=0.0, description="Максимальная задержка для LLM retry")

    # Confidence cap
    max_confidence_cap: float = Field(default=0.95, ge=0.0, le=1.0, description="Максимальный cap для confidence")

    # ===========================================
    # Router Ranking Settings (Hybrid)
    # ===========================================

    # Режим ранжирования: "vector", "llm", "hybrid"
    # - vector: только vector score + graph связи (быстро, универсально)
    # - llm: LLM ranking (медленно, точнее для сложных запросов)
    # - hybrid: vector для простых, LLM для сложных (баланс)
    router_ranking_mode: str = Field(default="vector", description="Режим ранжирования БД: vector, llm, hybrid")
    
    # Веса для vector ranking
    vector_score_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="Вес vector similarity score")
    graph_score_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Вес graph связей")
    row_count_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Вес количества строк в таблицах")
    
    # Порог для hybrid режима (если confidence < threshold → используем LLM)
    hybrid_llm_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Порог для переключения на LLM в hybrid режиме")

    # ===========================================
    # Properties
    # ===========================================

    @property
    def log_filepath(self) -> str:
        """Полный путь к файлу логов."""
        return str(Path(self.base_dir) / self.log_file)

    @property
    def db_full_path(self) -> str:
        """Полный путь к базе данных."""
        return str(Path(self.base_dir) / self.db_path)

    def get_local_model_path(self, model_name: str) -> str:
        """
        Получить путь к локальной модели.

        Args:
            model_name: Имя модели.

        Returns:
            Полный путь к папке с моделью.
        """
        return str(Path(self.models_path) / model_name)

    def get_local_embedding_path(self) -> str:
        """
        Получить путь к локальной embedding модели.

        Returns:
            Полный путь к папке с embedding моделью.
        """
        return str(Path(self.models_path) / self.embedding_model)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Получить значение настройки по ключу.

        Args:
            key: Ключ настройки.
            default: Значение по умолчанию.

        Returns:
            Значение настройки.
        """
        return getattr(self, key, default)

    model_config = SettingsConfigDict(
        env_prefix="TEXT2SQL_",
        extra="ignore",
        env_file=[BASE_DIR / ".env"],
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Глобальная переменная для переопределения настроек
_override_settings: Dict[str, Any] = {}


@lru_cache
def get_settings(model_name: Optional[str] = None, use_local: Optional[bool] = None) -> Settings:
    """
    Получить настройки с кэшированием.

    Args:
        model_name: Опционально переопределить модель.
        use_local: Опционально переопределить использование локальной модели.

    Returns:
        Настройки.
    """
    settings = Settings()

    if model_name is not None:
        settings.llm_model = model_name
    if use_local is not None:
        settings.use_local_model = use_local

    return settings


def override_settings(model_name: Optional[str] = None, use_local: Optional[bool] = None, **kwargs: Any) -> None:
    """
    Переопределить настройки.

    Args:
        model_name: Имя модели.
        use_local: Использовать локальную модель.
        **kwargs: Дополнительные параметры.
    """
    global _override_settings

    if model_name is not None:
        _override_settings["llm_model"] = model_name
    if use_local is not None:
        _override_settings["use_local_model"] = use_local

    _override_settings.update(kwargs)

    # Очищаем кэш чтобы применить новые настройки
    get_settings.cache_clear()
    logger.info("Settings overridden, cache cleared")


def get_settings_with_override() -> Settings:
    """
    Получить настройки с применёнными переопределениями.

    Returns:
        Настройки с переопределениями.
    """
    settings = get_settings()

    for key, value in _override_settings.items():
        if hasattr(settings, key):
            setattr(settings, key, value)

    return settings
