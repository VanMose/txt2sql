# src\services\pipeline_service.py
"""
Service слой для бизнес-логики Text-to-SQL пайплайна.

Вынесенная логика из app.py:
- Инициализация пайплайна
- Индексация баз данных
- Выполнение запросов
- Управление кэшем

Использование:
    from services.pipeline_service import PipelineService
    
    service = PipelineService(db_paths=["db1.sqlite", "db2.sqlite"])
    service.initialize()
    result = service.run_query("Show movies")
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.settings import get_settings
from ..db.schema_loader import SchemaLoader
from ..pipeline.langgraph_pipeline import MultiDBPipeline
from ..retrieval.vector_db import TableDocument

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Статистика пайплайна."""

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_latency_ms: float = 0
    avg_routing_ms: float = 0
    avg_generation_ms: float = 0
    avg_execution_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        avg_latency = (
            self.total_latency_ms / max(self.total_queries, 1)
        )
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": round(
                self.successful_queries / max(self.total_queries, 1) * 100, 2
            ),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_routing_ms": round(self.avg_routing_ms, 2),
            "avg_generation_ms": round(self.avg_generation_ms, 2),
            "avg_execution_ms": round(self.avg_execution_ms, 2),
        }


@dataclass
class QueryResult:
    """Результат запроса."""

    query: str
    sql: str
    confidence: float
    execution_result: Any
    selected_databases: List[Dict[str, Any]]
    latencies: Dict[str, float]
    success: bool
    error: Optional[str] = None
    refinement_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "sql": self.sql,
            "confidence": self.confidence,
            "execution_result": self.execution_result,
            "selected_databases": self.selected_databases,
            "latencies": self.latencies,
            "success": self.success,
            "error": self.error,
            "refinement_count": self.refinement_count,
        }


class PipelineService:
    """
    Сервис для управления Multi-DB пайплайном.

    Инкапсулирует:
    - Инициализацию Vector DB + Graph DB
    - Индексацию баз данных
    - Выполнение запросов
    - Статистику и мониторинг
    """

    def __init__(
        self,
        db_paths: Optional[List[str]] = None,
        use_local_qdrant: bool = True,
        qdrant_local_path: Optional[str] = None,
    ) -> None:
        """
        Инициализировать сервис.

        Args:
            db_paths: Пути к базам данных.
            use_local_qdrant: Использовать локальный Qdrant.
            qdrant_local_path: Путь к Qdrant.
        """
        settings = get_settings()
        self.db_paths = db_paths or settings.get("db_paths", [])
        self.use_local_qdrant = use_local_qdrant
        self.qdrant_local_path = qdrant_local_path or settings.get("qdrant_local_path")

        self._pipeline: Optional[MultiDBPipeline] = None
        self._initialized = False
        self._stats = PipelineStats()

        logger.info(f"PipelineService initialized with {len(self.db_paths)} databases")

    def initialize(self, warmup_model: bool = True) -> bool:
        """
        Инициализировать пайплайн.

        Args:
            warmup_model: Прогреть модель при инициализации.

        Returns:
            True если успешно.
        """
        try:
            settings = get_settings()

            self._pipeline = MultiDBPipeline(
                db_paths=self.db_paths,
                qdrant_url=settings.qdrant_url if not settings.qdrant_use_local else None,
                qdrant_api_key=settings.qdrant_api_key,
                neo4j_uri=settings.neo4j_uri,
                neo4j_username=settings.neo4j_username,
                neo4j_password=settings.neo4j_password,
                use_local_qdrant=self.use_local_qdrant,
                qdrant_local_path=self.qdrant_local_path,
            )

            # Проверка Vector DB
            vector_stats = self._pipeline.vector_db.get_stats()
            points_count = vector_stats.get("points_count", 0)

            if points_count == 0:
                logger.info("Vector DB empty, indexing databases...")
                self.index_databases()
            else:
                logger.info(f"Vector DB has {points_count} points")

            # Warm-up модели
            if warmup_model and settings.model_warmup:
                self._warmup_model()

            self._initialized = True
            logger.info("PipelineService initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PipelineService: {e}")
            return False

    def _warmup_model(self) -> None:
        """
        Прогреть модель перед первым использованием.

        Использует тестовые промпты для активации модели и кэшей.
        """
        try:
            from ..llm.model_loader import ModelLoader
            
            logger.info("Warming up model...")
            start_time = time.time()
            
            if ModelLoader.warmup():
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"Model warm-up completed in {elapsed:.0f}ms")
            else:
                logger.warning("Model warm-up skipped or failed")
                
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def index_databases(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Индексировать базы данных.

        Args:
            force_reindex: Принудительная переиндексация.

        Returns:
            Статистика индексации.
        """
        if not self._pipeline:
            raise RuntimeError("Pipeline not initialized")

        start_time = time.time()

        self._pipeline.index_databases(force_reindex=force_reindex)

        elapsed_ms = (time.time() - start_time) * 1000

        # Статистика
        vector_stats = self._pipeline.vector_db.get_stats()
        graph_stats = self._pipeline.graph_db.get_stats()

        result = {
            "elapsed_ms": round(elapsed_ms, 2),
            "vector_db": vector_stats,
            "graph_db": graph_stats,
            "databases_indexed": len(self.db_paths),
        }

        logger.info(
            f"Indexing completed in {elapsed_ms:.0f}ms: "
            f"{vector_stats.get('points_count', 0)} tables indexed"
        )

        return result

    def run_query(self, query: str) -> QueryResult:
        """
        Выполнить запрос.

        Args:
            query: Запрос пользователя.

        Returns:
            QueryResult.
        """
        if not self._pipeline:
            raise RuntimeError("Pipeline not initialized")

        start_time = time.time()
        self._stats.total_queries += 1

        try:
            result = self._pipeline.run(query)

            elapsed_ms = (time.time() - start_time) * 1000
            self._stats.successful_queries += 1
            self._stats.total_latency_ms += elapsed_ms

            # Обновление средней статистики
            latencies = result.get("latencies", {})
            self._update_avg_latencies(latencies)

            return QueryResult(
                query=query,
                sql=result.get("best_sql", ""),
                confidence=result.get("confidence", 0.0),
                execution_result=result.get("execution_result"),
                selected_databases=result.get("selected_databases", []),
                latencies=latencies,
                success=result.get("best_sql") is not None,
                refinement_count=result.get("refinement_count", 0),
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self._stats.failed_queries += 1
            self._stats.total_latency_ms += elapsed_ms

            logger.error(f"Query failed: {e}", exc_info=True)

            return QueryResult(
                query=query,
                sql="",
                confidence=0.0,
                execution_result=None,
                selected_databases=[],
                latencies={},
                success=False,
                error=str(e),
            )

    def _update_avg_latencies(self, latencies: Dict[str, float]) -> None:
        """Обновить средние значения латентности."""
        if "routing" in latencies:
            self._stats.avg_routing_ms = (
                self._stats.avg_routing_ms * (self._stats.total_queries - 1)
                + latencies["routing"]
            ) / self._stats.total_queries

        if "sql_generation" in latencies:
            self._stats.avg_generation_ms = (
                self._stats.avg_generation_ms * (self._stats.total_queries - 1)
                + latencies["sql_generation"]
            ) / self._stats.total_queries

        if "execution_judge" in latencies:
            self._stats.avg_execution_ms = (
                self._stats.avg_execution_ms * (self._stats.total_queries - 1)
                + latencies["execution_judge"]
            ) / self._stats.total_queries

    def get_vector_db_stats(self) -> Dict[str, Any]:
        """Получить статистику Vector DB."""
        if not self._pipeline:
            return {}
        return self._pipeline.vector_db.get_stats()

    def get_graph_db_stats(self) -> Dict[str, Any]:
        """Получить статистику Graph DB."""
        if not self._pipeline:
            return {}

        try:
            return self._pipeline.graph_db.get_stats()
        except Exception as e:
            logger.warning(f"Graph DB stats not available: {e}")
            return {}

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Получить общую статистику."""
        return {
            "initialized": self._initialized,
            "databases": len(self.db_paths),
            "database_paths": self.db_paths,
            "queries": self._stats.to_dict(),
            "vector_db": self.get_vector_db_stats(),
            "graph_db": self.get_graph_db_stats(),
        }

    def recreate_vector_db_collection(self) -> Dict[str, Any]:
        """Пересоздать коллекцию Vector DB."""
        if not self._pipeline:
            raise RuntimeError("Pipeline not initialized")

        self._pipeline.vector_db.recreate_collection()
        return self.index_databases(force_reindex=True)

    def clear_router_cache(self) -> None:
        """Очистить кэш Router Agent."""
        if not self._pipeline:
            return
        self._pipeline.router.invalidate_cache()
        logger.info("Router cache cleared")

    def reset(self) -> None:
        """Сбросить сервис."""
        if self._pipeline:
            self._pipeline.close()
            self._pipeline = None

        self._initialized = False
        self._stats = PipelineStats()

        logger.info("PipelineService reset")

    def close(self) -> None:
        """Закрыть сервис."""
        self.reset()
        logger.info("PipelineService closed")

    def __enter__(self) -> "PipelineService":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class DatabaseDiscoveryService:
    """Сервис для обнаружения баз данных."""

    @staticmethod
    def discover(data_dir: str, extensions: Optional[List[str]] = None) -> List[str]:
        """
        Обнаружить базы данных в директории.

        Args:
            data_dir: Директория для поиска.
            extensions: Расширения файлов.

        Returns:
            Список путей к БД.
        """
        extensions = extensions or [".sqlite", ".db", ".sqlite3"]
        db_paths: List[str] = []

        data_path = Path(data_dir)
        if not data_path.exists():
            return db_paths

        for ext in extensions:
            for db_file in data_path.rglob(f"*{ext}"):
                if not db_file.name.startswith("sqlite_"):
                    db_paths.append(str(db_file.resolve()))

        logger.info(f"Discovered {len(db_paths)} databases in {data_dir}")
        return db_paths

    @staticmethod
    def get_db_info(db_path: str) -> Dict[str, Any]:
        """
        Получить информацию о БД.

        Args:
            db_path: Путь к БД.

        Returns:
            Информация о БД.
        """
        try:
            loader = SchemaLoader(db_path)
            tables = loader.get_tables()
            stats = loader.get_stats()
            loader.close()

            return {
                "path": db_path,
                "name": Path(db_path).stem,
                "tables_count": len(tables),
                "tables": tables,
                "load_time_ms": stats.get("load_time_ms", 0),
            }

        except Exception as e:
            logger.error(f"Failed to get DB info for {db_path}: {e}")
            return {
                "path": db_path,
                "name": Path(db_path).stem,
                "error": str(e),
            }
