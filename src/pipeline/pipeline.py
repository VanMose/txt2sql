# src\pipeline\pipeline.py
"""
Unified Production Text-to-SQL Pipeline.

Architecture:
    User Query → Query Understanding → Router (Vector + Graph) →
    Schema Retrieval → Schema Compression → SQL Generation →
    Parallel Execution + Judge → Early Exit → Refiner → Final SQL

Features:
- Multi-DB support with Router Agent
- Query Understanding for intent detection
- Hybrid retrieval (Vector + Graph) with Russian synonyms
- Schema compression for token efficiency
- Parallel SQL execution with Early Exit
- Metrics tracking
- Comprehensive logging

Example:
    >>> from pipeline.pipeline import Text2SQLPipeline
    >>> with Text2SQLPipeline(db_paths=["db1.sqlite", "db2.sqlite"]) as pipeline:
    ...     pipeline.index_databases()
    ...     result = pipeline.run("Show movies and albums")
"""
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph

from ..agents.query_understanding import QueryUnderstandingAgent, QueryUnderstanding
from ..agents.router_agent import RouterAgent
from ..agents.sql_generator import SQLGenerator
from ..agents.sql_judge import SQLJudge
from ..agents.sql_refiner import SQLRefiner
from ..agents.sql_validator import SQLValidator
from ..config.settings import get_settings
from ..db.multi_db_executor import MultiDBExecutor
from ..db.schema_loader import SchemaLoader
from ..retrieval.graph_db import Neo4jGraphDB
from ..retrieval.vector_db import QdrantVectorDB, TableDocument
from ..retrieval.schema_compressor import SchemaCompressor, CompactTableInfo
from ..services.metrics import metrics, QueryMetrics, QueryStatus
from .state import SQLAttempt

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Результат выполнения пайплайна."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    sql: str = ""
    confidence: float = 0.0
    execution_result: Any = None
    selected_databases: List[Dict[str, Any]] = field(default_factory=list)
    relevant_tables: List[str] = field(default_factory=list)
    latencies: Dict[str, float] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None
    refinement_count: int = 0
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в dict."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "sql": self.sql,
            "confidence": self.confidence,
            "execution_result": self.execution_result,
            "selected_databases": self.selected_databases,
            "relevant_tables": self.relevant_tables,
            "latencies": self.latencies,
            "success": self.success,
            "error": self.error,
            "refinement_count": self.refinement_count,
            "cache_hit": self.cache_hit,
        }


class PipelineState(TypedDict):
    """State для LangGraph пайплайна."""
    query_id: str
    query: str
    understanding: Optional[QueryUnderstanding]
    selected_databases: List[Dict[str, Any]]
    schema: Optional[str]
    compressed_schema: Optional[str]
    relevant_tables: List[str]
    sql_candidates: List[str]
    best_sql: Optional[str]
    confidence: float
    execution_result: Optional[Any]
    needs_refinement: bool
    refinement_count: int
    latencies: Dict[str, float]
    errors: List[str]


class Text2SQLPipeline:
    """
    Unified Production Text-to-SQL Pipeline.

    Combines best features:
    - Multi-DB support with Router Agent (from langgraph_pipeline)
    - Query Understanding layer (from production_pipeline)
    - Schema compression for token efficiency
    - Parallel execution with Early Exit
    - Russian synonyms for better search
    - Metrics tracking
    """

    # Russian synonyms for table names
    TABLE_SYNONYMS = {
        'movie': 'фильм кино movies films',
        'film': 'фильм кино',
        'rating': 'рейтинг оценка rating stars',
        'reviewer': 'рецензент отзыв reviewer',
        'director': 'режиссер директор director',
        'actor': 'актер актёр actor',
        'song': 'песня трек song music',
        'music': 'музыка music',
        'artist': 'исполнитель артист artist',
        'album': 'альбом album',
        'files': 'файлы файлы files',
        'volume': 'том выпуск volume',
        'festival': 'фестиваль праздник festival',
    }

    def __init__(
        self,
        db_paths: Optional[List[str]] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        use_local_qdrant: bool = True,
        qdrant_local_path: Optional[str] = None,
        use_query_understanding: bool = True,
        use_schema_compression: bool = True,
        use_parallel_execution: bool = True,
    ) -> None:
        """
        Инициализировать пайплайн.

        Args:
            db_paths: Пути к базам данных.
            qdrant_url: URL Qdrant сервера.
            qdrant_api_key: API ключ Qdrant.
            neo4j_uri: URI Neo4j сервера.
            neo4j_username: Имя пользователя Neo4j.
            neo4j_password: Пароль Neo4j.
            use_local_qdrant: Использовать локальный Qdrant.
            qdrant_local_path: Путь к локальному Qdrant.
            use_query_understanding: Использовать Query Understanding.
            use_schema_compression: Использовать сжатие схемы.
            use_parallel_execution: Использовать параллельное выполнение.
        """
        settings = get_settings()
        self.db_paths = db_paths or settings.get("db_paths", [])
        self.use_query_understanding = use_query_understanding
        self.use_schema_compression = use_schema_compression
        self.use_parallel_execution = use_parallel_execution

        logger.info(f"Initializing Text2SQLPipeline with {len(self.db_paths)} databases")

        # Initialize Vector DB
        self.vector_db = QdrantVectorDB(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            use_local=use_local_qdrant,
            local_path=qdrant_local_path,
        )

        # Initialize Graph DB (with error handling)
        try:
            self.graph_db = Neo4jGraphDB(
                uri=neo4j_uri,
                username=neo4j_username,
                password=neo4j_password,
            )
            logger.info("Neo4j Graph DB initialized")
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}. Graph features disabled.")
            self.graph_db = None  # type: ignore

        # Initialize Router Agent
        self.router = RouterAgent(
            vector_db=self.vector_db,
            graph_db=self.graph_db if self.graph_db else self.vector_db,  # type: ignore
            use_parallel=use_parallel_execution,
            use_semantic_cache=settings.use_semantic_cache,
        )

        # Initialize Query Understanding Agent
        self.understanding_agent = QueryUnderstandingAgent() if use_query_understanding else None

        # Initialize other agents
        self.generator = SQLGenerator()
        self.validator = SQLValidator()
        self.judge = SQLJudge()
        self.refiner = SQLRefiner()

        # Initialize Schema Compressor
        self.compressor = SchemaCompressor(
            compression_level=2,
            include_join_hints=True,
            include_pk=True,
            include_types=True,
        ) if use_schema_compression else None

        # Parallel executor
        self._executor = ThreadPoolExecutor(max_workers=4) if use_parallel_execution else None

        logger.info(
            f"Text2SQLPipeline initialized: "
            f"parallel={use_parallel_execution}, "
            f"query_understanding={use_query_understanding}, "
            f"schema_compression={use_schema_compression}"
        )

    def _create_search_text(self, table: Any, db_name: str) -> str:
        """
        Создать текст для поиска с русскими синонимами.

        Проблема: embedding модель обучена на английском,
        но запросы могут быть на русском.

        Решение: добавляем русские синонимы для названий таблиц.
        """
        table_name = table.name

        # Формируем базовый текст
        col_parts = []
        for col in table.column_names:
            col_type = table.column_types.get(col, "")
            col_parts.append(f"{col} ({col_type})")

        fk_text = ""
        if table.foreign_keys:
            fk_parts = [f"{fk.from_column}→{fk.table}.{fk.to_column}" for fk in table.foreign_keys]
            fk_text = f" FK: {', '.join(fk_parts)}"

        base_text = f"Table: {table_name}, Database: {db_name}, Columns: {', '.join(col_parts)}{fk_text}"

        # Добавляем русские синонимы
        table_lower = table_name.lower()
        synonyms = []
        for eng, rus in self.TABLE_SYNONYMS.items():
            if eng in table_lower or table_lower in eng:
                synonyms.append(rus)

        if synonyms:
            base_text += f" Keywords: {' '.join(synonyms)}"

        return base_text

    def index_databases(self, force_reindex: bool = False) -> Dict[str, Any]:
        """Индексировать все базы данных в Vector и Graph DB.
        
        🔥 FIX: При force_reindex=True очищаем ОБА хранилища (Vector + Graph)
        """
        logger.info(f"Indexing {len(self.db_paths)} databases...")
        logger.info(f"Databases to index: {[Path(p).parent.name for p in self.db_paths]}")

        start_time = time.time()

        if force_reindex:
            # 🔥 FIX: Очищаем ОБА хранилища
            logger.info("Force reindex: Recreating Vector DB collection...")
            self.vector_db.recreate_collection()
            
            if self.graph_db:
                logger.info("Force reindex: Clearing Graph DB (Neo4j)...")
                self.graph_db.delete_all()
                logger.info("Force reindex: Graph DB cleared")

        for db_path in self.db_paths:
            db_name = Path(db_path).parent.name

            try:
                loader = SchemaLoader(db_path)
                tables = loader.load_full_schema(include_row_count=True)
                logger.info(f"Processing {db_name}: {len(tables)} tables")

                total_rows = sum(t.row_count or 0 for t in tables)
                logger.info(f"  Total rows in {db_name}: {total_rows}")

                # Vector DB - добавляем русские синонимы
                table_docs = [
                    TableDocument(
                        id=None,
                        db_path=db_path,
                        db_name=db_name,
                        table_name=table.name,
                        text=self._create_search_text(table, db_name),
                        columns=table.column_names,
                        column_types=table.column_types,
                        foreign_keys=[fk.to_dict() for fk in table.foreign_keys],
                        primary_key=table.primary_key,
                        row_count=table.row_count,
                    )
                    for table in tables
                ]
                self.vector_db.add_tables_batch(table_docs)

                # Graph DB
                graph_tables = [
                    {
                        "name": table.name,
                        "columns": table.column_names,
                        "column_types": table.column_types,
                        "primary_key": table.primary_key,
                        "row_count": table.row_count,
                        "foreign_keys": [
                            {"from_table": table.name, "from_column": fk.from_column, "to_table": fk.table, "to_column": fk.to_column}
                            for fk in table.foreign_keys
                        ],
                    }
                    for table in tables
                ]
                if self.graph_db:
                    self.graph_db.add_schema_batch(db_name, graph_tables)
                loader.close()

            except Exception as e:
                logger.error(f"Failed to index {db_name}: {e}")
                raise

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Indexing complete in {elapsed_ms:.0f}ms: {len(self.db_paths)} databases")

        vector_stats = self.vector_db.get_stats()
        graph_stats = self.graph_db.get_stats() if self.graph_db else None

        return {
            "elapsed_ms": round(elapsed_ms, 2),
            "vector_db": vector_stats,
            "graph_db": graph_stats,
            "databases_indexed": len(self.db_paths),
        }

    def _measure_time(self, func: Any, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
        """Измерить время выполнения функции."""
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.time() - start) * 1000
            return result, elapsed_ms
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error(f"Error in {func.__name__} after {elapsed_ms:.0f}ms: {e}")
            raise

    def run(self, query: str) -> PipelineResult:
        """Запустить пайплайн."""
        logger.info(f"Running Text2SQLPipeline for query: {query[:100]}...")

        start_time = time.time()
        query_id = str(uuid.uuid4())[:8]

        graph = self._build_graph()
        initial_state: PipelineState = {
            "query_id": query_id,
            "query": query,
            "understanding": None,
            "selected_databases": [],
            "schema": None,
            "compressed_schema": None,
            "relevant_tables": [],
            "sql_candidates": [],
            "best_sql": None,
            "confidence": 0.0,
            "execution_result": None,
            "needs_refinement": False,
            "refinement_count": 0,
            "latencies": {},
            "errors": [],
        }

        try:
            result = graph.invoke(initial_state)
            total_latency = (time.time() - start_time) * 1000

            # Record metrics
            self._record_metrics(result, total_latency)

            return PipelineResult(
                query_id=query_id,
                query=query,
                sql=result.get("best_sql", ""),
                confidence=result.get("confidence", 0.0),
                execution_result=result.get("execution_result"),
                selected_databases=result.get("selected_databases", []),
                relevant_tables=result.get("relevant_tables", []),
                latencies=result.get("latencies", {}),
                success=result.get("best_sql") is not None,
                error=result.get("errors", [None])[0] if result.get("errors") else None,
                refinement_count=result.get("refinement_count", 0),
            )

        except Exception as e:
            total_latency = (time.time() - start_time) * 1000
            logger.error(f"Pipeline error: {e}", exc_info=True)

            return PipelineResult(
                query_id=query_id,
                query=query,
                sql="",
                confidence=0.0,
                execution_result=None,
                selected_databases=[],
                relevant_tables=[],
                latencies={"total": total_latency},
                success=False,
                error=str(e),
            )

    def _execute_sql_parallel(
        self,
        query: str,
        sql_candidates: List[str],
        db_paths: List[str],
        db_aliases: List[str],
        early_exit_threshold: float,
        schema: Optional[str] = None,
    ) -> List[Tuple]:
        """
        Параллельное выполнение и оценка SQL кандидатов.

        Args:
            query: Запрос пользователя.
            sql_candidates: Список SQL кандидатов.
            db_paths: Пути к базам данным.
            db_aliases: Псевдонимы БД.
            early_exit_threshold: Порог для early exit.
            schema: Схема БД для case-sensitive валидации.

        Returns:
            Список кортежей (sql, confidence, result, error).
        """
        results = []

        def execute_single(sql: str) -> Tuple:
            """Выполнить один SQL и оценить."""
            try:
                with MultiDBExecutor(db_paths) as executor:
                    executor.attach_databases(db_aliases)

                    if not self.validator.validate(sql):
                        return sql, 0.0, None, "Validation failed"

                    ok, result = executor.execute_with_dataframe(sql)

                    if ok:
                        # 🔥 Передаём схему для case-sensitive валидации
                        confidence, reason = self.judge.evaluate(query, sql, schema=schema)
                    else:
                        confidence, reason, _ = self.judge.evaluate_with_error(
                            query, sql, str(result), schema=schema
                        )
                        confidence = 0.0

                    return sql, confidence, result if ok else None, None

            except Exception as e:
                return sql, 0.0, None, str(e)

        # Параллельное выполнение
        futures = {
            self._executor.submit(execute_single, sql): sql
            for sql in sql_candidates
        }

        for future in as_completed(futures):
            sql = futures[future]
            try:
                result = future.result(timeout=10.0)
                results.append(result)

                # Проверка Early Exit
                if result[1] >= early_exit_threshold:
                    logger.debug(f"Early exit candidate found: {result[1]:.2f}")
            except Exception as e:
                results.append((sql, 0.0, None, str(e)))

        # Сортировка по confidence (лучшие первыми)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _compress_schema(self, schema: str, relevant_tables: List[str]) -> str:
        """
        Сжать схему для эффективного использования токенов.
        
        🔥 PRODUCTION IMPLEMENTATION — экономия 50% токенов
        
        Args:
            schema: Полная схема БД.
            relevant_tables: Список релевантных таблиц.
        
        Returns:
            Сжатое представление схемы или оригинальная схема если сжатие не удалось.
        """
        if not self.compressor:
            return schema
        
        try:
            # Parse schema into CompactTableInfo
            tables = self._parse_schema_to_tables(schema)
            
            # Если не удалось распарсить, возвращаем оригинальную схему
            if not tables:
                logger.warning(f"Schema parsing returned 0 tables, using original schema ({len(schema)} chars)")
                return schema
            
            # Filter by relevant tables if specified
            if relevant_tables:
                relevant_table_names = set()
                for rt in relevant_tables:
                    if '.' in rt:
                        _, table_name = rt.split('.', 1)
                        relevant_table_names.add(table_name)
                    else:
                        relevant_table_names.add(rt)
                tables = [t for t in tables if t.name in relevant_table_names]
            
            # Если после фильтрации не осталось таблиц, возвращаем оригинал
            if not tables:
                logger.warning(f"No tables after filtering, using original schema ({len(schema)} chars)")
                return schema
            
            # Use compressor for LLM-optimized format
            compressed = self.compressor.compress_for_llm(tables)
            
            # Если сжатая схема слишком короткая, используем оригинал
            if len(compressed) < 50:
                logger.warning(f"Compressed schema too short ({len(compressed)} chars), using original ({len(schema)} chars)")
                return schema
            
            # Log compression stats
            stats = self.compressor.get_stats()
            if stats.get('compression_ratio', 0) > 0:
                logger.info(f"Schema compressed: {stats.get('original_tokens', 0)} → {stats.get('compressed_tokens', 0)} tokens ({stats['compression_ratio']:.1f}% saved)")
            
            return compressed
            
        except Exception as e:
            logger.error(f"Schema compression failed: {e}, using original schema")
            return schema
    
    def _parse_schema_to_tables(self, schema: str) -> List[CompactTableInfo]:
        """
        Распарсить схему БД в список CompactTableInfo.
        
        Args:
            schema: Текст схемы в формате:
                -- Database: music_1
                Table: song
                Columns: Id (INTEGER), Title (TEXT), ...
                Foreign Keys: AlbumId → album.AlbumId
        
        Returns:
            Список CompactTableInfo.
        """
        tables = []
        
        # Split by database sections
        db_sections = re.split(r'-- Database:\s*\w+', schema)
        
        for section in db_sections:
            if not section.strip():
                continue
            
            # Extract database name
            db_match = re.search(r'-- Database:\s*(\w+)', schema)
            db_name = db_match.group(1) if db_match else "main"
            
            # Find all tables in section
            table_pattern = r'Table:\s*(\w+)\s*\nColumns:\s*([^\n]+)(?:\nForeign Keys:\s*([^\n]+))?'
            table_matches = re.finditer(table_pattern, section, re.IGNORECASE)
            
            for table_match in table_matches:
                table_name = table_match.group(1)
                columns_str = table_match.group(2)
                fk_str = table_match.group(3) if table_match.lastindex >= 3 else None
                
                # Parse columns
                columns = []
                column_types = {}
                for col in columns_str.split(','):
                    col = col.strip()
                    col_match = re.match(r'(\w+)\s*\((\w+)\)', col)
                    if col_match:
                        col_name = col_match.group(1)
                        col_type = col_match.group(2)
                        columns.append(col_name)
                        column_types[col_name] = col_type
                
                # Parse foreign keys
                foreign_keys = []
                if fk_str:
                    for fk in fk_str.split(';'):
                        fk = fk.strip()
                        fk_match = re.match(r'(\w+)\s*→\s*(\w+)\.(\w+)', fk)
                        if fk_match:
                            foreign_keys.append({
                                "from_column": fk_match.group(1),
                                "to_table": fk_match.group(2),
                                "to_column": fk_match.group(3),
                            })
                
                table_info = CompactTableInfo(
                    name=table_name,
                    db_name=db_name,
                    columns=columns,
                    column_types=column_types,
                    primary_key=columns[0] if columns else None,  # First column is usually PK
                    foreign_keys=foreign_keys,
                )
                tables.append(table_info)
        
        return tables

    def _build_graph(self) -> StateGraph:
        """Построить LangGraph граф."""
        settings = get_settings()

        def understand_query(state: PipelineState) -> PipelineState:
            """Query Understanding: анализ намерений пользователя."""
            if not self.understanding_agent:
                return state

            logger.info("Step 0: Understanding query...")
            start = time.time()

            understanding = self.understanding_agent.analyze(state["query"])
            state["understanding"] = understanding
            state["latencies"]["understanding"] = (time.time() - start) * 1000

            logger.info(f"Understanding: intent={understanding.intent.value}")
            return state

        def route_databases(state: PipelineState) -> PipelineState:
            """Router Agent: выбор релевантных баз данных."""
            logger.info("Step 1: Routing databases...")

            selections, latency = self._measure_time(
                self.router.route,
                query=state["query"],
                top_k_dbs=settings.get("top_k_dbs", 2),
                top_k_tables=settings.get("top_k_tables", 5),
                use_llm_ranking=settings.get("use_llm_ranking", True),
            )

            state["latencies"]["routing"] = latency
            state["selected_databases"] = [
                {
                    "db_name": s.db_name,
                    "db_path": s.db_path,
                    "tables": s.tables,
                    "relevance_score": s.relevance_score,
                    "confidence": s.confidence,
                    "reason": s.reason,
                }
                for s in selections
            ]

            # Extract relevant tables
            state["relevant_tables"] = [
                f"{db_info['db_name']}.{t}"
                for db_info in state["selected_databases"]
                for t in db_info.get("tables", [])
            ]

            logger.info(f"Selected {len(selections)} databases: {[s.db_name for s in selections]}")
            return state

        def retrieve_schema(state: PipelineState) -> PipelineState:
            """Schema Retrieval с prefetch."""
            logger.info("Step 2: Retrieving schema...")

            if not state["selected_databases"]:
                state["errors"].append("No databases selected by router")
                return state

            logger.info(f"📊 Selected databases:")
            for db in state["selected_databases"]:
                logger.info(f"   - {db['db_name']}: tables={db.get('tables', [])}")

            schema_parts = []

            if self.use_parallel_execution and self._executor:
                # Параллельная загрузка схем
                def load_schema(db_info: Dict[str, Any]) -> Tuple:
                    db_path = db_info["db_path"]
                    db_name = db_info["db_name"]
                    tables_filter = db_info.get("tables", [])
                    try:
                        loader = SchemaLoader(db_path)
                        schema = loader.get_schema_for_tables(
                            tables_filter if tables_filter else None,
                            include_details=True
                        )
                        loader.close()
                        return db_name, f"-- Database: {db_name}\n{schema}", None
                    except Exception as e:
                        return db_name, None, f"Schema load error for {db_name}: {str(e)}"

                futures = {
                    self._executor.submit(load_schema, db_info): db_info
                    for db_info in state["selected_databases"]
                }

                for future in as_completed(futures):
                    db_name, schema, error = future.result(timeout=5.0)
                    if schema:
                        schema_parts.append(schema)
                    if error:
                        state["errors"].append(error)
            else:
                # Последовательная загрузка
                for db_info in state["selected_databases"]:
                    db_path = db_info["db_path"]
                    db_name = db_info["db_name"]
                    tables_filter = db_info.get("tables", [])

                    try:
                        loader = SchemaLoader(db_path)
                        schema = loader.get_schema_for_tables(
                            tables_filter if tables_filter else None,
                            include_details=True
                        )
                        schema_parts.append(f"-- Database: {db_name}\n{schema}")
                        loader.close()
                    except Exception as e:
                        logger.error(f"Failed to load schema for {db_name}: {e}")
                        state["errors"].append(f"Schema load error for {db_name}: {str(e)}")

            state["schema"] = "\n\n".join(schema_parts)

            # Schema compression
            if self.use_schema_compression and self.compressor:
                state["compressed_schema"] = self._compress_schema(
                    state["schema"],
                    state["relevant_tables"]
                )
            else:
                state["compressed_schema"] = state["schema"]

            logger.info(f"Retrieved schema for {len(schema_parts)} databases")
            return state

        def generate_sql(state: PipelineState) -> PipelineState:
            """SQL Generator."""
            logger.info("Step 3: Generating SQL...")

            if not state["schema"]:
                state["errors"].append("No schema available for SQL generation")
                return state

            # Use compressed schema if available
            schema_to_use = state["compressed_schema"] or state["schema"]

            logger.info(f"📝 Schema for SQL generation ({len(schema_to_use)} chars)")

            sql_candidates, latency = self._measure_time(
                self.generator.generate,
                query=state["query"],
                schema=schema_to_use,
                n_samples=settings.get("n_samples", 2),
                temperature=settings.get("temperature", 0.15),
                use_batch=True,
                early_stop=True,
            )

            state["latencies"]["sql_generation"] = latency
            state["sql_candidates"] = sql_candidates

            logger.info(f"Generated {len(sql_candidates)} SQL candidates")
            return state

        def execute_and_judge(state: PipelineState) -> PipelineState:
            """Execution + Judge с оптимизациями."""
            logger.info("Step 4: Executing and judging SQL...")

            if not state["sql_candidates"]:
                state["errors"].append("No SQL candidates to execute")
                return state

            best_sql = None
            best_confidence = 0.0
            best_result = None

            db_paths = [db["db_path"] for db in state["selected_databases"]]
            db_aliases = [db["db_name"] for db in state["selected_databases"]]
            early_exit_threshold = settings.get("early_exit_threshold", 0.6)

            start_exec = time.time()

            if self.use_parallel_execution and self._executor:
                # Параллельное выполнение + Early Exit
                results = self._execute_sql_parallel(
                    query=state["query"],
                    sql_candidates=state["sql_candidates"],
                    db_paths=db_paths,
                    db_aliases=db_aliases,
                    early_exit_threshold=early_exit_threshold,
                    schema=state.get("schema"),  # 🔥 Передаём схему для case-sensitive валидации
                )

                for sql, confidence, result, error in results:
                    if error:
                        confidence = 0.0

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_sql = sql
                        best_result = result if not error else None

                    # Early Exit
                    if confidence >= early_exit_threshold:
                        logger.info(f"Early exit: confidence {confidence:.2f} >= {early_exit_threshold}")
                        break
            else:
                # Последовательное выполнение
                with MultiDBExecutor(db_paths) as executor:
                    executor.attach_databases(db_aliases)

                    for i, sql in enumerate(state["sql_candidates"]):
                        if not self.validator.validate(sql):
                            logger.warning(f"Validation failed for SQL: {sql[:100]}")
                            continue

                        ok, result = executor.execute_with_dataframe(sql)

                        if ok:
                            confidence, reason = self.judge.evaluate(state["query"], sql)
                        else:
                            confidence, reason, _ = self.judge.evaluate_with_error(
                                state["query"], sql, str(result)
                            )
                            confidence = 0.0

                        logger.info(f"Candidate {i + 1}: confidence={confidence:.2f}")

                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_sql = sql
                            best_result = result if ok else None

                        # Early Exit
                        if confidence >= early_exit_threshold:
                            logger.info(f"Early exit: confidence {confidence:.2f} >= {early_exit_threshold}")
                            break

            state["latencies"]["execution_judge"] = (time.time() - start_exec) * 1000
            state["best_sql"] = best_sql
            state["confidence"] = best_confidence
            state["execution_result"] = best_result
            state["needs_refinement"] = best_confidence < early_exit_threshold

            logger.info(
                f"Best SQL: confidence={best_confidence:.2f}, "
                f"needs_refinement={state['needs_refinement']}"
            )
            return state

        def refine_sql(state: PipelineState) -> PipelineState:
            """SQL Refiner (только если confidence < threshold)."""
            logger.info("Step 5: Refining SQL...")

            if state["refinement_count"] >= settings.get("max_retries", 1):
                logger.info("Max refinement count reached")
                return state

            attempts = [
                SQLAttempt(
                    sql=state["best_sql"] or "",
                    confidence=state["confidence"],
                    error=state["needs_refinement"],
                    reason="Low confidence" if state["needs_refinement"] else "",
                )
            ]

            refined_sql, latency = self._measure_time(
                self.refiner.refine,
                state["query"],
                state["schema"],
                attempts
            )

            state["latencies"]["refinement"] = latency

            db_paths = [db["db_path"] for db in state["selected_databases"]]
            with MultiDBExecutor(db_paths) as executor:
                executor.attach_databases([db["db_name"] for db in state["selected_databases"]])
                ok, result = executor.execute_with_dataframe(refined_sql)

                if ok:
                    conf, _ = self.judge.evaluate(state["query"], refined_sql)
                    if conf > state["confidence"]:
                        logger.info(f"Refinement improved confidence: {state['confidence']:.2f} -> {conf:.2f}")
                        state["best_sql"] = refined_sql
                        state["confidence"] = conf
                        state["execution_result"] = result
                        state["needs_refinement"] = False

            state["refinement_count"] += 1
            return state

        def should_refine(state: PipelineState) -> str:
            """Проверка необходимости рефайнмента с Early Exit."""
            if state["confidence"] >= settings.get("early_exit_threshold", 0.6):
                logger.info(f"Early exit: confidence {state['confidence']:.2f} >= threshold")
                return "end"

            if state["needs_refinement"] and state["refinement_count"] < settings.get("max_retries", 1):
                return "refine"
            return "end"

        # Построение графа
        graph = StateGraph(PipelineState)

        # Add nodes
        if self.use_query_understanding:
            graph.add_node("understand", understand_query)
        graph.add_node("route", route_databases)
        graph.add_node("retrieve", retrieve_schema)
        graph.add_node("generate", generate_sql)
        graph.add_node("execute", execute_and_judge)
        graph.add_node("refine", refine_sql)

        # Set entry point
        if self.use_query_understanding:
            graph.set_entry_point("understand")
            graph.add_edge("understand", "route")
        else:
            graph.set_entry_point("route")

        # Add edges
        graph.add_edge("route", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "execute")

        # Conditional edge for refinement
        graph.add_conditional_edges(
            "execute",
            should_refine,
            {"refine": "refine", "end": END},
        )

        # After refinement, go back to execution
        graph.add_edge("refine", "execute")

        return graph.compile()

    def _record_metrics(self, result: PipelineState, total_latency: float) -> None:
        """Записать метрики."""
        query_metrics = QueryMetrics(
            query_id=result.get("query_id", "unknown"),
            query_text=result.get("query", ""),
            status=QueryStatus.SUCCESS if result.get("best_sql") else QueryStatus.FAILED,
            latency_ms=total_latency,
            sql_generated=result.get("best_sql"),
            sql_valid=result.get("confidence", 0.0) > 0,
            confidence=result.get("confidence", 0.0),
            tables_retrieved=len(result.get("selected_databases", [])),
            cache_hit=False,
            error_message=result.get("errors", [None])[0] if result.get("errors") else None,
            latencies_breakdown=result.get("latencies", {}),
        )
        metrics.record_query(query_metrics)

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику пайплайна."""
        return {
            "vector_db": self.vector_db.get_stats(),
            "graph_db": self.graph_db.get_stats() if self.graph_db else None,
            "databases": len(self.db_paths),
        }

    def close(self) -> None:
        """Закрыть соединения."""
        if self._executor:
            self._executor.shutdown(wait=False)
        self.vector_db.close()
        if self.graph_db:
            self.graph_db.close()
        logger.info("Text2SQLPipeline closed")

    def __enter__(self) -> "Text2SQLPipeline":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
