# src\pipeline\langgraph_pipeline.py
"""
LangGraph многоагентный пайплайн с оптимизациями.

Оптимизации:
1. Early Exit - пропуск рефайнмента при confidence > threshold
2. Parallel Execution - параллельное выполнение SQL кандидатов
3. Prefetch схем - асинхронная загрузка схем
4. Batch Generation - генерация за 1 проход GPU

Архитектура:
    User Query → Router Agent → Vector DB + Graph DB → Schema Retrieval →
    SQL Generator → Validator → Parallel Executor → Judge → Early Exit → Final SQL

Example:
    >>> from pipeline.langgraph_pipeline import MultiDBPipeline
    >>> with MultiDBPipeline(db_paths=["db1.sqlite", "db2.sqlite"]) as pipeline:
    ...     pipeline.index_databases()
    ...     result = pipeline.run("Show movies and albums")
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

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
from ..llm.prompts import Prompts
from ..utils.json_parser import parse_json
from .state import SQLAttempt

logger = logging.getLogger(__name__)


class MultiDBLangGraphState(TypedDict):
    """State для LangGraph пайплайна."""
    query: str
    selected_databases: List[Dict[str, Any]]
    schema: Optional[str]
    relevant_tables: List[str]
    sql_candidates: List[str]
    best_sql: Optional[str]
    attach_statements: List[str]
    confidence: float
    execution_result: Optional[Any]
    needs_refinement: bool
    refinement_count: int
    latencies: Dict[str, float]
    errors: List[str]


class MultiDBPipeline:
    """
    Многоагентный пайплайн с оптимизациями.

    Оптимизации:
    - Early Exit при confidence > threshold
    - Parallel Execution SQL кандидатов
    - Prefetch схем
    - Batch Generation
    """

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
    ) -> None:
        """Инициализировать пайплайн."""
        settings = get_settings()
        self.db_paths = db_paths or settings.get("db_paths", [])

        logger.info(f"Initializing MultiDBPipeline with {len(self.db_paths)} databases")

        self.vector_db = QdrantVectorDB(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            use_local=use_local_qdrant,
            local_path=qdrant_local_path,
        )
        self.graph_db = Neo4jGraphDB(uri=neo4j_uri, username=neo4j_username, password=neo4j_password)
        
        # Router с оптимизациями
        self.router = RouterAgent(
            vector_db=self.vector_db,
            graph_db=self.graph_db,
            use_parallel=settings.use_parallel_execution,
            use_semantic_cache=settings.use_semantic_cache,
        )

        self.embedder = None  # Lazy init
        self.generator = SQLGenerator()
        self.validator = SQLValidator()
        self.judge = SQLJudge()
        self.refiner = SQLRefiner()

        # Parallel executor
        self._use_parallel = settings.use_parallel_execution
        self._executor = ThreadPoolExecutor(max_workers=4) if self._use_parallel else None

        logger.info(
            f"MultiDBPipeline initialized: parallel={self._use_parallel}, "
            f"early_exit_threshold={settings.early_exit_threshold}"
        )

    def _execute_sql_parallel(
        self,
        query: str,
        sql_candidates: List[str],
        db_paths: List[str],
        db_aliases: List[str],
        early_exit_threshold: float,
    ) -> List[tuple]:
        """
        Параллельное выполнение и оценка SQL кандидатов.

        Args:
            query: Запрос пользователя.
            sql_candidates: Список SQL кандидатов.
            db_paths: Пути к базам данных.
            db_aliases: Алиасы баз данных.
            early_exit_threshold: Порог для ранней остановки.

        Returns:
            Список кортежей (sql, confidence, result, error).
        """
        results = []

        def execute_single(sql: str) -> tuple:
            """Выполнить один SQL и оценить."""
            try:
                with MultiDBExecutor(db_paths) as executor:
                    executor.attach_databases(db_aliases)

                    if not self.validator.validate(sql):
                        return sql, 0.0, None, "Validation failed"

                    ok, result = executor.execute_with_dataframe(sql)

                    if ok:
                        confidence, reason = self.judge.evaluate(query, sql)
                    else:
                        confidence, reason, _ = self.judge.evaluate_with_error(
                            query, sql, str(result)
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

    def _create_search_text(self, table: Any, db_name: str) -> str:
        """
        Создать текст для поиска с русскими синонимами.
        
        Проблема: embedding модель (all-MiniLM-L6-v2) обучена на английском,
        но запросы могут быть на русском.
        
        Решение: добавляем русские синонимы для названий таблиц.
        """
        # Словарь синонимов: английское название -> русские варианты
        table_synonyms = {
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
        for eng, rus in table_synonyms.items():
            if eng in table_lower or table_lower in eng:
                synonyms.append(rus)
        
        if synonyms:
            base_text += f" Keywords: {' '.join(synonyms)}"
        
        return base_text

    def index_databases(self, force_reindex: bool = False) -> None:
        """Индексировать все базы данных в Vector и Graph DB."""
        logger.info(f"Indexing {len(self.db_paths)} databases...")
        
        # 🔥 Логирование списка баз данных для индексации
        logger.info(f"Databases to index: {[Path(p).parent.name for p in self.db_paths]}")

        if force_reindex:
            logger.info("Force reindex: Recreating Vector DB collection...")
            self.vector_db.recreate_collection()
            logger.info("Force reindex: Clearing Graph DB...")
            self.graph_db.delete_all()

        for db_path in self.db_paths:
            db_name = Path(db_path).parent.name

            try:
                loader = SchemaLoader(db_path)
                tables = loader.load_full_schema(include_row_count=True)
                logger.info(f"Processing {db_name}: {len(tables)} tables")
                
                # 🔥 Логирование количества строк в таблицах
                total_rows = sum(t.row_count or 0 for t in tables)
                logger.info(f"  Total rows in {db_name}: {total_rows}")

                # Vector DB - добавляем русские синонимы для лучшего поиска
                table_docs = [
                    TableDocument(
                        id=None,
                        db_path=db_path,
                        db_name=db_name,
                        table_name=table.name,
                        text=self._create_search_text(table, db_name),  # ✅ Добавляем синонимы
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
                self.graph_db.add_schema_batch(db_name, graph_tables)
                loader.close()

            except Exception as e:
                logger.error(f"Failed to index {db_name}: {e}")
                raise

        logger.info(f"Indexing complete: {len(self.db_paths)} databases")
        
        # 🔥 Финальная статистика
        vector_stats = self.vector_db.get_stats()
        logger.info(f"Vector DB stats: {vector_stats.get('points_count', 0)} points")
        graph_stats = self.graph_db.get_stats()
        logger.info(f"Graph DB stats: {graph_stats.get('tables', 0)} tables")

    def _measure_time(self, func: Any, *args: Any, **kwargs: Any) -> tuple:
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

    def run(self, query: str) -> Dict[str, Any]:
        """Запустить пайплайн."""
        logger.info(f"Running MultiDBPipeline for query: {query[:100]}...")

        graph = self._build_graph()
        initial_state: MultiDBLangGraphState = {
            "query": query,
            "selected_databases": [],
            "schema": None,
            "relevant_tables": [],
            "sql_candidates": [],
            "best_sql": None,
            "attach_statements": [],
            "confidence": 0.0,
            "execution_result": None,
            "needs_refinement": False,
            "refinement_count": 0,
            "latencies": {},
            "errors": [],
        }

        result = graph.invoke(initial_state)
        logger.info(
            f"Pipeline completed: best_sql={result['best_sql'] is not None}, "
            f"confidence={result['confidence']:.2f}"
        )
        return result

    def _build_graph(self) -> StateGraph:
        """Построить LangGraph граф."""
        settings = get_settings()

        def route_databases(state: MultiDBLangGraphState) -> MultiDBLangGraphState:
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
            logger.info(f"Selected {len(selections)} databases: {[s.db_name for s in selections]}")
            return state

        def retrieve_schema(state: MultiDBLangGraphState) -> MultiDBLangGraphState:
            """Schema Retrieval с prefetch."""
            logger.info("Step 2: Retrieving schema...")
            if not state["selected_databases"]:
                state["errors"].append("No databases selected by router")
                return state

            # 🔥 Логирование выбранных таблиц
            logger.info(f"📊 Selected databases:")
            for db in state["selected_databases"]:
                logger.info(f"   - {db['db_name']}: tables={db.get('tables', [])}")

            schema_parts = []
            
            if self._use_parallel:
                # Параллельная загрузка схем
                def load_schema(db_info: Dict[str, Any]) -> tuple:
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
            state["relevant_tables"] = [
                f"{db_name}.{t}"
                for db_info in state["selected_databases"]
                for t in db_info.get("tables", [])
            ]
            logger.info(f"Retrieved schema for {len(schema_parts)} databases")
            return state

        def generate_sql(state: MultiDBLangGraphState) -> MultiDBLangGraphState:
            """SQL Generator с batch generation."""
            logger.info("Step 3: Generating SQL...")
            if not state["schema"]:
                state["errors"].append("No schema available for SQL generation")
                return state

            # 🔥 Логирование схемы для отладки
            logger.info(f"📝 Schema for SQL generation ({len(state['schema'])} chars):")
            logger.info(f"   {state['schema'][:1000]}...")

            # Используем SQLGenerator с batch generation и early stopping
            sql_candidates, latency = self._measure_time(
                self.generator.generate,
                query=state["query"],
                schema=state["schema"],
                n_samples=settings.get("n_samples", 2),
                temperature=settings.get("temperature", 0.15),
                use_batch=True,
                early_stop=True,
            )
            state["latencies"]["sql_generation"] = latency

            state["sql_candidates"] = sql_candidates
            logger.info(f"Generated {len(sql_candidates)} SQL candidates")
            return state

        def execute_and_judge(state: MultiDBLangGraphState) -> MultiDBLangGraphState:
            """
            Execution + Judge с оптимизациями.
            
            Оптимизации:
            1. Parallel Execution - параллельное выполнение SQL кандидатов
            2. Early Exit - остановка после нахождения хорошего SQL
            """
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

            if self._use_parallel:
                # Параллельное выполнение + Early Exit
                results = self._execute_sql_parallel(
                    query=state["query"],
                    sql_candidates=state["sql_candidates"],
                    db_paths=db_paths,
                    db_aliases=db_aliases,
                    early_exit_threshold=early_exit_threshold,
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

                        logger.info(f"Candidate {i + 1}: confidence={confidence:.2f}, error={not ok}")

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

            # Early Exit: пропуск рефайнмента при высоком confidence
            state["needs_refinement"] = best_confidence < early_exit_threshold

            logger.info(
                f"Best SQL: confidence={best_confidence:.2f}, "
                f"needs_refinement={state['needs_refinement']}"
            )
            return state

        def refine_sql(state: MultiDBLangGraphState) -> MultiDBLangGraphState:
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
                self.refiner.refine, state["query"], state["schema"], attempts
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

        def should_refine(state: MultiDBLangGraphState) -> str:
            """Проверка необходимости рефайнмента с Early Exit."""
            # Early Exit: если confidence >= threshold, пропускаем рефайнмент
            if state["confidence"] >= settings.get("early_exit_threshold", 0.6):
                logger.info(f"Early exit: confidence {state['confidence']:.2f} >= threshold, skipping refinement")
                return "end"
            
            # Рефайнмент только если нужно и не превышен лимит
            if state["needs_refinement"] and state["refinement_count"] < settings.get("max_retries", 1):
                return "refine"
            return "end"

        # Построение графа
        graph = StateGraph(MultiDBLangGraphState)
        graph.add_node("route_databases", route_databases)
        graph.add_node("retrieve_schema", retrieve_schema)
        graph.add_node("generate_sql", generate_sql)
        graph.add_node("execute_and_judge", execute_and_judge)
        graph.add_node("refine_sql", refine_sql)

        graph.set_entry_point("route_databases")
        graph.add_edge("route_databases", "retrieve_schema")
        graph.add_edge("retrieve_schema", "generate_sql")
        graph.add_edge("generate_sql", "execute_and_judge")
        graph.add_conditional_edges(
            "execute_and_judge",
            should_refine,
            {"refine": "refine_sql", "end": END},
        )
        graph.add_edge("refine_sql", END)

        return graph.compile()

    def close(self) -> None:
        """Закрыть соединения."""
        if hasattr(self, "vector_db"):
            self.vector_db.close()
        if hasattr(self, "graph_db"):
            self.graph_db.close()
        if self._executor:
            self._executor.shutdown(wait=False)
        if hasattr(self, "router"):
            self.router.close()
        logger.info("MultiDBPipeline connections closed")

    def __enter__(self) -> "MultiDBPipeline":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
