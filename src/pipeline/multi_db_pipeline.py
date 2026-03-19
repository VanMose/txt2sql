# src\pipeline\multi_db_pipeline.py
"""
Multi-DB LangGraph пайплайн с Router Agent, Vector DB и Graph DB."""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from ..agents.router_agent import DatabaseSelection, RouterAgent
from ..agents.sql_generator import SQLGenerator
from ..agents.sql_judge import SQLJudge
from ..agents.sql_refiner import SQLRefiner
from ..agents.sql_validator import SQLValidator
from ..config.settings import get_settings
from ..db.multi_db_executor import MultiDBExecutor
from ..db.schema_loader import SchemaLoader
from ..llm.inference import LLMService
from ..llm.prompts import Prompts
from ..retrieval.embedder import SchemaEmbedder
from ..retrieval.graph_db import Neo4jGraphDB
from ..retrieval.vector_db import QdrantVectorDB, TableDocument

logger = logging.getLogger(__name__)


@dataclass
class MultiDBSQLAttempt:
    """Попытка генерации SQL для Multi-DB."""
    sql: str
    attach_statements: List[str]
    confidence: float
    error: bool
    reason: str
    db_selections: List[DatabaseSelection] = field(default_factory=list)
    execution_result: Optional[Any] = None


class MultiDBPipelineState(TypedDict):
    """State для Multi-DB LangGraph пайплайна."""

    # Входные данные
    query: str

    # Выбор баз данных
    db_selections: List[Dict[str, Any]]
    selected_db_paths: List[str]

    # Схема
    schema: Optional[str]
    multi_db_schema: Optional[str]

    # SQL генерация
    sql_candidates: List[Dict[str, Any]]
    best_sql: Optional[str]
    best_attach_statements: List[str]
    confidence: float
    execution_result: Optional[Any]

    # Рефайнмент
    needs_refinement: bool
    refinement_count: int
    refinement_history: str

    # Метрики
    latencies: Dict[str, float]
    success: bool


class MultiDBPipeline:
    """
    Multi-DB пайплайн с использованием LangGraph.

    Pipeline:
        User Query
            ↓
        Router Agent (Vector DB + Graph DB)
            ↓
        Schema Loading (Multi-DB ATTACH)
            ↓
        SQL Generator (Multi-DB aware)
            ↓
        Validator + Executor + Judge
            ↓
        Best Selector
            ↓
        Refiner (если нужно)
            ↓
        Final SQL + Results
    """

    def __init__(
        self,
        db_paths: Optional[List[str]] = None,
        qdrant_use_local: bool = True,
        qdrant_local_path: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        """
        Инициализировать Multi-DB пайплайн.

        Args:
            db_paths: Список путей к базам данных.
            qdrant_use_local: Использовать локальный Qdrant.
            qdrant_local_path: Путь к локальному Qdrant.
            neo4j_uri: URI Neo4j сервера.
            neo4j_username: Имя пользователя Neo4j.
            neo4j_password: Пароль Neo4j.
        """
        settings = get_settings()

        # Пути к базам данных
        if db_paths is None:
            # Поиск всех БД в папке data/
            db_paths = self._discover_databases()

        # Конвертация путей в абсолютные
        self.db_paths = [str(Path(p).resolve()) for p in db_paths]
        logger.info(f"Multi-DB Pipeline initialized with {len(db_paths)} databases")

        # Vector DB
        self.vector_db = QdrantVectorDB(
            use_local=qdrant_use_local,
            local_path=qdrant_local_path or settings.get("qdrant_local_path", "qdrant_storage"),
        )

        # Graph DB
        self.graph_db = Neo4jGraphDB(
            uri=neo4j_uri or settings.get("neo4j_uri", "bolt://localhost:7687"),
            username=neo4j_username or settings.get("neo4j_username", "neo4j"),
            password=neo4j_password or settings.get("neo4j_password", "password"),
        )

        # Router Agent
        self.router = RouterAgent(
            vector_db=self.vector_db,
            graph_db=self.graph_db,
        )

        # LLM сервис
        self.llm = LLMService()

        # Другие агенты
        self.generator = SQLGenerator()
        self.validator = SQLValidator()
        self.judge = SQLJudge()
        self.refiner = SQLRefiner()

        # Executor (создаётся динамически)
        self._executor: Optional[MultiDBExecutor] = None

        # Кэширование
        self._schema_cache: Dict[str, str] = {}

        logger.info("Multi-DB Pipeline components initialized")

    def _discover_databases(self) -> List[str]:
        """Найти все SQLite базы данных в папке data/."""
        data_dir = Path("data")
        db_paths = []

        if data_dir.exists():
            for pattern in ["**/*.sqlite", "**/*.db", "**/*.sqlite3"]:
                for db_file in data_dir.glob(pattern):
                    db_paths.append(str(db_file.resolve()))

        logger.info(f"Discovered {len(db_paths)} database files")
        return db_paths

    def index_databases(self, force: bool = False):
        """
        Индексировать все базы данных в Vector DB и Graph DB.

        Args:
            force: Принудительная переиндексация.
        """
        logger.info(f"Starting indexing of {len(self.db_paths)} databases")

        for db_path in self.db_paths:
            try:
                db_name = Path(db_path).stem
                logger.info(f"Indexing database: {db_name} ({db_path})")

                # Загрузка схемы
                loader = SchemaLoader(db_path)
                tables = loader.load_full_schema(include_row_count=True)

                if not tables:
                    logger.warning(f"No tables found in {db_path}")
                    continue

                # Добавление в Vector DB
                table_docs = []
                for table in tables:
                    doc = TableDocument(
                        id=None,  # Будет сгенерирован автоматически через MD5 хеш
                        db_path=db_path,
                        db_name=db_name,
                        table_name=table.name,
                        text=table.to_schema_doc(),
                        columns=table.column_names,
                        column_types=table.column_types,
                        foreign_keys=[fk.to_dict() for fk in table.foreign_keys],
                        primary_key=table.primary_key,
                        row_count=table.row_count,
                    )
                    table_docs.append(doc)

                # Batch добавление в Vector DB
                self.vector_db.add_tables_batch(table_docs)

                # Добавление в Graph DB
                graph_tables = []
                for table in tables:
                    graph_tables.append({
                        "name": table.name,
                        "columns": table.column_names,
                        "column_types": table.column_types,
                        "primary_key": table.primary_key,
                        "row_count": table.row_count,
                        "foreign_keys": [
                            {
                                "from_table": table.name,
                                "from_column": fk.from_column,
                                "to_table": fk.table,
                                "to_column": fk.to_column,
                            }
                            for fk in table.foreign_keys
                        ],
                    })

                # Batch добавление в Graph DB
                self.graph_db.add_schema_batch(db_name, graph_tables)

                loader.close()

            except Exception as e:
                logger.error(f"Error indexing {db_path}: {e}")
                continue

        logger.info("Database indexing completed")

    def run(self, query: str) -> Dict[str, Any]:
        """
        Запустить пайплайн.

        Args:
            query: Запрос пользователя.

        Returns:
            Результаты пайплайна.
        """
        logger.info(f"Starting Multi-DB pipeline for query: {query}")
        start_time = time.time()

        # Построение графа
        graph = self._build_graph()

        # Начальное состояние
        initial_state: MultiDBPipelineState = {
            "query": query,
            "db_selections": [],
            "selected_db_paths": [],
            "schema": None,
            "multi_db_schema": None,
            "sql_candidates": [],
            "best_sql": None,
            "best_attach_statements": [],
            "confidence": 0.0,
            "execution_result": None,
            "needs_refinement": False,
            "refinement_count": 0,
            "refinement_history": "",
            "latencies": {},
            "success": False,
        }

        # Выполнение графа
        result = graph.invoke(initial_state)

        # Общее время
        total_time = time.time() - start_time
        result["latencies"]["total"] = total_time

        logger.info(
            f"Pipeline completed in {total_time * 1000:.0f}ms. "
            f"Success: {result['success']}, SQL: {result['best_sql'][:100] if result['best_sql'] else 'None'}"
        )

        return result

    def _build_graph(self) -> StateGraph:
        """Построить LangGraph граф."""
        settings = get_settings()

        def route_databases(state: MultiDBPipelineState) -> MultiDBPipelineState:
            """Узел маршрутизации к базам данных."""
            logger.info("Step 1: Routing to databases")

            selections = self.router.route(
                query=state["query"],
                top_k_dbs=settings.get("top_k_dbs", 3),
                top_k_tables=settings.get("top_k_tables", 10),
                use_llm_ranking=settings.get("use_llm_ranking", True),
            )

            # Сохранение выборов
            state["db_selections"] = [
                {
                    "db_name": s.db_name,
                    "db_path": s.db_path,
                    "tables": s.tables,
                    "relevance_score": s.relevance_score,
                    "reason": s.reason,
                }
                for s in selections
            ]
            state["selected_db_paths"] = [s.db_path for s in selections]

            logger.info(f"Selected {len(selections)} databases: {[s.db_name for s in selections]}")
            return state

        def load_schema(state: MultiDBPipelineState) -> MultiDBPipelineState:
            """Узел загрузки схемы."""
            logger.info("Step 2: Loading schema for selected databases")

            if not state["selected_db_paths"]:
                logger.warning("No databases selected")
                state["schema"] = ""
                state["multi_db_schema"] = ""
                return state

            schema_parts = []
            multi_db_parts = []

            for selection in state["db_selections"]:
                db_path = selection["db_path"]
                db_name = selection.get("db_name", Path(db_path).stem)
                tables = selection.get("tables", [])

                try:
                    loader = SchemaLoader(db_path)
                    schema = loader.get_schema_for_tables(tables) if tables else loader.get_full_schema()

                    schema_parts.append(f"-- Database: {db_name}\n{schema}")

                    # Для multi-DB генерации
                    multi_db_parts.append(f"Database: {db_name} (path: {db_path})\n{schema}")

                    loader.close()
                except Exception as e:
                    logger.error(f"Error loading schema for {db_path}: {e}")
                    continue

            state["schema"] = "\n\n".join(schema_parts)
            state["multi_db_schema"] = "\n\n".join(multi_db_parts)

            logger.info(f"Loaded schema for {len(schema_parts)} databases")
            return state

        def generate_sql(state: MultiDBPipelineState) -> MultiDBPipelineState:
            """Узел генерации SQL."""
            logger.info("Step 3: Generating SQL")

            # Проверка на multi-DB
            is_multi_db = len(state["selected_db_paths"]) > 1

            if is_multi_db:
                # Multi-DB генерация
                prompt = Prompts.format_multi_db_sql_generator(
                    query=state["query"],
                    databases_schema=state["multi_db_schema"],
                )

                outputs = self.llm.generate(prompt, n=3, temperature=0.5)

                candidates = []
                for output in outputs:
                    try:
                        import json
                        obj = json.loads(output)
                        candidates.append({
                            "sql": obj.get("sql", ""),
                            "attach_statements": obj.get("attach_statements", []),
                            "tables_used": obj.get("tables_used", {}),
                        })
                    except Exception:
                        candidates.append({
                            "sql": output,
                            "attach_statements": [],
                            "tables_used": {},
                        })
            else:
                # Single DB генерация
                candidates_raw = self.generator.generate(state["query"], state["schema"])
                candidates = [
                    {"sql": sql, "attach_statements": [], "tables_used": {}}
                    for sql in candidates_raw
                ]

            state["sql_candidates"] = candidates
            logger.info(f"Generated {len(candidates)} SQL candidates")
            return state

        def execute_and_judge(state: MultiDBPipelineState) -> MultiDBPipelineState:
            """Узел выполнения и оценки SQL."""
            logger.info("Step 4: Executing and judging SQL")

            best_sql = None
            best_attach = []
            best_confidence = 0.0
            best_result = None

            # Создание executor для выбранных БД
            if state["selected_db_paths"]:
                self._executor = MultiDBExecutor(state["selected_db_paths"])

            for candidate in state["sql_candidates"]:
                sql = candidate["sql"]
                attach_statements = candidate.get("attach_statements", [])

                # Формирование полного SQL
                full_sql = ""
                if attach_statements:
                    full_sql += ";\n".join(attach_statements) + ";\n"
                full_sql += sql

                # Валидация
                if not self.validator.validate(sql):
                    logger.debug(f"Validation failed for: {sql[:50]}...")
                    continue

                # Выполнение
                if self._executor:
                    ok, result = self._executor.execute(full_sql)
                else:
                    ok, result = False, "No executor"

                # Judge оценка
                if ok:
                    confidence, reason = self.judge.evaluate(state["query"], sql)
                else:
                    confidence, reason = 0.0, str(result)

                logger.debug(f"SQL confidence: {confidence:.2f} - {reason}")

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_sql = full_sql
                    best_attach = attach_statements
                    best_result = result if ok else None

            state["best_sql"] = best_sql
            state["best_attach_statements"] = best_attach
            state["confidence"] = best_confidence
            state["execution_result"] = best_result
            state["needs_refinement"] = best_confidence < settings.get("confidence_threshold", 0.5)

            logger.info(
                f"Best SQL confidence: {best_confidence:.2f}, "
                f"Needs refinement: {state['needs_refinement']}"
            )
            return state

        def refine_sql(state: MultiDBPipelineState) -> MultiDBPipelineState:
            """Узел рефайнмента SQL."""
            logger.info("Step 5: Refining SQL")

            if state["refinement_count"] >= settings.get("max_retries", 3):
                logger.warning("Max refinement attempts reached")
                state["needs_refinement"] = False
                return state

            # Создание истории попыток
            from .state import SQLAttempt

            attempts = [
                SQLAttempt(
                    sql=state["best_sql"] or "",
                    confidence=state["confidence"],
                    error=state["needs_refinement"],
                    reason="Low confidence" if state["needs_refinement"] else "",
                )
            ]

            # Рефайнмент
            refined_sql = self.refiner.refine(
                query=state["query"],
                schema=state["schema"],
                attempts=attempts,
            )

            # Выполнение рефайненного SQL
            if self._executor:
                ok, result = self._executor.execute(refined_sql)
            else:
                ok, result = False, "No executor"

            if ok:
                conf, reason = self.judge.evaluate(state["query"], refined_sql)
                if conf > state["confidence"]:
                    state["best_sql"] = refined_sql
                    state["confidence"] = conf
                    state["execution_result"] = result
                    state["needs_refinement"] = False
                    logger.info(f"Refinement improved confidence to {conf:.2f}")

            state["refinement_count"] += 1
            return state

        def should_refine(state: MultiDBPipelineState) -> str:
            """Проверка необходимости рефайнмента."""
            if state["needs_refinement"] and state["refinement_count"] < settings.get("max_retries", 3):
                return "refine"
            return "end"

        def finalize(state: MultiDBPipelineState) -> MultiDBPipelineState:
            """Финализация."""
            state["success"] = state["best_sql"] is not None and state["confidence"] > 0

            # Очистка executor
            if self._executor:
                self._executor.close()
                self._executor = None

            return state

        # Построение графа
        graph = StateGraph(MultiDBPipelineState)

        # Узлы
        graph.add_node("route_databases", route_databases)
        graph.add_node("load_schema", load_schema)
        graph.add_node("generate_sql", generate_sql)
        graph.add_node("execute_and_judge", execute_and_judge)
        graph.add_node("refine_sql", refine_sql)
        graph.add_node("finalize", finalize)

        # Рёбра
        graph.set_entry_point("route_databases")
        graph.add_edge("route_databases", "load_schema")
        graph.add_edge("load_schema", "generate_sql")
        graph.add_edge("generate_sql", "execute_and_judge")

        # Условный переход для рефайнмента
        graph.add_conditional_edges(
            "execute_and_judge",
            should_refine,
            {"refine": "refine_sql", "end": "finalize"},
        )

        # После рефайнмента - повторная оценка
        graph.add_edge("refine_sql", "execute_and_judge")

        # Финализация
        graph.add_edge("finalize", END)

        return graph.compile()

    def run_simple(self, query: str) -> str:
        """
        Упрощённый запуск - вернуть только SQL.

        Args:
            query: Запрос пользователя.

        Returns:
            SQL запрос.
        """
        result = self.run(query)
        return result.get("best_sql", "")

    def run_with_result(self, query: str) -> Dict[str, Any]:
        """
        Запуск с возвратом полного результата.

        Args:
            query: Запрос пользователя.

        Returns:
            Полный результат.
        """
        return self.run(query)

    def close(self):
        """Закрыть соединения."""
        if self._executor:
            self._executor.close()
        self.vector_db.close()
        self.graph_db.close()
        logger.info("Multi-DB Pipeline closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
