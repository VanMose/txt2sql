# src\pipeline\text2sql_pipeline.py
"""
Основный Text-to-SQL пайплайн.

Pipeline:
    User Query → Schema Retrieval → SQL Generator (N samples) →
    Validator → Executor → Judge → Best Selector → Refiner → Final SQL

Example:
    >>> from pipeline.text2sql_pipeline import Text2SQLPipeline
    >>> pipeline = Text2SQLPipeline(db_path="data/movie.sqlite")
    >>> result = pipeline.run_with_result("Show all movies")
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from ..agents.sql_generator import SQLGenerator
from ..agents.sql_judge import SQLJudge
from ..agents.sql_refiner import SQLRefiner
from ..agents.sql_validator import SQLValidator
from ..config.settings import get_settings
from ..db.executor import SQLExecutor
from ..db.schema_loader import SchemaLoader
from .state import PipelineState, SQLAttempt
from ..retrieval.embedder import SchemaEmbedder
from ..retrieval.schema_retriever import SchemaRetriever

logger = logging.getLogger(__name__)


class Text2SQLPipeline:
    """Основный пайплайн Text-to-SQL."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Инициализировать пайплайн.

        Args:
            db_path: Путь к SQLite базе данных.
        """
        self.settings = get_settings()

        self.schema_loader = SchemaLoader(db_path)
        self.executor = SQLExecutor(db_path)
        self.embedder = SchemaEmbedder()

        self.generator = SQLGenerator()
        self.validator = SQLValidator()
        self.judge = SQLJudge()
        self.refiner = SQLRefiner()

        logger.info(f"Pipeline initialized with db_path={db_path}")

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

    def run(self, query: str) -> PipelineState:
        """
        Выполнить пайплайн.

        Args:
            query: Запрос пользователя.

        Returns:
            PipelineState с результатами.
        """
        logger.info(f"Starting pipeline for query: '{query}'")
        state = PipelineState(query=query)
        latencies: Dict[str, float] = {}

        try:
            # 1. Schema Retrieval
            logger.info("Step 1: Schema Retrieval")
            schema_docs, retrieval_time = self._measure_time(self._retrieve_schema, query)
            latencies["schema_retrieval"] = retrieval_time
            logger.info(f"Retrieved {len(schema_docs)} tables in {retrieval_time:.0f}ms")

            state.relevant_tables = schema_docs
            table_names = self._extract_table_names(schema_docs)
            state.schema = self.schema_loader.get_schema_for_tables(table_names)
            latencies["schema_loading"] = 0
            logger.info(f"Schema loaded for tables: {table_names}")

            # 2. SQL Generation
            logger.info("Step 2: SQL Generation")
            candidates, gen_time = self._measure_time(self.generator.generate, query, state.schema)
            latencies["sql_generation"] = gen_time
            logger.info(f"Generated {len(candidates)} SQL candidates in {gen_time:.0f}ms")

            for i, sql in enumerate(candidates, 1):
                logger.info(f"  Candidate {i}: {sql}")

            # 3. Validation, Execution, Judge
            logger.info("Step 3: Validation, Execution, Judge")
            attempts, exec_time = self._measure_time(self._validate_and_judge, query, candidates)
            latencies["validation_execution_judge"] = exec_time
            logger.info(f"Validated {len(attempts)} attempts in {exec_time:.0f}ms")

            state.attempts = attempts

            # 4. Выбор лучшего SQL
            logger.info("Step 4: Selecting best SQL")
            if attempts:
                best = max(attempts, key=lambda x: x.confidence)
                state.best_sql = best.sql
                state.execution_results = best.execution_result
                logger.info(f"Best SQL (confidence={best.confidence:.2f}): {best.sql}")

                # 5. Проверка необходимости рефайнмента
                if best.confidence < self.settings.confidence_threshold:
                    logger.info(f"Confidence {best.confidence:.2f} < threshold {self.settings.confidence_threshold}")
                    state.needs_refinement = True
                    state.retry_count = 1

                    refined_sql, refine_time = self._measure_time(
                        self.refiner.refine, query, state.schema, attempts
                    )
                    latencies["refinement"] = refine_time
                    logger.info(f"Refined SQL: {refined_sql}")

                    ok, result = self.executor.execute(refined_sql)
                    if ok:
                        conf, reason = self.judge.evaluate(query, refined_sql)
                        logger.info(f"Refined confidence: {conf:.2f}")
                        if conf > best.confidence:
                            state.best_sql = refined_sql
                            state.execution_results = result
                            logger.info("Using refined SQL")
            else:
                logger.warning("No valid SQL attempts generated!")

            state.latencies = latencies
            logger.info("Pipeline completed successfully")

        except Exception as e:
            state.latencies = latencies
            logger.error(f"Pipeline error: {e}", exc_info=True)
            raise

        return state

    def _retrieve_schema(self, query: str) -> List[str]:
        """Retrieve релевантных таблиц схемы."""
        schema_docs = self.schema_loader.get_schema_docs()
        retriever = SchemaRetriever(self.embedder, schema_docs)
        return retriever.retrieve(query, top_k=self.settings.top_k_tables)

    def _extract_table_names(self, schema_docs: List[str]) -> List[str]:
        """Извлечь имена таблиц из документов схемы."""
        table_names = []
        for doc in schema_docs:
            if doc.startswith("Table: "):
                name = doc.split("\n")[0].replace("Table: ", "")
                table_names.append(name)
        return table_names

    def _validate_and_judge(self, query: str, candidates: List[str]) -> List[SQLAttempt]:
        """Валидация, выполнение и оценка SQL кандидатов."""
        attempts: List[SQLAttempt] = []

        for i, sql in enumerate(candidates, 1):
            logger.info(f"  Processing candidate {i}: {sql}")

            if not self.validator.validate(sql):
                logger.warning(f"  Validation failed for: {sql}")
                continue

            ok, result = self.executor.execute(sql)
            logger.info(f"  Execution: {'OK' if ok else 'FAILED'} - {result if not ok else f'{len(result)} rows'}")

            if ok:
                confidence, reason = self.judge.evaluate(query, sql)
                try:
                    df = pl.DataFrame(result)
                except Exception as e:
                    logger.warning(f"Failed to create DataFrame: {e}")
                    df = result
            else:
                judge_result = self.judge.evaluate_with_error(query, sql, str(result))
                confidence = 0.0
                reason = judge_result[1] if len(judge_result) >= 2 else "Execution error"
                df = None

            logger.info(f"  Confidence: {confidence:.2f} - {reason}")

            attempts.append(
                SQLAttempt(
                    sql=sql,
                    confidence=confidence,
                    error=not ok,
                    reason=reason,
                    execution_result=df,
                )
            )

        return attempts

    def run_simple(self, query: str) -> str:
        """Упрощенный запуск пайплайна."""
        state = self.run(query)
        return state.best_sql or ""

    def run_with_result(self, query: str) -> Dict[str, Any]:
        """Запустить пайплайн и вернуть полный результат."""
        state = self.run(query)

        result: Dict[str, Any] = {
            "query": state.query,
            "sql_query": state.best_sql or "",
            "sql_result": state.execution_results,
            "confidence": state.attempts[0].confidence if state.attempts else 0.0,
            "relevant_tables": state.relevant_tables,
            "latencies": state.latencies,
            "retry_count": state.retry_count,
            "success": state.best_sql is not None,
            "attempts_count": len(state.attempts),
        }

        logger.info(f"Result: success={result['success']}, sql_query='{result['sql_query']}', attempts={result['attempts_count']}")
        return result
