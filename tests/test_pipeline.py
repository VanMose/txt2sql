"""
Integration tests для основного пайплайна.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.pipeline.text2sql_pipeline import Text2SQLPipeline
from src.pipeline.state import PipelineState, SQLAttempt


@pytest.mark.integration
class TestText2SQLPipeline:
    """Интеграционные тесты для Text2SQLPipeline."""

    def test_pipeline_init(self, test_db_path):
        """Инициализация пайплайна."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        assert pipeline is not None
        assert pipeline.schema_loader is not None
        assert pipeline.executor is not None
        assert pipeline.embedder is not None
        assert pipeline.generator is not None
        assert pipeline.validator is not None
        assert pipeline.judge is not None
        assert pipeline.refiner is not None

    def test_pipeline_run_simple(self, test_db_path, sample_queries):
        """Запуск пайплайна с простым запросом."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        query = sample_queries["simple"]
        
        state = pipeline.run(query)
        
        assert isinstance(state, PipelineState)
        assert state.query == query
        assert state.best_sql is not None
        assert len(state.attempts) > 0

    def test_pipeline_run_with_result(self, test_db_path, sample_queries):
        """Запуск пайплайна с результатом."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        query = sample_queries["simple"]
        
        result = pipeline.run_with_result(query)
        
        assert isinstance(result, dict)
        assert "query" in result
        assert "sql_query" in result
        assert "sql_result" in result
        assert "confidence" in result
        assert "success" in result
        assert result["query"] == query
        assert result["success"] is True

    def test_pipeline_run_simple_sql(self, test_db_path):
        """Упрощенный запуск пайплайна."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        query = "Показать всех пользователей"
        
        sql = pipeline.run_simple(query)
        
        assert isinstance(sql, str)
        assert len(sql) > 0
        assert "SELECT" in sql.upper()

    def test_pipeline_count_query(self, test_db_path, sample_queries):
        """Запуск пайплайна с COUNT запросом."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        query = sample_queries["count"]
        
        result = pipeline.run_with_result(query)
        
        assert result["success"] is True

    def test_pipeline_filter_query(self, test_db_path, sample_queries):
        """Запуск пайплайна с WHERE запросом."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        query = sample_queries["filter"]
        
        result = pipeline.run_with_result(query)
        
        assert result["success"] is True

    def test_pipeline_state_structure(self, test_db_path):
        """Проверка структуры PipelineState."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        query = "Тестовый запрос"
        
        state = pipeline.run(query)
        
        assert hasattr(state, "query")
        assert hasattr(state, "schema")
        assert hasattr(state, "relevant_tables")
        assert hasattr(state, "attempts")
        assert hasattr(state, "best_sql")
        assert hasattr(state, "execution_results")
        assert hasattr(state, "needs_refinement")
        assert hasattr(state, "retry_count")
        assert hasattr(state, "latencies")
        
        assert state.query == query
        assert isinstance(state.relevant_tables, list)
        assert isinstance(state.attempts, list)
        assert isinstance(state.latencies, dict)

    def test_pipeline_sql_attempt_structure(self, test_db_path):
        """Проверка структуры SQLAttempt."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        query = "Тестовый запрос"
        
        state = pipeline.run(query)
        
        if state.attempts:
            attempt = state.attempts[0]
            assert isinstance(attempt, SQLAttempt)
            assert hasattr(attempt, "sql")
            assert hasattr(attempt, "confidence")
            assert hasattr(attempt, "error")
            assert hasattr(attempt, "reason")
            assert hasattr(attempt, "execution_result")
            
            assert isinstance(attempt.sql, str)
            assert isinstance(attempt.confidence, float)
            assert 0 <= attempt.confidence <= 1

    def test_pipeline_latency_tracking(self, test_db_path):
        """Отслеживание latency."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        query = "Тестовый запрос"
        
        state = pipeline.run(query)
        
        assert "schema_retrieval" in state.latencies
        assert "sql_generation" in state.latencies
        assert "validation_execution_judge" in state.latencies
        
        for step, latency in state.latencies.items():
            assert latency >= 0, f"{step} has negative latency"

    def test_pipeline_schema_retrieval(self, test_db_path):
        """Проверка retrieval схемы."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        query = "Показать пользователей"
        
        state = pipeline.run(query)
        
        assert len(state.relevant_tables) > 0
        for doc in state.relevant_tables:
            assert "Table:" in doc or "table:" in doc.lower()

    def test_pipeline_execution_result(self, test_db_path):
        """Проверка результата выполнения."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        query = "Показать всех пользователей"
        
        result = pipeline.run_with_result(query)
        
        assert result["success"] is True
        assert result["sql_result"] is not None
        if hasattr(result["sql_result"], "__len__"):
            assert len(result["sql_result"]) > 0


@pytest.mark.slow
class TestText2SQLPipelineSlow:
    """Медленные интеграционные тесты."""

    def test_pipeline_multiple_queries(self, test_db_path, sample_queries):
        """Запуск пайплайна с несколькими запросами."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        
        results = []
        for query_name, query in sample_queries.items():
            result = pipeline.run_with_result(query)
            results.append((query_name, result))
        
        assert len(results) == len(sample_queries)
        
        for query_name, result in results:
            assert isinstance(result, dict)
            assert "sql_query" in result

    def test_pipeline_refinement_trigger(self, test_db_path):
        """Проверка запуска refinement."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        
        query = "Сложный запрос с несколькими условиями"
        
        state = pipeline.run(query)
        
        assert isinstance(state.needs_refinement, bool)
        assert isinstance(state.retry_count, int)


@pytest.mark.unit
class TestPipelineEdgeCases:
    """Тесты граничных случаев для пайплайна."""

    def test_pipeline_special_characters_query(self, test_db_path):
        """Пайплайн с спецсимволами в запросе."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        query = "Показать пользователей с именем 'Test@#$%' "
        
        state = pipeline.run(query)
        assert isinstance(state, PipelineState)

    def test_pipeline_very_long_query(self, test_db_path):
        """Пайплайн с очень длинным запросом."""
        pipeline = Text2SQLPipeline(db_path=test_db_path)
        query = "Показать всех пользователей " + "очень длинное описание " * 100
        
        state = pipeline.run(query)
        assert isinstance(state, PipelineState)

    def test_pipeline_nonexistent_db_path(self):
        """Пайплайн с несуществующим путем к БД."""
        with pytest.raises(Exception):
            Text2SQLPipeline(db_path="/nonexistent/path/to/db.sqlite")
