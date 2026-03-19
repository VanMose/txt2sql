"""
Unit tests для database компонентов.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.db.schema_loader import SchemaLoader
from src.db.executor import SQLExecutor


@pytest.mark.unit
class TestSchemaLoader:
    """Тесты для SchemaLoader."""

    def test_init(self, test_db_path):
        """Инициализация SchemaLoader."""
        loader = SchemaLoader(test_db_path)
        assert loader.db_path == test_db_path

    def test_get_tables(self, test_db_path):
        """Получение списка таблиц."""
        loader = SchemaLoader(test_db_path)
        tables = loader.get_tables()
        assert isinstance(tables, list)
        assert "users" in tables
        assert "orders" in tables
        assert "products" in tables

    def test_get_table_schema(self, test_db_path):
        """Получение схемы таблицы."""
        loader = SchemaLoader(test_db_path)
        schema = loader.get_table_schema("users")
        assert schema is not None
        assert "table_name" in schema
        assert "columns" in schema
        assert schema["table_name"] == "users"
        assert len(schema["columns"]) > 0

    def test_get_table_schema_columns(self, test_db_path):
        """Проверка колонок таблицы."""
        loader = SchemaLoader(test_db_path)
        schema = loader.get_table_schema("users")
        column_names = [col["name"] for col in schema["columns"]]
        assert "id" in column_names
        assert "name" in column_names
        assert "email" in column_names
        assert "age" in column_names

    def test_get_table_schema_nonexistent(self, test_db_path):
        """Получение схемы несуществующей таблицы."""
        loader = SchemaLoader(test_db_path)
        schema = loader.get_table_schema("nonexistent_table")
        assert schema is None or schema.get("columns") == []

    def test_get_full_schema(self, test_db_path):
        """Получение полной схемы БД."""
        loader = SchemaLoader(test_db_path)
        schema = loader.get_full_schema()
        assert isinstance(schema, dict)
        assert "users" in schema
        assert "orders" in schema

    def test_get_schema_docs(self, test_db_path):
        """Получение документов схемы."""
        loader = SchemaLoader(test_db_path)
        docs = loader.get_schema_docs()
        assert isinstance(docs, list)
        assert len(docs) > 0
        docs_text = " ".join(docs)
        assert "users" in docs_text.lower() or "Users" in docs_text

    def test_get_schema_for_tables(self, test_db_path):
        """Получение схемы для выбранных таблиц."""
        loader = SchemaLoader(test_db_path)
        schema = loader.get_schema_for_tables(["users", "orders"])
        assert isinstance(schema, str)
        assert "users" in schema.lower()
        assert "orders" in schema.lower()

    def test_get_schema_for_tables_empty(self, test_db_path):
        """Получение схемы для пустого списка таблиц."""
        loader = SchemaLoader(test_db_path)
        schema = loader.get_schema_for_tables([])
        assert schema == "" or schema is None


@pytest.mark.unit
class TestSQLExecutor:
    """Тесты для SQLExecutor."""

    def test_init(self, test_db_path):
        """Инициализация SQLExecutor."""
        executor = SQLExecutor(test_db_path)
        assert executor.db_path == test_db_path

    def test_execute_select(self, test_db_path):
        """Выполнение SELECT запроса."""
        executor = SQLExecutor(test_db_path)
        ok, result = executor.execute("SELECT * FROM users")
        assert ok is True
        assert isinstance(result, list)
        assert len(result) == 3

    def test_execute_select_columns(self, test_db_path):
        """Выполнение SELECT с колонками."""
        executor = SQLExecutor(test_db_path)
        ok, result = executor.execute("SELECT id, name FROM users")
        assert ok is True
        assert len(result) == 3
        assert len(result[0]) == 2

    def test_execute_with_where(self, test_db_path):
        """Выполнение SELECT с WHERE."""
        executor = SQLExecutor(test_db_path)
        ok, result = executor.execute("SELECT * FROM users WHERE age > 30")
        assert ok is True
        assert len(result) == 1

    def test_execute_count(self, test_db_path):
        """Выполнение COUNT запроса."""
        executor = SQLExecutor(test_db_path)
        ok, result = executor.execute("SELECT COUNT(*) FROM users")
        assert ok is True
        assert len(result) == 1
        assert result[0][0] == 3

    def test_execute_sum(self, test_db_path):
        """Выполнение SUM запроса."""
        executor = SQLExecutor(test_db_path)
        ok, result = executor.execute("SELECT SUM(amount) FROM orders")
        assert ok is True
        assert len(result) == 1
        assert result[0][0] == pytest.approx(1409.96, rel=0.01)

    def test_execute_join(self, test_db_path):
        """Выполнение JOIN запроса."""
        executor = SQLExecutor(test_db_path)
        ok, result = executor.execute("""
            SELECT u.name, o.product
            FROM users u
            JOIN orders o ON u.id = o.user_id
        """)
        assert ok is True
        assert len(result) == 4

    def test_execute_invalid_sql(self, test_db_path):
        """Выполнение невалидного SQL."""
        executor = SQLExecutor(test_db_path)
        ok, result = executor.execute("INVALID SQL QUERY")
        assert ok is False
        assert isinstance(result, str)

    def test_execute_empty_result(self, test_db_path):
        """Выполнение запроса с пустым результатом."""
        executor = SQLExecutor(test_db_path)
        ok, result = executor.execute("SELECT * FROM users WHERE age > 100")
        assert ok is True
        assert len(result) == 0

    def test_execute_with_order(self, test_db_path):
        """Выполнение запроса с ORDER BY."""
        executor = SQLExecutor(test_db_path)
        ok, result = executor.execute("SELECT * FROM users ORDER BY age DESC")
        assert ok is True
        assert len(result) == 3
        assert result[0][3] == 35

    def test_execute_with_limit(self, test_db_path):
        """Выполнение запроса с LIMIT."""
        executor = SQLExecutor(test_db_path)
        ok, result = executor.execute("SELECT * FROM users LIMIT 2")
        assert ok is True
        assert len(result) == 2

    def test_execute_columns_and_rows(self, test_db_path):
        """Проверка execute_with_columns."""
        executor = SQLExecutor(test_db_path)
        ok, result = executor.execute_with_columns("SELECT id, name FROM users")
        assert ok is True
        assert "columns" in result
        assert "rows" in result
        assert result["columns"] == ["id", "name"]
        assert len(result["rows"]) == 3


@pytest.mark.integration
class TestSchemaLoaderIntegration:
    """Интеграционные тесты для database компонентов."""

    def test_schema_loader_and_executor(self, test_db_path):
        """Совместная работа SchemaLoader и SQLExecutor."""
        loader = SchemaLoader(test_db_path)
        executor = SQLExecutor(test_db_path)

        tables = loader.get_tables()
        assert "users" in tables

        schema = loader.get_table_schema("users")
        column_names = [col["name"] for col in schema["columns"]]

        if "name" in column_names:
            ok, result = executor.execute("SELECT name FROM users")
            assert ok is True
            assert len(result) == 3

    def test_full_schema_retrieval(self, test_db_path):
        """Полный цикл получения схемы."""
        loader = SchemaLoader(test_db_path)
        tables = loader.get_tables()

        for table in tables:
            schema = loader.get_table_schema(table)
            assert schema is not None
            assert "columns" in schema

        docs = loader.get_schema_docs()
        assert len(docs) == len(tables)
