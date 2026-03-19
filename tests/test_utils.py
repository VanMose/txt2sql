"""
Unit tests для утилит.
"""
import pytest
import sys
from pathlib import Path

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.json_parser import parse_json, safe_parse_json
from src.utils.sql_parser import (
    extract_tables,
    extract_columns,
    get_query_type,
    is_select_query,
)


class TestJsonParser:
    """Тесты для JSON парсера."""

    def test_parse_clean_json(self):
        """Парсинг чистого JSON."""
        json_str = '{"sql": "SELECT * FROM users", "confidence": 0.9}'
        result = parse_json(json_str)
        assert result["sql"] == "SELECT * FROM users"
        assert result["confidence"] == 0.9

    def test_parse_json_in_text(self):
        """Парсинг JSON внутри текста."""
        text = """
        Вот результат:
        ```json
        {"sql": "SELECT * FROM users", "confidence": 0.9}
        ```
        Конец.
        """
        result = parse_json(text)
        assert result["sql"] == "SELECT * FROM users"

    def test_parse_json_with_backticks(self):
        """Парсинг JSON с markdown backticks."""
        text = '```json\n{"key": "value"}\n```'
        result = parse_json(text)
        assert result["key"] == "value"

    def test_safe_parse_json_valid(self):
        """Безопасный парсинг валидного JSON."""
        json_str = '{"valid": true}'
        default = {"default": "value"}
        result = safe_parse_json(json_str, default)
        assert result["valid"] is True

    def test_safe_parse_json_invalid(self):
        """Безопасный парсинг невалидного JSON."""
        invalid_str = '{invalid json}'
        default = {"default": "value"}
        result = safe_parse_json(invalid_str, default)
        assert result == default

    def test_safe_parse_json_empty(self):
        """Безопасный парсинг пустой строки."""
        result = safe_parse_json("", {"default": "value"})
        assert result["default"] == "value"


class TestSqlParser:
    """Тесты для SQL парсера."""

    def test_extract_tables_simple(self):
        """Извлечение таблиц из простого запроса."""
        sql = "SELECT * FROM users"
        tables = extract_tables(sql)
        assert "users" in tables

    def test_extract_tables_multiple(self):
        """Извлечение нескольких таблиц."""
        sql = "SELECT * FROM users u JOIN orders o ON u.id = o.user_id"
        tables = extract_tables(sql)
        assert "users" in tables
        assert "orders" in tables

    def test_extract_columns_simple(self):
        """Извлечение колонок из простого запроса."""
        sql = "SELECT id, name, email FROM users"
        columns = extract_columns(sql)
        assert "id" in columns
        assert "name" in columns
        assert "email" in columns

    def test_extract_columns_star(self):
        """Извлечение *."""
        sql = "SELECT * FROM users"
        columns = extract_columns(sql)
        assert "*" in columns

    def test_get_query_type_select(self):
        """Определение типа SELECT запроса."""
        sql = "SELECT * FROM users"
        query_type = get_query_type(sql)
        assert query_type.upper() == "SELECT"

    def test_get_query_type_insert(self):
        """Определение типа INSERT запроса."""
        sql = "INSERT INTO users (name) VALUES ('Test')"
        query_type = get_query_type(sql)
        assert query_type.upper() == "INSERT"

    def test_get_query_type_update(self):
        """Определение типа UPDATE запроса."""
        sql = "UPDATE users SET name = 'Test'"
        query_type = get_query_type(sql)
        assert query_type.upper() == "UPDATE"

    def test_get_query_type_delete(self):
        """Определение типа DELETE запроса."""
        sql = "DELETE FROM users WHERE id = 1"
        query_type = get_query_type(sql)
        assert query_type.upper() == "DELETE"

    def test_is_select_query_true(self):
        """Проверка SELECT запроса - True."""
        sql = "SELECT * FROM users"
        assert is_select_query(sql) is True

    def test_is_select_query_false(self):
        """Проверка не-SELECT запроса - False."""
        sql = "INSERT INTO users (name) VALUES ('Test')"
        assert is_select_query(sql) is False

    def test_is_select_query_lowercase(self):
        """Проверка SELECT в нижнем регистре."""
        sql = "select * from users"
        assert is_select_query(sql) is True

    def test_extract_tables_complex(self):
        """Извлечение таблиц из сложного запроса."""
        sql = """
            SELECT u.name, COUNT(o.id) as order_count
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            GROUP BY u.id
        """
        tables = extract_tables(sql)
        assert "users" in tables
        assert "orders" in tables

    def test_invalid_sql_handling(self):
        """Обработка невалидного SQL."""
        sql = "INVALID SQL QUERY"
        tables = extract_tables(sql)
        assert isinstance(tables, list)


@pytest.mark.unit
class TestSqlParserEdgeCases:
    """Тесты граничных случаев для SQL парсера."""

    def test_empty_sql(self):
        """Пустой SQL запрос."""
        assert extract_tables("") == []
        assert extract_columns("") == []
        assert get_query_type("") == "UNKNOWN"
        assert is_select_query("") is False

    def test_whitespace_sql(self):
        """SQL с лишними пробелами."""
        sql = "   SELECT   *   FROM   users   "
        tables = extract_tables(sql)
        assert "users" in tables

    def test_multiline_sql(self):
        """Многострочный SQL."""
        sql = """
            SELECT id, name
            FROM users
            WHERE age > 30
        """
        tables = extract_tables(sql)
        assert "users" in tables

    def test_sql_with_comments(self):
        """SQL с комментариями."""
        sql = "-- Comment\nSELECT * FROM users"
        tables = extract_tables(sql)
        assert "users" in tables

    def test_case_insensitive_keywords(self):
        """Регистронезависимые ключевые слова."""
        for sql in ["SELECT * FROM users", "select * from users", "SeLeCt * FrOm users"]:
            assert is_select_query(sql) is True
            assert get_query_type(sql).upper() == "SELECT"
