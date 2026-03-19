"""
Unit tests для agents компонентов.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents.sql_validator import SQLValidator
from src.agents.sql_generator import SQLGenerator


@pytest.mark.unit
class TestSQLValidator:
    """Тесты для SQLValidator."""

    def test_init(self):
        """Инициализация SQLValidator."""
        validator = SQLValidator()
        assert validator is not None

    def test_validate_select(self):
        """Валидация SELECT запроса."""
        validator = SQLValidator()
        sql = "SELECT * FROM users"
        assert validator.validate(sql) is True

    def test_validate_select_complex(self):
        """Валидация сложного SELECT запроса."""
        validator = SQLValidator()
        sql = """
            SELECT u.name, COUNT(o.id) as order_count
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            GROUP BY u.id
            HAVING COUNT(o.id) > 0
            ORDER BY order_count DESC
        """
        assert validator.validate(sql) is True

    def test_validate_insert(self):
        """Валидация INSERT запроса (должен быть отклонен)."""
        validator = SQLValidator()
        sql = "INSERT INTO users (name) VALUES ('Test')"
        assert validator.validate(sql) is False

    def test_validate_update(self):
        """Валидация UPDATE запроса (должен быть отклонен)."""
        validator = SQLValidator()
        sql = "UPDATE users SET name = 'Test'"
        assert validator.validate(sql) is False

    def test_validate_delete(self):
        """Валидация DELETE запроса (должен быть отклонен)."""
        validator = SQLValidator()
        sql = "DELETE FROM users WHERE id = 1"
        assert validator.validate(sql) is False

    def test_validate_drop(self):
        """Валидация DROP запроса (должен быть отклонен)."""
        validator = SQLValidator()
        sql = "DROP TABLE users"
        assert validator.validate(sql) is False

    def test_validate_invalid_sql(self):
        """Валидация невалидного SQL."""
        validator = SQLValidator()
        sql = "INVALID SQL QUERY"
        assert validator.validate(sql) is False

    def test_validate_empty_sql(self):
        """Валидация пустого SQL."""
        validator = SQLValidator()
        sql = ""
        assert validator.validate(sql) is False

    def test_validate_whitespace_sql(self):
        """Валидация SQL с пробелами."""
        validator = SQLValidator()
        sql = "   SELECT   *   FROM   users   "
        assert validator.validate(sql) is True

    def test_validate_lowercase(self):
        """Валидация SQL в нижнем регистре."""
        validator = SQLValidator()
        sql = "select * from users"
        assert validator.validate(sql) is True

    def test_validate_mixed_case(self):
        """Валидация SQL со смешанным регистром."""
        validator = SQLValidator()
        sql = "SeLeCt * FrOm users"
        assert validator.validate(sql) is True

    def test_validate_with_comments(self):
        """Валидация SQL с комментариями."""
        validator = SQLValidator()
        sql = "-- Comment\nSELECT * FROM users"
        assert validator.validate(sql) is True

    def test_validate_subquery(self):
        """Валидация SQL с подзапросом."""
        validator = SQLValidator()
        sql = """
            SELECT name FROM users
            WHERE id IN (SELECT user_id FROM orders WHERE amount > 100)
        """
        assert validator.validate(sql) is True

    def test_validate_cte(self):
        """Валидация SQL с CTE (Common Table Expression)."""
        validator = SQLValidator()
        sql = """
            WITH user_orders AS (
                SELECT user_id, COUNT(*) as order_count
                FROM orders
                GROUP BY user_id
            )
            SELECT u.name, uo.order_count
            FROM users u
            JOIN user_orders uo ON u.id = uo.user_id
        """
        assert validator.validate(sql) is True


@pytest.mark.unit
class TestSQLGenerator:
    """Тесты для SQLGenerator."""

    def test_init(self):
        """Инициализация SQLGenerator."""
        generator = SQLGenerator()
        assert generator is not None

    def test_generate_mock(self, sample_queries):
        """Генерация SQL в mock режиме."""
        generator = SQLGenerator()
        schema = """
        Table: users
        Columns: id (INTEGER), name (TEXT), email (TEXT), age (INTEGER)
        
        Table: orders
        Columns: id (INTEGER), user_id (INTEGER), product (TEXT), amount (REAL)
        """
        
        query = sample_queries["simple"]
        candidates = generator.generate(query, schema, n=2)
        
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        for candidate in candidates:
            assert isinstance(candidate, str)
            assert len(candidate) > 0

    def test_generate_multiple_samples(self, sample_queries):
        """Генерация нескольких вариантов SQL."""
        generator = SQLGenerator()
        schema = """
        Table: users
        Columns: id, name, email, age
        """
        
        query = sample_queries["simple"]
        candidates = generator.generate(query, schema, n=5)
        
        assert len(candidates) <= 5
        assert len(candidates) > 0

    def test_generate_with_schema_tables(self, sample_queries):
        """Генерация SQL с использованием таблиц из схемы."""
        generator = SQLGenerator()
        schema = """
        Table: users
        Columns: id, name, email
        
        Table: products
        Columns: id, name, price
        """
        
        query = "Показать всех пользователей"
        candidates = generator.generate(query, schema, n=1)
        
        assert len(candidates) > 0
        sql_lower = candidates[0].lower()
        assert "users" in sql_lower or "select" in sql_lower


@pytest.mark.integration
class TestAgentsIntegration:
    """Интеграционные тесты для agents."""

    def test_validator_with_generated_sql(self, sample_queries):
        """Валидация сгенерированного SQL."""
        generator = SQLGenerator()
        validator = SQLValidator()
        
        schema = """
        Table: users
        Columns: id, name, email, age
        """
        
        query = sample_queries["simple"]
        candidates = generator.generate(query, schema, n=2)
        
        for candidate in candidates:
            result = validator.validate(candidate)
            assert isinstance(result, bool)

    def test_generate_and_validate_count_query(self, sample_queries):
        """Генерация и валидация COUNT запроса."""
        generator = SQLGenerator()
        validator = SQLValidator()
        
        schema = """
        Table: users
        Columns: id, name, email
        """
        
        query = sample_queries["count"]
        candidates = generator.generate(query, schema, n=1)
        
        if candidates:
            sql = candidates[0]
            is_valid = validator.validate(sql)
            assert isinstance(is_valid, bool)
