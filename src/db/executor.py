# src\db\executor.py
"""
Исполнитель SQL запросов к SQLite базе данных.

Возвращает статус выполнения и результаты/ошибку.
Поддерживает валидацию на наличие опасных операций.

Example:
    >>> from db.executor import SQLExecutor
    >>> executor = SQLExecutor("data/movie.sqlite")
    >>> success, result = executor.execute("SELECT * FROM Movie")
"""
import logging
import sqlite3
from typing import Any, List, Optional, Tuple

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class SQLExecutor:
    """
    Исполнитель SQL запросов к SQLite базе данных.

    Attributes:
        db_path: Путь к SQLite базе данных.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Инициализировать исполнитель.

        Args:
            db_path: Путь к SQLite базе данных.
        """
        settings = get_settings()
        self.db_path = db_path or settings.db_full_path

    def execute(self, sql: str) -> Tuple[bool, Any]:
        """
        Выполнить SQL запрос.

        Args:
            sql: SQL запрос для выполнения.

        Returns:
            Кортеж (успех, результат):
            - (True, List[rows]) при успехе
            - (False, error_message) при ошибке
        """
        import sqlite3

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            conn.close()

            logger.debug(f"Executed SQL: {sql[:100]}..., rows={len(rows)}")
            return True, rows

        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return False, str(e)

    def execute_with_columns(
        self,
        sql: str
    ) -> Tuple[bool, List[str], Any]:
        """
        Выполнить SQL запрос с возвратом имен колонок.

        Args:
            sql: SQL запрос для выполнения.

        Returns:
            Кортеж (успех, columns, rows).
        """
        import sqlite3

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()

            return True, columns, rows

        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return False, [], str(e)

    def validate_execution(self, sql: str) -> Tuple[bool, str]:
        """
        Проверить возможность выполнения запроса без побочных эффектов.

        Args:
            sql: SQL запрос для проверки.

        Returns:
            (True, "") если запрос безопасен,
            (False, reason) если запрос опасен.
        """
        sql_upper = sql.upper().strip()

        dangerous_keywords = [
            "DROP", "DELETE", "TRUNCATE",
            "ALTER", "CREATE", "INSERT",
            "UPDATE", "REPLACE"
        ]

        for keyword in dangerous_keywords:
            if sql_upper.startswith(keyword):
                return False, f"Dangerous operation: {keyword}"

        return True, ""


# Импорты для type hints
from typing import Optional
