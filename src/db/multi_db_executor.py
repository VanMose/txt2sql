# src\db\multi_db_executor.py
"""
Multi-DB Executor для выполнения SQL в нескольких базах данных.

Особенности:
- Поддержка ATTACH DATABASE
- Автоматическое управление соединениями
- Кэширование результатов
- Обработка ошибок

Example:
    >>> from db.multi_db_executor import MultiDBExecutor
    >>> with MultiDBExecutor(["db1.sqlite", "db2.sqlite"]) as executor:
    ...     success, result = executor.execute("SELECT * FROM db1.table1")
"""
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class MultiDBExecutor:
    """
    Исполнитель SQL запросов для нескольких баз данных.

    Attributes:
        db_paths: Список путей к базам данных.
        db_aliases: Словарь {alias: path}.
        _connection: Главное соединение.
        _attached: Флаг подключенных БД.
        _result_cache: Кэш результатов.
    """

    def __init__(self, db_paths: List[str]) -> None:
        """
        Инициализировать исполнитель.

        Args:
            db_paths: Список путей к базам данных.
        """
        self.db_paths = [str(Path(p).resolve()) for p in db_paths]
        self.db_aliases = {Path(p).stem: str(Path(p).resolve()) for p in db_paths}

        self._connection: Optional[sqlite3.Connection] = None
        self._attached = False
        self._result_cache: Dict[str, Any] = {}
        self._cache_max_size = 100

        logger.info(f"MultiDBExecutor initialized with {len(db_paths)} databases")

    def _get_connection(self) -> sqlite3.Connection:
        """Получить соединение (in-memory для ATTACH)."""
        if self._connection is None:
            self._connection = sqlite3.connect(":memory:")
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def attach_databases(self, aliases: Optional[List[str]] = None) -> None:
        """
        Подключить базы данных через ATTACH.

        Args:
            aliases: Псевдонимы для баз данных.
        """
        if self._attached:
            logger.debug("Databases already attached")
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        if aliases is None:
            aliases = list(self.db_aliases.keys())

        for alias, db_path in zip(aliases, self.db_paths):
            try:
                safe_path = db_path.replace("'", "''")
                cursor.execute(f"ATTACH DATABASE '{safe_path}' AS {alias}")
                logger.debug(f"Attached database {alias} -> {db_path}")
            except Exception as e:
                logger.error(f"Failed to attach {alias}: {e}")
                raise

        self._attached = True
        logger.info(f"Attached {len(self.db_paths)} databases")

    def detach_databases(self) -> None:
        """Отключить базы данных."""
        if not self._attached:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        for alias in self.db_aliases.keys():
            try:
                cursor.execute(f"DETACH DATABASE {alias}")
            except Exception:
                pass

        self._attached = False
        logger.debug("Databases detached")

    def execute(
        self,
        sql: str,
        params: Optional[Tuple] = None,
        use_cache: bool = True,
    ) -> Tuple[bool, Any]:
        """
        Выполнить SQL запрос.

        Args:
            sql: SQL запрос.
            params: Параметры для запроса.
            use_cache: Использовать кэш.

        Returns:
            Кортеж (success, result/error).
        """
        cache_key = f"{sql}:{params}"

        # Guardrails
        from .guardrails import SQLGuardrails
        is_safe, validated = SQLGuardrails.validate(sql)
        if not is_safe:
            logger.warning(f"SQL blocked by guardrails: {validated}")
            return False, f"Заблокировано: {validated}"
        sql = validated


        if use_cache and cache_key in self._result_cache:
            logger.debug(f"Cache hit for SQL: {sql[:50]}...")
            return True, self._result_cache[cache_key]

        try:
            if not self._attached and "ATTACH" not in sql.upper():
                self.attach_databases()

            conn = self._get_connection()
            cursor = conn.cursor()

            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            if sql.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]

                if use_cache and result:
                    if len(self._result_cache) >= self._cache_max_size:
                        keys_to_delete = list(self._result_cache.keys())[: len(self._result_cache) // 2]
                        for k in keys_to_delete:
                            del self._result_cache[k]
                    self._result_cache[cache_key] = result

                logger.info(f"Query executed successfully: {len(result)} rows")
                return True, result
            else:
                conn.commit()
                logger.info(f"Non-SELECT query executed: {cursor.rowcount} rows affected")
                return True, {"rows_affected": cursor.rowcount}

        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return False, str(e)

    def execute_with_dataframe(
        self,
        sql: str,
        params: Optional[Tuple] = None,
    ) -> Tuple[bool, Any]:
        """
        Выполнить SQL и вернуть DataFrame.

        Args:
            sql: SQL запрос.
            params: Параметры.

        Returns:
            Кортеж (success, DataFrame/error).
        """
        success, result = self.execute(sql, params)

        if success and isinstance(result, list):
            try:
                df = pl.DataFrame(result)
                return True, df
            except Exception as e:
                logger.warning(f"Failed to create DataFrame: {e}")
                return True, result

        return success, result

    def get_attached_databases(self) -> List[str]:
        """Получить список подключенных баз данных."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA database_list")
        return [row[1] for row in cursor.fetchall() if row[1] != "main"]

    def test_connections(self) -> Dict[str, bool]:
        """Протестировать подключение к базам данных."""
        results: Dict[str, bool] = {}

        for alias, db_path in self.db_aliases.items():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                conn.close()
                results[alias] = True
            except Exception as e:
                logger.warning(f"Connection test failed for {alias}: {e}")
                results[alias] = False

        return results

    def close(self) -> None:
        """Закрыть соединение."""
        self.detach_databases()
        if self._connection:
            self._connection.close()
            self._connection = None
        logger.info("MultiDBExecutor connection closed")

    def __enter__(self) -> "MultiDBExecutor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
