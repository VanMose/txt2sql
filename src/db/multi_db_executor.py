# src\db\multi_db_executor.py
"""
Multi-DB Executor для выполнения SQL в нескольких базах данных.

Production features:
- SQL Guardrails integration (SELECT-only, auto-LIMIT, SQL injection protection)
- Connection Pooling для эффективного управления соединениями
- Result Cache для кэширования результатов запросов
- Поддержка ATTACH DATABASE
- Автоматическое управление соединениями
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
from queue import Queue, Empty
import threading

import polars as pl

from ..config.settings import get_settings
from .guardrails import SQLGuardrails

logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    Пул соединений для SQLite.
    
    Production feature: эффективное переиспользование соединений.
    """
    
    def __init__(self, db_path: str, pool_size: int = 20, timeout: int = 30) -> None:
        """
        Инициализировать пул соединений.
        
        Args:
            db_path: Путь к базе данных.
            pool_size: Размер пула.
            timeout: Таймаут ожидания соединения (сек).
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout
        
        # Queue для соединений
        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created = 0
        
        # Pre-create connections
        for _ in range(min(pool_size, 5)):
            conn = self._create_connection()
            if conn:
                self._pool.put(conn)
        
        logger.info(f"ConnectionPool initialized for {Path(db_path).name}: size={pool_size}")
    
    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Создать новое соединение."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            return None
    
    def get(self) -> Optional[sqlite3.Connection]:
        """
        Получить соединение из пула.
        
        Returns:
            Соединение или None.
        """
        try:
            # Try to get from pool
            conn = self._pool.get(timeout=self.timeout)
            logger.debug(f"Connection acquired from pool (size={self._pool.qsize()})")
            return conn
        except Empty:
            # Pool empty, create new if under limit
            with self._lock:
                if self._created < self.pool_size:
                    conn = self._create_connection()
                    if conn:
                        self._created += 1
                        logger.debug(f"New connection created (total={self._created})")
                        return conn
            # Pool at max, wait for available
            try:
                conn = self._pool.get(timeout=self.timeout)
                return conn
            except Empty:
                logger.warning("Connection pool exhausted")
                return None
    
    def put(self, conn: sqlite3.Connection) -> None:
        """
        Вернуть соединение в пул.
        
        Args:
            conn: Соединение для возврата.
        """
        try:
            self._pool.put_nowait(conn)
            logger.debug(f"Connection returned to pool (size={self._pool.qsize()})")
        except Exception:
            # Pool full, close connection
            try:
                conn.close()
            except:
                pass
            logger.debug("Connection pool full, connection closed")
    
    def close_all(self) -> None:
        """Закрыть все соединения."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except:
                pass
        logger.info(f"ConnectionPool closed (created={self._created})")


# Глобальный пул соединений
_connection_pools: Dict[str, ConnectionPool] = {}
_pool_lock = threading.Lock()


def get_connection_pool(db_path: str) -> ConnectionPool:
    """
    Получить или создать пул соединений для БД.
    
    Args:
        db_path: Путь к базе данных.
    
    Returns:
        ConnectionPool экземпляр.
    """
    db_path = str(Path(db_path).resolve())
    
    with _pool_lock:
        if db_path not in _connection_pools:
            settings = get_settings()
            _connection_pools[db_path] = ConnectionPool(
                db_path=db_path,
                pool_size=settings.connection_pool_size,
                timeout=settings.connection_pool_timeout,
            )
        
        return _connection_pools[db_path]


class MultiDBExecutor:
    """
    Исполнитель SQL запросов для нескольких баз данных.
    
    Production features:
    - Connection Pooling для эффективного управления соединениями
    - Result Cache для кэширования результатов
    - SQL Guardrails для безопасности

    Attributes:
        db_paths: Список путей к базам данным.
        db_aliases: Словарь {alias: path}.
        _connection: Главное соединение.
        _attached: Флаг подключенных БД.
        _use_cache: Флаг использования кэша.
        _result_cache: Кэш результатов.
    """

    def __init__(self, db_paths: List[str], use_cache: bool = True) -> None:
        """
        Инициализировать исполнитель.

        Args:
            db_paths: Список путей к базам данным.
            use_cache: Использовать кэш результатов.
        """
        self.db_paths = [str(Path(p).resolve()) for p in db_paths]
        self.db_aliases = {Path(p).stem: str(Path(p).resolve()) for p in db_paths}
        self.use_cache = use_cache

        self._connection: Optional[sqlite3.Connection] = None
        self._attached = False
        
        # Result cache
        self._result_cache: Dict[str, Any] = {}
        self._cache_max_size = 100
        
        # Connection pools
        self._pools: Dict[str, ConnectionPool] = {}
        
        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._queries_executed = 0

        logger.info(f"MultiDBExecutor initialized with {len(db_paths)} databases, cache={use_cache}")

    def _get_connection(self) -> sqlite3.Connection:
        """Получить соединение (in-memory для ATTACH)."""
        if self._connection is None:
            self._connection = sqlite3.connect(":memory:")
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _get_pool(self, db_path: str) -> ConnectionPool:
        """Получить пул соединений для БД."""
        if db_path not in self._pools:
            settings = get_settings()
            self._pools[db_path] = ConnectionPool(
                db_path=db_path,
                pool_size=settings.connection_pool_size,
                timeout=settings.connection_pool_timeout,
            )
        return self._pools[db_path]

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
        use_cache = use_cache and self.use_cache
        cache_key = f"{sql}:{params}"
        
        # Проверка кэша
        if use_cache and cache_key in self._result_cache:
            self._cache_hits += 1
            logger.debug(f"Cache HIT for SQL: {sql[:50]}...")
            return True, self._result_cache[cache_key]
        
        self._cache_misses += 1

        # Guardrails
        is_safe, validated = SQLGuardrails.validate(sql)
        if not is_safe:
            logger.warning(f"SQL blocked by guardrails: {validated}")
            return False, f"Заблокировано: {validated}"
        sql = validated

        try:
            if not self._attached and "ATTACH" not in sql.upper():
                self.attach_databases()

            conn = self._get_connection()
            cursor = conn.cursor()

            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            self._queries_executed += 1

            if sql.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]

                # Сохранение в кэш
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
        use_cache: bool = True,
    ) -> Tuple[bool, Any]:
        """
        Выполнить SQL и вернуть DataFrame.

        Args:
            sql: SQL запрос.
            params: Параметры.
            use_cache: Использовать кэш.

        Returns:
            Кортеж (success, DataFrame/error).
        """
        success, result = self.execute(sql, params, use_cache)

        if success and isinstance(result, list):
            try:
                df = pl.DataFrame(result)
                return True, df
            except Exception as e:
                logger.warning(f"Failed to create DataFrame: {e}")
                return True, result

        return success, result

    def execute_on_db(
        self,
        db_path: str,
        sql: str,
        use_cache: bool = True,
    ) -> Tuple[bool, Any]:
        """
        Выполнить SQL на конкретной БД с использованием connection pool.
        
        Production feature: эффективное переиспользование соединений.

        Args:
            db_path: Путь к базе данных.
            sql: SQL запрос.
            use_cache: Использовать кэш.

        Returns:
            Кортеж (success, result).
        """
        cache_key = f"{db_path}:{sql}"
        
        # Проверка кэша
        if use_cache and self.use_cache and cache_key in self._result_cache:
            self._cache_hits += 1
            logger.debug(f"Cache HIT for {Path(db_path).name}: {sql[:30]}...")
            return True, self._result_cache[cache_key]
        
        self._cache_misses += 1

        # Guardrails
        is_safe, validated = SQLGuardrails.validate(sql)
        if not is_safe:
            return False, f"Заблокировано: {validated}"
        sql = validated

        try:
            # Get connection from pool
            pool = self._get_pool(db_path)
            conn = pool.get()
            
            if conn is None:
                return False, "Failed to get connection from pool"
            
            try:
                cursor = conn.cursor()
                cursor.execute(sql)
                
                self._queries_executed += 1
                
                if sql.strip().upper().startswith("SELECT"):
                    rows = cursor.fetchall()
                    result = [dict(row) for row in rows]
                    
                    # Save to cache
                    if use_cache and self.use_cache and result:
                        if len(self._result_cache) >= self._cache_max_size:
                            keys_to_delete = list(self._result_cache.keys())[: len(self._result_cache) // 2]
                            for k in keys_to_delete:
                                del self._result_cache[k]
                        self._result_cache[cache_key] = result
                    
                    # Return connection to pool
                    pool.put(conn)
                    
                    logger.info(f"Query executed on {Path(db_path).name}: {len(result)} rows")
                    return True, result
                else:
                    conn.commit()
                    pool.put(conn)
                    return True, {"rows_affected": cursor.rowcount}
            
            except Exception as e:
                pool.put(conn)
                raise
                
        except Exception as e:
            logger.error(f"SQL execution error on {db_path}: {e}")
            return False, str(e)

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
                pool = self._get_pool(db_path)
                conn = pool.get()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    pool.put(conn)
                    results[alias] = True
                else:
                    results[alias] = False
            except Exception as e:
                logger.warning(f"Connection test failed for {alias}: {e}")
                results[alias] = False

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику исполнителя."""
        total_queries = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_queries * 100 if total_queries > 0 else 0
        
        return {
            "databases": len(self.db_paths),
            "queries_executed": self._queries_executed,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(hit_rate, 2),
            "cache_size": len(self._result_cache),
            "connection_pools": len(self._pools),
        }

    def close(self) -> None:
        """Закрыть соединения и пулы."""
        # Close connection pools
        for db_path, pool in self._pools.items():
            pool.close_all()
        self._pools.clear()
        
        # Close main connection
        self.detach_databases()
        if self._connection:
            self._connection.close()
            self._connection = None
        
        logger.info(f"MultiDBExecutor closed (executed={self._queries_executed}, cache_hit_rate={self._cache_hits / (self._cache_hits + self._cache_misses) * 100 if (self._cache_hits + self._cache_misses) > 0 else 0:.1f}%)")

    def __enter__(self) -> "MultiDBExecutor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
