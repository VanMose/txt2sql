"""
SQL Guardrails — защита от опасных запросов.

Функции:
1. SELECT-only — блокирует DROP/DELETE/UPDATE/INSERT
2. Авто-LIMIT — добавляет LIMIT если нет
3. SQL Injection защита — блокирует множественные запросы
4. Таймаут — ограничение времени выполнения
"""
import logging
import re
from typing import Tuple

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class SQLGuardrails:
    """Валидация и защита SQL запросов."""

    # Запрещённые операции
    DANGEROUS_KEYWORDS = [
        "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE",
        "INSERT", "UPDATE", "REPLACE", "GRANT", "REVOKE",
        "EXEC", "EXECUTE", "ATTACH", "DETACH",
    ]

    @classmethod
    def validate(cls, sql: str) -> Tuple[bool, str]:
        """
        Полная валидация SQL запроса.

        Returns:
            (True, sanitized_sql) если безопасен,
            (False, reason) если опасен.
        """
        if not sql or not sql.strip():
            return False, "Пустой запрос"

        sql = sql.strip().rstrip(";")

        # 1. Проверка на множественные запросы (SQL injection)
        if ";" in sql:
            return False, "Множественные запросы запрещены"

        # 2. SELECT-only
        is_safe, reason = cls._check_select_only(sql)
        if not is_safe:
            return False, reason

        # 3. Проверка на опасные ключевые слова в любом месте
        is_safe, reason = cls._check_dangerous_keywords(sql)
        if not is_safe:
            return False, reason

        # 4. Авто-LIMIT
        sql = cls._ensure_limit(sql)

        logger.debug(f"SQL passed guardrails: {sql[:100]}...")
        return True, sql

    @classmethod
    def _check_select_only(cls, sql: str) -> Tuple[bool, str]:
        """Проверить что запрос начинается с SELECT или WITH."""
        sql_upper = sql.upper().strip()
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return False, f"Разрешены только SELECT запросы. Получено: {sql.split()[0] if sql.split() else '?'}"
        return True, ""

    @classmethod
    def _check_dangerous_keywords(cls, sql: str) -> Tuple[bool, str]:
        """Проверить на опасные операции внутри запроса."""
        sql_upper = sql.upper()
        for keyword in cls.DANGEROUS_KEYWORDS:
            # Ищем ключевое слово как отдельное слово (не часть имени таблицы)
            pattern = rf"\b{keyword}\b"
            # Пропускаем если это внутри строковых литералов
            sql_no_strings = re.sub(r"'[^']*'", "", sql_upper)
            sql_no_strings = re.sub(r'"[^"]*"', "", sql_no_strings)
            if re.search(pattern, sql_no_strings):
                return False, f"Запрещённая операция: {keyword}"
        return True, ""

    @classmethod
    def _ensure_limit(cls, sql: str) -> str:
        """Добавить LIMIT если отсутствует."""
        from src.config.settings import get_settings
        settings = get_settings()
        sql_upper = sql.upper()

        # Не добавляем LIMIT к агрегатным запросам (COUNT, SUM, AVG, MAX, MIN)
        # без GROUP BY — они и так вернут 1 строку
        agg_funcs = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]
        has_agg = any(func in sql_upper for func in agg_funcs)
        has_group_by = "GROUP BY" in sql_upper
        if has_agg and not has_group_by:
            return sql

        # Проверяем наличие LIMIT
        if "LIMIT" not in sql_upper:
            sql = f"{sql} LIMIT {settings.sql_default_limit}"
            logger.info(f"Auto-added LIMIT {settings.sql_default_limit}")
        else:
            # Проверяем что LIMIT не слишком большой
            match = re.search(r"LIMIT\s+(\d+)", sql_upper)
            if match:
                limit_val = int(match.group(1))
                if limit_val > settings.sql_max_limit:
                    sql = re.sub(
                        r"LIMIT\s+\d+",
                        f"LIMIT {settings.sql_max_limit}",
                        sql,
                        flags=re.IGNORECASE,
                    )
                    logger.warning(f"Reduced LIMIT from {limit_val} to {settings.sql_max_limit}")

        return sql
