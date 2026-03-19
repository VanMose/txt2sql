"""
SQL Parser утилиты для анализа SQL запросов.

Использование:
    from utils.sql_parser import extract_tables, get_query_type
    
    tables = extract_tables("SELECT * FROM users")
    query_type = get_query_type("INSERT INTO users ...")
"""
import re
from typing import List, Set


def extract_tables(sql: str) -> List[str]:
    """
    Извлечь имена таблиц из SQL запроса.

    Args:
        sql: SQL запрос.

    Returns:
        Список имён таблиц.
    """
    if not sql or not sql.strip():
        return []

    tables: Set[str] = set()
    sql_upper = sql.upper()

    # Паттерн: FROM table_name
    from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    for match in re.finditer(from_pattern, sql, re.IGNORECASE):
        table_name = match.group(1).lower()
        if table_name not in ('select', 'where', 'group', 'order', 'having'):
            tables.add(table_name)

    # Паттерн: JOIN table_name
    join_pattern = r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    for match in re.finditer(join_pattern, sql, re.IGNORECASE):
        tables.add(match.group(1).lower())

    # Паттерн: INTO table_name
    into_pattern = r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    for match in re.finditer(into_pattern, sql, re.IGNORECASE):
        tables.add(match.group(1).lower())

    # Паттерн: UPDATE table_name
    update_pattern = r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    for match in re.finditer(update_pattern, sql, re.IGNORECASE):
        tables.add(match.group(1).lower())

    return sorted(list(tables))


def extract_columns(sql: str) -> List[str]:
    """
    Извлечь имена колонок из SQL запроса.

    Args:
        sql: SQL запрос.

    Returns:
        Список имён колонок.
    """
    if not sql or not sql.strip():
        return []

    columns: Set[str] = set()

    # Паттерн: SELECT column1, column2, ...
    select_pattern = r'\bSELECT\s+(.+?)\s+FROM\b'
    match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)

    if match:
        columns_str = match.group(1)

        # Проверка на *
        if '*' in columns_str:
            columns.add('*')

        # Извлечение имён колонок
        col_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)'
        for col_match in re.finditer(col_pattern, columns_str):
            col = col_match.group(1)
            # Исключение ключевых слов
            if col.upper() not in ('AS', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN'):
                columns.add(col.lower())

    # Паттерн: SET column = value
    set_pattern = r'\bSET\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*='
    for match in re.finditer(set_pattern, sql, re.IGNORECASE):
        columns.add(match.group(1).lower())

    # Паттерн: WHERE column
    where_pattern = r'\bWHERE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>]'
    for match in re.finditer(where_pattern, sql, re.IGNORECASE):
        columns.add(match.group(1).lower())

    return sorted(list(columns))


def get_query_type(sql: str) -> str:
    """
    Определить тип SQL запроса.

    Args:
        sql: SQL запрос.

    Returns:
        Тип запроса (SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, UNKNOWN).
    """
    if not sql or not sql.strip():
        return "UNKNOWN"

    sql_trimmed = sql.strip().upper()

    # Проверка на BEGIN/COMMIT/ROLLBACK
    if sql_trimmed.startswith("BEGIN"):
        return "BEGIN"
    if sql_trimmed.startswith("COMMIT"):
        return "COMMIT"
    if sql_trimmed.startswith("ROLLBACK"):
        return "ROLLBACK"

    # Проверка на основные типы запросов
    query_types = [
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "CREATE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "REPLACE",
    ]

    for qtype in query_types:
        if sql_trimmed.startswith(qtype):
            return qtype

    return "UNKNOWN"


def is_select_query(sql: str) -> bool:
    """
    Проверить, является ли запрос SELECT запросом.

    Args:
        sql: SQL запрос.

    Returns:
        True если SELECT запрос.
    """
    return get_query_type(sql).upper() == "SELECT"


def is_valid_sql(sql: str) -> bool:
    """
    Быстрая проверка валидности SQL.

    Args:
        sql: SQL запрос.

    Returns:
        True если запрос выглядит валидным.
    """
    if not sql or not sql.strip():
        return False

    sql_upper = sql.upper().strip()

    # Проверка на наличие ключевого слова
    valid_starts = [
        "SELECT", "INSERT", "UPDATE", "DELETE",
        "CREATE", "DROP", "ALTER", "TRUNCATE"
    ]

    for start in valid_starts:
        if sql_upper.startswith(start):
            return True

    return False


def has_dangerous_operations(sql: str) -> bool:
    """
    Проверить наличие опасных операций.

    Args:
        sql: SQL запрос.

    Returns:
        True если есть опасные операции.
    """
    if not sql or not sql.strip():
        return False

    sql_upper = sql.upper()

    dangerous = [
        "DROP ", "DELETE ", "TRUNCATE ",
        "ALTER ", "CREATE ", "REPLACE "
    ]

    for op in dangerous:
        if sql_upper.startswith(op):
            return True

    return False


def normalize_sql(sql: str) -> str:
    """
    Нормализовать SQL запрос (удалить лишние пробелы).

    Args:
        sql: SQL запрос.

    Returns:
        Нормализованный SQL.
    """
    if not sql or not sql.strip():
        return ""

    # Удаление лишних пробелов
    normalized = ' '.join(sql.split())

    return normalized


def count_tables(sql: str) -> int:
    """
    Подсчитать количество таблиц в запросе.

    Args:
        sql: SQL запрос.

    Returns:
        Количество таблиц.
    """
    return len(extract_tables(sql))


def count_columns(sql: str) -> int:
    """
    Подсчитать количество уникальных колонок в запросе.

    Args:
        sql: SQL запрос.

    Returns:
        Количество колонок.
    """
    columns = extract_columns(sql)
    return len([c for c in columns if c != '*'])
