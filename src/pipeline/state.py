# src\pipeline\state.py
"""
State для передачи данных между агентами пайплайна.

Определяет структуры данных для обмена информацией между этапами:
- SQLAttempt - результат одной попытки генерации SQL
- PipelineState - полное состояние пайплайна

Example:
    >>> from pipeline.state import PipelineState, SQLAttempt
    >>> state = PipelineState(query="Show movies")
    >>> attempt = SQLAttempt(sql="SELECT * FROM Movie", confidence=0.9, error=False, reason="OK")
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SQLAttempt:
    """
    Результат одной попытки генерации SQL.

    Attributes:
        sql: SQL запрос.
        confidence: Уровень уверенности (0.0-1.0).
        error: Флаг ошибки.
        reason: Описание результата/ошибки.
        execution_result: Результат выполнения SQL.
    """
    sql: str
    confidence: float
    error: bool
    reason: str
    execution_result: Optional[Any] = None


@dataclass
class PipelineState:
    """
    Единый state для передачи данных между этапами пайплайна.

    Pipeline:
        User Query → Schema Retrieval → SQL Generator → Validator →
        Executor → Judge → Best Selector → Refiner → Final SQL

    Attributes:
        query: Входной запрос пользователя.
        schema: Схема БД (полная или отфильтрованная).
        relevant_tables: Релевантные таблицы после retrieval.
        attempts: Все попытки генерации SQL.
        best_sql: Лучший SQL запрос.
        execution_results: Результаты выполнения SQL.
        needs_refinement: Флаг необходимости рефайнмента.
        refinement_history: История попыток для рефайнмента.
        retry_count: Количество попыток рефайнмента.
        latencies: Метрики времени выполнения этапов.
    """
    query: str = ""
    schema: Optional[str] = None
    relevant_tables: List[str] = field(default_factory=list)
    attempts: List[SQLAttempt] = field(default_factory=list)
    best_sql: Optional[str] = None
    execution_results: Optional[Any] = None
    needs_refinement: bool = False
    refinement_history: str = ""
    retry_count: int = 0
    latencies: Dict[str, float] = field(default_factory=dict)
