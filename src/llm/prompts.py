# src\llm\prompts.py
"""
Prompts for LLM agents.

This module now uses PromptLoader for versioned prompts from config files.
The old hardcoded prompts are kept as fallbacks.

Example:
    >>> from llm.prompts import Prompts
    >>>
    >>> # Using PromptLoader (recommended)
    >>> prompt = Prompts.get_sql_generator(query="Show songs", schema="Table: song...")
    >>>
    >>> # Specify version
    >>> prompt_v1 = Prompts.get_sql_generator(query="...", schema="...", version="v1")
"""
import logging
from typing import Any, Dict, List, Optional

from .prompt_loader import get_prompt_loader

logger = logging.getLogger(__name__)


class Prompts:
    """
    Коллекция промптов для различных агентов.

    Uses PromptLoader for versioned prompts from YAML configs.
    Falls back to hardcoded prompts if configs are not available.
    """

    # Lazy-loaded PromptLoader
    _loader = None

    @classmethod
    def _get_loader(cls) -> "PromptLoader":
        """Get PromptLoader singleton."""
        if cls._loader is None:
            cls._loader = get_prompt_loader()
        return cls._loader

    # =========================================================================
    # Router Prompts
    # =========================================================================

    @classmethod
    def format_router(
        cls,
        query: str,
        databases: List[Dict[str, Any]],
        version: Optional[str] = None,
    ) -> str:
        """
        Сформировать промпт для Router Agent.

        Args:
            query: Запрос.
            databases: Список баз данных.
            version: Версия промпта (None = из settings).

        Returns:
            Промпт.
        """
        # Format databases info
        db_info = "".join(
            f"\n### Database: {db['db_name']}\n"
            f"Path: {db['db_path']}\n"
            f"Tables: {', '.join(db['tables'])}\n"
            f"Relevance: {db['avg_score']:.3f}\n\n"
            for db in databases
        )

        loader = cls._get_loader()
        return loader.get_router(
            query=query,
            databases_info=db_info,
            version=version,
        )

    # =========================================================================
    # SQL Generator Prompts
    # =========================================================================

    @classmethod
    def format_sql_generator(
        cls,
        query: str,
        schema: str,
        version: Optional[str] = None,
    ) -> str:
        """
        Сформировать промпт для генерации SQL.

        Args:
            query: Запрос.
            schema: Схема.
            version: Версия промпта (None = из settings).

        Returns:
            Промпт.
        """
        loader = cls._get_loader()
        return loader.get_sql_generator(
            query=query,
            schema=schema,
            version=version,
        )

    # =========================================================================
    # SQL Judge Prompts
    # =========================================================================

    @classmethod
    def format_sql_judge(
        cls,
        query: str,
        sql: str,
        schema: Optional[str] = None,  # 🔥 Добавлен параметр schema
        version: Optional[str] = None,
    ) -> str:
        """
        Сформировать промпт для judge.

        Args:
            query: Запрос.
            sql: SQL запрос.
            schema: Схема БД для case-sensitive валидации (опционально).
            version: Версия промпта (None = из settings).

        Returns:
            Промпт.
        """
        loader = cls._get_loader()
        return loader.get_sql_judge(
            query=query,
            sql=sql,
            schema=schema,  # 🔥 Передаём схему
            version=version,
        )

    # =========================================================================
    # SQL Refiner Prompts
    # =========================================================================

    @classmethod
    def format_sql_refiner(
        cls,
        query: str,
        schema: str,
        history: str,
        version: Optional[str] = None,
    ) -> str:
        """
        Сформировать промпт для рефайнмента.

        Args:
            query: Запрос.
            schema: Схема.
            history: История попыток.
            version: Версия промпта (None = из settings).

        Returns:
            Промпт.
        """
        loader = cls._get_loader()
        return loader.get_sql_refiner(
            query=query,
            schema=schema,
            history=history,
            version=version,
        )

    # =========================================================================
    # Database Descriptions
    # =========================================================================

    @classmethod
    def get_db_description(
        cls,
        db_name: str,
        language: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """
        Получить описание базы данных.

        Args:
            db_name: Имя базы данных.
            language: Язык (en/ru). None = из settings.
            version: Версия описания. None = из settings.

        Returns:
            Описание базы данных.
        """
        from ..config.settings import get_settings

        if language is None:
            settings = get_settings()
            language = settings.db_description_language

        loader = cls._get_loader()
        return loader.get_db_description(
            db_name=db_name,
            language=language,
            version=version,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @classmethod
    def get_available_versions(cls, prompt_type: str) -> List[str]:
        """
        Получить доступные версии промпта.

        Args:
            prompt_type: Тип промпта ("router", "sql_generator", "sql_judge", "sql_refiner").

        Returns:
            Список версий.
        """
        loader = cls._get_loader()
        return loader.get_available_versions(prompt_type)

    @classmethod
    def get_prompt_info(cls, prompt_type: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Получить информацию о промпте.

        Args:
            prompt_type: Тип промпта.
            version: Версия промпта.

        Returns:
            Информация о промпте.
        """
        loader = cls._get_loader()
        return loader.get_prompt_info(prompt_type, version)

    @classmethod
    def add_few_shot_examples(
        cls,
        prompt: str,
        examples: List[Dict[str, str]],
    ) -> str:
        """
        Добавить few-shot примеры.

        Args:
            prompt: Основной промпт.
            examples: Примеры [{"question": "...", "sql": "..."}].

        Returns:
            Промпт с примерами.
        """
        if not examples:
            return prompt

        examples_text = "## Examples\n\n"
        for i, ex in enumerate(examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"  Q: {ex['question']}\n"
            examples_text += f"  A: {ex['sql']}\n\n"

        return examples_text + prompt

    @classmethod
    def format_error_recovery(
        cls,
        query: str,
        schema: str,
        error_sql: str,
        error_message: str,
    ) -> str:
        """
        Сформировать промпт для восстановления после ошибки.

        Args:
            query: Запрос.
            schema: Схема.
            error_sql: Ошибочный SQL.
            error_message: Сообщение об ошибке.

        Returns:
            Промпт.
        """
        return f"""
You are an SQL error recovery expert.

## Question
{query}

## Schema
{schema}

## Failed SQL
{error_sql}

## Error Message
{error_message}

## Task
Generate corrected SQL.

## Output Format
{{"corrected_sql": "...", "error_analysis": "...", "fix_description": "..."}}

## Corrected SQL (JSON)
"""
