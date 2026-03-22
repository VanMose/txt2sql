# src\llm\prompt_loader.py
"""
Prompt Loader with Version Support.

Loads prompts from YAML config files with version selection.
Supports:
- Multiple prompt versions
- Database descriptions (EN/RU)
- Template variable injection
- Fallback to default version

Example:
    >>> from llm.prompt_loader import PromptLoader
    >>> loader = PromptLoader()
    >>> 
    >>> # Get SQL generator prompt (uses version from settings)
    >>> prompt = loader.get_sql_generator(query="Show songs", schema="Table: song...")
    >>> 
    >>> # Get specific version
    >>> prompt_v1 = loader.get_sql_generator(query="...", schema="...", version="v1")
    >>> 
    >>> # Get database description
    >>> desc = loader.get_db_description("music_1", language="en")
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Загрузчик промптов из YAML конфигов с поддержкой версий.
    
    Attributes:
        config_dir: Директория с конфигами промптов.
        prompts: Кэш загруженных промптов.
        db_descriptions: Кэш описаний баз данных.
    """
    
    def __init__(self, config_dir: Optional[str] = None) -> None:
        """
        Инициализировать загрузчик промптов.
        
        Args:
            config_dir: Директория с конфигами промптов.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "configs" / "prompts"
        self.config_dir = Path(config_dir)
        
        self.prompts: Dict[str, Dict[str, Any]] = {}
        self.db_descriptions: Dict[str, Dict[str, str]] = {}
        
        logger.info(f"PromptLoader initialized with config_dir: {self.config_dir}")
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Загрузить YAML файл.
        
        Args:
            filename: Имя файла.
            
        Returns:
            Загруженные данные.
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Prompt config not found: {filepath}")
            return {}
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        logger.debug(f"Loaded prompt config: {filename}")
        return data
    
    def _get_prompt_template(
        self,
        config_name: str,
        version: Optional[str] = None,
    ) -> Optional[str]:
        """
        Получить шаблон промпта по версии.
        
        Args:
            config_name: Имя конфига (без .yaml).
            version: Версия промпта. Если None, используется версия из settings.
            
        Returns:
            Шаблон промпта или None.
        """
        # Кэширование
        if config_name not in self.prompts:
            self.prompts[config_name] = self._load_yaml(f"{config_name}_prompts.yaml")
        
        config = self.prompts[config_name]
        
        if not config:
            return None
        
        # Определение версии
        if version is None:
            # Версия из settings
            version_key = f"{config_name}_prompt_version"
            settings = get_settings()
            version = getattr(settings, version_key, None)
        
        # Если версии нет в settings, используем default_version из конфига
        if version is None:
            version = config.get("default_version", "v1")
        
        # Получение промпта
        version_data = config.get(version)
        
        if version_data is None:
            logger.warning(
                f"Prompt version '{version}' not found in {config_name}. "
                f"Available versions: {[k for k in config.keys() if k != 'default_version']}"
            )
            # Fallback на default_version
            version_data = config.get(config.get("default_version", "v1"))
        
        if version_data is None:
            return None
        
        logger.debug(f"Using prompt: {config_name} version {version}")
        return version_data.get("template", "")
    
    def format_prompt(
        self,
        template: str,
        **kwargs: Any,
    ) -> str:
        """
        Форматировать промпт, подставив переменные.
        
        Args:
            template: Шаблон промпта.
            **kwargs: Переменные для подстановки.
            
        Returns:
            Отформатированный промпт.
        """
        if not template:
            return ""
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing prompt variable: {e}")
            return template
    
    # =========================================================================
    # Router Prompts
    # =========================================================================
    
    def get_router(
        self,
        query: str,
        databases_info: str,
        version: Optional[str] = None,
    ) -> str:
        """
        Получить промпт для Router Agent.
        
        Args:
            query: Запрос пользователя.
            databases_info: Информация о базах данных.
            version: Версия промпта.
            
        Returns:
            Отформатированный промпт.
        """
        template = self._get_prompt_template("router", version)
        
        if not template:
            # Fallback на хардкодный промпт
            template = """
You are a database router. Select relevant databases for the query.

## Available Databases
{databases_info}

## Question
{query}

## Output Format
{{"ranked_databases": [{{"db_name": "...", "tables": [...], "confidence": 0.9, "reason": "..."}}]}}

## Your Selection (JSON):
"""
        
        return self.format_prompt(
            template,
            query=query,
            databases_info=databases_info,
        )
    
    # =========================================================================
    # SQL Generator Prompts
    # =========================================================================
    
    def get_sql_generator(
        self,
        query: str,
        schema: str,
        version: Optional[str] = None,
    ) -> str:
        """
        Получить промпт для SQL Generator.
        
        Args:
            query: Запрос пользователя.
            schema: Схема базы данных.
            version: Версия промпта.
            
        Returns:
            Отформатированный промпт.
        """
        template = self._get_prompt_template("sql_generator", version)
        
        if not template:
            # Fallback на хардкодный промпт
            template = """
You are an expert SQL developer. Generate SQL query using the schema.

## Schema
{schema}

## Question
{query}

## Rules
1. Use EXACT table/column names from schema
2. Do NOT add values not in the question
3. Output: {{"sql": "query"}}

## Your Answer:
"""
        
        return self.format_prompt(
            template,
            query=query,
            schema=schema,
        )
    
    # =========================================================================
    # SQL Judge Prompts
    # =========================================================================

    def get_sql_judge(
        self,
        query: str,
        sql: str,
        schema: Optional[str] = None,  # 🔥 Добавлен параметр schema
        version: Optional[str] = None,
    ) -> str:
        """
        Получить промпт для SQL Judge.
        
        Production feature: schema-aware validation для case-sensitive проверки.

        Args:
            query: Запрос пользователя.
            sql: SQL запрос.
            schema: Схема БД для валидации (опционально).
            version: Версия промпта.

        Returns:
            Отформатированный промпт.
        """
        template = self._get_prompt_template("sql_judge", version)

        if not template:
            # Fallback на хардкодный промпт
            template = """
You are an SQL evaluator. Rate how well the query answers the question.

## Question
{query}

## SQL Query
{sql}

## Output Format
{{"confidence": 0.0-1.0, "reason": "explanation"}}

## Your Evaluation:
"""

        # 🔥 Если схема передана, добавляем её в промпт
        if schema:
            return self.format_prompt(
                template,
                query=query,
                sql=sql,
                schema=schema,
            )
        else:
            return self.format_prompt(
                template,
                query=query,
                sql=sql,
            )
    
    # =========================================================================
    # SQL Refiner Prompts
    # =========================================================================
    
    def get_sql_refiner(
        self,
        query: str,
        schema: str,
        history: str,
        version: Optional[str] = None,
    ) -> str:
        """
        Получить промпт для SQL Refiner.
        
        Args:
            query: Запрос пользователя.
            schema: Схема базы данных.
            history: История предыдущих попыток.
            version: Версия промпта.
            
        Returns:
            Отформатированный промпт.
        """
        template = self._get_prompt_template("sql_refiner", version)
        
        if not template:
            # Fallback на хардкодный промпт
            template = """
You are an SQL debugger. Fix the query based on previous errors.

## Schema
{schema}

## Question
{query}

## Previous Attempts
{history}

## Output Format
{{"sql": "corrected SQL", "fixes_applied": [...], "explanation": "..."}}

## Your Response:
"""
        
        return self.format_prompt(
            template,
            query=query,
            schema=schema,
            history=history,
        )
    
    # =========================================================================
    # Database Descriptions
    # =========================================================================
    
    def _load_db_descriptions(self) -> None:
        """Загрузить описания баз данных."""
        if self.db_descriptions:
            return
        
        data = self._load_yaml("database_descriptions.yaml")
        
        if not data:
            logger.warning("No database descriptions found")
            return
        
        for db_name, db_data in data.items():
            if db_name == "default_version":
                continue
            
            self.db_descriptions[db_name] = {
                "name": db_data.get("name", db_name),
                "en": db_data.get("en", ""),
                "ru": db_data.get("ru", ""),
            }
        
        logger.info(f"Loaded descriptions for {len(self.db_descriptions)} databases")
    
    def get_db_description(
        self,
        db_name: str,
        language: str = "en",
        version: Optional[str] = None,
    ) -> str:
        """
        Получить описание базы данных.
        
        Args:
            db_name: Имя базы данных.
            language: Язык описания ("en" или "ru").
            version: Версия описания (пока не используется, зарезервировано).
            
        Returns:
            Описание базы данных или пустая строка.
        """
        self._load_db_descriptions()
        
        if db_name not in self.db_descriptions:
            logger.debug(f"No description found for database: {db_name}")
            return ""
        
        db_data = self.db_descriptions[db_name]
        description = db_data.get(language, db_data.get("en", ""))
        
        logger.debug(f"Got {language} description for {db_name}")
        return description
    
    def get_all_db_descriptions(
        self,
        language: str = "en",
    ) -> Dict[str, str]:
        """
        Получить все описания баз данных.
        
        Args:
            language: Язык описания.
            
        Returns:
            Словарь {db_name: description}.
        """
        self._load_db_descriptions()
        
        return {
            db_name: db_data.get(language, db_data.get("en", ""))
            for db_name, db_data in self.db_descriptions.items()
        }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_available_versions(self, prompt_type: str) -> List[str]:
        """
        Получить доступные версии промпта.
        
        Args:
            prompt_type: Тип промпта ("router", "sql_generator", "sql_judge", "sql_refiner").
            
        Returns:
            Список версий.
        """
        config = self._load_yaml(f"{prompt_type}_prompts.yaml")
        
        if not config:
            return []
        
        versions = [k for k in config.keys() if k != "default_version"]
        return versions
    
    def get_prompt_info(self, prompt_type: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Получить информацию о промпте.
        
        Args:
            prompt_type: Тип промпта.
            version: Версия промпта.
            
        Returns:
            Информация о промпте.
        """
        config = self._load_yaml(f"{prompt_type}_prompts.yaml")
        
        if not config:
            return {"error": "Config not found"}
        
        if version is None:
            version = config.get("default_version", "v1")
        
        version_data = config.get(version, {})
        
        return {
            "version": version,
            "description": version_data.get("description", ""),
            "template_length": len(version_data.get("template", "")),
            "available_versions": [k for k in config.keys() if k != "default_version"],
        }
    
    def clear_cache(self) -> None:
        """Очистить кэш."""
        self.prompts.clear()
        self.db_descriptions.clear()
        logger.info("PromptLoader cache cleared")


# Singleton instance
_prompt_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """
    Получить singleton экземпляр PromptLoader.
    
    Returns:
        PromptLoader экземпляр.
    """
    global _prompt_loader
    
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    
    return _prompt_loader
