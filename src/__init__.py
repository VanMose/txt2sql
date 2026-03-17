"""
Text-to-SQL Pipeline - Оптимизированная версия.

Модульная архитектура:
- config: Настройки через Pydantic
- db: Database компоненты (SchemaLoader, Executor)
- llm: LLM сервисы с retry и rate limiting
- retrieval: Vector DB, Graph DB, Schema Retrieval
- agents: SQL Generator, Validator, Judge, Refiner, Router
- pipeline: Оркестрация пайплайнов
- services: Service слой (PipelineService, Metrics)
- utils: Утилиты (кэширование, retry, rate limiting)

Example:
    from src.services import PipelineService, metrics
    
    with PipelineService(db_paths=["db.sqlite"]) as service:
        result = service.run_query("Show movies")
        print(result.sql)
        print(metrics.get_summary())
"""
__version__ = "2.0.0"
__author__ = "Text2SQL Team"
