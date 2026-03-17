"""Агенты пайплайна."""
# Избегаем циклических импортов - импортируем только по запросу
__all__ = [
    "SQLGenerator",
    "SQLValidator",
    "SQLJudge",
    "SQLRefiner",
    "RouterAgent",
    "DatabaseSelection",
    "QueryUnderstandingAgent",
    "QueryUnderstanding",
    "QueryIntent",
]

# Lazy imports
def __getattr__(name):
    if name == "SQLGenerator":
        from .sql_generator import SQLGenerator
        return SQLGenerator
    elif name == "SQLValidator":
        from .sql_validator import SQLValidator
        return SQLValidator
    elif name == "SQLJudge":
        from .sql_judge import SQLJudge
        return SQLJudge
    elif name == "SQLRefiner":
        from .sql_refiner import SQLRefiner
        return SQLRefiner
    elif name == "RouterAgent":
        from .router_agent import RouterAgent
        return RouterAgent
    elif name == "DatabaseSelection":
        from .router_agent import DatabaseSelection
        return DatabaseSelection
    elif name == "QueryUnderstandingAgent":
        from .query_understanding import QueryUnderstandingAgent
        return QueryUnderstandingAgent
    elif name == "QueryUnderstanding":
        from .query_understanding import QueryUnderstanding
        return QueryUnderstanding
    elif name == "QueryIntent":
        from .query_understanding import QueryIntent
        return QueryIntent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
