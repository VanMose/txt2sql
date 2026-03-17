"""Пайплайны."""
# Lazy imports для избежания циклических импортов
__all__ = [
    "PipelineState",
    "SQLAttempt",
    "Text2SQLPipeline",
    "MultiDBPipeline",
]


def __getattr__(name):
    if name == "PipelineState":
        from .state import PipelineState
        return PipelineState
    elif name == "SQLAttempt":
        from .state import SQLAttempt
        return SQLAttempt
    elif name == "Text2SQLPipeline":
        from .text2sql_pipeline import Text2SQLPipeline
        return Text2SQLPipeline
    elif name == "MultiDBPipeline":
        from .langgraph_pipeline import MultiDBPipeline
        return MultiDBPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
