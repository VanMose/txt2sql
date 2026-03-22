"""Пайплайны."""
# Lazy imports для избежания циклических импортов
__all__ = [
    "PipelineState",
    "SQLAttempt",
    "Text2SQLPipeline",
    "PipelineResult",
]


def __getattr__(name):
    if name == "PipelineState":
        from .state import PipelineState
        return PipelineState
    elif name == "SQLAttempt":
        from .state import SQLAttempt
        return SQLAttempt
    elif name == "Text2SQLPipeline":
        from .pipeline import Text2SQLPipeline
        return Text2SQLPipeline
    elif name == "PipelineResult":
        from .pipeline import PipelineResult
        return PipelineResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
