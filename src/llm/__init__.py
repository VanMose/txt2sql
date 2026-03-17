"""LLM сервисы."""
from .model_loader import ModelLoader, LLMBackend, VLLM_AVAILABLE, TRANSFORMERS_AVAILABLE

__all__ = [
    "ModelLoader",
    "LLMBackend",
    "LLMService",
    "Prompts",
    "TransformersService",
    "VLLM_AVAILABLE",
    "TRANSFORMERS_AVAILABLE",
]


def __getattr__(name: str):
    """Ленивый импорт для избежания циклических зависимостей."""
    if name == "LLMService":
        from .inference import LLMService
        return LLMService
    elif name == "Prompts":
        from .prompts import Prompts
        return Prompts
    elif name == "TransformersService":
        from .transformers_service import TransformersService
        return TransformersService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
