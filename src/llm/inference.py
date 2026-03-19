# src\llm\inference.py
"""
Оптимизированный LLM сервис с Ollama backend.

Оптимизации:
1. Ollama API (быстрый llama.cpp backend)
2. Retry with exponential backoff
3. Rate limiting
4. Batch generation
5. Metrics tracking

Использование:
    from llm.inference import LLMService

    service = LLMService()
    outputs = service.generate(prompt, n=5)
"""
import logging
from typing import Any, List, Optional

from ..config.settings import get_settings
from ..utils.rate_limiter import llm_rate_limiter, RateLimitExceeded
from ..utils.retry import retry_with_backoff, llm_retry_executor
from .model_loader import LLMBackend, ModelLoader
from .prompts import Prompts

logger = logging.getLogger(__name__)


# Экспорт для тестов
from .model_loader import MockSamplingParams  # noqa: F401


class LLMService:
    """
    Оптимизированный сервис для генерации текста.

    Features:
    - Ollama API (primary backend)
    - Retry с exponential backoff
    - Rate limiting
    - Batch generation
    - Metrics tracking
    """

    def __init__(
        self,
        force_backend: Optional[LLMBackend] = None,
        use_retry: bool = True,
        use_rate_limit: bool = True,
    ) -> None:
        """
        Инициализировать сервис.

        Args:
            force_backend: Принудительно выбрать backend.
            use_retry: Использовать retry logic.
            use_rate_limit: Использовать rate limiting.
        """
        settings = get_settings()
        
        # Приоритет: Ollama > vLLM > Transformers > Mock
        if settings.use_ollama:
            self.backend = LLMBackend.OLLAMA
            self.ollama_service = self._init_ollama(settings)
        else:
            self.backend = ModelLoader.get_backend()
            if force_backend:
                self.backend = force_backend
            self.ollama_service = None

        self.use_retry = use_retry
        self.use_rate_limit = use_rate_limit

        logger.info(f"LLMService initialized with backend: {self.backend.value}")

    def _init_ollama(self, settings: Any) -> Any:
        """Инициализировать Ollama сервис."""
        try:
            from .ollama_service import OllamaService
            
            return OllamaService(
                model_name=settings.llm_model,
                base_url=settings.ollama_base_url,
                timeout=120,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            logger.warning("Falling back to Transformers/Mock backend")
            return None

    @retry_with_backoff(
        max_retries=3,
        base_delay=0.5,
        max_delay=10.0,
        jitter=True,
        retryable_exceptions=(RateLimitExceeded, RuntimeError),
    )
    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        use_rate_limit: Optional[bool] = None,
    ) -> List[str]:
        """
        Сгенерировать текст по промпту.

        Args:
            prompt: Входной промпт.
            n: Количество сэмплов.
            temperature: Температура генерации.
            use_rate_limit: Использовать rate limiting.

        Returns:
            Список сгенерированных текстов.
        """
        # Rate limiting
        if use_rate_limit is None:
            use_rate_limit = self.use_rate_limit

        if use_rate_limit:
            if not llm_rate_limiter.acquire(blocking=True, timeout=60.0):
                raise RateLimitExceeded("Rate limit exceeded after timeout")

        settings = get_settings()

        # Ollama backend (primary)
        if self.backend == LLMBackend.OLLAMA and self.ollama_service:
            return self._generate_ollama(prompt, n, temperature, settings)
        # vLLM backend
        elif self.backend == LLMBackend.VLLM:
            return self._generate_vllm(prompt, n, temperature, settings)
        # Transformers backend
        elif self.backend == LLMBackend.TRANSFORMERS:
            return self._generate_transformers(prompt, n, temperature, settings)
        # Mock backend (fallback)
        else:
            return self._generate_mock(prompt, n, temperature, settings)

    def generate_batch(
        self,
        prompts: List[str],
        n: int = 1,
        temperature: float = 0.7,
        batch_size: int = 5,
    ) -> List[List[str]]:
        """
        Batch генерация для нескольких промптов.

        Args:
            prompts: Список промптов.
            n: Количество сэмплов на промпт.
            temperature: Температура.
            batch_size: Размер батча.

        Returns:
            Список списков сгенерированных текстов.
        """
        results: List[List[str]] = []

        # Ollama batch generation
        if self.backend == LLMBackend.OLLAMA and self.ollama_service:
            return self.ollama_service.generate_batch(
                prompts=prompts,
                n=n,
                temperature=temperature,
                max_tokens=get_settings().max_tokens,
            )

        # Обработка батчами для других backend'ов
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            # Batch генерация через transformers
            if self.backend == LLMBackend.TRANSFORMERS:
                from .transformers_service import TransformersService

                batch_results = TransformersService.generate(
                    prompts=batch_prompts,
                    max_tokens=get_settings().max_tokens,
                    temperature=temperature,
                    n=n,
                )

                # Разделение результатов
                for j in range(len(batch_prompts)):
                    start_idx = j * n
                    end_idx = start_idx + n
                    results.append(batch_results[start_idx:end_idx])
            else:
                # Последовательная генерация для других backend'ов
                for prompt in batch_prompts:
                    result = self.generate(prompt, n=n, temperature=temperature)
                    results.append(result)

        return results

    def _generate_ollama(
        self,
        prompt: str,
        n: int,
        temperature: float,
        settings: Any,
    ) -> List[str]:
        """Генерация через Ollama."""
        if not self.ollama_service:
            logger.error("Ollama service not initialized")
            return []
        
        try:
            texts = self.ollama_service.generate(
                prompt=prompt,
                n=n,
                temperature=temperature,
                max_tokens=settings.max_tokens,
            )
            logger.info(f"Ollama generated {len(texts)} outputs")
            return texts
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return []

    def _generate_vllm(
        self,
        prompt: str,
        n: int,
        temperature: float,
        settings: Any,
    ) -> List[str]:
        """Генерация через vLLM."""
        try:
            from vllm import SamplingParams

            model = ModelLoader.get_model()
            params = SamplingParams(
                temperature=temperature,
                max_tokens=settings.max_tokens,
                n=n,
            )
            outputs = model.generate([prompt], params)

            texts = [o.text for output in outputs for o in output.outputs]
            logger.info(f"vLLM generated {len(texts)} outputs")
            return texts

        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            return []

    def _generate_transformers(
        self,
        prompt: str,
        n: int,
        temperature: float,
        settings: Any,
    ) -> List[str]:
        """Генерация через Transformers."""
        from .transformers_service import TransformersService

        texts = TransformersService.generate(
            prompts=[prompt],
            max_tokens=settings.max_tokens,
            temperature=temperature,
            n=n,
        )
        logger.info(f"Transformers generated {len(texts)} outputs")
        return texts

    def _generate_mock(
        self,
        prompt: str,
        n: int,
        temperature: float,
        settings: Any,
    ) -> List[str]:
        """Генерация через Mock."""
        model = ModelLoader.get_model()

        from ..config.settings import MockSamplingParams

        params = MockSamplingParams(
            temperature=temperature,
            max_tokens=settings.max_tokens,
            n=n,
        )
        outputs = model.generate([prompt], params)

        texts = [o.text for output in outputs for o in output.outputs]
        logger.info(f"Mock generated {len(texts)} outputs")
        return texts

    def generate_with_retry(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
    ) -> List[str]:
        """
        Генерация с retry logic.

        Args:
            prompt: Промпт.
            n: Количество сэмплов.
            temperature: Температура.

        Returns:
            Список текстов.
        """
        return llm_retry_executor.execute(
            self.generate,
            prompt,
            n=n,
            temperature=temperature,
            use_rate_limit=True,
        )

    def get_stats(self) -> dict:
        """Получить статистику."""
        stats = {
            "backend": self.backend.value,
            "retry_enabled": self.use_retry,
            "rate_limit_enabled": self.use_rate_limit,
        }
        
        if self.backend == LLMBackend.OLLAMA and self.ollama_service:
            stats.update(self.ollama_service.get_stats())
        
        return stats
