# src\llm\model_loader.py
"""
Загрузчик LLM моделей с оптимизациями.

Оптимизации:
1. 4-bit квантизация (BitsAndBytes)
2. Torch Compile (torch.compile)
3. Flash Attention 2 (если доступна)
4. Model Warm-up (прогрев при инициализации)
5. Prompt Caching (кэширование префиксов)

Модуль предоставляет автоматическое определение доступного backend:
- vLLM (Linux с GPU, максимальная производительность)
- Transformers (кроссплатформенный, CPU/GPU)
- Mock (тестовый режим, без зависимостей)

Example:
    >>> from llm.model_loader import ModelLoader
    >>> model = ModelLoader.get_model()
    >>> outputs = model.generate(["prompt"], params)
"""
import logging
import platform
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class LLMBackend(Enum):
    """Типы LLM backend."""
    OLLAMA = "ollama"
    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    MOCK = "mock"


# Определяем доступность vLLM (только Linux)
try:
    from vllm import LLM as VLLMModel
    VLLM_AVAILABLE = True
    logger.info("vLLM is available")
except (ImportError, ModuleNotFoundError):
    VLLM_AVAILABLE = False
    logger.info("vLLM is not available (Windows/Mac or missing dependencies)")

# Определяем доступность Transformers
try:
    import transformers
    import torch
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers is available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.info("Transformers is not available")

# Определяем доступность Flash Attention 2
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
    logger.info("Flash Attention 2 is available")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logger.info("Flash Attention 2 is not installed")

# Определяем доступность BitsAndBytes (4-bit квантизация)
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
    logger.info("BitsAndBytes is available (4-bit quantization)")
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.info("BitsAndBytes is not installed (4-bit quantization unavailable)")


def detect_platform() -> str:
    """
    Определить платформу.

    Returns:
        Название платформы: 'linux', 'windows', 'macos', 'unknown'.
    """
    system = platform.system().lower()
    platform_map = {"linux": "linux", "windows": "windows", "darwin": "macos"}
    return platform_map.get(system, "unknown")


def get_recommended_backend() -> LLMBackend:
    """
    Получить рекомендуемый backend на основе платформы.

    Returns:
        Рекомендуемый LLMBackend.
    """
    plat = detect_platform()
    logger.info(f"Detected platform: {plat}")

    if plat == "linux" and VLLM_AVAILABLE:
        return LLMBackend.VLLM
    if TRANSFORMERS_AVAILABLE:
        return LLMBackend.TRANSFORMERS
    return LLMBackend.MOCK


class MockSamplingParams:
    """Mock SamplingParams для Windows/Mac."""

    def __init__(self, temperature: float = 0.2, max_tokens: int = 256, n: int = 1, **kwargs: Any) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockOutput:
    """Mock output для совместимости."""

    def __init__(self, text: str) -> None:
        self.text = text


class MockRequestOutput:
    """Mock request output для совместимости."""

    def __init__(self, outputs: List[MockOutput]) -> None:
        self.outputs = outputs


class MockLLM:
    """
    Заглушка LLM для тестирования.

    Возвращает тестовые SQL запросы в JSON формате на основе ключевых слов в промпте.
    """

    def __init__(self, model: str, gpu_memory_utilization: float = 0.9, **kwargs: Any) -> None:
        self.model = model
        logger.info(f"[MockLLM] Initialized with model: {model}")

    def generate(self, prompts: List[str], params: MockSamplingParams) -> List[MockRequestOutput]:
        """
        Генерирует mock ответы в JSON формате.

        Args:
            prompts: Список промптов.
            params: Параметры генерации.

        Returns:
            Список output объектов.
        """
        outputs = []

        for prompt in prompts:
            n = getattr(params, 'n', 1)
            for _ in range(n):
                mock_sql, tables_used = self._generate_mock_sql(prompt)
                json_response = self._create_json_response(mock_sql, tables_used)
                outputs.append(MockOutput(json_response))

        return [MockRequestOutput(outputs)]

    def _generate_mock_sql(self, prompt: str) -> tuple:
        """Сгенерировать mock SQL на основе промпта."""
        prompt_lower = prompt.lower()

        if "movie" in prompt_lower or "фильм" in prompt_lower:
            return "SELECT * FROM Movie LIMIT 10", ["Movie"]
        elif "count" in prompt_lower or "количеств" in prompt_lower:
            return "SELECT COUNT(*) FROM Movie", ["Movie"]
        elif "review" in prompt_lower or "оценк" in prompt_lower or "rating" in prompt_lower:
            return "SELECT * FROM Rating LIMIT 10", ["Rating"]
        elif "all" in prompt_lower or "все" in prompt_lower or "show" in prompt_lower or "показать" in prompt_lower:
            return "SELECT * FROM Movie", ["Movie"]
        elif "max" in prompt_lower or "максимальн" in prompt_lower:
            return "SELECT MAX(stars) FROM Rating", ["Rating"]

        return "SELECT * FROM test_table LIMIT 10", ["test_table"]

    def _create_json_response(self, sql: str, tables: List[str]) -> str:
        """Создать JSON ответ."""
        import json
        return json.dumps({
            "sql": sql,
            "tables_used": tables,
            "explanation": f"Generated SQL using table(s): {', '.join(tables)}"
        }, ensure_ascii=False)


class PromptCache:
    """
    Кэширование префиксов промптов.

    Кэширует общие части промптов (schema, инструкции)
    для ускорения повторных запросов.
    """

    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        logger.info(f"PromptCache initialized: max_size={max_size}")

    def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша."""
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            logger.debug(f"PromptCache HIT: {key[:30]}...")
            return self._cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Положить значение в кэш."""
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = value
        self._access_order.append(key)
        logger.debug(f"PromptCache PUT: {key[:30]}...")

    def clear(self) -> None:
        """Очистить кэш."""
        self._cache.clear()
        self._access_order.clear()
        logger.info("PromptCache cleared")


class OptimizedModel:
    """
    Обёртка для оптимизированной модели.

    Применяет:
    - 4-bit квантизацию
    - Torch compile
    - Flash Attention 2
    """

    def __init__(
        self,
        model: Any,
        use_4bit: bool = True,
        use_torch_compile: bool = True,
        use_flash_attention: bool = False,
    ) -> None:
        self.model = model
        self.use_4bit = use_4bit and BITSANDBYTES_AVAILABLE
        self.use_torch_compile = use_torch_compile
        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE

        # Применяем оптимизации
        self._apply_optimizations()

        logger.info(
            f"OptimizedModel: 4bit={self.use_4bit}, "
            f"torch_compile={self.use_torch_compile}, "
            f"flash_attn={self.use_flash_attention}"
        )

    def _apply_optimizations(self) -> None:
        """Применить оптимизации к модели."""
        if not TRANSFORMERS_AVAILABLE:
            return

        # 4-bit квантизация уже применена при загрузке
        # Torch compile применяется к forward методу
        if self.use_torch_compile and hasattr(self.model, 'forward'):
            try:
                self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead")
                logger.info("Torch compile applied to model.forward")
            except Exception as e:
                logger.warning(f"Torch compile failed: {e}")

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Генерация через модель."""
        return self.model.generate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Делегирование атрибутов модели."""
        return getattr(self.model, name)


class ModelLoader:
    """
    Singleton для загрузки LLM модели с оптимизациями.

    Автоопределение backend:
    - vLLM (Linux с GPU) - максимальная производительность
    - Transformers (Windows/Mac/Linux) - кроссплатформенный
    - Mock (fallback) - для тестирования

    Оптимизации:
    - 4-bit квантизация (BitsAndBytes)
    - Torch compile (torch.compile)
    - Flash Attention 2
    - Model warm-up
    - Prompt caching

    Attributes:
        _model: Кэш модели.
        _backend: Текущий backend.
        _prompt_cache: Кэш промптов.
        _warmed_up: Флаг warm-up.
    """

    _model: Optional[Any] = None
    _backend: Optional[LLMBackend] = None
    _prompt_cache: Optional[PromptCache] = None
    _warmed_up: bool = False
    _current_model_name: Optional[str] = None


    @classmethod
    def get_model(
        cls,
        model_name: Optional[str] = None,
        force_backend: Optional[LLMBackend] = None,
        use_4bit: bool = True,
        use_torch_compile: bool = True,
        use_flash_attention: bool = False,
    ) -> Any:
        """
        Получить экземпляр модели (ленивая загрузка).

        Args:
            model_name: Имя модели. Если None, используется из настроек.
            force_backend: Принудительно выбрать backend.
            use_4bit: Использовать 4-битную квантизацию.
            use_torch_compile: Использовать torch.compile.
            use_flash_attention: Использовать Flash Attention 2.

        Returns:
            Экземпляр LLM.
        """
        if cls._model is None:
            from ..config.settings import get_settings
            settings = get_settings()

            model_path = model_name or settings.llm_model
            cls._current_model_name = model_name or settings.llm_model
            if settings.use_local_model:
                model_path = settings.get_local_model_path(model_name or settings.llm_model)

            cls._backend = force_backend or get_recommended_backend()
            logger.info(f"Using backend: {cls._backend.value}")

            # Переопределяем настройки из параметров
            use_4bit = use_4bit and settings.use_4bit_quantization
            use_torch_compile = use_torch_compile and settings.use_torch_compile
            use_flash_attention = use_flash_attention and settings.use_flash_attention

            if cls._backend == LLMBackend.VLLM:
                cls._model = VLLMModel(
                    model=model_path,
                    gpu_memory_utilization=settings.gpu_memory_utilization,
                    trust_remote_code=True
                )
            elif cls._backend == LLMBackend.TRANSFORMERS:
                from .transformers_service import TransformersService
                cls._model = TransformersService.load_model(
                    model_path=model_path,
                    gpu_memory_utilization=settings.gpu_memory_utilization,
                    use_4bit=use_4bit,
                    use_flash_attention=use_flash_attention,
                )
                # Применяем torch.compile
                if use_torch_compile:
                    cls._model = OptimizedModel(
                        cls._model,
                        use_4bit=use_4bit,
                        use_torch_compile=use_torch_compile,
                        use_flash_attention=use_flash_attention,
                    )
                logger.info(f"Loaded Transformers model: {model_path}")
            else:
                cls._model = MockLLM(
                    model=model_path,
                    gpu_memory_utilization=settings.gpu_memory_utilization
                )

            # Инициализация prompt cache
            if settings.use_prompt_cache:
                cls._prompt_cache = PromptCache()

        return cls._model

    @classmethod
    def reload(cls, model_name: str, force_backend: Optional[LLMBackend] = None) -> Any:
        """
        Перезагрузить модель.

        Args:
            model_name: Имя новой модели.
            force_backend: Принудительно выбрать backend.

        Returns:
            Новый экземпляр LLM.
        """
        cls._model = None
        cls._warmed_up = False
        if cls._backend == LLMBackend.TRANSFORMERS:
            from .transformers_service import TransformersService
            TransformersService.clear_cache()
        if cls._prompt_cache:
            cls._prompt_cache.clear()
        return cls.get_model(model_name, force_backend)

    @classmethod
    def warmup(cls, sample_prompts: Optional[List[str]] = None) -> bool:
        """
        Прогреть модель перед первым использованием.

        Args:
            sample_prompts: Тестовые промпты для warm-up.

        Returns:
            True если warm-up успешен.
        """
        if cls._warmed_up:
            logger.debug("Model already warmed up")
            return True

        # Загружаем модель если не загружена
        if cls._model is None:
            logger.info("Loading model for warm-up...")
            cls.get_model()  # Загружаем модель
            
        if cls._model is None:
            logger.warning("Cannot warm-up: model not loaded")
            return False

        if cls._backend == LLMBackend.MOCK:
            logger.info("Mock model: warm-up skipped")
            cls._warmed_up = True
            return True

        sample_prompts = sample_prompts or [
            "SELECT * FROM test",
            "Покажи все таблицы",
            "Count all records",
        ]

        logger.info(f"Warming up model with {len(sample_prompts)} sample prompts...")

        try:
            for prompt in sample_prompts:
                if cls._backend == LLMBackend.VLLM:
                    from vllm import SamplingParams
                    params = SamplingParams(temperature=0.1, max_tokens=50)
                    cls._model.generate([prompt], params)
                elif cls._backend == LLMBackend.TRANSFORMERS:
                    from .transformers_service import TransformersService
                    TransformersService.generate(
                        prompts=[prompt],
                        max_tokens=50,
                        temperature=0.1,
                        n=1,
                    )

            cls._warmed_up = True
            logger.info("Model warm-up completed successfully")
            return True

        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
            return False

    @classmethod
    def get_prompt_cache(cls) -> Optional[PromptCache]:
        """Получить кэш промптов."""
        return cls._prompt_cache

    @classmethod
    def is_warmed_up(cls) -> bool:
        """Проверить, прогрета ли модель."""
        return cls._warmed_up

    @classmethod
    def get_backend(cls) -> LLMBackend:
        """Получить текущий backend."""
        if cls._backend is None:
            cls._backend = get_recommended_backend()
        return cls._backend

    @classmethod
    def is_vllm_available(cls) -> bool:
        """Проверить доступность vLLM."""
        return VLLM_AVAILABLE

    @classmethod
    def is_transformers_available(cls) -> bool:
        """Проверить доступность Transformers."""
        return TRANSFORMERS_AVAILABLE

    @classmethod
    def is_flash_attention_available(cls) -> bool:
        """Проверить доступность Flash Attention 2."""
        return FLASH_ATTN_AVAILABLE

    @classmethod
    def is_bitsandbytes_available(cls) -> bool:
        """Проверить доступность BitsAndBytes."""
        return BITSANDBYTES_AVAILABLE

    @classmethod
    def get_platform_info(cls) -> Dict[str, Any]:
        """
        Получить информацию о платформе.

        Returns:
            Информация о платформе и доступных backend.
        """
        return {
            "platform": detect_platform(),
            "vllm_available": VLLM_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "flash_attention_available": FLASH_ATTN_AVAILABLE,
            "bitsandbytes_available": BITSANDBYTES_AVAILABLE,
            "recommended_backend": get_recommended_backend().value,
            "cuda_available": torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False,
            "mps_available": torch.backends.mps.is_available() if TRANSFORMERS_AVAILABLE else False,
        }
