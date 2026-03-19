# src\llm\transformers_service.py
"""
LLM сервис на базе HuggingFace Transformers с оптимизациями.

Оптимизации:
1. 4-bit квантование (BitsAndBytes)
2. Flash Attention 2 (если доступна)
3. Torch compile (torch.compile)
4. Пакетная генерация
5. Prompt caching

Example:
    >>> from llm.transformers_service import TransformersService
    >>> outputs = TransformersService.generate(["prompt"], max_tokens=256)
"""
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class TransformersService:
    """
    Сервис для генерации текста через HuggingFace Transformers.

    Оптимизации:
    - 4-bit квантование (BitsAndBytes)
    - Flash Attention 2
    - Torch compile
    - Пакетная генерация

    Attributes:
        _model: Кэш модели.
        _tokenizer: Кэш токенизатора.
        _device: Устройство (cuda/cpu/mps).
        _current_model_path: Путь к текущей модели.
        _use_flash_attention: Флаг использования Flash Attention.
    """

    _model: Optional[AutoModelForCausalLM] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _device: Optional[torch.device] = None
    _current_model_path: Optional[str] = None
    _use_flash_attention: bool = False

    @classmethod
    def _is_local_path(cls, path: str) -> bool:
        """Проверить, является ли путь локальным."""
        return os.path.isabs(path) or Path(path).exists()

    @classmethod
    def get_model(
        cls,
        model_name: Optional[str] = None,
        use_4bit: bool = True,
        use_flash_attention: bool = False,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
        """
        Получить модель и токенизатор (ленивая загрузка).

        Args:
            model_name: Имя модели. Если None, используется из настроек.
            use_4bit: Использовать 4-битную квантизацию.
            use_flash_attention: Использовать Flash Attention 2.

        Returns:
            Кортеж (model, tokenizer, device).
        """
        if cls._model is None or cls._tokenizer is None:
            settings = get_settings()
            model_path = model_name or settings.llm_model

            if settings.use_local_model:
                local_path = settings.get_local_model_path(model_name or settings.llm_model)
                if Path(local_path).exists():
                    model_path = local_path
                    logger.info(f"Using local model: {model_path}")

            cls._current_model_path = model_path
            cls._use_flash_attention = use_flash_attention

            # Определяем устройство
            if torch.cuda.is_available():
                cls._device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                cls._device = torch.device("mps")
                logger.info("Using Apple MPS")
            else:
                cls._device = torch.device("cpu")
                logger.info("Using CPU")

            logger.info(f"Loading model: {model_path}")

            try:
                # Загружаем токенизатор
                cls._tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=True,
                    local_files_only=cls._is_local_path(model_path)
                )

                # Загружаем модель с оптимизациями
                if cls._device.type == "cuda":
                    cls._model = cls._load_with_optimizations(
                        model_path,
                        use_4bit=use_4bit,
                        use_flash_attention=use_flash_attention,
                    )
                else:
                    cls._model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map=cls._device,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        local_files_only=cls._is_local_path(model_path)
                    )
                    logger.info("Loaded model for CPU/MPS")

                logger.info(f"Model loaded successfully on {cls._device}")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

        return cls._model, cls._tokenizer, cls._device

    @classmethod
    def _load_with_optimizations(
        cls,
        model_path: str,
        use_4bit: bool = True,
        use_flash_attention: bool = False,
    ) -> AutoModelForCausalLM:
        """
        Загрузить модель с оптимизациями.

        Args:
            model_path: Путь к модели.
            use_4bit: Использовать 4-битную квантизацию.
            use_flash_attention: Использовать Flash Attention 2.

        Returns:
            Загруженная модель.
        """
        # Проверяем доступность BitsAndBytes
        bitsandbytes_available = True
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            bitsandbytes_available = False
            use_4bit = False
            logger.warning("bitsandbytes not available, disabling 4-bit quantization")

        # Проверяем доступность Flash Attention
        flash_attn_available = True
        attn_implementation = None
        
        if use_flash_attention:
            try:
                # Проверяем, поддерживает ли модель Flash Attention
                import flash_attn
                attn_implementation = "flash_attention_2"
                logger.info("Flash Attention 2 is available")
            except ImportError:
                flash_attn_available = False
                use_flash_attention = False
                logger.warning("Flash Attention 2 not installed, using sdpa")
            
            # Fallback на sdpa (PyTorch 2.0+)
            if not flash_attn_available:
                try:
                    attn_implementation = "sdpa"
                    logger.info("Using SDPA (scaled dot-product attention)")
                except:
                    attn_implementation = None

        # 4-bit квантизация
        quantization_config = None
        if use_4bit and bitsandbytes_available:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Loading model with 4-bit quantization")

        # Подготовка аргументов
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "local_files_only": cls._is_local_path(model_path),
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        if attn_implementation is not None:
            try:
                model_kwargs["attn_implementation"] = attn_implementation
                logger.info(f"Using attention implementation: {attn_implementation}")
            except Exception as e:
                logger.warning(f"Cannot use {attn_implementation}: {e}")

        return AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    @classmethod
    def load_model(
        cls,
        model_path: str,
        gpu_memory_utilization: Optional[float] = None,
        use_4bit: bool = True,
        use_flash_attention: bool = False,
    ) -> Any:
        """
        Загрузить модель с настройками.

        Args:
            model_path: Путь к модели.
            gpu_memory_utilization: Утилизация GPU памяти.
            use_4bit: Использовать 4-битную квантизацию.
            use_flash_attention: Использовать Flash Attention 2.

        Returns:
            Модель.
        """
        from src.config.settings import get_settings
        settings = get_settings()
        
        if gpu_memory_utilization is None:
            gpu_memory_utilization = settings.gpu_memory_utilization
        
        # Ограничение памяти GPU (если возможно)
        if torch.cuda.is_available() and gpu_memory_utilization < 1.0:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            limit_bytes = int(total_memory * gpu_memory_utilization)
            torch.cuda.set_per_process_memory_fraction(gpu_memory_utilization)
            logger.info(f"GPU memory limit set to {gpu_memory_utilization * 100:.0f}%")

        cls.clear_cache()
        return cls.get_model(
            model_name=model_path,
            use_4bit=use_4bit,
            use_flash_attention=use_flash_attention,
        )

    @classmethod
    def generate(
        cls,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        n: int = 1
    ) -> List[str]:
        """
        Сгенерировать текст.

        Args:
            prompts: Список промптов.
            max_tokens: Максимальное количество токенов.
            temperature: Температура генерации.
            n: Количество вариантов на промпт.

        Returns:
            Список сгенерированных текстов.
        """
        from src.config.settings import get_settings
        settings = get_settings()
        
        if max_tokens is None:
            max_tokens = settings.max_tokens
        if temperature is None:
            temperature = settings.temperature
        
        model, tokenizer, device = cls.get_model()
        all_outputs: List[str] = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=0.95,
                num_return_sequences=n,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=generation_config)

            # Декодируем ответы
            for output in outputs:
                generated_text = tokenizer.decode(
                    output[inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                all_outputs.append(generated_text)

        logger.info(f"Generated {len(all_outputs)} outputs")
        return all_outputs

    @classmethod
    def clear_cache(cls) -> None:
        """Очистить кэш модели."""
        cls._model = None
        cls._tokenizer = None
        cls._device = None
        cls._current_model_path = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model cache cleared")

    @classmethod
    def is_available(cls) -> bool:
        """Проверить доступность Transformers."""
        try:
            import transformers
            return True
        except ImportError:
            return False
