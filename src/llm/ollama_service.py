# src\llm\ollama_service.py
"""
Ollama сервис для генерации текста.

Быстрый и легковесный сервис на основе Ollama API.
Использует llama.cpp backend с оптимизацией для CPU/GPU.

Использование:
    from llm.ollama_service import OllamaService
    
    service = OllamaService(model="qwen2.5-coder:1.5b")
    texts = service.generate("SELECT * FROM users", n=3)
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OllamaService:
    """
    Сервис для генерации текста через Ollama API.
    
    Features:
    - Быстрая генерация (llama.cpp backend)
    - Низкое потребление памяти
    - Поддержка CPU и GPU
    - Встроенная квантизация (GGUF format)
    
    Attributes:
        model_name: Имя модели в Ollama.
        base_url: URL Ollama сервера.
        timeout: Таймаут запроса в секундах.
    """
    
    def __init__(
        self,
        model_name: str = "qwen2.5-coder:1.5b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        """
        Инициализировать сервис.
        
        Args:
            model_name: Имя модели (должна быть установлена в Ollama).
            base_url: URL Ollama сервера.
            timeout: Таймаут запроса.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        
        # Проверяем доступность Ollama
        self._check_ollama()
        
        logger.info(
            f"OllamaService initialized: model={model_name}, "
            f"url={base_url}"
        )
    
    def _check_ollama(self) -> None:
        """Проверить доступность Ollama сервера."""
        try:
            import ollama
            client = ollama.Client(host=self.base_url)
            client.list()
            logger.info("Ollama server is accessible")
        except ImportError:
            logger.error("ollama package not installed. Run: pip install ollama")
            raise
        except Exception as e:
            logger.warning(f"Ollama server check failed: {e}")
    
    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> List[str]:
        """
        Сгенерировать текст.
        
        Args:
            prompt: Входной промпт.
            n: Количество сэмплов.
            temperature: Температура генерации.
            max_tokens: Максимум токенов.
            
        Returns:
            Список сгенерированных текстов.
        """
        import ollama
        
        results = []
        client = ollama.Client(host=self.base_url)
        
        for i in range(n):
            try:
                response = client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "top_p": 0.9,
                    },
                )
                results.append(response['response'])
                
            except Exception as e:
                logger.error(f"Ollama generation error: {e}")
                results.append("")
        
        return results
    
    def generate_batch(
        self,
        prompts: List[str],
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> List[List[str]]:
        """
        Batch генерация.
        
        Args:
            prompts: Список промптов.
            n: Количество сэмплов на промпт.
            temperature: Температура.
            max_tokens: Максимум токенов.
            
        Returns:
            Список списков текстов.
        """
        results = []
        for prompt in prompts:
            batch_result = self.generate(
                prompt,
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results.append(batch_result)
        return results
    
    def is_available(self) -> bool:
        """Проверить доступность сервера."""
        try:
            import ollama
            client = ollama.Client(host=self.base_url)
            client.list()
            return True
        except:
            return False
    
    def list_models(self) -> List[str]:
        """Получить список доступных моделей."""
        try:
            import ollama
            client = ollama.Client(host=self.base_url)
            response = client.list()
            return [model['name'] for model in response.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def pull_model(self, model_name: Optional[str] = None) -> bool:
        """
        Скачать модель.
        
        Args:
            model_name: Имя модели для скачивания.
            
        Returns:
            True если успешно.
        """
        import ollama
        
        model = model_name or self.model_name
        logger.info(f"Pulling model: {model}")
        
        try:
            response = ollama.pull(model, stream=True)
            for status in response:
                if 'status' in status:
                    logger.info(f"Pull status: {status['status']}")
            logger.info(f"Model {model} pulled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        return {
            "model": self.model_name,
            "base_url": self.base_url,
            "available": self.is_available(),
            "models": self.list_models(),
        }
