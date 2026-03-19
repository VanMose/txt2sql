# text2sql_baseline\src\utils\config_loader.py
"""Загрузчик YAML конфигураций."""
import yaml
from functools import lru_cache
from pathlib import Path


class ConfigLoader:
    """Загрузчик YAML конфигов с кэшированием."""

    def __init__(self, config_dir: str | None = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "configs"
        self.config_dir = Path(config_dir)

    @lru_cache(maxsize=16)
    def load_yaml(self, filename: str) -> dict:
        """Загрузить YAML файл с кэшированием."""
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Config not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_sql_prompt(self, model_name: str) -> tuple[str, str]:
        """Получить промпт для SQL генератора."""
        config = self.load_yaml("sql_generator_prompts.yaml")
        model_config = config.get(model_name, config.get("Qwen2.5-Coder-3B"))
        return (
            model_config.get("system", ""),
            model_config.get("user_template", ""),
        )

    def get_judge_prompt(self, model_name: str) -> tuple[str, str]:
        """Получить промпт для Judge."""
        config = self.load_yaml("judge_prompts.yaml")
        model_config = config.get(model_name, config.get("Qwen2.5-Coder-3B"))
        return (
            model_config.get("system", ""),
            model_config.get("user_template", ""),
        )

    def get_model_params(self, model_name: str) -> dict:
        """Получить параметры модели."""
        config = self.load_yaml("model_params.yaml")
        defaults = config.get("defaults", {})
        model_params = config.get("models", {}).get(model_name, {})
        return {**defaults, **model_params}

    def get_all_models(self) -> list[str]:
        """Получить список всех доступных моделей."""
        sql_config = self.load_yaml("sql_generator_prompts.yaml")
        return list(sql_config.keys())


@lru_cache
def get_config_loader() -> ConfigLoader:
    """Singleton для загрузчика конфигов."""
    return ConfigLoader()
