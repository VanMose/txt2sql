"""
Unit tests для config и settings.
"""
import os
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config.settings import Settings, get_settings, override_settings, get_settings_with_override, BASE_DIR


@pytest.mark.unit
class TestSettings:
    """Тесты для Settings."""

    def test_settings_init(self):
        """Инициализация Settings."""
        settings = Settings()
        assert settings is not None
        assert hasattr(settings, "llm_model")
        assert hasattr(settings, "n_samples")
        assert hasattr(settings, "temperature")

    def test_settings_default_values(self):
        """Значения по умолчанию."""
        settings = Settings()
        assert settings.llm_model == "Qwen2.5-Coder-3B"
        assert settings.n_samples == 5
        assert settings.temperature == 0.7
        assert settings.top_k_tables == 5
        assert settings.confidence_threshold == 0.5
        assert settings.max_retries == 5

    def test_settings_base_dir(self):
        """Проверка base_dir."""
        settings = Settings()
        assert "text2sql_baseline" in settings.base_dir

    def test_settings_paths(self):
        """Проверка путей."""
        settings = Settings()
        assert "llm_models" in settings.models_path
        assert "data" in settings.data_path
        assert "configs" in settings.configs_path

    def test_settings_log_filepath(self):
        """Проверка log_filepath."""
        settings = Settings()
        assert settings.log_filepath.endswith(".log")
        assert "text2sql" in settings.log_filepath.lower()

    def test_settings_get_local_model_path(self):
        """Проверка get_local_model_path."""
        settings = Settings()
        path = settings.get_local_model_path("Qwen2.5-Coder-3B")
        assert "Qwen2.5-Coder-3B" in path
        assert "llm_models" in path


@pytest.mark.unit
class TestSettingsFunctions:
    """Тесты для функций настроек."""

    def test_get_settings_cached(self):
        """Кэширование get_settings."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_get_settings_with_model_override(self):
        """Переопределение модели."""
        get_settings.cache_clear()
        settings = get_settings(model_name="TestModel")
        assert settings.llm_model == "TestModel"

    def test_get_settings_with_local_override(self):
        """Переопределение use_local_model."""
        get_settings.cache_clear()
        settings = get_settings(use_local=False)
        assert settings.use_local_model is False

    def test_override_settings_function(self):
        """Функция override_settings."""
        get_settings.cache_clear()
        
        override_settings(model_name="OverriddenModel")
        settings = get_settings_with_override()
        
        assert settings.llm_model == "OverriddenModel"
        
        get_settings.cache_clear()

    def test_override_settings_multiple(self):
        """Множественное переопределение."""
        get_settings.cache_clear()
        
        override_settings(
            model_name="TestModel",
            use_local=True,
            n_samples=3,
            temperature=0.5
        )
        settings = get_settings_with_override()
        
        assert settings.llm_model == "TestModel"
        assert settings.use_local_model is True
        assert settings.n_samples == 3
        assert settings.temperature == 0.5
        
        get_settings.cache_clear()

    def test_base_dir_constant(self):
        """Проверка BASE_DIR."""
        assert isinstance(BASE_DIR, Path)
        assert BASE_DIR.exists()
        assert "text2sql_baseline" in str(BASE_DIR)


@pytest.mark.integration
class TestSettingsIntegration:
    """Интеграционные тесты для настроек."""

    def test_settings_with_env_fixture(self, setup_env):
        """Настройки с fixture setup_env."""
        get_settings.cache_clear()
        settings = get_settings_with_override()
        
        assert settings.n_samples == 2
        assert settings.temperature == 0.1
        assert settings.confidence_threshold == 0.3

    def test_settings_paths_exist(self):
        """Проверка существования путей."""
        settings = Settings()
        
        base_path = Path(settings.base_dir)
        assert base_path.exists()
        
        configs_path = Path(settings.configs_path)
        assert configs_path.exists()
        
        data_path = Path(settings.data_path)
        assert data_path.exists()

    def test_settings_env_prefix(self):
        """Проверка префикса переменных окружения."""
        settings = Settings()
        assert settings.Config.env_prefix == "TEXT2SQL_"
