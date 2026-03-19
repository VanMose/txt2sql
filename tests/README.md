# Tests for Text-to-SQL Pipeline

## Запуск тестов

### Все тесты
```bash
pytest tests/ -v
```

### С покрытием кода (требуется pytest-cov)
```bash
# Установка
pip install pytest-cov

# Запуск с покрытием
pytest tests/ -v --cov=src --cov-report=html
```

### Только unit тесты
```bash
pytest tests/ -v -m "unit"
```

### Только integration тесты
```bash
pytest tests/ -v -m "integration"
```

### Без медленных тестов
```bash
pytest tests/ -v -m "not slow"
```

### Конкретный тест файл
```bash
pytest tests/test_utils.py -v
pytest tests/test_db.py -v
pytest tests/test_pipeline.py -v
```

### Конкретный тест
```bash
pytest tests/test_utils.py::TestJsonParser::test_parse_clean_json -v
```

## Структура тестов

```
tests/
├── conftest.py           # Fixtures и общие настройки
├── test_utils.py         # Тесты утилит (JSON, SQL parsers)
├── test_db.py            # Тесты database компонентов
├── test_retrieval.py     # Тесты retrieval компонентов
├── test_agents.py        # Тесты agents
├── test_config.py        # Тесты config и settings
└── test_pipeline.py      # Integration тесты пайплайна
```

## Категории тестов

- **unit** - Быстрые изолированные тесты отдельных компонентов
- **integration** - Интеграционные тесты взаимодействия компонентов
- **slow** - Медленные тесты (запуск с реальной моделью)

## Fixtures

- `test_db_path` - Тестовая SQLite БД с таблицами users, orders, products
- `temp_sqlite_db` - Временная пустая SQLite БД
- `sample_queries` - Примеры запросов на естественном языке
- `sample_sql_queries` - Примеры SQL запросов
- `setup_env` - Настройка переменных окружения для тестов

## Быстрая проверка

```bash
# Запуск быстрого теста (без pytest)
python scripts/test_pipeline.py
```
