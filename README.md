# 🗄️ Text-to-SQL v0.1.0

**Text-to-SQL система с Hybrid RAG, Graph reasoning и LLM**

---


## 🚀 Быстрый старт

### 1. Установка

```bash
conda create -n llm_env python=3.11
conda activate llm_env
pip install -r requirements.txt
```

### 2. Настройка

```bash
# Скопируйте .env.example в .env
cp .env.example .env

# Отредактируйте .env под вашу конфигурацию
```

### 3. Подготовка данных

```
data/
├── movie_1/
│   ├── movie_1.sqlite
│   └── schema.sql
└── music_1/
    ├── music_1.sqlite
    └── schema.sql
```

### 4. Запуск

```bash
# Streamlit UI
conda activate llm_env
streamlit run app/main.py

# Или через Python
conda activate llm_env
python -c "from src.pipeline.production_pipeline import ProductionPipeline; p = ProductionPipeline(); p.run('Покажи фильмы')"
```

---

## 🏗️ Архитектура

### Production Pipeline

```
User Question
     │
     ▼
Query Understanding
(Intent, Entities, Filters)
     │
     ▼
Hybrid Retrieval
(Vector + Graph)
     │
     ▼
Reranking
(Cross-Encoder)
     │
     ▼
Schema Compression
(PK/FK/Types format)
     │
     ▼
SQL Generation
(LLM: Qwen2.5-Coder)
     │
     ▼
SQL Validation
(Syntax, Schema, JOINs)
     │
     ▼
Execution
(SQLite/Postgres)
     │
     ▼
Result
```

### Компоненты

| Компонент | Технология | Назначение |
|-----------|------------|------------|
| **Query Understanding** | Custom Agent | Intent classification, entity extraction |
| **Vector DB** | Qdrant | Semantic поиск таблиц |
| **Graph DB** | Neo4j | JOIN path discovery |
| **Reranker** | Cross-Encoder | Точный reranking |
| **Schema Compressor** | Custom | Формат с PK/FK/types |
| **LLM** | Qwen2.5-Coder | SQL генерация |
| **Validator** | Custom Agent | SQL валидация |
| **Orchestration** | LangGraph | Workflow management |

---

## 📁 Структура проекта

```
text2sql_baseline/
├── app/                        # Streamlit UI
│   ├── main.py                 # Главное приложение
│   ├── components/             # UI компоненты
│   └── pages/                  # Дополнительные страницы
│
├── src/
│   ├── agents/                 # AI агенты
│   │   ├── query_understanding.py  # Query Understanding
│   │   ├── sql_validator.py        # SQL Validator
│   │   ├── sql_generator.py        # SQL Generator
│   │   ├── sql_judge.py            # SQL Judge
│   │   └── sql_refiner.py          # SQL Refiner
│   │
│   ├── config/                 # Конфигурация
│   │   └── settings.py         # Pydantic settings
│   │
│   ├── db/                     # Database layer
│   │   ├── executor.py         # SQL execution
│   │   └── schema_loader.py    # Schema loading
│   │
│   ├── llm/                    # LLM layer
│   │   ├── inference.py        # LLM service
│   │   ├── model_loader.py     # Model loading
│   │   └── prompts.py          # Prompt templates
│   │
│   ├── pipeline/               # Pipelines
│   │   ├── production_pipeline.py  # Production pipeline
│   │   ├── text2sql_pipeline.py    # Legacy pipeline
│   │   └── state.py                # State definitions
│   │
│   ├── retrieval/              # Retrieval layer
│   │   ├── vector_db.py        # Qdrant vector DB
│   │   ├── graph_db.py         # Neo4j graph DB
│   │   ├── schema_retriever.py # Hybrid retriever
│   │   ├── schema_compressor.py# Schema compression
│   │   └── embedder.py         # Embeddings с TTL cache
│   │
│   ├── services/               # Services
│   │   ├── metrics.py          # Metrics & observability
│   │   └── pipeline_service.py # Pipeline service
│   │
│   └── utils/                  # Utilities
│       ├── optimizations.py    # Caching, retry
│       └── rate_limiter.py     # Rate limiting
│
├── configs/                    # Config files
│   ├── judge_prompts.yaml
│   ├── sql_generator_prompts.yaml
│   └── model_params.yaml
│
├── data/                       # SQLite databases
├── qdrant_storage/             # Qdrant vector storage
├── cache/                      # Disk cache
├── logs/                       # Logs
└── llm_models/                 # Local LLM models
```

---

## ⚙️ Конфигурация (.env)

```ini
# LLM Model
TEXT2SQL_LLM_MODEL=Qwen2.5-Coder-1.5B
TEXT2SQL_USE_LOCAL_MODEL=true

# Generation
TEXT2SQL_N_SAMPLES=2
TEXT2SQL_TEMPERATURE=0.15
TEXT2SQL_MAX_TOKENS=256

# Retrieval
TEXT2SQL_TOP_K_TABLES=5
TEXT2SQL_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Graph Expansion
TEXT2SQL_USE_GRAPH_EXPANSION=true
TEXT2SQL_GRAPH_EXPANSION_DEPTH=2

# Reranking
TEXT2SQL_USE_RERANKING=true
TEXT2SQL_RERANKER_MODEL=bge-reranker-base

# Validation
TEXT2SQL_USE_VALIDATION=true
TEXT2SQL_CONFIDENCE_THRESHOLD=0.6

# Qdrant
TEXT2SQL_QDRANT_USE_LOCAL=true
TEXT2SQL_QDRANT_LOCAL_PATH=qdrant_storage

# Neo4j
TEXT2SQL_NEO4J_URI=bolt://localhost:7687
TEXT2SQL_NEO4J_USERNAME=neo4j
TEXT2SQL_NEO4J_PASSWORD=password

# Optimizations
TEXT2SQL_USE_4BIT_QUANTIZATION=true
TEXT2SQL_USE_TORCH_COMPILE=true
TEXT2SQL_USE_SEMANTIC_CACHE=true
TEXT2SQL_CACHE_TTL=3600

# Logging
TEXT2SQL_LOG_LEVEL=INFO
```

---

## 🔧 API Usage

### Production Pipeline

```python
from src.pipeline.production_pipeline import ProductionPipeline

pipeline = ProductionPipeline(
    db_paths=["data/movie_1/movie_1.sqlite"],
    use_graph_expansion=True,
    use_reranking=True,
    use_validation=True,
)

result = pipeline.run("Покажи топ 5 фильмов с рейтингом выше 8")

print(f"SQL: {result.generated_sql}")
print(f"Success: {result.execution_success}")
print(f"Latency: {result.latencies}")
```

