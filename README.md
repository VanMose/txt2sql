# 🗄️ Text-to-SQL Pipeline v0.1.5

**Natural language → SQL** with Hybrid RAG, Graph reasoning, and Ollama LLM.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
conda create -n llm_env python=3.11
conda activate llm_env
pip install -r requirements.txt
```

### 2. Setup Ollama (Fast LLM Backend)

```bash
# Install Ollama: https://ollama.ai
ollama pull qwen2.5-coder:1.5b
```

### 3. Start Docker Services

```bash
docker-compose up -d
```

### 4. Run UI

```bash
streamlit run app/main.py
```

---

## 📁 Project Structure

```
text2sql_baseline/
├── app/                    # Streamlit UI
│   ├── main.py            # Main app
│   ├── pages/             # Database Viewer
│   └── components/        # UI components
├── src/
│   ├── agents/            # AI agents (Router, SQL Gen, Validator)
│   ├── llm/               # Ollama & Transformers backends
│   ├── pipeline/          # LangGraph workflows
│   ├── retrieval/         # Vector DB (Qdrant) + Graph DB (Neo4j)
│   ├── db/                # Database executors + Guardrails
│   ├── services/          # Metrics, Caching, Result Cache
│   └── config/            # Settings management
├── configs/prompts/        # Prompt templates 
├── data/                   # SQLite databases
├── docker-compose.yml     # Neo4j + Qdrant
└── .env                   # Configuration
```

---

---

## 🏗️ Architecture

```
User Query
    ↓
Query Understanding → Intent, Entities, DB Hints
    ↓
Router Agent → Selects relevant databases (Vector + Graph + Rerank)
    ↓
Schema Retrieval → Load schemas + Compress (54.6% savings)
    ↓
SQL Generator → Batch generation + Early stopping
    ↓
Validator + Guardrails → Syntax + Security checks
    ↓
Executor → Parallel execution with connection pooling
    ↓
Judge → Quality evaluation + Early exit
    ↓
Refiner (if needed) → Error correction
    ↓
Result + Metrics
```

---
### Streamlit UI

```bash
streamlit run app/main.py
```

