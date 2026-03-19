# 🗄️ Text-to-SQL Pipeline v0.1.2

**Natural language → SQL** with Hybrid RAG, Graph reasoning, and Ollama LLM.

---

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
│   └── db/                # Database executors
├── data/                   # SQLite databases
├── docker-compose.yml     # Neo4j + Qdrant
└── docs/                   # Documentation
```

---


## 🛠️ Configuration (.env)

```ini
# LLM - Ollama (Primary)
TEXT2SQL_USE_OLLAMA=true
TEXT2SQL_LLM_MODEL=qwen2.5-coder:1.5b
TEXT2SQL_OLLAMA_BASE_URL=http://localhost:11434

# Databases
TEXT2SQL_AUTO_DISCOVER_DBS=true
TEXT2SQL_TOP_K_DBS=2

# Vector DB
TEXT2SQL_QDRANT_USE_LOCAL=false
TEXT2SQL_QDRANT_URL=http://localhost:6333

# Graph DB
TEXT2SQL_NEO4J_URI=bolt://localhost:7687
```

---


## 🏗️ Architecture

```
User Query
    ↓
Router Agent → Selects relevant databases
    ↓
Schema Retrieval → Vector DB + Graph DB
    ↓
SQL Generator (Ollama) → Generates SQL
    ↓
Validator → Checks syntax & schema
    ↓
Executor → Runs SQL
    ↓
Result
```

---
