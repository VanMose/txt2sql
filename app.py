# text2sql_baseline\app.py
"""Streamlit UI для Text-to-SQL пайплайна."""
import sys
import logging
import argparse
from pathlib import Path

import streamlit as st

# Добавляем src в path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from config.settings import get_settings, Settings
from pipeline.text2sql_pipeline import Text2SQLPipeline
from db.schema_loader import SchemaLoader
from llm.model_loader import LLMBackend, ModelLoader

# Настройки страницы
st.set_page_config(
    page_title="Text-to-SQL",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_debug_info(result: dict) -> None:
    """Отобразить отладочную информацию."""
    with st.expander("🐛 Debug Info", expanded=False):
        st.write("**Raw Result:**")
        st.json({
            "query": result["query"],
            "sql_query": result["sql_query"],
            "success": result["success"],
            "confidence": result["confidence"],
            "retry_count": result["retry_count"],
            "attempts_count": result.get("attempts_count", 0),
            "latencies": result["latencies"],
        })
        
        st.write("**Relevant Tables:**")
        for table in result.get("relevant_tables", []):
            st.text(table)


def render_result(result: dict) -> None:
    """Отобразить результаты пайплайна."""
    logger.info(f"Rendering result: success={result['success']}, sql_query='{result['sql_query']}'")
    
    # SQL Query
    with st.expander("📝 SQL Query", expanded=True):
        if result["sql_query"]:
            st.code(result["sql_query"], language="sql", line_numbers=False)
            st.success("✅ SQL успешно сгенерирован")
        else:
            st.warning("⚠️ SQL не сгенерирован")
            st.info("Возможные причины:")
            st.markdown("""
            - Ошибка валидации SQL
            - Ошибка при генерации кандидатов
            - Низкий confidence threshold
            """)
        st.caption(f"Model: `{get_settings().llm_model}`")

    # Результат выполнения
    with st.expander("📊 Execution Result", expanded=True):
        if result["success"] and result["sql_result"] is not None:
            try:
                st.dataframe(result["sql_result"], width="stretch", hide_index=True)
                st.success(f"✅ Выполнено успешно, строк: {len(result['sql_result'])}")
            except Exception as e:
                st.write(result["sql_result"])
                st.warning(f"Результат не является DataFrame: {e}")
        else:
            st.warning("⚠️ Пустой результат или ошибка выполнения")
            
        if not result["success"]:
            st.error("❌ Пайплайн не смог сгенерировать валидный SQL")

    # Metrics
    with st.expander("📈 Metrics", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            conf = result["confidence"]
            if conf >= 0.7:
                st.success(f"**Confidence:** {conf:.2f}")
            elif conf >= 0.5:
                st.warning(f"**Confidence:** {conf:.2f}")
            else:
                st.error(f"**Confidence:** {conf:.2f}")

        with col2:
            st.metric("Retry Count", result["retry_count"])
            
        with col3:
            st.metric("Attempts", result.get("attempts_count", 0))

        total_latency = sum(result["latencies"].values())
        st.divider()
        st.metric("Total Latency", f"{total_latency:.0f} ms")

        # Latency by step
        st.write("**⏱️ Latency by step:**")
        if result["latencies"]:
            cols = st.columns(len(result["latencies"]))
            for idx, (step, latency) in enumerate(result["latencies"].items()):
                with cols[idx]:
                    step_name = step.replace("_", " ").title()
                    st.metric(step_name, f"{latency:.0f} ms")
        else:
            st.info("Latency данные отсутствуют")

        # Релевантные таблицы
        if result.get("relevant_tables"):
            st.divider()
            st.write("**📋 Релевантные таблицы:**")
            for table in result["relevant_tables"]:
                st.text(table)
        else:
            st.info("Релевантные таблицы не найдены")
    
    # Debug info
    render_debug_info(result)


def render_dataset_info(db_path: str) -> None:
    """Отобразить информацию о базе данных."""
    try:
        loader = SchemaLoader(db_path)
        tables = loader.get_tables()

        with st.expander(f"📁 Database Schema ({len(tables)} tables)", expanded=False):
            for table in tables:
                schema_info = loader.get_table_schema(table)
                st.write(f"**{table}** ({len(schema_info['columns'])} columns)")

                # Показываем колонки
                col_names = [col["name"] for col in schema_info["columns"]]
                st.caption(", ".join(col_names))
    except Exception as e:
        st.error(f"Error loading schema: {e}")
        logger.error(f"Error loading schema: {e}")


def render_backend_info(backend: LLMBackend) -> None:
    """Отобразить информацию о backend."""
    platform_info = ModelLoader.get_platform_info()
    
    with st.expander("🤖 LLM Backend Info", expanded=False):
        st.write(f"**Current Backend:** `{backend.value}`")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Platform:**")
            st.json({
                "OS": platform_info["platform"],
                "CUDA Available": platform_info["cuda_available"],
                "MPS Available": platform_info["mps_available"],
            })
        
        with col2:
            st.write("**Availability:**")
            if platform_info["vllm_available"]:
                st.success("✅ vLLM available")
            else:
                st.info("⚪ vLLM not available (Linux only)")
            
            if platform_info["transformers_available"]:
                st.success("✅ Transformers available")
            else:
                st.error("❌ Transformers not installed")
        
        st.divider()
        st.info(f"""
        **Recommendation:**
        - Linux + GPU → vLLM (fastest)
        - Windows/Mac → Transformers (full LLM)
        - Testing → Mock (instant, no LLM)
        """)


def main(mock_mode: bool = False, force_backend: str = None) -> None:
    logger.info("Starting Streamlit app")
    
    st.title("🗄️ Text-to-SQL Pipeline")
    st.markdown(
        "Преобразование естественного языка в SQL с использованием **self-consistency decoding**"
    )

    # Session state
    if "result" not in st.session_state:
        st.session_state.result = None
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    if "pipeline_created" not in st.session_state:
        st.session_state.pipeline_created = False

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Информация о настройках
        settings = get_settings()
        
        # Определяем backend
        if mock_mode:
            backend = LLMBackend.MOCK
            st.warning("⚠️ MOCK MODE (no real LLM)")
        elif force_backend:
            backend = LLMBackend(force_backend)
            st.info(f"Force backend: `{backend.value}`")
        else:
            backend = ModelLoader.get_backend()
            st.success(f"✅ Backend: `{backend.value}`")
        
        st.info(f"""
        **Model:** {settings.llm_model}
        
        **N Samples:** {settings.n_samples}
        
        **Temperature:** {settings.temperature}
        
        **Confidence Threshold:** {settings.confidence_threshold}
        
        **Data Path:** {settings.data_path}
        """)
        
        # Показываем статус backend
        platform_info = ModelLoader.get_platform_info()
        
        if backend == LLMBackend.VLLM:
            st.success("✅ vLLM (fastest, Linux only)")
        elif backend == LLMBackend.TRANSFORMERS:
            st.info("ℹ️ Transformers (full LLM, all platforms)")
        else:
            st.warning("⚠️ Mock mode (testing only)")
        
        render_backend_info(backend)

        st.divider()

        # Выбор базы данных
        db_files = list(Path(settings.data_path).glob("**/*.db"))
        db_files.extend(Path(settings.data_path).glob("**/*.sqlite"))

        logger.info(f"Found {len(db_files)} database files")

        if db_files:
            selected_db = st.selectbox(
                "📁 Database",
                options=[str(db) for db in db_files],
                format_func=lambda x: Path(x).name,
                index=0,
            )
            settings.db_path = selected_db
            logger.info(f"Selected database: {selected_db}")
        else:
            st.error("❌ No databases found in data/")
            selected_db = None

        # Информация о датасете
        if selected_db:
            render_dataset_info(selected_db)

        st.divider()

        # Кнопки управления
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.result = None
                st.session_state.query_input = ""
                st.session_state.pipeline_created = False
                logger.info("Reset session state")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                # Очистка кэша Streamlit
                st.cache_resource.clear()
                st.cache_data.clear()
                # Очистка кэша модели
                if backend == LLMBackend.TRANSFORMERS:
                    from llm.transformers_service import TransformersService
                    TransformersService.clear_cache()
                logger.info("Cleared caches")
                st.success("Cache cleared!")

        st.divider()

        # Примеры запросов
        st.header("💡 Примеры запросов")
        examples = [
            "Показать все фильмы",
            "Найти максимальную оценку",
            "Посчитать количество фильмов",
            "Показать все рейтинги",
        ]

        for example in examples:
            if st.button(example, use_container_width=True, key=f"ex_{example}"):
                st.session_state.query_input = example
                st.session_state.result = None
                logger.info(f"Selected example: {example}")
                st.rerun()

    # Основная панель
    query = st.text_area(
        "Введите ваш запрос:",
        value=st.session_state.query_input,
        placeholder="Например: Показать все фильмы",
        height=100,
        key="query_textarea",
        on_change=lambda: st.session_state.update(query_input=st.session_state.query_textarea),
    )

    # Кнопка запуска
    col1, col2 = st.columns([1, 5])
    with col1:
        run_button = st.button(
            "🚀 Запустить",
            type="primary",
            use_container_width=True,
            disabled=not query or not selected_db,
        )

    if run_button and query and selected_db:
        logger.info(f"Running pipeline with query: '{query}', db: {selected_db}, backend: {backend.value}")
        
        with st.spinner("⏳ Обработка запроса..."):
            try:
                # Создаем пайплайн
                pipeline = Text2SQLPipeline(db_path=selected_db)
                st.session_state.pipeline_created = True

                # Запускаем
                result = pipeline.run_with_result(query)

                st.session_state.result = result
                st.session_state.query_input = ""
                
                logger.info(f"Pipeline result: success={result['success']}, sql='{result['sql_query']}'")
                
                render_result(result)

            except Exception as e:
                logger.error(f"Pipeline error: {e}", exc_info=True)
                st.error(f"❌ Ошибка: {e}")
                with st.expander("📋 Stack Trace"):
                    st.exception(e)
                st.session_state.result = None

    elif st.session_state.result:
        logger.info("Rendering cached result")
        render_result(st.session_state.result)

    elif not query:
        st.info("💬 Введите запрос или выберите пример из sidebar")

    elif not selected_db:
        st.warning("⚠️ Выберите базу данных в sidebar")

    # Footer
    st.divider()
    st.markdown(
        """
        **Pipeline:** Query → Schema Retrieval → SQL Generation (N samples) → Execute → Judge → Best Selection → Refinement

        *Architecture: vLLM/Transformers + Self-Consistency + Schema Linking + Execution-Guided Decoding*
        """,
        help="Схема работы пайплайна",
    )


def parse_args():
    """Парсить аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Streamlit UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Запуск с автоопределением backend (рекомендуется)
  streamlit run app.py
  
  # Запуск в Mock режиме (для тестов, без LLM)
  streamlit run app.py -- --mock
  
  # Запуск с Transformers backend
  streamlit run app.py -- --backend transformers
  
  # Запуск с vLLM backend (только Linux)
  streamlit run app.py -- --backend vllm
  
  # Запуск с локальной моделью
  streamlit run app.py -- --model Qwen2.5-Coder-3B
  
  # Запуск с указанием пути к модели
  streamlit run app.py -- --model-path llm_models/Qwen2.5-Coder-3B
        """
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Запустить в Mock режиме (без реальной LLM, для тестов)"
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers", "mock"],
        help="Принудительно выбрать backend"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Имя локальной модели (например, Qwen2.5-Coder-3B)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Полный путь к модели (переопределяет --model)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Переопределяем настройки если указаны
    from config.settings import override_settings
    
    if args.model_path:
        # Используем указанный путь
        override_settings(
            llm_model=args.model_path,
            use_local_model=True
        )
    elif args.model:
        # Используем имя модели из llm_models
        override_settings(
            llm_model=args.model,
            use_local_model=True
        )
    
    # Запускаем с аргументами
    main(
        mock_mode=args.mock,
        force_backend=args.backend
    )
