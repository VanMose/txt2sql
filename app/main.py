# app\main.py
"""
Streamlit UI для Text-to-SQL пайплайна v2.0.

Production features:
- Hybrid Retrieval (Vector + Graph)
- Query Understanding
- SQL Validation
- Metrics Dashboard
- Pipeline Progress

Запуск:
    conda activate llm_env
    streamlit run app/main.py
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Добавляем project root в path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import get_settings
from src.services.pipeline_service import PipelineService, DatabaseDiscoveryService, QueryResult
from src.services.metrics import metrics, get_metrics_summary
from src.db.schema_loader import SchemaLoader

# Импортируем компоненты
from app.components.sidebar import render_sidebar
from app.components.query_input import render_query_input
from app.components.results_view import render_result
from app.components.metrics_panel import render_metrics_dashboard

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Настройки страницы
st.set_page_config(
    page_title="Text-to-SQL v2.0 | Production Pipeline",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Session State
# ============================================================================
def init_session_state() -> None:
    """Инициализировать session state."""
    defaults = {
        "service": None,
        "db_paths": [],
        "result": None,
        "last_query": "",
        "initialized": False,
        "show_metrics": False,
        "show_debug": False,
        "query_history": [],
        "query_input": "",
        "show_schema_for": None,  # 🔥 Для отображения схемы БД
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# Helper Functions
# ============================================================================
def discover_databases(data_dir: Path) -> List[str]:
    """Автоматически обнаружить базы данных."""
    return DatabaseDiscoveryService.discover(str(data_dir))


def initialize_pipeline(db_paths: List[str]) -> bool:
    """Инициализировать PipelineService."""
    try:
        settings = get_settings()

        with st.spinner("🔧 Инициализация пайплайна..."):
            service = PipelineService(
                db_paths=db_paths,
                use_local_qdrant=settings.qdrant_use_local,
                qdrant_local_path=settings.qdrant_local_path,
            )

            if service.initialize():
                st.session_state.service = service
                st.session_state.db_paths = db_paths
                st.session_state.initialized = True

                logger.info(f"Pipeline initialized with {len(db_paths)} databases")
                return True
            return False

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        st.error(f"❌ Initialization failed: {e}")
        return False


def run_query(query: str) -> Optional[QueryResult]:
    """Выполнить запрос через PipelineService."""
    if not st.session_state.service:
        return None

    # Pipeline Progress
    progress_bar = st.progress(0, text="🔀 Роутинг баз данных...")
    status_text = st.empty()
    
    with st.spinner("⚡ Обработка запроса..."):
        result = st.session_state.service.run_query(query)
        
        # Update progress по этапам
        if result and result.latencies:
            total_ms = sum(result.latencies.values())
            elapsed = 0
            
            for step, latency_ms in result.latencies.items():
                elapsed += latency_ms
                progress = min(elapsed / total_ms, 1.0)
                
                step_names = {
                    "routing": "🔀 Роутинг",
                    "sql_generation": "✏️ Генерация SQL",
                    "execution_judge": "⚖️ Выполнение + Оценка",
                    "refinement": "🔧 Рефайнмент",
                }
                
                status_text.text(f"{step_names.get(step, step)}...")
                progress_bar.progress(progress)
        
        status_text.text("✅ Готово!")
        progress_bar.progress(1.0)

    # Запись метрик
    from src.services.metrics import QueryMetrics, QueryStatus
    import uuid
    
    if result:
        query_metrics = QueryMetrics(
            query_id=str(uuid.uuid4())[:8],
            query_text=query,
            status=QueryStatus.SUCCESS if result.success else QueryStatus.FAILED,
            latency_ms=total_ms if result and result.latencies else 0,
            sql_generated=result.sql if result else None,
            sql_valid=result.success if result else False,
            confidence=result.confidence if result else 0.0,
            tables_retrieved=len(result.selected_databases) if result else 0,
            cache_hit=False,
            error_message=result.error if result and not result.success else None,
            latencies_breakdown=result.latencies if result else {},
        )
        metrics.record_query(query_metrics)

    # Добавление в историю
    if query and result and result.success:
        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)
            if len(st.session_state.query_history) > 20:
                st.session_state.query_history.pop(0)

    return result


# ============================================================================
# Main App
# ============================================================================
def main() -> None:
    """Main application."""
    logger.info("Starting Streamlit app")

    st.title("🗄️ Text-to-SQL v2.0")
    st.markdown(
        """
    **Production система** преобразования естественного языка в SQL:
    - ⚡ **Hybrid Retrieval**: Vector (Qdrant) + Graph (Neo4j)
    - 🎯 **Query Understanding**: Intent, entities, filters
    - 🔄 **Cross-encoder Reranking**: Точный выбор таблиц
    - 📐 **Schema Compression**: Формат с PK/FK/types
    - ✅ **SQL Validation**: Валидация перед выполнением
    - 💾 **TTL Cache**: Ускорение повторных запросов
    """
    )

    init_session_state()
    render_sidebar()

    # Initialization
    if not st.session_state.initialized:
        st.subheader("🚀 Начальная настройка")
        settings = get_settings()
        data_dir = Path(settings.base_dir) / "data"

        # Проверка существования директории
        if not data_dir.exists():
            st.error(f"❌ Директория `{data_dir.name}/` не найдена")
            st.info("Создайте папку data/ и поместите туда SQLite базы данных")
            return

        db_paths = discover_databases(data_dir)

        if db_paths:
            st.success(f"✅ Найдено {len(db_paths)} БД в `{data_dir.name}/`")

            # Initialize
            if st.button(
                "🚀 Инициализировать пайплайн",
                type="primary",
                use_container_width=True,
            ):
                if initialize_pipeline(db_paths):
                    st.success("✅ Пайплайн инициализирован!")
                    st.rerun()
        else:
            st.error(f"❌ БД не найдены в `{data_dir.name}/`")
            st.info(
                """
                Добавьте SQLite базы данных в папку `data/`:
                ```
                data/
                ├── movie_1/
                │   └── movie_1.sqlite
                └── music_1/
                    └── music_1.sqlite
                ```
                """
            )

    # Main interface
    if st.session_state.initialized:
        st.divider()

        # Query input
        query = render_query_input()

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(
                "🚀 Выполнить запрос",
                type="primary",
                use_container_width=True,
                disabled=not query,
            ):
                result = run_query(query)
                if result:
                    st.session_state.result = result
                    st.session_state.last_query = query
                else:
                    st.error("❌ Запрос не выполнен")

        with col2:
            if st.button("🗑️ Очистить", use_container_width=True):
                st.session_state.result = None
                st.session_state.last_query = ""
                st.rerun()

        if st.session_state.last_query:
            st.info(f"📌 Последний запрос: **{st.session_state.last_query}**")

        if st.session_state.result:
            render_result(st.session_state.result, db_paths=st.session_state.db_paths)

        # Metrics Dashboard
        render_metrics_dashboard()

    # Footer
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        from src.llm.model_loader import ModelLoader
        platform_info = ModelLoader.get_platform_info()
        st.caption(
            f"**Backend:** {platform_info['recommended_backend']} | "
            f"**CUDA:** {'✅' if platform_info.get('cuda_available') else '❌'}"
        )

    with col2:
        settings = get_settings()
        st.caption(
            f"**Модель:** {settings.llm_model} | "
            f"**4-bit:** {'✅' if settings.use_4bit_quantization else '❌'}"
        )

    with col3:
        st.caption("**v2.0** | Production Pipeline")
    
    with col4:
        # Pipeline status
        pipeline_status = "🟢 Ready" if st.session_state.initialized else "🔴 Not initialized"
        st.caption(f"**Status:** {pipeline_status}")


if __name__ == "__main__":
    main()
