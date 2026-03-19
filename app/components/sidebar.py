"""
Sidebar компонент для Text-to-SQL UI.

Исправления v3.0.4:
- @st.cache_data для render_database_preview
- Полная схема БД с колонками и типами
- Уникальные key для кнопок
- Правильное управление session_state
"""
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src.services.pipeline_service import PipelineService, DatabaseDiscoveryService
from src.db.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300)
def load_database_info(db_path: str) -> Dict[str, Any]:
    """
    Загрузить информацию о БД с кэшированием.
    
    Args:
        db_path: Путь к базе данных.
        
    Returns:
        Информация о БД.
    """
    try:
        loader = SchemaLoader(db_path)
        tables = loader.load_full_schema(use_cache=True)
        stats = loader.get_stats()
        loader.close()

        return {
            "path": db_path,
            "name": Path(db_path).stem,
            "tables": tables,  # List[TableInfo]
            "tables_count": len(tables),
            "load_time_ms": stats.get("load_time_ms", 0),
            "error": None,
        }
    except Exception as e:
        return {
            "path": db_path,
            "name": Path(db_path).stem,
            "error": str(e),
        }


def render_database_schema(db_path: str, db_name: str) -> None:
    """
    Отрисовать полную схему БД.
    
    Args:
        db_path: Путь к базе данных.
        db_name: Имя базы данных.
    """
    info = load_database_info(db_path)
    
    if info.get("error"):
        st.error(f"Ошибка: {info['error']}")
        return
    
    tables = info.get("tables", [])
    
    for table in tables:
        with st.expander(f"📄 {table.name}", expanded=False):
            # Columns
            st.markdown("**Колонки:**")
            for col in table.columns:
                pk_marker = "🔑 " if col.pk else ""
                fk_marker = "🔗 " if any(fk.from_column == col.name for fk in table.foreign_keys) else ""
                st.caption(f"{pk_marker}{fk_marker}{col.name}: {col.type}")
            
            # Foreign Keys
            if table.foreign_keys:
                st.markdown("**Внешние ключи:**")
                for fk in table.foreign_keys:
                    st.caption(f"{fk.from_column} → {fk.table}.{fk.to_column}")


def render_sidebar() -> None:
    """Отрисовать боковую панель."""
    with st.sidebar:
        st.title("⚙️ Настройки")


        # Model Selection
        st.subheader("🤖 Модель")
        available_models = [
            "qwen2.5-coder:1.5b",
            "qwen2.5:3b",
            "qwen2.5:7b",
            "sqlcoder:7b",
        ]
        # Текущая модель
        from src.config.settings import get_settings, override_settings
        current_model = st.session_state.get("current_model", get_settings().llm_model)
        if current_model not in available_models:
            available_models.insert(0, current_model)

        selected_model = st.selectbox(
            "Выберите модель",
            available_models,
            index=available_models.index(current_model),
            key="model_selector",
        )
        # Кастомная модель
        custom_model = st.text_input(
            "Или введите имя модели",
            placeholder="например: llama3:8b",
            key="custom_model_input",
        )
        if custom_model:
            selected_model = custom_model

        # Применить если модель изменилась
        if selected_model != st.session_state.get("current_model", current_model):
            st.session_state.current_model = selected_model
            override_settings(model_name=selected_model)
            from src.llm.model_loader import ModelLoader
            ModelLoader._current_model_name = selected_model
            if st.session_state.get("service") and st.session_state.service._pipeline:
                st.session_state.service._pipeline.llm_service = None
            st.success(f"Модель: {selected_model}")
            st.rerun()


        st.divider()



        # Status
        st.subheader("📊 Статус")
        if st.session_state.get("initialized", False):
            db_count = len(st.session_state.get("db_paths", []))
            st.success(f"✅ {db_count} БД подключено")
        else:
            st.warning("⚠️ Не инициализировано")

        # Databases
        st.subheader("🗄️ Базы данных")
        db_paths = st.session_state.get("db_paths", [])

        if db_paths:
            with st.expander("📁 Файлы БД", expanded=False):
                st.caption(f"Всего БД: {len(db_paths)}")
                for db_path in db_paths:
                    db_name = Path(db_path).stem
                    info = load_database_info(db_path)

                    if info.get("error"):
                        st.error(f"• {db_name}: {info['error']}")
                    else:
                        cols = st.columns([3, 1])
                        with cols[0]:
                            st.text(f"• {db_name}: {info['tables_count']} таблиц")
                        with cols[1]:
                            st.text(f"{info['load_time_ms']:.0f}мс")

                        # Schema viewer - 🔥 уникальная кнопка
                        key = f"schema_{hashlib.md5(db_path.encode()).hexdigest()[:8]}"
                        if st.button("👁️ Схема", key=key, use_container_width=True):
                            st.session_state.show_schema_for = db_path
                            st.rerun()

                        # Показываем схему если выбрана
                        if st.session_state.get("show_schema_for") == db_path:
                            with st.expander(f"Схема {db_name}", expanded=True):
                                render_database_schema(db_path, db_name)

                                # Кнопка закрыть
                                if st.button("✕ Закрыть", key=f"close_{key}"):
                                    st.session_state.show_schema_for = None
                                    st.rerun()
        else:
            st.info("Нет подключенных БД")

        # 🔥 Vector DB Content - показываем что реально индексировано
        if st.session_state.get("service"):
            with st.expander("🔍 Vector DB содержимое", expanded=False):
                stats = st.session_state.service.get_vector_db_stats()
                points_count = stats.get("points_count", 0)
                st.caption(f"Всего точек: {points_count}")
                
                if points_count > 0 and st.session_state.get("service") and st.session_state.service._pipeline:
                    # Показываем распределение по базам данных
                    try:
                        all_tables = st.session_state.service._pipeline.vector_db.get_all_tables()
                        db_counts = {}
                        for table in all_tables:
                            db_name = table.db_name
                            db_counts[db_name] = db_counts.get(db_name, 0) + 1
                        
                        if db_counts:
                            st.markdown("**Распределение по БД:**")
                            for db_name, count in sorted(db_counts.items()):
                                st.text(f"  {db_name}: {count} таблиц")
                    except Exception as e:
                        st.caption(f"Не удалось получить детали: {e}")

        st.divider()

        # Vector DB Stats
        st.subheader("🔍 Vector DB (Qdrant)")
        if st.session_state.get("service"):
            stats = st.session_state.service.get_vector_db_stats()
            st.metric("Таблиц индексировано", stats.get("points_count", 0))
        else:
            st.info("Не инициализировано")

        st.divider()

        # Graph DB Stats
        st.subheader("🕷️ Graph DB (Neo4j)")
        if st.session_state.get("service"):
            stats = st.session_state.service.get_graph_db_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Таблиц", stats.get("tables", 0))
            with col2:
                st.metric("FK связей", stats.get("foreign_keys", 0))
        else:
            st.info("Не инициализировано")

        st.divider()

        # Metrics Toggle
        st.subheader("📈 Метрики")
        st.session_state.show_metrics = st.checkbox(
            "Показывать панель метрик",
            value=st.session_state.get("show_metrics", False),
        )

        st.session_state.show_debug = st.checkbox(
            "Режим отладки",
            value=st.session_state.get("show_debug", False),
        )

        st.divider()

        # Actions
        st.subheader("🔧 Действия")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Индекс", use_container_width=True):
                if st.session_state.get("service"):
                    with st.spinner("Индексация..."):
                        # 🔥 FIX: Переоткрываем базы данных перед переиндексацией
                        from src.config.settings import get_settings
                        
                        settings = get_settings()
                        data_dir = Path(settings.base_dir) / "data"
                        current_db_paths = DatabaseDiscoveryService.discover(str(data_dir))
                        
                        # Проверяем, изменился ли список баз данных
                        if set(current_db_paths) != set(st.session_state.get("db_paths", [])):
                            logger.info(f"Database list changed: {len(st.session_state.get('db_paths', []))} → {len(current_db_paths)}")
                            st.session_state.db_paths = current_db_paths
                            
                            # Переинициализируем сервис с новым списком баз данных
                            st.session_state.service.db_paths = current_db_paths
                            if st.session_state.service._pipeline:
                                st.session_state.service._pipeline.db_paths = current_db_paths
                        
                        # Переиндексация с обновленным списком
                        st.session_state.service.index_databases(force_reindex=True)
                        
                        # 🔥 Перезагружаем страницу для обновления Vector DB клиента
                        st.success(f"✅ Переиндексировано ({len(current_db_paths)} БД)")
                        st.info("🔄 Перезагрузка страницы для применения изменений...")
                        st.rerun()

        with col2:
            if st.button("🗑️ Кэш", use_container_width=True):
                if st.session_state.get("service"):
                    st.session_state.service.clear_router_cache()
                    st.success("✅ Кэш очищен")

        if st.button("🔄 Сброс", use_container_width=True):
            if st.session_state.get("service"):
                st.session_state.service.close()
            st.session_state.service = None
            st.session_state.db_paths = []
            st.session_state.initialized = False
            st.session_state.result = None
            st.session_state.show_schema_for = None
            st.success("✅ Сброшено")
            st.rerun()
