# Database Viewer Page

"""
Страница просмотра баз данных и таблиц.

Запуск:
    streamlit run app/pages/01_📊_Database_Viewer.py
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

# Добавляем project root в path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.pipeline_service import DatabaseDiscoveryService
from app.components.database_viewer import (
    load_table_schema,
    render_table_schema_card,
    render_table_data_viewer,
    render_custom_sql_executor,
)

# Настройка страницы
st.set_page_config(
    page_title="📊 Database Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)


def get_database_tables(db_path: str) -> List[Dict[str, Any]]:
    """Получить список таблиц из базы данных."""
    try:
        from src.db.schema_loader import SchemaLoader
        
        loader = SchemaLoader(db_path)
        tables = loader.load_full_schema(include_row_count=True)
        loader.close()
        
        return [
            {
                "name": table.name,
                "columns": table.column_names,
                "column_types": table.column_types,
                "row_count": table.row_count,
                "foreign_keys": table.foreign_keys,
                "primary_key": table.primary_key,
            }
            for table in tables
        ]
    except Exception as e:
        st.error(f"❌ Error loading tables: {e}")
        return []


def render_database_selector() -> Optional[str]:
    """Отрисовать селектор базы данных."""
    st.sidebar.subheader("🗄️ Выбор базы данных")
    
    # Discover databases
    data_dir = Path(__file__).parent.parent.parent / "data"
    db_paths = DatabaseDiscoveryService.discover(str(data_dir))
    
    if not db_paths:
        st.sidebar.warning("❌ Базы данных не найдены")
        return None
    
    # Создаем словарь с именами для отображения
    db_options = {}
    for db_path in db_paths:
        db_name = Path(db_path).parent.name
        db_options[db_name] = db_path
    
    # Селектор
    selected_name = st.sidebar.selectbox(
        "Выберите БД:",
        options=list(db_options.keys()),
        index=0,
    )
    
    # Показываем информацию о выбранной БД
    if selected_name:
        db_path = db_options[selected_name]
        st.sidebar.info(f"**Путь:** `{db_path}`")
        
        # Считаем количество таблиц
        try:
            from src.db.schema_loader import SchemaLoader
            loader = SchemaLoader(db_path)
            tables = loader.get_tables()
            loader.close()
            st.sidebar.caption(f"Таблиц: {len(tables)}")
        except:
            pass
    
    return db_options.get(selected_name)


def render_table_viewer(db_path: str) -> None:
    """Отрисовать просмотрщик таблиц."""
    st.subheader("📋 Таблицы")
    
    # Загружаем список таблиц
    tables = get_database_tables(db_path)
    
    if not tables:
        st.warning("Нет таблиц в этой базе данных")
        return
    
    # Создаем вкладки для каждой таблицы
    tab_names = []
    for t in tables:
        row_count = t.get('row_count', 0)
        if row_count:
            tab_names.append(f"📄 {t['name']} ({row_count:,})")
        else:
            tab_names.append(f"📄 {t['name']} (0)")
    
    tabs = st.tabs(tab_names)
    
    for tab, table in zip(tabs, tables):
        with tab:
            render_single_table(db_path, table)


def render_single_table(db_path: str, table: Dict[str, Any]) -> None:
    """Отрисовать одну таблицу."""
    table_name = table["name"]
    
    # Загружаем полную схему
    table_info = load_table_schema(db_path, table_name)
    
    if table_info:
        render_table_schema_card(table_info)
        st.divider()
    
    # Просмотрщик данных
    render_table_data_viewer(db_path, table_name, table_info, default_limit=1000)
    
    st.divider()
    
    # Кастомный SQL
    render_custom_sql_executor(db_path, table_name)


def render_main_view() -> None:
    """Основной просмотрщик."""
    st.title("📊 Database Viewer")
    st.markdown("**Просмотр таблиц и данных из всех баз данных**")
    
    st.divider()
    
    # Селектор базы данных
    db_path = render_database_selector()
    
    if not db_path:
        st.info("👈 Выберите базу данных в боковой панели")
        return
    
    st.divider()
    
    # Просмотрщик таблиц
    render_table_viewer(db_path)


if __name__ == "__main__":
    render_main_view()
