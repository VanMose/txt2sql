"""
Database Viewer компонент.

Компоненты для просмотра баз данных и таблиц.
"""
import logging
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

from src.db.schema_loader import SchemaLoader
from src.db.executor import SQLExecutor

logger = logging.getLogger(__name__)


def load_table_schema(db_path: str, table_name: str) -> Optional[Dict[str, Any]]:
    """
    Загрузить схему таблицы.
    
    Args:
        db_path: Путь к базе данных.
        table_name: Имя таблицы.
    
    Returns:
        Информация о таблице или None.
    """
    try:
        loader = SchemaLoader(db_path)
        tables = loader.load_full_schema(include_row_count=True)
        loader.close()
        
        for table in tables:
            if table.name == table_name:
                return {
                    "name": table.name,
                    "columns": table.column_names,
                    "column_types": table.column_types,
                    "row_count": table.row_count,
                    "foreign_keys": table.foreign_keys,
                    "primary_key": table.primary_key,
                }
        
        return None
    except Exception as e:
        logger.error(f"Failed to load table schema: {e}")
        return None


def execute_table_query(db_path: str, table_name: str, limit: int = 1000) -> tuple[bool, Any, str]:
    """
    Выполнить SELECT * FROM table.
    
    Args:
        db_path: Путь к базе данных.
        table_name: Имя таблицы.
        limit: Лимит строк.
    
    Returns:
        (success, result, error_message)
    """
    sql = f"SELECT * FROM {table_name} LIMIT {limit}"
    
    try:
        executor = SQLExecutor(db_path)
        success, result = executor.execute(sql)
        
        if success:
            # Convert list of tuples to DataFrame
            if result:
                df = pd.DataFrame(result)
                return True, df, ""
            else:
                return True, pd.DataFrame(), ""
        else:
            return False, None, str(result)
    except Exception as e:
        return False, None, str(e)


def render_table_schema_card(table_info: Dict[str, Any]) -> None:
    """
    Отрисовать карточку схемы таблицы.
    
    Args:
        table_info: Информация о таблице.
    """
    table_name = table_info["name"]
    row_count = table_info["row_count"]
    columns = table_info["columns"]
    column_types = table_info["column_types"]
    primary_key = table_info["primary_key"]
    foreign_keys = table_info["foreign_keys"]
    
    # Заголовок
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📄 {table_name}")
    with col2:
        st.metric("Строк", row_count if row_count else "N/A")
    
    # Схема
    with st.expander("📐 Схема таблицы", expanded=False):
        # Таблица колонок
        schema_data = []
        for col in columns:
            pk_marker = "🔑 PK" if col == primary_key else ""
            fk_marker = "🔗 FK" if any(fk.from_column == col for fk in foreign_keys) else ""
            col_type = column_types.get(col, "UNKNOWN") if column_types else "UNKNOWN"
            
            markers = []
            if pk_marker:
                markers.append(pk_marker)
            if fk_marker:
                markers.append(fk_marker)
            
            schema_data.append({
                "Column": col,
                "Type": col_type,
                "Keys": " ".join(markers),
            })
        
        st.dataframe(
            pd.DataFrame(schema_data),
            width="stretch",
            hide_index=True,
        )
        
        # Foreign Keys
        if foreign_keys:
            st.markdown("**🔗 Внешние ключи:**")
            fk_data = []
            for fk in foreign_keys:
                fk_data.append({
                    "From Column": fk.from_column,
                    "To Table": fk.table,
                    "To Column": fk.to_column,
                })
            st.dataframe(
                pd.DataFrame(fk_data),
                width="stretch",
                hide_index=True,
            )


def render_table_data_viewer(
    db_path: str,
    table_name: str,
    table_info: Optional[Dict[str, Any]] = None,
    default_limit: int = 1000,
) -> None:
    """
    Отрисовать просмотрщик данных таблицы.
    
    Args:
        db_path: Путь к базе данных.
        table_name: Имя таблицы.
        table_info: Информация о таблице (опционально).
        default_limit: Лимит строк по умолчанию.
    """
    # Загружаем схему если не передана
    if table_info is None:
        table_info = load_table_schema(db_path, table_name)
    
    if table_info:
        render_table_schema_card(table_info)
        st.divider()
    
    # Кнопки управления
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        limit = st.number_input(
            "Лимит:",
            min_value=10,
            max_value=10000,
            value=default_limit,
            step=100,
            key=f"limit_{table_name}",
        )
    with col2:
        if st.button("🔄 Обновить", key=f"refresh_{table_name}", width="stretch"):
            st.rerun()
    
    # Выполняем запрос
    with st.spinner(f"⏳ Загрузка данных из {table_name}..."):
        success, result, error = execute_table_query(db_path, table_name, limit)
    
    if success and result is not None:
        # Показываем данные
        st.dataframe(result, width="stretch", height=400)
        
        # Статистика
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"✅ Показано строк: {len(result)}")
        with col2:
            st.caption(f"📊 Столбцов: {len(result.columns)}")
        with col3:
            if table_info and table_info.get("row_count"):
                total = table_info["row_count"]
                progress = min(len(result) / total, 1.0) if total > 0 else 0
                st.progress(progress, text=f"Загружено: {len(result)}/{total}")
    else:
        st.error(f"❌ Ошибка выполнения запроса: {error}")


def render_custom_sql_executor(db_path: str, table_name: str) -> None:
    """
    Отрисовать исполнитель кастомных SQL запросов.
    
    Args:
        db_path: Путь к базе данных.
        table_name: Имя таблицы для шаблона.
    """
    with st.expander("✏️ Пользовательский SQL запрос", expanded=False):
        custom_sql = st.text_area(
            "SQL:",
            value=f"SELECT * FROM {table_name} LIMIT 100",
            height=100,
            key=f"custom_sql_{table_name}",
            help="Введите SQL запрос для выполнения",
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("▶️ Выполнить", key=f"execute_{table_name}", width="stretch"):
                try:
                    executor = SQLExecutor(db_path)
                    success, result = executor.execute(custom_sql)
                    
                    if success and result:
                        df = pd.DataFrame(result)
                        st.success(f"✅ Успешно! Строк: {len(df)}")
                        st.dataframe(df, width="stretch", height=400)
                    else:
                        st.error(f"❌ Ошибка: {result}")
                except Exception as e:
                    st.error(f"❌ Ошибка выполнения: {e}")
