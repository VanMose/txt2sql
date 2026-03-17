"""
Results View компонент для отображения результатов.

Исправления v3.0.4:
- Исправлена переменная cols → columns
- Polars CSV encode
- SQL highlighting через st.code
- Row limit для результатов
- Уникальный key для SQL editor
"""
import logging
from typing import Any, Dict, Optional

import streamlit as st

from src.services.pipeline_service import QueryResult

logger = logging.getLogger(__name__)


def render_sql_editor(result: QueryResult) -> Optional[str]:
    """
    Отрисовать SQL editor с возможностью редактирования.

    Returns:
        Изменённый SQL или None.
    """
    st.markdown("### 💾 Сгенерированный SQL")

    # 🔥 Уникальный key для SQL editor
    sql_key = f"sql_editor_{hash(result.sql)}"

    # SQL highlighting через st.code
    st.code(result.sql, language="sql", line_numbers=False)

    # Editor для редактирования
    edited_sql = st.text_area(
        "Редактировать SQL",
        value=result.sql,
        height=150,
        key=sql_key,
        help="Можно редактировать и выполнить заново"
    )

    cols = st.columns([1, 4])
    with cols[0]:
        if st.button("▶️ Выполнить", key="rerun_sql"):
            if edited_sql != result.sql:
                return edited_sql

    return None


def execute_edited_sql(edited_sql: str, db_paths: list) -> tuple[bool, Any]:
    """
    Выполнить отредактированный SQL запрос.
    
    Args:
        edited_sql: Отредактированный SQL.
        db_paths: Пути к базам данных.
    
    Returns:
        (success, result) кортеж.
    """
    from src.db.multi_db_executor import MultiDBExecutor
    
    try:
        executor = MultiDBExecutor(db_paths)
        success, result = executor.execute(edited_sql)
        executor.close()
        return success, result
    except Exception as e:
        logger.error(f"Edited SQL execution failed: {e}")
        return False, str(e)


def render_metrics(result: QueryResult) -> None:
    """Отрисовать метрики запроса."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Уверенность", f"{result.confidence:.0%}")
    with col2:
        st.metric("Выбрано БД", len(result.selected_databases))
    with col3:
        st.metric("Рефайнментов", result.refinement_count)


def render_selected_databases(result: QueryResult) -> None:
    """Отрисовать выбранные базы данных."""
    if not result.selected_databases:
        return
    
    st.markdown("### 🎯 Выбранные базы данных")
    
    for db in result.selected_databases:
        db_name = db.get("db_name", "Unknown")
        confidence = db.get("confidence", 0)
        
        with st.expander(f"📁 {db_name} ({confidence:.0%})", expanded=False):
            st.text(f"Путь: {db.get('db_path', 'N/A')}")
            st.text(f"Таблицы: {', '.join(db.get('tables', []))}")
            st.text(f"Причина: {db.get('reason', 'N/A')}")


def render_execution_result(result: QueryResult) -> None:
    """Отрисовать результаты выполнения."""
    st.markdown("### 📋 Результаты выполнения")
    
    if result.execution_result is not None:
        # Проверка типа DataFrame (polars или pandas)
        if hasattr(result.execution_result, "head"):
            # DataFrame (polars или pandas)
            st.dataframe(result.execution_result, use_container_width=True)
            
            # Info - 🔥 Исправлено: columns → num_columns
            if hasattr(result.execution_result, "shape"):
                rows, num_columns = result.execution_result.shape
            elif hasattr(result.execution_result, "__len__"):
                rows = len(result.execution_result)
                num_columns = len(result.execution_result.columns) if hasattr(result.execution_result, "columns") else 0
            else:
                rows, num_columns = 0, 0
            
            cols = st.columns(2)
            with cols[0]:
                st.caption(f"Строк: {rows}")
            with cols[1]:
                st.caption(f"Столбцов: {num_columns}")
            
            # Download button - 🔥 Polars CSV encode
            try:
                if hasattr(result.execution_result, "write_csv"):
                    # Polars DataFrame → bytes
                    csv = result.execution_result.write_csv().encode('utf-8')
                elif hasattr(result.execution_result, "to_csv"):
                    # Pandas DataFrame
                    csv = result.execution_result.to_csv(index=False).encode('utf-8')
                else:
                    csv = None
                
                if csv:
                    st.download_button(
                        label="📥 Скачать CSV",
                        data=csv,
                        file_name="query_result.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                logger.warning(f"Failed to generate CSV: {e}")
            
        elif isinstance(result.execution_result, list):
            if result.execution_result:
                # 🔥 Row limit для списков
                if len(result.execution_result) > 1000:
                    st.info(f"Показано 1000 из {len(result.execution_result)} результатов")
                    st.dataframe(result.execution_result[:1000], use_container_width=True)
                else:
                    st.dataframe(result.execution_result, use_container_width=True)
            else:
                st.info("Запрос не вернул результатов")
        else:
            st.json(result.execution_result)
    else:
        if result.error:
            st.warning(f"Ошибка выполнения: {result.error}")
        else:
            st.warning("Нет результатов выполнения")


def render_latencies(result: QueryResult) -> None:
    """Отрисовать время выполнения этапов."""
    if not result.latencies:
        return
    
    st.markdown("### ⏱️ Время выполнения этапов")
    
    # Pipeline visualization
    steps = {
        "routing": "🔀 Роутинг",
        "sql_generation": "✏️ Генерация SQL",
        "execution_judge": "⚖️ Выполнение + Оценка",
        "refinement": "🔧 Рефайнмент",
    }
    
    cols = st.columns(len(result.latencies))
    
    for idx, (step, latency_ms) in enumerate(result.latencies.items()):
        step_name = steps.get(step, step.replace("_", " ").title())
        with cols[idx]:
            st.metric(step_name, f"{latency_ms:.0f}мс")
    
    # Progress bar
    total_ms = sum(result.latencies.values())
    st.progress(min(total_ms / 30000, 1.0), text=f"Общее время: {total_ms:.0f}мс")


def render_result(result: QueryResult, db_paths: Optional[list] = None) -> None:
    """
    Отрисовать результат запроса.

    Args:
        result: Результат запроса.
        db_paths: Пути к базам данных для выполнения edited SQL.
    """
    st.subheader("📊 Результат запроса")

    # SQL Editor
    new_sql = render_sql_editor(result)
    if new_sql and new_sql != result.sql:
        # ✅ Выполнить запрос с новым SQL
        st.info(f"Выполнение нового SQL: {new_sql[:100]}...")
        
        if db_paths:
            with st.spinner("⚡ Выполнение SQL..."):
                success, exec_result = execute_edited_sql(new_sql, db_paths)
            
            if success:
                st.success("✅ SQL выполнен успешно!")
                # Обновляем result в session state
                result.sql = new_sql
                result.execution_result = exec_result
                result.success = True
                st.rerun()
            else:
                st.error(f"❌ Ошибка выполнения: {exec_result}")
                result.sql = new_sql
                result.execution_result = None
                result.success = False
                result.error = str(exec_result)

    st.divider()

    # Metrics
    render_metrics(result)

    st.divider()

    # Selected databases
    render_selected_databases(result)

    st.divider()

    # Execution result
    render_execution_result(result)

    st.divider()

    # Latencies
    render_latencies(result)

    # Debug info
    if st.session_state.get("show_debug", False):
        st.divider()
        st.markdown("### 🐛 Отладочная информация")
        st.json(result.to_dict())
