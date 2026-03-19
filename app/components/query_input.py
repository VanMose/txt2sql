"""
Query Input компонент с примерами и историей.

Исправления v3.0.4:
- Убран value= из text_input (Streamlit сам управляет)
- Уникальные key для кнопок (hash)
- Нет дублирования session_state
"""
import hashlib
import streamlit as st

# Примеры запросов
QUERY_EXAMPLES = {
    "Фильмы": [
        "Покажи все фильмы",
        "Фильмы с рейтингом выше 7",
        "Количество фильмов",
        "Найди фильмы 1990 года",
    ],
    "Музыка": [
        "Покажи все песни",
        "Песни 1990 года",
        "Артисты по алфавиту",
        "Количество альбомов",
    ],
    "Общие": [
        "Показать все таблицы",
        "Структура базы данных",
    ],
}


def render_query_input() -> str:
    """
    Отрисовать ввод запроса с примерами.

    Returns:
        Текст запроса или пустая строка.
    """
    st.subheader("📝 Введите запрос")

    # Query examples
    with st.expander("💡 Примеры запросов", expanded=False):
        cols = st.columns(len(QUERY_EXAMPLES))
        
        for idx, (category, examples) in enumerate(QUERY_EXAMPLES.items()):
            with cols[idx]:
                st.markdown(f"**{category}**")
                for example in examples:
                    # 🔥 Уникальный key через hash
                    key = f"ex_{hashlib.md5(example.encode()).hexdigest()[:8]}"
                    if st.button(example, key=key, use_container_width=True):
                        st.session_state.query_input = example
                        st.rerun()

    # Query history
    query_history = st.session_state.get("query_history", [])
    if query_history:
        with st.expander(f"📚 История запросов ({len(query_history)})", expanded=False):
            for i, query in enumerate(reversed(query_history[-10:]), 1):
                # 🔥 Уникальный key через hash
                key = f"hist_{hashlib.md5(query.encode()).hexdigest()[:8]}"
                if st.button(f"{i}. {query[:50]}...", key=key, use_container_width=True):
                    st.session_state.query_input = query
                    st.rerun()

    # 🔥 Query input - Streamlit сам управляет значением через session_state
    # Убран value= - Streamlit автоматически берёт из session_state по key
    st.text_input(
        "Запрос на естественном языке",
        placeholder="Например: Покажи все фильмы или Найди песни 1990 года",
        key="query_input",
        label_visibility="collapsed",
    )

    return st.session_state.get("query_input", "")
