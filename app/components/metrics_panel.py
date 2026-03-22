# app\components\metrics_panel.py
"""
Metrics Panel компонент для Text-to-SQL v0.1.5

Production metrics:
- Query latency (p50, p95, p99)
- Success rate
- Cache hit rate
- SQL validity rate
- QPS
"""
import streamlit as st

from src.services.metrics import metrics, get_metrics_summary


def render_metrics_dashboard() -> None:
    """Отрисовать панель метрик."""
    if not st.session_state.get("show_metrics", False):
        return

    st.subheader("📊 Панель метрик")

    summary = get_metrics_summary()

    # Query Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Всего запросов",
            summary.get("total_queries", 0),
            delta=f"{summary.get('success_rate', 0):.1f}% успех",
        )
    
    with col2:
        st.metric(
            "Cache Hit Rate",
            f"{summary.get('cache_hit_rate', 0):.1f}%",
        )
    
    with col3:
        st.metric(
            "SQL Validity",
            f"{summary.get('sql_validity_rate', 0):.1f}%",
        )
    
    with col4:
        st.metric(
            "QPS",
            f"{summary.get('qps', 0):.2f}",
        )

    # Latency Stats
    latency = summary.get("latency_ms", {})
    if latency and any(v > 0 for v in latency.values()):
        st.divider()
        st.markdown("### ⏱️ Распределение времени (ms)")

        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Min", f"{latency.get('min', 0):.0f}мс")
        
        with col2:
            st.metric("p50", f"{latency.get('p50', 0):.0f}мс")
        
        with col3:
            st.metric("p95", f"{latency.get('p95', 0):.0f}мс")
        
        with col4:
            st.metric("p99", f"{latency.get('p99', 0):.0f}мс")
        
        with col5:
            st.metric("Mean", f"{latency.get('mean', 0):.0f}мс")

    # Error tracking
    top_errors = summary.get("top_errors", [])
    if top_errors:
        st.divider()
        st.markdown("### ❌ Топ ошибок")
        
        for error, count in top_errors[:5]:
            st.text(f"• {error[:80]}... ({count} раз)")

    # Tokens
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Всего токенов",
            summary.get("total_tokens", 0),
        )
    
    with col2:
        st.metric(
            "Токенов/сек",
            f"{summary.get('tokens_per_second', 0):.1f}",
        )
