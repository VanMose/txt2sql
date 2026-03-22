"""
Service слой для Text-to-SQL пайплайна.

Note: PipelineService is NOT imported here to avoid circular imports.
Import it directly: from services.pipeline_service import PipelineService
"""
from .metrics import metrics, QueryMetrics, QueryStatus, MetricsCollector, get_metrics_summary, record_query_latency
from .result_cache import ResultCache, get_result_cache

__all__ = [
    # Metrics
    "metrics",
    "QueryMetrics",
    "QueryStatus",
    "MetricsCollector",
    "get_metrics_summary",
    "record_query_latency",
    # Result Cache
    "ResultCache",
    "get_result_cache",
]
