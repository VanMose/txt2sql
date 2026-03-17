# src\services\metrics.py
"""
Метрики и Observability для Text-to-SQL системы.

Production features:
- Query latency tracking (p50, p95, p99)
- SQL accuracy metrics (EM, EX)
- Retrieval metrics (Recall@k, Precision@k)
- Token usage tracking
- Error rate monitoring
- Prometheus-compatible export

Metrics:
    - query_latency_ms: Latency распределение
    - query_success_rate: % успешных запросов
    - sql_validity_rate: % валидного SQL
    - retrieval_recall: Retrieval точность
    - token_usage: Использование токенов
    - cache_hit_rate: % попаданий в кэш
"""
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class QueryStatus(Enum):
    """Статус запроса."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    CACHED = "cached"


@dataclass
class QueryMetrics:
    """Метрики одного запроса."""
    query_id: str
    query_text: str
    status: QueryStatus
    latency_ms: float
    sql_generated: Optional[str] = None
    sql_valid: bool = False
    confidence: float = 0.0
    tables_retrieved: int = 0
    tokens_used: int = 0
    cache_hit: bool = False
    error_message: Optional[str] = None
    latencies_breakdown: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в dict."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "sql_generated": self.sql_generated,
            "sql_valid": self.sql_valid,
            "confidence": self.confidence,
            "tables_retrieved": self.tables_retrieved,
            "tokens_used": self.tokens_used,
            "cache_hit": self.cache_hit,
            "error_message": self.error_message,
            "latencies_breakdown": self.latencies_breakdown,
            "timestamp": self.timestamp.isoformat(),
        }


class MetricsCollector:
    """
    Collector для сбора и агрегации метрик.
    
    Production features:
    - Rolling window statistics
    - Percentile calculations (p50, p95, p99)
    - Error rate tracking
    - Cache hit rate
    - Prometheus-compatible export
    """
    
    def __init__(self, window_size: int = 1000) -> None:
        """
        Инициализировать collector.
        
        Args:
            window_size: Размер rolling window для статистики.
        """
        self.window_size = window_size
        self._queries: List[QueryMetrics] = []
        self._latencies: List[float] = []
        self._success_count = 0
        self._failed_count = 0
        self._cached_count = 0
        self._total_tokens = 0
        self._errors: Dict[str, int] = defaultdict(int)
        
        # Retrieval metrics
        self._retrieval_hits: List[bool] = []
        
        # Start time for rate calculations
        self._start_time = datetime.now()
        
        logger.info(f"MetricsCollector initialized (window_size={window_size})")
    
    def record_query(self, metrics: QueryMetrics) -> None:
        """
        Записать метрики запроса.
        
        Args:
            metrics: QueryMetrics для записи.
        """
        self._queries.append(metrics)
        self._latencies.append(metrics.latency_ms)
        
        if metrics.status == QueryStatus.SUCCESS:
            self._success_count += 1
        elif metrics.status == QueryStatus.FAILED:
            self._failed_count += 1
            if metrics.error_message:
                self._errors[metrics.error_message[:100]] += 1
        elif metrics.status == QueryStatus.CACHED:
            self._cached_count += 1
        
        if metrics.cache_hit:
            self._cached_count += 1
        
        self._total_tokens += metrics.tokens_used
        
        # Rolling window cleanup
        if len(self._queries) > self.window_size:
            oldest = self._queries.pop(0)
            self._latencies.pop(0)
        
        logger.debug(f"Recorded query: status={metrics.status.value}, latency={metrics.latency_ms:.0f}ms")
    
    def record_retrieval_result(self, is_relevant: bool) -> None:
        """
        Записать результат retrieval.
        
        Args:
            is_relevant: релевантен ли результат.
        """
        self._retrieval_hits.append(is_relevant)
        if len(self._retrieval_hits) > self.window_size:
            self._retrieval_hits.pop(0)
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """
        Получить percentiles latency.
        
        Returns:
            Dict с p50, p95, p99.
        """
        if not self._latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        
        sorted_latencies = sorted(self._latencies)
        n = len(sorted_latencies)
        
        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        
        return {
            "p50": sorted_latencies[min(p50_idx, n - 1)],
            "p95": sorted_latencies[min(p95_idx, n - 1)],
            "p99": sorted_latencies[min(p99_idx, n - 1)],
            "mean": sum(sorted_latencies) / n,
            "min": sorted_latencies[0],
            "max": sorted_latencies[-1],
        }
    
    def get_success_rate(self) -> float:
        """Получить % успешных запросов."""
        total = self._success_count + self._failed_count
        if total == 0:
            return 1.0
        return self._success_count / total
    
    def get_cache_hit_rate(self) -> float:
        """Получить % попаданий в кэш."""
        total = len(self._queries)
        if total == 0:
            return 0.0
        return self._cached_count / total
    
    def get_sql_validity_rate(self) -> float:
        """Получить % валидного SQL."""
        valid_count = sum(1 for q in self._queries if q.sql_valid)
        total = len(self._queries)
        if total == 0:
            return 0.0
        return valid_count / total
    
    def get_retrieval_recall(self) -> float:
        """Получить retrieval recall."""
        if not self._retrieval_hits:
            return 0.0
        return sum(self._retrieval_hits) / len(self._retrieval_hits)
    
    def get_error_rate(self) -> float:
        """Получить % ошибок."""
        total = self._success_count + self._failed_count
        if total == 0:
            return 0.0
        return self._failed_count / total
    
    def get_top_errors(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Получить топ ошибок."""
        sorted_errors = sorted(self._errors.items(), key=lambda x: x[1], reverse=True)
        return sorted_errors[:limit]
    
    def get_queries_per_second(self) -> float:
        """Получить QPS."""
        elapsed = (datetime.now() - self._start_time).total_seconds()
        if elapsed <= 0:
            return 0.0
        return len(self._queries) / elapsed
    
    def get_tokens_per_second(self) -> float:
        """Получить токенов в секунду."""
        elapsed = (datetime.now() - self._start_time).total_seconds()
        if elapsed <= 0:
            return 0.0
        return self._total_tokens / elapsed
    
    def get_summary(self) -> Dict[str, Any]:
        """Получить сводную статистику."""
        percentiles = self.get_latency_percentiles()
        
        return {
            "total_queries": len(self._queries),
            "success_rate": round(self.get_success_rate() * 100, 2),
            "cache_hit_rate": round(self.get_cache_hit_rate() * 100, 2),
            "sql_validity_rate": round(self.get_sql_validity_rate() * 100, 2),
            "retrieval_recall": round(self.get_retrieval_recall() * 100, 2),
            "error_rate": round(self.get_error_rate() * 100, 2),
            "latency_ms": {
                "p50": round(percentiles["p50"], 2),
                "p95": round(percentiles["p95"], 2),
                "p99": round(percentiles["p99"], 2),
                "mean": round(percentiles["mean"], 2),
                "min": round(percentiles["min"], 2),
                "max": round(percentiles["max"], 2),
            },
            "qps": round(self.get_queries_per_second(), 2),
            "tokens_per_second": round(self.get_tokens_per_second(), 2),
            "total_tokens": self._total_tokens,
            "top_errors": self.get_top_errors(),
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
        }
    
    def export_prometheus(self) -> str:
        """
        Экспортировать метрики в Prometheus format.
        
        Returns:
            Метрики в Prometheus text format.
        """
        summary = self.get_summary()
        lines = []
        
        # Latency histogram
        lines.append("# HELP text2sql_query_latency_ms Query latency in milliseconds")
        lines.append("# TYPE text2sql_query_latency_ms gauge")
        lines.append(f"text2sql_query_latency_ms{{quantile=\"0.50\"}} {summary['latency_ms']['p50']}")
        lines.append(f"text2sql_query_latency_ms{{quantile=\"0.95\"}} {summary['latency_ms']['p95']}")
        lines.append(f"text2sql_query_latency_ms{{quantile=\"0.99\"}} {summary['latency_ms']['p99']}")
        
        # Success rate
        lines.append("# HELP text2sql_success_rate Query success rate")
        lines.append("# TYPE text2sql_success_rate gauge")
        lines.append(f"text2sql_success_rate {summary['success_rate'] / 100}")
        
        # Cache hit rate
        lines.append("# HELP text2sql_cache_hit_rate Cache hit rate")
        lines.append("# TYPE text2sql_cache_hit_rate gauge")
        lines.append(f"text2sql_cache_hit_rate {summary['cache_hit_rate'] / 100}")
        
        # QPS
        lines.append("# HELP text2sql_queries_per_second Queries per second")
        lines.append("# TYPE text2sql_queries_per_second gauge")
        lines.append(f"text2sql_queries_per_second {summary['qps']}")
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Сбросить все метрики."""
        self._queries.clear()
        self._latencies.clear()
        self._success_count = 0
        self._failed_count = 0
        self._cached_count = 0
        self._total_tokens = 0
        self._errors.clear()
        self._retrieval_hits.clear()
        self._start_time = datetime.now()
        logger.info("MetricsCollector reset")


# Глобальный экземпляр
metrics = MetricsCollector(window_size=1000)


def record_query_latency(latency_ms: float, success: bool, confidence: float = 0.0, latencies: Optional[Dict[str, float]] = None) -> None:
    """
    Записать latency запроса.
    
    Args:
        latency_ms: Latency в мс.
        success: Успешен ли запрос.
        confidence: Confidence score.
        latencies: Breakdown по этапам.
    """
    import uuid
    
    query_metrics = QueryMetrics(
        query_id=str(uuid.uuid4())[:8],
        query_text="",
        status=QueryStatus.SUCCESS if success else QueryStatus.FAILED,
        latency_ms=latency_ms,
        confidence=confidence,
        latencies_breakdown=latencies or {},
    )
    metrics.record_query(query_metrics)


def get_metrics_summary() -> Dict[str, Any]:
    """Получить сводку метрик."""
    return metrics.get_summary()
