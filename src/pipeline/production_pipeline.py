# src\pipeline\production_pipeline.py
"""
Production Text-to-SQL Pipeline с LangGraph orchestration.

Architecture:
    User Query → Query Understanding → Hybrid Retrieval → 
    Schema Compression → SQL Generation → SQL Validation → 
    Execution → Result

Features:
- Query Understanding layer
- Hybrid retrieval (vector + graph)
- Cross-encoder reranking
- SQL validator agent
- Execution с error recovery
- Metrics tracking
"""
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from ..config.settings import get_settings
from ..retrieval.vector_db import QdrantVectorDB, TableDocument
from ..retrieval.graph_db import Neo4jGraphDB
from ..retrieval.schema_retriever import HybridSchemaRetriever
from ..retrieval.schema_compressor import SchemaCompressor, CompactTableInfo
from ..retrieval.embedder import SchemaEmbedder
from ..agents.query_understanding import QueryUnderstandingAgent, QueryUnderstanding, QueryIntent
from ..agents.sql_validator import SQLValidator, ValidationResult
from ..llm.prompts import Prompts
from ..llm.inference import LLMService
from ..db.executor import SQLExecutor
from ..services.metrics import metrics, QueryMetrics, QueryStatus

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Состояние пайплайна."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    understanding: Optional[QueryUnderstanding] = None
    retrieved_tables: List[Tuple[TableDocument, float]] = field(default_factory=list)
    join_paths: List[Dict[str, Any]] = field(default_factory=list)
    compressed_schema: str = ""
    generated_sql: Optional[str] = None
    sql_candidates: List[str] = field(default_factory=list)
    validation_result: Optional[ValidationResult] = None
    execution_result: Any = None
    execution_success: bool = False
    confidence: float = 0.0
    error: Optional[str] = None
    latencies: Dict[str, float] = field(default_factory=dict)
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в dict."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "understanding": self.understanding.to_dict() if self.understanding else None,
            "retrieved_tables": len(self.retrieved_tables),
            "join_paths": len(self.join_paths),
            "compressed_schema": self.compressed_schema[:200] + "..." if len(self.compressed_schema) > 200 else self.compressed_schema,
            "generated_sql": self.generated_sql,
            "sql_candidates": self.sql_candidates,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "execution_success": self.execution_success,
            "confidence": self.confidence,
            "error": self.error,
            "latencies": self.latencies,
            "cache_hit": self.cache_hit,
        }


class ProductionPipeline:
    """
    Production Text-to-SQL Pipeline.
    
    Architecture:
        Query Understanding → Hybrid Retrieval → Schema Compression → 
        SQL Generation → SQL Validation → Execution
    """
    
    def __init__(
        self,
        db_paths: Optional[List[str]] = None,
        use_graph_expansion: bool = True,
        use_reranking: bool = True,
        use_validation: bool = True,
    ) -> None:
        """
        Инициализировать пайплайн.
        
        Args:
            db_paths: Пути к базам данных.
            use_graph_expansion: Использовать graph expansion.
            use_reranking: Использовать reranking.
            use_validation: Использовать SQL валидацию.
        """
        self.settings = get_settings()
        self.db_paths = db_paths or []
        self.use_graph_expansion = use_graph_expansion
        self.use_reranking = use_reranking
        self.use_validation = use_validation
        
        # Initialize components
        logger.info("Initializing ProductionPipeline components...")
        
        self.vector_db = QdrantVectorDB(
            use_local=self.settings.qdrant_use_local,
            local_path=self.settings.qdrant_local_path,
        )
        
        self.graph_db = None
        if use_graph_expansion:
            try:
                self.graph_db = Neo4jGraphDB(
                    uri=self.settings.neo4j_uri,
                    username=self.settings.neo4j_username,
                    password=self.settings.neo4j_password,
                )
                logger.info("Neo4jGraphDB initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Neo4j: {e}. Graph expansion disabled.")
                self.graph_db = None
        
        self.retriever = HybridSchemaRetriever(
            vector_db=self.vector_db,
            graph_db=self.graph_db,
            use_graph_expansion=use_graph_expansion and self.graph_db is not None,
            use_reranking=use_reranking,
        )
        
        self.compressor = SchemaCompressor(
            compression_level=2,
            include_join_hints=True,
            include_pk=True,
            include_types=True,
        )
        
        self.understanding_agent = QueryUnderstandingAgent()
        self.llm_service = LLMService()
        self.embedder = SchemaEmbedder()
        
        # Build LangGraph workflow
        self._build_graph()
        
        logger.info("ProductionPipeline initialized")
    
    def _build_graph(self) -> None:
        """Построить LangGraph workflow."""
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("understanding", self._node_understanding)
        workflow.add_node("retrieval", self._node_retrieval)
        workflow.add_node("compression", self._node_compression)
        workflow.add_node("generation", self._node_generation)
        workflow.add_node("validation", self._node_validation)
        workflow.add_node("execution", self._node_execution)
        
        # Set entry point
        workflow.set_entry_point("understanding")
        
        # Add edges
        workflow.add_edge("understanding", "retrieval")
        workflow.add_edge("retrieval", "compression")
        workflow.add_edge("compression", "generation")
        
        if self.use_validation:
            workflow.add_edge("generation", "validation")
            workflow.add_conditional_edges(
                "validation",
                self._should_refine,
                {
                    "refine": "generation",  # Refine if validation fails
                    "execute": "execution",
                },
            )
        else:
            workflow.add_edge("generation", "execution")
        
        workflow.add_edge("execution", END)
        
        self.graph = workflow.compile()
        logger.info("LangGraph workflow built")
    
    def run(self, query: str) -> PipelineState:
        """
        Выполнить пайплайн.
        
        Args:
            query: Запрос пользователя.
        
        Returns:
            PipelineState с результатами.
        """
        start_time = time.time()
        logger.info(f"Starting pipeline for query: '{query}'")
        
        initial_state = PipelineState(query=query)
        
        try:
            result = self.graph.invoke(initial_state)
            
            # Record metrics
            total_latency = (time.time() - start_time) * 1000
            self._record_metrics(result, total_latency)
            
            logger.info(f"Pipeline completed: success={result.execution_success}, latency={total_latency:.0f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            initial_state.error = str(e)
            return initial_state
    
    def _node_understanding(self, state: PipelineState) -> Dict[str, Any]:
        """Query Understanding node."""
        logger.info("Node: Query Understanding")
        start = time.time()
        
        understanding = self.understanding_agent.analyze(state.query)
        
        state.understanding = understanding
        state.latencies["understanding"] = (time.time() - start) * 1000
        
        logger.info(f"Understanding: intent={understanding.intent.value}, tables={understanding.tables}")
        return {"understanding": understanding}
    
    def _node_retrieval(self, state: PipelineState) -> Dict[str, Any]:
        """Hybrid Retrieval node."""
        logger.info("Node: Hybrid Retrieval")
        start = time.time()
        
        # Get DB filter from understanding
        db_filter = None
        if state.understanding and state.understanding.db_hints:
            db_filter = state.understanding.db_hints if state.understanding.db_hints != ["all"] else None
        
        # Retrieve with graph expansion
        results, join_paths = self.retriever.retrieve_with_join_paths(
            query=state.query,
            top_k=self.settings.top_k_tables,
            db_filter=db_filter,
        )
        
        state.retrieved_tables = results
        state.join_paths = join_paths
        state.latencies["retrieval"] = (time.time() - start) * 1000
        
        logger.info(f"Retrieval: {len(results)} tables, {len(join_paths)} join paths")
        return {"retrieved_tables": results, "join_paths": join_paths}
    
    def _node_compression(self, state: PipelineState) -> Dict[str, Any]:
        """Schema Compression node."""
        logger.info("Node: Schema Compression")
        start = time.time()
        
        # Convert retrieved tables to CompactTableInfo
        tables = []
        for doc, _ in state.retrieved_tables:
            compact = CompactTableInfo(
                name=doc.table_name,
                db_name=doc.db_name,
                columns=doc.columns,
                column_types=doc.column_types or {},
                primary_key=doc.primary_key,
                foreign_keys=doc.foreign_keys or [],
            )
            tables.append(compact)
        
        # Compress with join hints
        compressed = self.compressor.compress_for_llm(tables, state.join_paths)
        
        state.compressed_schema = compressed
        state.latencies["compression"] = (time.time() - start) * 1000
        
        logger.info(f"Compression: {len(compressed)} chars")
        return {"compressed_schema": compressed}
    
    def _node_generation(self, state: PipelineState) -> Dict[str, Any]:
        """SQL Generation node."""
        logger.info("Node: SQL Generation")
        start = time.time()
        
        # Build prompt
        prompt = Prompts.format_sql_generator(
            query=state.query,
            schema=state.compressed_schema,
            use_compact=False,
        )
        
        # Generate SQL
        outputs = self.llm_service.generate(
            prompt=prompt,
            n=self.settings.n_samples,
            temperature=self.settings.temperature,
        )
        
        # Parse SQL from JSON
        sql_candidates = []
        for output in outputs:
            sql = self._parse_sql_from_output(output)
            if sql:
                sql_candidates.append(sql)
        
        state.sql_candidates = sql_candidates
        state.generated_sql = sql_candidates[0] if sql_candidates else None
        state.latencies["generation"] = (time.time() - start) * 1000
        
        logger.info(f"Generation: {len(sql_candidates)} candidates")
        return {"sql_candidates": sql_candidates, "generated_sql": state.generated_sql}
    
    def _node_validation(self, state: PipelineState) -> Dict[str, Any]:
        """SQL Validation node."""
        logger.info("Node: SQL Validation")
        start = time.time()
        
        if not state.generated_sql:
            state.validation_result = ValidationResult(
                valid=False,
                sql="",
                errors=["No SQL generated"],
                warnings=[],
                suggestions=[],
            )
            return {"validation_result": state.validation_result}
        
        validator = SQLValidator()
        result = validator.validate_with_details(state.generated_sql)
        
        state.validation_result = result
        state.latencies["validation"] = (time.time() - start) * 1000
        
        logger.info(f"Validation: valid={result.valid}, errors={len(result.errors)}")
        return {"validation_result": result}
    
    def _node_execution(self, state: PipelineState) -> Dict[str, Any]:
        """SQL Execution node."""
        logger.info("Node: SQL Execution")
        start = time.time()
        
        if not state.generated_sql:
            state.execution_success = False
            state.error = "No SQL to execute"
            return {"execution_success": False}
        
        # Execute on first available DB
        if not self.db_paths:
            state.execution_success = False
            state.error = "No database paths configured"
            return {"execution_success": False}
        
        executor = SQLExecutor(self.db_paths[0])
        success, result = executor.execute(state.generated_sql)
        
        state.execution_result = result
        state.execution_success = success
        state.latencies["execution"] = (time.time() - start) * 1000
        
        if not success:
            state.error = str(result) if isinstance(result, str) else "Execution failed"
        
        logger.info(f"Execution: success={success}")
        return {"execution_result": result, "execution_success": success}
    
    def _should_refine(self, state: PipelineState) -> str:
        """Проверить нужно ли рефинить SQL."""
        if state.validation_result and not state.validation_result.valid:
            return "refine"
        return "execute"
    
    def _parse_sql_from_output(self, output: str) -> Optional[str]:
        """Распарсить SQL из LLM вывода."""
        import json
        import re
        
        # Try JSON parsing first
        try:
            # Find JSON in output
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("sql")
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: extract SQL-like pattern
        sql_match = re.search(r'(SELECT\s+.+?)(?:;|$)', output, re.IGNORECASE | re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        return None
    
    def _record_metrics(self, state: PipelineState, total_latency: float) -> None:
        """Записать метрики."""
        query_metrics = QueryMetrics(
            query_id=state.query_id,
            query_text=state.query,
            status=QueryStatus.SUCCESS if state.execution_success else QueryStatus.FAILED,
            latency_ms=total_latency,
            sql_generated=state.generated_sql,
            sql_valid=state.validation_result.valid if state.validation_result else False,
            confidence=state.confidence,
            tables_retrieved=len(state.retrieved_tables),
            cache_hit=state.cache_hit,
            error_message=state.error,
            latencies_breakdown=state.latencies,
        )
        metrics.record_query(query_metrics)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику пайплайна."""
        return {
            "vector_db": self.vector_db.get_stats(),
            "graph_db": self.graph_db.get_stats() if self.graph_db else None,
            "retriever": self.retriever.get_stats(),
            "embedder": self.embedder.get_cache_stats(),
        }
    
    def close(self) -> None:
        """Закрыть соединения."""
        self.vector_db.close()
        if self.graph_db:
            self.graph_db.close()
        logger.info("ProductionPipeline closed")
