# src\retrieval\schema_retriever.py
"""
Hybrid Retriever для поиска релевантных таблиц.

Production Architecture:
    User Query → Vector Search → Graph Expansion → Rerank → Tables

Features:
- Hybrid retrieval (vector + graph)
- Cross-encoder reranking
- Normalized embeddings для cosine similarity
- TTL cache для результатов
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .embedder import SchemaEmbedder
from .graph_db import Neo4jGraphDB
from .vector_db import QdrantVectorDB, TableDocument

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker для reranking таблиц.
    
    Production feature: более точный reranking через cross-encoder.
    Поддерживает различные модели через настройки.
    """
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Инициализировать reranker.
        
        Args:
            model_name: Название модели (из settings или переопределение).
        """
        from ..config.settings import get_settings
        
        settings = get_settings()
        self.model_name = model_name or settings.get("reranker_model", "BAAI/bge-reranker-base")
        self.use_local = settings.get("use_local_reranker", True)
        
        # Проверка локальной модели
        if self.use_local:
            local_path = settings.get_local_embedding_path().replace(
                settings.embedding_model, 
                self.model_name
            )
            if Path(local_path).exists():
                self.model_path = local_path
                logger.info(f"Using local reranker model: {self.model_path}")
            else:
                self.model_path = self.model_name
                logger.warning(f"Local reranker model not found: {local_path}")
        else:
            self.model_path = self.model_name
        
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_path)
            self.enabled = True
            logger.info(f"CrossEncoderReranker initialized with {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder '{self.model_name}': {e}. Using simple reranking.")
            self.enabled = False
            self.model = None
    
    def compute_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Вычислить scores для пар (query, document).
        
        Args:
            pairs: Список пар (query, document).
        
        Returns:
            Список scores.
        """
        if not self.enabled or not self.model:
            # Fallback: uniform scores
            return [1.0] * len(pairs)
        
        try:
            scores = self.model.predict(pairs)
            return scores.tolist() if hasattr(scores, "tolist") else scores
        except Exception as e:
            logger.error(f"Cross-encoder prediction error: {e}")
            return [1.0] * len(pairs)
    
    def rerank(
        self,
        query: str,
        documents: List[Tuple[TableDocument, float]],
        top_k: int,
    ) -> List[Tuple[TableDocument, float]]:
        """
        Rerank документов.
        
        Pipeline:
            1. Create pairs (query, doc.text)
            2. Compute cross-encoder scores
            3. Combine with vector scores
            4. Return top k
        
        Args:
            query: Запрос пользователя.
            documents: Список (TableDocument, vector_score).
            top_k: Количество результатов.
        
        Returns:
            Список (TableDocument, combined_score).
        """
        if not documents:
            return []
        
        if not self.enabled:
            # Fallback: simple reranking
            return self._simple_rerank(documents, top_k)
        
        # Step 1: Create pairs
        pairs = [(query, doc.text) for doc, _ in documents]
        
        # Step 2: Compute scores
        rerank_scores = self.compute_score(pairs)
        
        # Step 3: Combine scores
        reranked = []
        for (doc, _), rerank_score in zip(documents, rerank_scores):
            # Combined score: 60% reranker, 40% vector + FK boost
            fk_boost = 1.15 if doc.foreign_keys else 1.0
            combined_score = (0.6 * rerank_score + 0.4 * 1.0) * fk_boost
            reranked.append((doc, combined_score))
        
        # Step 4: Sort and return top k
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
    
    def _simple_rerank(
        self,
        documents: List[Tuple[TableDocument, float]],
        top_k: int,
    ) -> List[Tuple[TableDocument, float]]:
        """Простой reranking без cross-encoder."""
        if not documents:
            return []
        
        reranked = []
        for doc, score in documents:
            if doc.foreign_keys:
                score *= 1.15
            if doc.description:
                score *= 1.05
            reranked.append((doc, score))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


class HybridSchemaRetriever:
    """
    Hybrid Retriever для поиска релевантных таблиц.
    
    Production Pipeline:
        User Query
             ↓
        Vector Search (Qdrant) → top 2k
             ↓
        Graph Expansion (Neo4j) → add join tables
             ↓
        Reranker (Cross-Encoder) → top k
             ↓
        Return Tables
    
    Attributes:
        vector_db: QdrantVectorDB для vector search.
        graph_db: Neo4jGraphDB для graph expansion.
        embedder: SchemaEmbedder для embeddings.
        reranker: CrossEncoderReranker для reranking.
    """
    
    def __init__(
        self,
        vector_db: QdrantVectorDB,
        graph_db: Optional[Neo4jGraphDB] = None,
        use_graph_expansion: bool = True,
        use_reranking: Optional[bool] = None,
        expansion_depth: int = 2,
    ) -> None:
        """
        Инициализировать hybrid retriever.
        
        Args:
            vector_db: QdrantVectorDB экземпляр.
            graph_db: Neo4jGraphDB экземпляр (опционально).
            use_graph_expansion: Использовать graph expansion.
            use_reranking: Использовать cross-encoder reranking (из settings если None).
            expansion_depth: Глубина обхода графа.
        """
        from ..config.settings import get_settings
        
        settings = get_settings()
        
        self.vector_db = vector_db
        self.graph_db = graph_db if use_graph_expansion else None
        self.use_graph_expansion = use_graph_expansion and graph_db is not None
        
        # Reranking настройки из settings
        if use_reranking is None:
            self.use_reranking = settings.get("use_reranking", True)
        else:
            self.use_reranking = use_reranking
        
        self.expansion_depth = expansion_depth
        
        self.embedder = SchemaEmbedder()
        
        # Reranker инициализация с моделью из settings
        self.reranker = None
        if self.use_reranking:
            reranker_model = settings.get("reranker_model", "BAAI/bge-reranker-base")
            self.reranker = CrossEncoderReranker(model_name=reranker_model)
            logger.info(f"HybridSchemaRetriever: reranking={self.use_reranking}, model={reranker_model}")
        else:
            logger.info(f"HybridSchemaRetriever: reranking={self.use_reranking} (disabled)")
        
        # TTL cache для результатов
        self._cache: Dict[str, Tuple[List[Tuple[TableDocument, float]], float]] = {}
        self._cache_ttl = 300  # 5 минут
        
        logger.info(
            f"HybridSchemaRetriever initialized: "
            f"graph_expansion={self.use_graph_expansion}, "
            f"reranking={self.use_reranking}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        db_filter: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> List[Tuple[TableDocument, float]]:
        """
        Найти релевантные таблицы.
        
        Args:
            query: Запрос пользователя.
            top_k: Количество результатов.
            db_filter: Фильтр по БД.
            use_cache: Использовать кэш.
        
        Returns:
            Список (TableDocument, score).
        """
        # Check cache
        cache_key = f"{query}:{','.join(sorted(db_filter or []))}:{top_k}"
        if use_cache and cache_key in self._cache:
            cached_result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result
        
        # Step 1: Vector search
        start_time = time.time()
        
        if self.use_graph_expansion and self.graph_db:
            vector_results = self.vector_db.search_with_graph_expansion(
                query=query,
                top_k=top_k,
                db_filter=db_filter,
                graph_db=self.graph_db,
                expansion_depth=self.expansion_depth,
            )
        else:
            vector_results = self.vector_db.search(
                query=query,
                top_k=top_k * 2,
                db_filter=db_filter,
            )
        
        vector_time = (time.time() - start_time) * 1000
        logger.info(f"Vector search found {len(vector_results)} tables in {vector_time:.0f}ms")
        
        if not vector_results:
            return []
        
        # Step 2: Reranking
        if self.use_reranking and self.reranker:
            rerank_start = time.time()
            reranked = self.reranker.rerank(query, vector_results, top_k)
            rerank_time = (time.time() - rerank_start) * 1000
            logger.info(f"Reranking completed in {rerank_time:.0f}ms")
        else:
            reranked = self._simple_rerank(vector_results, top_k)
        
        # Save to cache
        if use_cache:
            self._cache[cache_key] = (reranked, time.time())
        
        logger.info(f"Retrieved {len(reranked)} tables for query: {query[:50]}...")
        return reranked
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        db_filter: Optional[List[str]] = None,
    ) -> List[Tuple[TableDocument, float]]:
        """
        Найти релевантные таблицы с scores.
        
        Args:
            query: Запрос пользователя.
            top_k: Количество результатов.
            db_filter: Фильтр по БД.
        
        Returns:
            Список (TableDocument, score).
        """
        return self.retrieve(query, top_k=top_k, db_filter=db_filter)
    
    def retrieve_with_join_paths(
        self,
        query: str,
        top_k: int = 5,
        db_filter: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[TableDocument, float]], List[Dict[str, Any]]]:
        """
        Найти релевантные таблицы с join paths.
        
        Production feature: возвращает join paths для SQL generation.
        
        Args:
            query: Запрос пользователя.
            top_k: Количество результатов.
            db_filter: Фильтр по БД.
        
        Returns:
            (таблицы, join_paths).
        """
        # Retrieve tables
        results = self.retrieve(query, top_k=top_k, db_filter=db_filter)
        
        if not results or not self.graph_db:
            return results, []
        
        # Find join paths
        table_names = [(doc.db_name, doc.table_name) for doc, _ in results]
        join_paths = self.graph_db.find_join_path(table_names, max_depth=3)
        
        logger.info(f"Found {len(join_paths)} join paths")
        return results, join_paths
    
    def _simple_rerank(
        self,
        documents: List[Tuple[TableDocument, float]],
        top_k: int,
    ) -> List[Tuple[TableDocument, float]]:
        """Простой reranking без cross-encoder."""
        if not documents:
            return []
        
        reranked = []
        for doc, score in documents:
            # Boost для таблиц с foreign keys
            if doc.foreign_keys:
                score *= 1.15
            # Boost для таблиц с description
            if doc.description:
                score *= 1.05
            reranked.append((doc, score))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
    
    def clear_cache(self) -> None:
        """Очистить кэш."""
        self._cache.clear()
        logger.info("HybridSchemaRetriever cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику retriever."""
        return {
            "use_graph_expansion": self.use_graph_expansion,
            "use_reranking": self.use_reranking,
            "expansion_depth": self.expansion_depth,
            "cache_size": len(self._cache),
            "graph_db_connected": self.graph_db is not None,
        }


class LegacySchemaRetriever:
    """
    Legacy Retriever для обратной совместимости.
    
    Используется в старом пайплайне без hybrid retrieval.
    """
    
    def __init__(self, embedder: SchemaEmbedder, schema_docs: List[str]) -> None:
        self.embedder = embedder
        self.schema_docs = schema_docs
        
        # OPTIMIZATION: нормализация embeddings
        self.embeddings = []
        for doc in schema_docs:
            emb = embedder.embed(doc)
            # Нормализация для cosine similarity
            emb_norm = emb / np.linalg.norm(emb)
            self.embeddings.append(emb_norm)
        
        logger.info(f"LegacySchemaRetriever initialized with {len(schema_docs)} docs")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Найти релевантные таблицы."""
        q = self.embedder.embed(query)
        q_norm = q / np.linalg.norm(q)  # FIX: нормализация
        
        # Cosine similarity (normalized dot product)
        scores = [np.dot(q_norm, emb) for emb in self.embeddings]
        idx = np.argsort(scores)[::-1][:top_k]
        
        return [self.schema_docs[i] for i in idx]
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Найти релевантные таблицы с scores."""
        q = self.embedder.embed(query)
        q_norm = q / np.linalg.norm(q)  # FIX: нормализация
        
        scores = [np.dot(q_norm, emb) for emb in self.embeddings]
        idx = np.argsort(scores)[::-1][:top_k]
        
        return [(self.schema_docs[i], float(scores[i])) for i in idx]
