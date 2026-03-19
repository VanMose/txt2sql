# src\retrieval\vector_db.py
"""
Vector DB на основе Qdrant для хранения и поиска схем таблиц.

Production features:
- Hybrid retrieval (vector + graph expansion)
- Batch загрузка с оптимизацией
- TTL cache для результатов
- Нормализация embeddings для cosine similarity

Architecture:
    User Query → Vector Search → Graph Expansion → Rerank → Tables
"""
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SearchParams,
    VectorParams,
)

from ..config.settings import get_settings
from .embedder import SchemaEmbedder

logger = logging.getLogger(__name__)


@dataclass
class TableDocument:
    """Документ таблицы для векторного поиска."""
    id: str
    db_path: str
    db_name: str
    table_name: str
    text: str
    columns: List[str]
    column_types: Optional[Dict[str, str]] = None
    foreign_keys: Optional[List[Dict[str, str]]] = None
    primary_key: Optional[str] = None
    row_count: Optional[int] = None
    description: Optional[str] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_payload(self) -> Dict[str, Any]:
        """Конвертировать в payload для Qdrant."""
        return {
            "id": self.id,
            "db_path": self.db_path,
            "db_name": self.db_name,
            "table_name": self.table_name,
            "text": self.text,  # FIX: сохраняем text в payload
            "columns": self.columns,
            "column_types": self.column_types or {},
            "foreign_keys": self.foreign_keys or [],
            "primary_key": self.primary_key,
            "row_count": self.row_count,
            "description": self.description or "",
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "TableDocument":
        """Создать из payload Qdrant."""
        return cls(
            id=payload.get("id", ""),
            db_path=payload["db_path"],
            db_name=payload["db_name"],
            table_name=payload["table_name"],
            text=payload.get("text", ""),  # FIX: читаем text из payload
            columns=payload.get("columns", []),
            column_types=payload.get("column_types", {}),
            foreign_keys=payload.get("foreign_keys", []),
            primary_key=payload.get("primary_key"),
            row_count=payload.get("row_count"),
            description=payload.get("description", ""),
        )


class QdrantVectorDB:
    """Vector DB на основе Qdrant для поиска схем таблиц."""

    COLLECTION_NAME = "text2sql_schemas"
    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
    HNSW_EF = 64  # OPTIMIZATION: снижено с 128 для производительности

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        use_local: bool = True,
        local_path: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = collection_name or settings.get("qdrant_collection_name", self.COLLECTION_NAME)
        self.use_local = use_local or settings.get("qdrant_use_local", True)

        self.embedder = SchemaEmbedder()

        if self.use_local:
            local_path = local_path or settings.get("qdrant_local_path", "qdrant_storage")
            self.client = QdrantClient(path=local_path)
            logger.info(f"QdrantVectorDB initialized with local storage: {local_path}")
        else:
            qdrant_url = qdrant_url or settings.get("qdrant_url")
            if not qdrant_url:
                raise ValueError("Qdrant URL required for remote mode")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            logger.info(f"QdrantVectorDB initialized with remote URL: {qdrant_url}")

        self._create_collection()

    def _create_collection(self) -> None:
        """Создать коллекцию если не существует."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.EMBEDDING_DIM,
                        distance=Distance.COSINE,  # COSINE для нормальнойized similarity
                    ),
                )
                logger.info(f"Collection '{self.collection_name}' created")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def recreate_collection(self) -> None:
        """Пересоздать коллекцию."""
        import shutil
        from pathlib import Path
        
        try:
            # 🔥 Для локального Qdrant - сначала закрываем клиент и удаляем ВСЕ хранилище
            if self.use_local:
                # Получаем путь к хранилищу ДО закрытия клиента
                local_path = None
                if hasattr(self.client, 'storage_path'):
                    local_path = Path(self.client.storage_path)
                elif hasattr(self.client, '_client') and hasattr(self.client._client, 'storage_path'):
                    local_path = Path(self.client._client.storage_path)
                else:
                    local_path = Path("qdrant_storage")
                
                # Закрываем текущий клиент (освобождаем lock файл)
                if hasattr(self.client, 'close'):
                    self.client.close()
                
                # Удаляем ВСЕ хранилище пока клиент закрыт
                if local_path and local_path.exists():
                    try:
                        shutil.rmtree(local_path)
                        logger.info(f"Deleted local Qdrant storage: {local_path}")
                    except Exception as e:
                        logger.warning(f"Could not delete local storage: {e}")
                
                # Пересоздаем клиент с чистым хранилищем
                self.client = QdrantClient(path=str(local_path))
                logger.info("Recreated Qdrant client with fresh storage")
                
                # Создаем новую коллекцию
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.EMBEDDING_DIM, distance=Distance.COSINE),
                )
                logger.info(f"Created new collection '{self.collection_name}'")
            else:
                # 🔥 Для Docker Qdrant - используем API (без удаления файлов)
                logger.info(f"Recreating collection in Docker Qdrant at {self.client._client._host}...")
                
                collections = self.client.get_collections().collections
                if self.collection_name in [c.name for c in collections]:
                    self.client.delete_collection(collection_name=self.collection_name)
                    logger.info(f"Deleted existing collection '{self.collection_name}' via API")
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.EMBEDDING_DIM, distance=Distance.COSINE),
                )
                logger.info(f"Created new collection '{self.collection_name}' in Docker Qdrant")
                
        except Exception as e:
            logger.error(f"Error recreating collection: {e}")
            raise

    def _generate_id(self, db_path: str, table_name: str) -> str:
        """Сгенерировать уникальный ID для документа."""
        key = f"{db_path}:{table_name}"
        return hashlib.md5(key.encode()).hexdigest()

    def add_table(self, doc: TableDocument) -> str:
        """Добавить документ таблицы в Vector DB."""
        doc_id = doc.id or self._generate_id(doc.db_path, doc.table_name)
        vector = self.embedder.embed(doc.text)
        vector_array = vector.tolist() if hasattr(vector, "tolist") else vector

        point = PointStruct(
            id=doc_id,
            vector=vector_array,
            payload=doc.to_payload(),  # FIX: payload включает text
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])

        logger.debug(f"Added table '{doc.db_name}.{doc.table_name}' to Vector DB")
        return doc_id

    def add_tables_batch(self, docs: List[TableDocument]) -> List[str]:
        """
        Добавить несколько таблиц батчем.
        
        OPTIMIZATION: использует batch embeddings вместо поэлементного.
        """
        if not docs:
            return []

        # FIX: генерируем IDs заранее
        ids: List[str] = []
        for doc in docs:
            doc_id = doc.id or self._generate_id(doc.db_path, doc.table_name)
            doc.id = doc_id
            ids.append(doc_id)

        # OPTIMIZATION: batch embeddings
        texts = [doc.text for doc in docs]
        batch_embeddings = self.embedder.embed_batch(texts)

        # FIX: корректно связываем docs и embeddings
        points = []
        for doc, embedding in zip(docs, batch_embeddings):
            vector_array = embedding.tolist() if hasattr(embedding, "tolist") else embedding
            points.append(PointStruct(
                id=doc.id,
                vector=vector_array,
                payload=doc.to_payload(),
            ))

        self.client.upsert(collection_name=self.collection_name, points=points)

        logger.info(f"Batch added {len(docs)} tables to Vector DB")
        return ids

    def add_schema_batch(self, db_name: str, tables: List[Dict[str, Any]]) -> List[str]:
        """
        Добавить схему базы данных батчем.
        
        FIX: корректно обрабатывает foreign_keys для каждой таблицы.
        """
        if not tables:
            return []

        docs = []
        for table in tables:
            # FIX: foreign_keys берётся из каждой таблицы, а не последней
            foreign_keys = table.get("foreign_keys", [])
            
            # Формируем текстовое описание
            col_parts = []
            for col in table.get("columns", []):
                col_type = table.get("column_types", {}).get(col, "")
                col_parts.append(f"{col} ({col_type})")
            
            fk_text = ""
            if foreign_keys:
                fk_parts = [f"{fk['from_column']}→{fk['to_table']}.{fk['to_column']}" for fk in foreign_keys]
                fk_text = f" FK: {', '.join(fk_parts)}"
            
            text = f"Table: {table['name']}, Database: {db_name}, Columns: {', '.join(col_parts)}{fk_text}"
            
            doc = TableDocument(
                id="",
                db_path=table.get("db_path", ""),
                db_name=db_name,
                table_name=table["name"],
                text=text,
                columns=table.get("columns", []),
                column_types=table.get("column_types", {}),
                foreign_keys=foreign_keys,
                primary_key=table.get("primary_key"),
                row_count=table.get("row_count"),
            )
            docs.append(doc)

        return self.add_tables_batch(docs)

    def search(
        self,
        query: str,
        top_k: int = 10,
        db_filter: Optional[List[str]] = None,
        score_threshold: float = 0.0,
    ) -> List[Tuple[TableDocument, float]]:
        """
        Поиск релевантных таблиц.
        
        OPTIMIZATION: нормализация embeddings для cosine similarity.
        """
        query_vector = self.embedder.embed(query)
        
        # OPTIMIZATION: нормализация вектора для COSINE distance
        if hasattr(query_vector, "tolist"):
            query_array = query_vector.tolist()
        else:
            query_array = query_vector
        
        # Нормализация
        query_norm = np.array(query_array) / np.linalg.norm(query_array)
        query_array = query_norm.tolist()

        query_filter = None
        if db_filter:
            query_filter = Filter(must=[
                FieldCondition(key="db_name", match=MatchValue(value=db))
                for db in db_filter
            ])

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_array,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            search_params=SearchParams(
                hnsw_ef=self.HNSW_EF,  # OPTIMIZATION: снижено с 128
                exact=False,
            ),
        )

        # FIX: читаем text из payload
        documents = [
            (TableDocument.from_payload(r.payload), r.score)
            for r in results.points
        ]

        logger.info(f"Found {len(documents)} tables for query: {query[:50]}...")
        return documents

    def search_with_graph_expansion(
        self,
        query: str,
        top_k: int = 10,
        db_filter: Optional[List[str]] = None,
        graph_db: Optional[Any] = None,
        expansion_depth: int = 2,
    ) -> List[Tuple[TableDocument, float]]:
        """
        Поиск с graph expansion для hybrid retrieval.
        
        Architecture:
            1. Vector search → top 2*k tables
            2. Graph expansion → find related tables via FK
            3. Deduplicate + rerank → top k tables
        
        Args:
            query: Запрос пользователя.
            top_k: Количество результатов.
            db_filter: Фильтр по БД.
            graph_db: Neo4jGraphDB экземпляр.
            expansion_depth: Глубина обхода графа.
        
        Returns:
            Список (TableDocument, score).
        """
        # Step 1: Vector search (берём больше для expansion)
        vector_results = self.search(query, top_k=top_k * 2, db_filter=db_filter)
        
        if not vector_results or graph_db is None:
            return vector_results[:top_k]

        # Step 2: Graph expansion
        expanded_tables: set = set()
        expanded_docs: Dict[str, Tuple[TableDocument, float]] = {}
        
        for doc, score in vector_results:
            key = f"{doc.db_name}.{doc.table_name}"
            expanded_tables.add(key)
            expanded_docs[key] = (doc, score)
            
            # Находим связанные таблицы через граф
            related = graph_db.find_related_tables(
                db_name=doc.db_name,
                table_name=doc.table_name,
                max_depth=expansion_depth,
            )
            
            for rel in related:
                rel_key = f"{rel['db_name']}.{rel['table_name']}"
                if rel_key not in expanded_tables:
                    expanded_tables.add(rel_key)
                    # Создаём документ для связанной таблицы
                    rel_doc = TableDocument(
                        id="",
                        db_path="",
                        db_name=rel["db_name"],
                        table_name=rel["table_name"],
                        text=f"Table: {rel['table_name']}, Database: {rel['db_name']}",
                        columns=rel.get("columns", []),
                        column_types={},
                        foreign_keys=rel.get("join_conditions", []),
                    )
                    # Score ниже для expanded таблиц
                    expanded_docs[rel_key] = (rel_doc, score * 0.7)

        # Step 3: Return top k
        sorted_docs = sorted(expanded_docs.values(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

    def search_with_reranking(
        self,
        query: str,
        top_k: int = 10,
        db_filter: Optional[List[str]] = None,
        reranker: Optional[Any] = None,
    ) -> List[Tuple[TableDocument, float]]:
        """
        Поиск с reranking.
        
        Pipeline:
            1. Vector search → 2*k candidates
            2. Reranker (cross-encoder) → top k
        
        Args:
            query: Запрос пользователя.
            top_k: Количество результатов.
            db_filter: Фильтр по БД.
            reranker: Cross-encoder reranker.
        """
        # Step 1: Vector retrieval (2*k candidates)
        candidates = self.search(query, top_k=top_k * 2, db_filter=db_filter)
        
        if not candidates or reranker is None:
            # Fallback: простой reranking по foreign keys
            return self._simple_rerank(candidates, top_k)

        # Step 2: Cross-encoder reranking
        pairs = [(query, doc.text) for doc, _ in candidates]
        rerank_scores = reranker.compute_score(pairs)
        
        # Combining scores
        reranked = []
        for (doc, _), rerank_score in zip(candidates, rerank_scores):
            combined_score = 0.6 * rerank_score + 0.4 * doc.foreign_keys is not None
            reranked.append((doc, combined_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def _simple_rerank(
        self,
        candidates: List[Tuple[TableDocument, float]],
        top_k: int,
    ) -> List[Tuple[TableDocument, float]]:
        """Простой reranking без cross-encoder."""
        if not candidates:
            return []

        reranked = []
        for doc, score in candidates:
            # Boost для таблиц с foreign keys
            if doc.foreign_keys:
                score *= 1.15  # Усиленный boost
            
            # Boost для таблиц с description
            if doc.description:
                score *= 1.05
            
            reranked.append((doc, score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def get_all_tables(self, db_filter: Optional[List[str]] = None) -> List[TableDocument]:
        """Получить все таблицы."""
        query_filter = None
        if db_filter:
            query_filter = Filter(must=[
                FieldCondition(key="db_name", match=MatchValue(value=db))
                for db in db_filter
            ])

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=10000,
        )
        documents = [TableDocument.from_payload(r.payload) for r in results]

        logger.info(f"Retrieved {len(documents)} tables from Vector DB")
        return documents

    def delete_tables(self, db_path: str) -> int:
        """Удалить все таблицы из базы данных."""
        tables = self.get_all_tables(db_filter=[Path(db_path).stem])
        ids = [self._generate_id(db_path, t.table_name) for t in tables]
        self.client.delete(collection_name=self.collection_name, points_selector=ids)

        logger.info(f"Deleted {len(tables)} tables from {db_path}")
        return len(tables)

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику коллекции."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": getattr(info, "vectors_count", 0),
                "points_count": getattr(info, "points_count", 0),
                "collection_name": self.collection_name,
            }
        except Exception as e:
            logger.warning(f"Failed to get vector DB stats: {e}")
            return {"vectors_count": 0, "points_count": 0, "collection_name": self.collection_name}

    def close(self) -> None:
        """Закрыть соединение с Qdrant."""
        if hasattr(self.client, "close"):
            self.client.close()
        logger.info("QdrantVectorDB connection closed")
