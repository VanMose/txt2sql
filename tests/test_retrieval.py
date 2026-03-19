"""
Unit tests для retrieval компонентов.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.retrieval.embedder import SchemaEmbedder
from src.retrieval.schema_retriever import SchemaRetriever


@pytest.mark.unit
class TestSchemaEmbedder:
    """Тесты для SchemaEmbedder."""

    def test_init(self):
        """Инициализация SchemaEmbedder."""
        embedder = SchemaEmbedder()
        assert embedder is not None

    def test_embed_single_text(self):
        """Эмбеддинг одного текста."""
        embedder = SchemaEmbedder()
        text = "Table: users, Columns: id, name, email"
        embedding = embedder.embed(text)
        
        assert embedding is not None
        assert len(embedding) > 0
        assert len(embedding) == 384

    def test_embed_batch(self):
        """Пакетный эмбеддинг."""
        embedder = SchemaEmbedder()
        texts = [
            "Table: users, Columns: id, name",
            "Table: orders, Columns: id, product, amount",
        ]
        embeddings = embedder.embed_batch(texts)
        
        assert len(embeddings) == 2
        for emb in embeddings:
            assert len(emb) == 384

    def test_embed_empty_string(self):
        """Эмбеддинг пустой строки."""
        embedder = SchemaEmbedder()
        embedding = embedder.embed("")
        assert embedding is not None
        assert len(embedding) == 384

    def test_embed_special_chars(self):
        """Эмбеддинг текста со спецсимволами."""
        embedder = SchemaEmbedder()
        text = "Table: test_123, Columns: id, name (utf8: тест)"
        embedding = embedder.embed(text)
        assert len(embedding) == 384


@pytest.mark.unit
class TestSchemaRetriever:
    """Тесты для SchemaRetriever."""

    def test_init(self):
        """Инициализация SchemaRetriever."""
        embedder = SchemaEmbedder()
        schema_docs = ["Table: users, Columns: id, name"]
        retriever = SchemaRetriever(embedder, schema_docs)
        assert retriever is not None

    def test_retrieve_single_query(self):
        """Поиск релевантных таблиц для одного запроса."""
        embedder = SchemaEmbedder()
        schema_docs = [
            "Table: users, Columns: id, name, email, age",
            "Table: orders, Columns: id, user_id, product, amount",
            "Table: products, Columns: id, name, price, category",
        ]
        retriever = SchemaRetriever(embedder, schema_docs)
        
        query = "Показать всех пользователей"
        results = retriever.retrieve(query, top_k=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2
        assert any("users" in doc.lower() for doc in results)

    def test_retrieve_with_scores(self):
        """Поиск с оценками релевантности."""
        embedder = SchemaEmbedder()
        schema_docs = [
            "Table: users, Columns: id, name, email",
            "Table: orders, Columns: id, product, amount",
        ]
        retriever = SchemaRetriever(embedder, schema_docs)
        
        query = "Найти заказы"
        results, scores = retriever.retrieve_with_scores(query, top_k=2)
        
        assert isinstance(results, list)
        assert isinstance(scores, list)
        assert len(results) == len(scores)
        for score in scores:
            assert 0 <= score <= 1

    def test_retrieve_top_k_limit(self):
        """Проверка ограничения top_k."""
        embedder = SchemaEmbedder()
        schema_docs = [
            "Table: users, Columns: id, name",
            "Table: orders, Columns: id, product",
            "Table: products, Columns: id, name, price",
        ]
        retriever = SchemaRetriever(embedder, schema_docs)
        
        query = "Показать все таблицы"
        results = retriever.retrieve(query, top_k=1)
        
        assert len(results) == 1

    def test_retrieve_exceeds_docs(self):
        """Запрос top_k больше количества документов."""
        embedder = SchemaEmbedder()
        schema_docs = [
            "Table: users, Columns: id, name",
        ]
        retriever = SchemaRetriever(embedder, schema_docs)
        
        query = "Показать пользователей"
        results = retriever.retrieve(query, top_k=5)
        
        assert len(results) == 1

    def test_retrieve_relevant_tables(self):
        """Проверка релевантности результатов."""
        embedder = SchemaEmbedder()
        schema_docs = [
            "Table: users, Columns: id, name, email, age",
            "Table: orders, Columns: id, user_id, product, amount, created_at",
            "Table: products, Columns: id, name, price, category",
        ]
        retriever = SchemaRetriever(embedder, schema_docs)
        
        query = "Показать все заказы с продуктами"
        results = retriever.retrieve(query, top_k=3)
        
        assert len(results) > 0
        first_result_lower = results[0].lower()
        assert "orders" in first_result_lower or "products" in first_result_lower


@pytest.mark.integration
class TestRetrievalIntegration:
    """Интеграционные тесты для retrieval компонентов."""

    def test_embedder_and_retriever_pipeline(self, test_db_path):
        """Полный пайплайн retrieval."""
        from src.db.schema_loader import SchemaLoader
        
        schema_loader = SchemaLoader(test_db_path)
        schema_docs = schema_loader.get_schema_docs()
        
        embedder = SchemaEmbedder()
        retriever = SchemaRetriever(embedder, schema_docs)
        
        query = "Найти всех пользователей и их заказы"
        results = retriever.retrieve(query, top_k=2)
        
        assert len(results) > 0
        for result in results:
            assert "Table:" in result or "table:" in result.lower()

    def test_cosine_similarity_calculation(self):
        """Проверка расчета косинусного сходства."""
        embedder = SchemaEmbedder()
        
        text1 = "Table: users with columns id, name, email"
        text2 = "Table: users with columns id, name, age"
        
        emb1 = embedder.embed(text1)
        emb2 = embedder.embed(text2)
        
        import numpy as np
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        assert similarity > 0.5
        
        text3 = "Table: products with columns price, category, stock"
        emb3 = embedder.embed(text3)
        
        similarity_different = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        
        assert similarity_different < similarity
