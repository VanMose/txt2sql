"""Retrieval modules."""
# Lazy imports to avoid circular imports
__all__ = [
    "SchemaEmbedder",
    "QdrantVectorDB",
    "TableDocument",
    "Neo4jGraphDB",
    "ForeignKey",
    "TableNode",
    "SchemaCompressor",
    "CompactTableInfo",
    "HybridSchemaRetriever",
    "CrossEncoderReranker",
    "LegacySchemaRetriever",
]


def __getattr__(name):
    if name == "SchemaEmbedder":
        from .embedder import SchemaEmbedder
        return SchemaEmbedder
    elif name == "QdrantVectorDB":
        from .vector_db import QdrantVectorDB
        return QdrantVectorDB
    elif name == "TableDocument":
        from .vector_db import TableDocument
        return TableDocument
    elif name == "Neo4jGraphDB":
        from .graph_db import Neo4jGraphDB
        return Neo4jGraphDB
    elif name == "ForeignKey":
        from .graph_db import ForeignKey
        return ForeignKey
    elif name == "TableNode":
        from .graph_db import TableNode
        return TableNode
    elif name == "SchemaCompressor":
        from .schema_compressor import SchemaCompressor
        return SchemaCompressor
    elif name == "CompactTableInfo":
        from .schema_compressor import CompactTableInfo
        return CompactTableInfo
    elif name == "HybridSchemaRetriever":
        from .schema_retriever import HybridSchemaRetriever
        return HybridSchemaRetriever
    elif name == "CrossEncoderReranker":
        from .schema_retriever import CrossEncoderReranker
        return CrossEncoderReranker
    elif name == "LegacySchemaRetriever":
        from .schema_retriever import LegacySchemaRetriever
        return LegacySchemaRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
