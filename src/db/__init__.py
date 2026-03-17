"""Database модули."""
# Lazy imports для избежания циклических импортов
__all__ = [
    "SchemaLoader",
    "TableInfo",
    "ColumnInfo",
    "ForeignKeyInfo",
    "SQLExecutor",
    "MultiDBExecutor",
]


def __getattr__(name):
    if name == "SchemaLoader":
        from .schema_loader import SchemaLoader
        return SchemaLoader
    elif name == "TableInfo":
        from .schema_loader import TableInfo
        return TableInfo
    elif name == "ColumnInfo":
        from .schema_loader import ColumnInfo
        return ColumnInfo
    elif name == "ForeignKeyInfo":
        from .schema_loader import ForeignKeyInfo
        return ForeignKeyInfo
    elif name == "SQLExecutor":
        from .executor import SQLExecutor
        return SQLExecutor
    elif name == "MultiDBExecutor":
        from .multi_db_executor import MultiDBExecutor
        return MultiDBExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
