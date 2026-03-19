# src\retrieval\schema_compressor.py
"""
Компрессор схем БД для уменьшения количества токенов.

Production features:
- Compact Format с PK/FK/types для LLM
- Table Pruning
- Column Summarization
- Join Path Hints для SQL generation

Format Examples:
    movie(
        id INTEGER PK,
        title TEXT,
        year INTEGER
    ) FK: mID→Rating.mID
    
    rating(
        mID INTEGER FK→Movie.mID,
        uID INTEGER FK→User.uID,
        stars INTEGER
    )
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CompactTableInfo:
    """Компактное представление таблицы для LLM."""
    name: str
    db_name: str = "main"
    columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    primary_key: Optional[str] = None
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    row_count: Optional[int] = None
    description: Optional[str] = None
    
    def to_sql_format(self) -> str:
        """
        Формат для SQL generation (оптимальный для LLM).
        
        Example:
            movie(
                id INTEGER PK,
                title TEXT,
                year INTEGER
            ) FK: mID→Rating.mID
        """
        lines = [f"{self.name}("]
        
        col_parts = []
        for col in self.columns:
            col_type = self.column_types.get(col, "")
            pk_marker = " PK" if self.primary_key == col else ""
            
            # Проверяем если колонка это FK
            fk_marker = ""
            for fk in self.foreign_keys:
                if fk.get("from_column") == col:
                    to_table = fk.get("to_table", "")
                    to_column = fk.get("to_column", "")
                    fk_marker = f" FK→{to_table}.{to_column}"
                    break
            
            col_parts.append(f"    {col} {col_type}{pk_marker}{fk_marker}")
        
        lines.append(",\n".join(col_parts))
        lines.append(")")
        
        return "".join(lines)
    
    def to_compact_string(self) -> str:
        """Сжатое строковое представление."""
        parts = [f"{self.name}("]
        
        col_parts = []
        for col in self.columns:
            col_type = self.column_types.get(col, "")
            type_short = self._shorten_type(col_type)
            
            pk_marker = " PK" if self.primary_key == col else ""
            
            fk_marker = ""
            for fk in self.foreign_keys:
                if fk.get("from_column") == col:
                    to_table = fk.get("to_table", "")
                    to_column = fk.get("to_column", "")
                    fk_marker = f"→{to_table}.{to_column}"
                    break
            
            col_parts.append(f"{col}:{type_short}{pk_marker}{fk_marker}")
        
        parts.append(",".join(col_parts))
        parts.append(")")
        
        return "".join(parts)
    
    def _shorten_type(self, col_type: str) -> str:
        """Сократить тип данных."""
        type_map = {
            "INTEGER": "INT",
            "TEXT": "TXT",
            "REAL": "FLT",
            "NUMERIC": "NUM",
            "BLOB": "BLB",
            "BOOLEAN": "BLN",
            "DATE": "DAT",
            "DATETIME": "DTS",
            "TIMESTAMP": "TS",
        }
        col_upper = col_type.upper()
        for full, short in type_map.items():
            if full in col_upper:
                return short
        return col_type[:3].upper() if col_type else "UNK"
    
    def get_join_hints(self) -> List[str]:
        """Получить подсказки для JOIN."""
        hints = []
        for fk in self.foreign_keys:
            from_col = fk.get("from_column", "")
            to_table = fk.get("to_table", "")
            to_column = fk.get("to_column", "")
            hints.append(f"{self.name}.{from_col} = {to_table}.{to_column}")
        return hints


@dataclass
class JoinPathHint:
    """Подсказка для JOIN пути между таблицами."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    depth: int = 1
    
    def to_sql_hint(self) -> str:
        """SQL hint для JOIN."""
        return f"{self.from_table}.{self.from_column} = {self.to_table}.{self.to_column}"
    
    def to_natural_language(self) -> str:
        """Natural language описание."""
        return f"{self.from_table} joins to {self.to_table} via {self.from_column} → {self.to_column}"


class SchemaCompressor:
    """
    Компрессор схем БД для LLM.
    
    Production features:
    - Формат с PK/FK/types для точной SQL генерации
    - Join path hints из графа
    - Table pruning по релевантности
    - Column summarization по типам
    """
    
    COMPRESSION_LEVELS = {
        0: "full",      # Полная схема
        1: "standard",  # Стандартная
        2: "compact",   # Компактная  
        3: "minimal",   # Минимальная
    }
    
    def __init__(
        self,
        compression_level: int = 2,
        include_join_hints: bool = True,
        include_pk: bool = True,
        include_types: bool = True,
    ) -> None:
        """
        Инициализировать компрессор.
        
        Args:
            compression_level: Уровень компрессии (0-3).
            include_join_hints: Включать подсказки для JOIN.
            include_pk: Включать primary keys.
            include_types: Включать типы данных.
        """
        self.compression_level = compression_level
        self.include_join_hints = include_join_hints
        self.include_pk = include_pk
        self.include_types = include_types
        
        self.stats = {
            "original_tokens": 0,
            "compressed_tokens": 0,
            "compression_ratio": 0.0,
        }
    
    def compress(
        self,
        tables: List[Any],
        relevant_tables: Optional[Set[str]] = None,
        join_paths: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Сжать схему БД.
        
        Args:
            tables: Список TableInfo или CompactTableInfo.
            relevant_tables: Множество релевантных таблиц.
            join_paths: Join paths из графа.
        
        Returns:
            Сжатое представление схемы.
        """
        if not tables:
            return ""
        
        # Фильтрация релевантных таблиц
        if relevant_tables:
            tables = [t for t in tables if getattr(t, "name", None) in relevant_tables]
        
        if self.compression_level >= 2:
            return self._compress_compact(tables, join_paths)
        elif self.compression_level == 1:
            return self._compress_standard(tables, join_paths)
        else:
            return self._compress_full(tables)
    
    def _compress_full(self, tables: List[Any]) -> str:
        """Полное представление схемы."""
        parts = []
        for table in tables:
            part = self._format_table_full(table)
            parts.append(part)
        return "\n\n".join(parts)
    
    def _compress_standard(self, tables: List[Any], join_paths: Optional[List[Dict[str, Any]]] = None) -> str:
        """Стандартное сжатое представление с JOIN hints."""
        parts = []
        for table in tables:
            part = self._format_table_standard(table)
            parts.append(part)
        
        # Добавляем JOIN hints
        if join_paths and self.include_join_hints:
            parts.append("\n## JOIN Paths")
            for path in join_paths:
                for join in path.get("joins", []):
                    hint = f"{join.get('from', '')}.{join.get('from_column', '')} = {join.get('to', '')}.{join.get('to_column', '')}"
                    parts.append(f"  {hint}")
        
        return "\n".join(parts)
    
    def _compress_compact(
        self,
        tables: List[Any],
        join_paths: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Компактное представление (максимальное сжатие).
        
        Format:
            db: {table(col:TYPE PK→FK,...)}; {table2(...)}
        """
        # Группировка по БД
        db_tables: Dict[str, List[CompactTableInfo]] = {}
        
        for table in tables:
            db_name = getattr(table, "db_name", "main")
            if db_name not in db_tables:
                db_tables[db_name] = []
            
            # Конвертируем в CompactTableInfo если нужно
            if isinstance(table, CompactTableInfo):
                compact_table = table
            else:
                compact_table = self._convert_to_compact(table)
            
            db_tables[db_name].append(compact_table)
        
        # Форматирование
        db_parts = []
        for db_name, db_table_list in db_tables.items():
            table_strs = [t.to_compact_string() for t in db_table_list]
            db_parts.append(f"{db_name}: {{{', '.join(table_strs)}}}")
        
        result = "; ".join(db_parts)
        
        # Добавляем JOIN hints
        if join_paths and self.include_join_hints:
            join_hints = []
            for path in join_paths:
                for join in path.get("joins", []):
                    hint = f"{join.get('from', '')}.{join.get('from_column', '')} = {join.get('to', '')}.{join.get('to_column', '')}"
                    join_hints.append(hint)
            if join_hints:
                result += f"\n\nJOIN: {', '.join(join_hints)}"
        
        return result
    
    def compress_for_llm(
        self,
        tables: List[Any],
        join_paths: Optional[List[Dict[str, Any]]] = None,
        include_examples: bool = False,
    ) -> str:
        """
        Сжатие оптимизированное для LLM SQL generation.
        
        Format:
            ## Schema
            movie(
                id INTEGER PK,
                title TEXT,
                year INTEGER
            )
            
            rating(
                mID INTEGER FK→Movie.mID,
                uID INTEGER FK→User.uID,
                stars INTEGER
            )
            
            ## JOIN Paths
            Movie.mID = Rating.mID
        """
        parts = ["## Schema\n"]
        
        for table in tables:
            if isinstance(table, CompactTableInfo):
                compact_table = table
            else:
                compact_table = self._convert_to_compact(table)
            
            parts.append(compact_table.to_sql_format())
            parts.append("")  # пустая строка между таблицами
        
        # Добавляем JOIN hints
        if join_paths and self.include_join_hints:
            parts.append("## JOIN Paths\n")
            for path in join_paths:
                for join in path.get("joins", []):
                    hint = f"{join.get('from', '')}.{join.get('from_column', '')} = {join.get('to', '')}.{join.get('to_column', '')}"
                    parts.append(f"- {hint}")
            parts.append("")
        
        return "\n".join(parts)
    
    def _format_table_full(self, table: Any) -> str:
        """Полное форматирование таблицы."""
        lines = [f"Table: {table.name}", f"Database: {getattr(table, 'db_name', 'main')}", "Columns:"]
        
        for col in getattr(table, "columns", []):
            pk_marker = " [PK]" if getattr(table, "primary_key", None) == col else ""
            col_type = getattr(table, "column_types", {}).get(col, "")
            lines.append(f"  - {col} ({col_type}){pk_marker}")
        
        if getattr(table, "foreign_keys", []):
            lines.append("Foreign Keys:")
            for fk in table.foreign_keys:
                lines.append(f"  - {fk.get('from_column', '')} → {fk.get('to_table', '')}.{fk.get('to_column', '')}")
        
        if getattr(table, "row_count", None):
            lines.append(f"Rows: {table.row_count}")
        
        return "\n".join(lines)
    
    def _format_table_standard(self, table: Any) -> str:
        """Стандартное форматирование."""
        parts = [f"Table: {table.name}"]
        
        # Колонки с типами
        col_strs = []
        for col in getattr(table, "columns", []):
            col_type = getattr(table, "column_types", {}).get(col, "")
            pk_marker = " [PK]" if getattr(table, "primary_key", None) == col else ""
            col_strs.append(f"{col}({col_type}){pk_marker}")
        
        parts.append(f"Columns: {', '.join(col_strs)}")
        
        # Foreign keys
        if getattr(table, "foreign_keys", []):
            fk_str = ", ".join(
                f"{fk.get('from_column', '')}→{fk.get('to_table', '')}.{fk.get('to_column', '')}"
                for fk in table.foreign_keys
            )
            parts.append(f"FK: {fk_str}")
        
        return "\n".join(parts)
    
    def _convert_to_compact(self, table: Any) -> CompactTableInfo:
        """Конвертировать таблицу в CompactTableInfo."""
        return CompactTableInfo(
            name=getattr(table, "name", ""),
            db_name=getattr(table, "db_name", "main"),
            columns=getattr(table, "columns", []) or getattr(table, "column_names", []),
            column_types=getattr(table, "column_types", {}),
            primary_key=getattr(table, "primary_key", None),
            foreign_keys=[fk.to_dict() if hasattr(fk, "to_dict") else fk for fk in getattr(table, "foreign_keys", [])],
            row_count=getattr(table, "row_count", None),
            description=getattr(table, "description", None),
        )
    
    def compress_from_schema_docs(
        self,
        schema_docs: List[str],
        top_k: int = 5,
    ) -> str:
        """Сжать список документов схемы."""
        if not schema_docs:
            return ""
        
        selected = schema_docs[:top_k]
        
        if self.compression_level >= 2:
            tables = []
            for doc in selected:
                table = self._parse_schema_doc(doc)
                if table:
                    tables.append(table)
            return self._compress_compact(tables)
        else:
            return "\n".join(selected)
    
    def _parse_schema_doc(self, doc: str) -> Optional[CompactTableInfo]:
        """Распарсить документ схемы в CompactTableInfo."""
        try:
            table_match = re.search(r"Table:\s*(\w+)", doc)
            if not table_match:
                return None
            table_name = table_match.group(1)
            
            db_match = re.search(r"Database:\s*(\w+)", doc)
            db_name = db_match.group(1) if db_match else "main"
            
            cols_match = re.search(r"Columns:\s*(.+?)(?:\n|$)", doc)
            columns = []
            column_types = {}
            if cols_match:
                cols_str = cols_match.group(1)
                for col in cols_str.split(","):
                    col = col.strip()
                    col_type_match = re.match(r"(\w+)\s*\((\w+)\)", col)
                    if col_type_match:
                        columns.append(col_type_match.group(1))
                        column_types[col_type_match.group(1)] = col_type_match.group(2)
                    else:
                        columns.append(col)
                        column_types[col] = ""
            
            foreign_keys = []
            fk_match = re.search(r"Foreign Keys:\s*(.+?)(?:\n|$)", doc)
            if fk_match:
                fk_str = fk_match.group(1)
                for fk in fk_str.split(","):
                    fk = fk.strip()
                    fk_parts = re.match(r"(\w+)\s*→?\s*(\w+)\.(\w+)", fk)
                    if fk_parts:
                        foreign_keys.append({
                            "from_column": fk_parts.group(1),
                            "to_table": fk_parts.group(2),
                            "to_column": fk_parts.group(3),
                        })
            
            return CompactTableInfo(
                name=table_name,
                db_name=db_name,
                columns=columns,
                column_types=column_types,
                primary_key=None,
                foreign_keys=foreign_keys,
                row_count=None,
            )
        except Exception as e:
            logger.warning(f"Failed to parse schema doc: {e}")
            return None
    
    def estimate_tokens(self, text: str) -> int:
        """Оценить количество токенов в тексте."""
        return len(text) // 4
    
    def compress_with_stats(
        self,
        tables: List[Any],
        relevant_tables: Optional[Set[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Сжать схему и вернуть статистику."""
        original = self._compress_full(tables)
        original_tokens = self.estimate_tokens(original)
        
        compressed = self.compress(tables, relevant_tables)
        compressed_tokens = self.estimate_tokens(compressed)
        
        compression_ratio = (
            (original_tokens - compressed_tokens) / original_tokens * 100
            if original_tokens > 0
            else 0
        )
        
        self.stats = {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": round(compression_ratio, 2),
            "original_chars": len(original),
            "compressed_chars": len(compressed),
        }
        
        return compressed, self.stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику компрессии."""
        return self.stats
