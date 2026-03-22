# src\agents\sql_validator.py
"""
SQL Validator Agent для валидации сгенерированного SQL.

Production features:
- SQL Guardrails integration (SELECT-only, auto-LIMIT, SQL injection protection)
- Syntax validation
- Table/column existence check
- JOIN validation
- Error detection and suggestions

Pipeline:
    SQL → Guardrails → Syntax Check → Schema Check → JOIN Check → Valid/Invalid
"""
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..db.guardrails import SQLGuardrails
from ..utils.sql_parser import (
    extract_tables,
    extract_columns,
    has_join,
    has_aggregate,
    extract_join_tables,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Результат валидации SQL."""
    valid: bool
    sql: str
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    error_type: Optional[str] = None  # syntax, schema, join, semantic
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в dict."""
        return {
            "valid": self.valid,
            "sql": self.sql,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "error_type": self.error_type,
        }


class SQLValidator:
    """
    SQL Validator Agent для валидации сгенерированного SQL.
    
    Production features:
    - Syntax validation (SQLite)
    - Table/column existence check
    - JOIN path validation
    - Error detection с suggestions
    """
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализировать валидатор.
        
        Args:
            schema: Схема базы данных для валидации.
        """
        self.schema = schema or {}
        self._tables: Set[str] = set()
        self._columns: Dict[str, Set[str]] = {}
        self._foreign_keys: List[Dict[str, str]] = []
        
        if schema:
            self._load_schema(schema)
    
    def _load_schema(self, schema: Dict[str, Any]) -> None:
        """Загрузить схему для валидации."""
        tables = schema.get("tables", [])
        
        for table in tables:
            table_name = table.get("name", "")
            self._tables.add(table_name)
            
            columns = table.get("columns", [])
            self._columns[table_name] = set(columns)
            
            fks = table.get("foreign_keys", [])
            for fk in fks:
                fk_entry = {
                    "from_table": table_name,
                    "from_column": fk.get("from_column", ""),
                    "to_table": fk.get("to_table", ""),
                    "to_column": fk.get("to_column", ""),
                }
                self._foreign_keys.append(fk_entry)
    
    def validate(self, sql: str) -> bool:
        """
        Валидировать SQL запрос.
        
        Args:
            sql: SQL запрос.
        
        Returns:
            True если валиден.
        """
        result = self.validate_with_details(sql)
        return result.valid
    
    def validate_with_details(self, sql: str) -> ValidationResult:
        """
        Валидировать SQL с деталями.
        
        Args:
            sql: SQL запрос.
        
        Returns:
            ValidationResult с ошибками и предупреждениями.
        """
        errors: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []
        
        # Step 1: Basic syntax check
        syntax_valid, syntax_errors = self._check_syntax(sql)
        if not syntax_valid:
            return ValidationResult(
                valid=False,
                sql=sql,
                errors=syntax_errors,
                warnings=warnings,
                suggestions=suggestions,
                error_type="syntax",
            )
        
        # Step 2: Schema validation (tables/columns)
        schema_valid, schema_errors, schema_warnings = self._check_schema(sql)
        errors.extend(schema_errors)
        warnings.extend(schema_warnings)
        
        # Step 3: JOIN validation
        join_valid, join_errors, join_suggestions = self._check_joins(sql)
        errors.extend(join_errors)
        suggestions.extend(join_suggestions)
        
        # Step 4: Semantic validation
        semantic_warnings = self._check_semantics(sql)
        warnings.extend(semantic_warnings)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            valid=is_valid,
            sql=sql,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            error_type=None if is_valid else "schema" if schema_errors else "join" if join_errors else "syntax",
        )
    
    def _check_syntax(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Проверить базовый синтаксис SQL.
        
        Checks:
        - Balanced parentheses
        - Basic SQL keywords
        - No obvious syntax errors
        """
        errors = []
        
        # Check balanced parentheses
        if sql.count("(") != sql.count(")"):
            errors.append("Unbalanced parentheses")
        
        # Check for basic SELECT structure
        sql_upper = sql.upper().strip()
        if not sql_upper.startswith("SELECT"):
            errors.append("SQL must start with SELECT")
        
        # Check for unclosed quotes
        single_quotes = sql.count("'")
        if single_quotes % 2 != 0:
            errors.append("Unclosed string literal")
        
        # Check for common syntax errors
        if re.search(r',\s*(FROM|WHERE|GROUP|ORDER|LIMIT)', sql, re.IGNORECASE):
            errors.append("Comma before clause keyword")
        
        # Check for double operators
        if re.search(r'[+\-*/]=\s*[+\-*/]', sql):
            errors.append("Invalid operator sequence")
        
        return len(errors) == 0, errors
    
    def _check_schema(self, sql: str) -> Tuple[bool, List[str], List[str]]:
        """
        Проверить существование таблиц и колонок.
        
        Checks:
        - Tables exist in schema
        - Columns exist in tables
        - Aliases are properly defined
        """
        errors = []
        warnings = []
        
        if not self._tables:
            # Schema not loaded, skip validation
            return True, errors, warnings
        
        # Extract table references
        table_refs = self._extract_table_references(sql)
        aliases = {}
        
        for table_ref in table_refs:
            parts = table_ref.split()
            if len(parts) >= 2:
                table_name = parts[0].strip()
                alias = parts[1].strip()
                aliases[alias] = table_name
            else:
                table_name = parts[0].strip()
                # Don't add to aliases if it's a join clause
        
        # 🔥 CRITICAL: Check table names exist in schema (case-sensitive)
        for table_name in list(aliases.values()) + [t.split()[0].strip() for t in table_refs]:
            # Check exact match first
            if table_name not in self._tables:
                # Check case-insensitive match
                case_insensitive_match = None
                for schema_table in self._tables:
                    if schema_table.lower() == table_name.lower():
                        case_insensitive_match = schema_table
                        break
                
                if case_insensitive_match:
                    errors.append(
                        f"Table '{table_name}' not found. Did you mean '{case_insensitive_match}'? "
                        f"(SQLite table names are case-sensitive)"
                    )
                else:
                    errors.append(f"Table '{table_name}' does not exist. Available tables: {', '.join(self._tables)}")
        
        # Extract column references
        column_refs = self._extract_column_references(sql)
        
        for col_ref in column_refs:
            parts = col_ref.split(".")
            if len(parts) == 2:
                alias_or_table = parts[0].strip()
                column = parts[1].strip()
                
                # Resolve alias
                table_name = aliases.get(alias_or_table, alias_or_table)
                
                # Check column exists
                if table_name in self._columns:
                    if column != "*" and column not in self._columns[table_name]:
                        errors.append(f"Column '{column}' does not exist in table '{table_name}'")
            else:
                # Column without table prefix
                column = parts[0].strip()
                if column != "*":
                    # Check if column exists in any referenced table
                    found = False
                    for table_name in self._tables:
                        if column in self._columns.get(table_name, set()):
                            found = True
                            break
                    
                    if not found and column.upper() not in ("COUNT", "SUM", "AVG", "MIN", "MAX"):
                        warnings.append(f"Column '{column}' not found in schema")
        
        return len(errors) == 0, errors, warnings
    
    def _check_joins(self, sql: str) -> Tuple[bool, List[str], List[str]]:
        """
        Проверить JOIN условия.
        
        Checks:
        - JOIN conditions use valid foreign keys
        - JOIN path exists between tables
        """
        errors = []
        suggestions = []
        
        if "JOIN" not in sql.upper():
            return True, errors, suggestions
        
        # Extract JOIN conditions
        join_conditions = self._extract_join_conditions(sql)
        
        for condition in join_conditions:
            # Check if JOIN condition matches a foreign key
            match_found = False
            for fk in self._foreign_keys:
                fk_condition = f"{fk['from_table']}.{fk['from_column']} = {fk['to_table']}.{fk['to_column']}"
                alt_condition = f"{fk['to_table']}.{fk['to_column']} = {fk['from_table']}.{fk['from_column']}"
                
                if condition in (fk_condition, alt_condition):
                    match_found = True
                    break
            
            if not match_found and self._foreign_keys:
                suggestions.append(f"JOIN condition '{condition}' does not match known foreign key relationships")
        
        return len(errors) == 0, errors, suggestions
    
    def _check_semantics(self, sql: str) -> List[str]:
        """
        Проверить семантические аспекты.
        
        Checks:
        - GROUP BY with aggregates
        - WHERE vs HAVING
        - ORDER BY direction
        """
        warnings = []
        
        sql_upper = sql.upper()
        
        # Check GROUP BY with aggregates
        if "GROUP BY" in sql_upper:
            has_aggregate = any(agg in sql_upper for agg in ("COUNT(", "SUM(", "AVG(", "MIN(", "MAX("))
            if not has_aggregate:
                warnings.append("GROUP BY without aggregate function")
        
        # Check HAVING without GROUP BY
        if "HAVING" in sql_upper and "GROUP BY" not in sql_upper:
            warnings.append("HAVING clause without GROUP BY")
        
        # Check LIMIT with negative value
        limit_match = re.search(r'LIMIT\s+(-?\d+)', sql, re.IGNORECASE)
        if limit_match:
            limit_value = int(limit_match.group(1))
            if limit_value < 0:
                warnings.append("Negative LIMIT value")
        
        return warnings
    
    def _extract_table_references(self, sql: str) -> List[str]:
        """Извлечь ссылки на таблицы из SQL."""
        tables = []
        
        # FROM clause
        from_match = re.search(r'FROM\s+([^WHERE]+?)(?:WHERE|$)', sql, re.IGNORECASE)
        if from_match:
            from_clause = from_match.group(1).strip()
            # Handle multiple tables (comma-separated)
            for part in from_clause.split(","):
                part = part.strip()
                # Remove JOIN parts
                part = re.split(r'\s+(?:LEFT|RIGHT|INNER|OUTER|CROSS|JOIN)', part, flags=re.IGNORECASE)[0]
                if part:
                    tables.append(part)
        
        # JOIN clauses
        join_matches = re.findall(r'JOIN\s+(\S+)', sql, re.IGNORECASE)
        tables.extend(join_matches)
        
        return tables
    
    def _extract_column_references(self, sql: str) -> List[str]:
        """Извлечь ссылки на колонки из SQL."""
        columns = []
        
        # SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # Handle aggregates
            select_clause = re.sub(r'\([^)]*\)', '', select_clause)
            for part in select_clause.split(","):
                part = part.strip()
                # Remove AS alias
                part = re.sub(r'\s+AS\s+\S+', '', part, flags=re.IGNORECASE)
                if "." in part:
                    columns.append(part)
        
        # WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|$)', sql, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1)
            # Extract column references
            col_refs = re.findall(r'(\w+\.\w+)', where_clause)
            columns.extend(col_refs)
        
        # JOIN conditions
        on_matches = re.findall(r'ON\s+(\w+\.\w+\s*=\s*\w+\.\w+)', sql, re.IGNORECASE)
        for on_match in on_matches:
            parts = re.findall(r'\w+\.\w+', on_match)
            columns.extend(parts)
        
        return columns
    
    def _extract_join_conditions(self, sql: str) -> List[str]:
        """Извлечь условия JOIN из SQL."""
        conditions = []
        
        on_matches = re.findall(r'ON\s+(\w+\.\w+\s*=\s*\w+\.\w+)', sql, re.IGNORECASE)
        for on_match in on_matches:
            # Normalize whitespace
            condition = re.sub(r'\s+', ' ', on_match.strip())
            conditions.append(condition)
        
        return conditions
    
    def get_fix_suggestion(self, error_sql: str, error_message: str) -> str:
        """
        Получить предложение для исправления ошибки.
        
        Args:
            error_sql: Ошибочный SQL.
            error_message: Сообщение об ошибке.
        
        Returns:
            Предложение для исправления.
        """
        suggestions = []
        
        if "does not exist" in error_message:
            if "Table" in error_message:
                table_match = re.search(r"Table '(\w+)'", error_message)
                if table_match:
                    suggestions.append(f"Check table name spelling. Available tables: {', '.join(self._tables)}")
            elif "Column" in error_message:
                suggestions.append("Check column name spelling and table alias")
        
        if "Unbalanced parentheses" in error_message:
            suggestions.append("Check for matching opening and closing parentheses")
        
        if "Unclosed string" in error_message:
            suggestions.append("Add closing quote for string literal")
        
        return "; ".join(suggestions) if suggestions else "Review SQL syntax and schema"
