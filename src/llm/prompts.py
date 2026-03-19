# src\llm\prompts.py
"""
Промпты для LLM агентов с оптимизациями.

Оптимизации:
1. Compact Format - сжатые промпты
2. Structured Output - структурированный JSON вывод
3. Few-Shot Examples - примеры в компактном формате
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Prompts:
    """Коллекция промптов для различных агентов."""

    # ===========================================
    # SQL Generator - Improved with Strict Rules
    # ===========================================

    SQL_GENERATOR_TEMPLATE = """You are an expert SQL compiler for SQLite.

## Task
Generate SQL query for the question using ONLY the schema below.

## Schema
{schema}

## Question
{query}

## CRITICAL RULES

### Rule 1: Use EXACT table/column names from schema
- Check schema for EXACT table names (case-sensitive!)
- If schema shows "song" → use "song" NOT "Songs"
- If schema shows "Movie" → use "Movie" NOT "movie"
- **NEVER invent table names!**

### Rule 2: DO NOT hallucinate values
- Use ONLY values from the question
- NEVER use hardcoded years/values unless explicitly in question

### Rule 3: For "rating" questions
- JOIN Movie and Rating tables
- Use `stars` column from Rating table

### Rule 4: For specific movie/artist
- Use WHERE title = 'Name' or WHERE name = 'Name'

### Rule 5: Return ONLY valid JSON
- Format: {{"sql": "YOUR_SQL_HERE"}}

## CORRECT Examples

Q: "What is the rating of the movie Avatar?"
Schema: Movie(mID, title, year), Rating(rID, mID, stars)
A: {{"sql": "SELECT R.stars FROM Movie M JOIN Rating R ON M.mID = R.mID WHERE M.title = 'Avatar'"}}

Q: "Show all songs"
Schema: song(song_name, artist_name, rating)
A: {{"sql": "SELECT * FROM song"}}

Q: "Show movies with rating above 7"
Schema: Movie(mID, title), Rating(mID, stars)
A: {{"sql": "SELECT M.title FROM Movie M JOIN Rating R ON M.mID = R.mID WHERE R.stars > 7"}}

## INCORRECT Examples (DO NOT DO THIS!)

❌ SELECT * FROM Songs WHERE schema shows "song" (wrong case!)
❌ SELECT * FROM Movie WHERE year = 1990 (year not in question!)
❌ SELECT * FROM Film (table must be "Movie" from schema!)

## Your Answer (JSON with YOUR SQL):
"""

    # Компактный формат для ускорения генерации
    SQL_GENERATOR_COMPACT = """SQL Generator. Output JSON: {{"sql": "...", "tables_used": [...]}}

Schema: {schema}
Question: {query}
SQL:
"""

    # ===========================================
    # SQL Judge - Improved with Semantic Check
    # ===========================================

    SQL_JUDGE_TEMPLATE = """You are an expert SQL verifier.

## Task
Evaluate whether the SQL query CORRECTLY ANSWERS the question.

## Critical Criteria
1. **SELECT Columns**: Does SELECT return what the question asks for?
2. **Table Usage**: Does it use the RIGHT tables for the question?
3. **Filter Correctness**: Are WHERE conditions from the question (not hallucinated)?
4. **Syntax**: Is the SQL syntactically valid?

## Output Format
{{"confidence": 0.0-1.0, "error": false, "reason": "explanation"}}

## Question
{query}

## SQL Query
{sql}

## CRITICAL Evaluation Rules

### Rule 1: Check SELECT columns match question intent
- Question asks for "rating" → SELECT must return rating/stars column
- Question asks for "title" → SELECT must return title
- Question asks for "count" → SELECT must use COUNT()
- Question asks for "name" → SELECT must return name column

### Rule 2: Check table usage
- Question asks for "rating" → MUST use Rating table with JOIN
- Question asks for specific movie → MUST filter by title (WHERE title = '...')
- Question asks for director → MUST use Movie table with director column

### Rule 3: Detect hallucinated values
- SQL has hardcoded values (year=1990, title='X') NOT in question → confidence < 0.3

### Rule 4: Verify JOIN for related data
- Question asks for movie + rating → MUST JOIN Movie and Rating tables
- Question asks for artist + song → MUST JOIN Artist and Song tables

## Scoring Examples

### GOOD (confidence > 0.8)
Q: "What is the rating of the movie Avatar?"
A: SELECT R.stars FROM Movie M JOIN Rating R ON M.mID = R.mID WHERE M.title = 'Avatar'
→ Returns stars (rating) ✅, Uses Rating table ✅, Correct filter ✅

### BAD (confidence < 0.3)
Q: "What is the rating of the movie Avatar?"
A: SELECT title FROM Movie WHERE title = 'Avatar'
→ Returns title (not rating) ❌, Doesn't use Rating table ❌

### BAD (confidence < 0.3)
Q: "Show movies with rating above 7"
A: SELECT * FROM Movie WHERE year = 1990
→ Hallucinated year ❌, Doesn't filter by rating ❌

## Evaluation (JSON)
"""

    # ===========================================
    # SQL Refiner
    # ===========================================

    SQL_REFINER_TEMPLATE = """You are an expert SQL debugger.

## Task
Fix the SQL query based on previous failed attempts.

## Schema
{schema}

## Question
{query}

## Previous Attempts
{history}

## Output Format
{{"sql": "corrected SQL", "fixes_applied": ["fix1"], "explanation": "why it works"}}

## Corrected SQL (JSON)
"""

    # ===========================================
    # Router Agent
    # ===========================================

    ROUTER_TEMPLATE = """You are a database router for multi-database text-to-SQL.

## Task
Select the most relevant databases to answer the question.

## Available Databases
{databases_info}

## Instructions
1. Review each database and its tables
2. Rank from most to least relevant based on the query semantics
3. Assign confidence scores (0.0-1.0)
4. Return ONLY valid JSON without any extra text

## Question
{query}

## Output Format
{{"ranked_databases": [{{"db_name": "database_name", "tables": ["table1", "table2"], "confidence": 0.9, "reason": "brief explanation"}}]}}

## Important
- Return ONLY the JSON object
- Do not include markdown code blocks
- Do not include any explanation before or after the JSON
- Ensure all strings are properly quoted

## Selection (JSON)
"""

    # ===========================================
    # Schema Retrieval
    # ===========================================

    SCHEMA_RETRIEVAL_TEMPLATE = """You are a schema analysis expert.

## Task
Identify which tables and columns are relevant to the question.

## Schema
{schema}

## Question
{query}

## Output Format
{{"relevant_tables": [{{"table_name": "...", "columns": [...], "reason": "..."}}]}}

## Analysis (JSON)
"""

    # ===========================================
    # Multi-DB SQL Generator
    # ===========================================

    MULTI_DB_SQL_GENERATOR_TEMPLATE = """You are an expert SQL developer for multiple SQLite databases.

## Task
Generate SQL that may access multiple databases.

## Available Databases
{databases_schema}

## Instructions
1. Use ATTACH DATABASE syntax for multi-db queries
2. Use database aliases (e.g., db1.table_name)
3. Ensure proper JOIN conditions

## ATTACH Example
ATTACH DATABASE 'path/to/db1.sqlite' AS db1;
SELECT db1.table1.col1 FROM db1.table1;

## Question
{query}

## Output Format
{{"attach_statements": ["ATTACH ..."], "sql": "SELECT ...", "tables_used": {{}}}}

## SQL (JSON)
"""

    @classmethod
    def format_sql_generator(
        cls,
        query: str,
        schema: str,
        use_compact: bool = False,
    ) -> str:
        """
        Сформировать промпт для генерации SQL.

        Args:
            query: Запрос.
            schema: Схема.
            use_compact: Использовать компактный формат.

        Returns:
            Промпт.
        """
        if use_compact:
            return cls.SQL_GENERATOR_COMPACT.format(query=query, schema=schema)
        return cls.SQL_GENERATOR_TEMPLATE.format(query=query, schema=schema)

    @classmethod
    def format_sql_judge(cls, query: str, sql: str) -> str:
        """Сформировать промпт для judge."""
        return cls.SQL_JUDGE_TEMPLATE.format(query=query, sql=sql)

    @classmethod
    def format_sql_refiner(
        cls,
        query: str,
        schema: str,
        history: str,
    ) -> str:
        """Сформировать промпт для рефайнмента."""
        return cls.SQL_REFINER_TEMPLATE.format(
            query=query,
            schema=schema,
            history=history,
        )

    @classmethod
    def format_router(
        cls,
        query: str,
        databases: List[Dict[str, Any]],
    ) -> str:
        """Сформировать промпт для Router Agent."""
        db_info = "".join(
            f"\n### Database: {db['db_name']}\n"
            f"Path: {db['db_path']}\n"
            f"Tables: {', '.join(db['tables'])}\n"
            f"Relevance: {db['avg_score']:.3f}\n\n"
            for db in databases
        )
        return cls.ROUTER_TEMPLATE.format(query=query, databases_info=db_info)

    @classmethod
    def format_schema_retrieval(cls, query: str, schema: str) -> str:
        """Сформировать промпт для retrieval схемы."""
        return cls.SCHEMA_RETRIEVAL_TEMPLATE.format(query=query, schema=schema)

    @classmethod
    def format_multi_db_sql_generator(
        cls,
        query: str,
        databases_schema: str,
    ) -> str:
        """Сформировать промпт для multi-DB генерации SQL."""
        return cls.MULTI_DB_SQL_GENERATOR_TEMPLATE.format(
            query=query,
            databases_schema=databases_schema,
        )

    @classmethod
    def add_few_shot_examples(
        cls,
        prompt: str,
        examples: List[Dict[str, str]],
    ) -> str:
        """
        Добавить few-shot примеры.

        Args:
            prompt: Основной промпт.
            examples: Примеры [{"question": "...", "sql": "..."}].

        Returns:
            Промпт с примерами.
        """
        if not examples:
            return prompt

        examples_text = "## Examples\n\n"
        for i, ex in enumerate(examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"  Q: {ex['question']}\n"
            examples_text += f"  A: {ex['sql']}\n\n"

        return examples_text + prompt

    @classmethod
    def format_error_recovery(
        cls,
        query: str,
        schema: str,
        error_sql: str,
        error_message: str,
    ) -> str:
        """Сформировать промпт для восстановления после ошибки."""
        return f"""
You are an SQL error recovery expert.

## Question
{query}

## Schema
{schema}

## Failed SQL
{error_sql}

## Error Message
{error_message}

## Task
Generate corrected SQL.

## Output Format
{{"corrected_sql": "...", "error_analysis": "...", "fix_description": "..."}}

## Corrected SQL (JSON)
"""
