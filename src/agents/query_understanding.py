# src\agents\query_understanding.py
"""
Query Understanding Layer для анализа запроса пользователя.

Production features:
- Intent classification
- Entity extraction
- Filter detection
- Aggregation detection
- Database routing hints

Pipeline:
    User Query → Intent → Entities → Filters → Routing
"""
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Типы интентов запроса."""
    SELECT = "select"           # Простой выбор данных
    AGGREGATE = "aggregate"     # Агрегация (COUNT, SUM, AVG)
    RANKING = "ranking"         # Ранжирование (TOP N, ORDER BY)
    FILTER = "filter"           # Фильтрация (WHERE)
    JOIN = "join"               # JOIN между таблицами
    GROUP = "group"             # GROUP BY
    COMPARISON = "comparison"   # Сравнение (больше, меньше)
    TEMPORAL = "temporal"       # Временные фильтры (год, дата)
    UNKNOWN = "unknown"         # Неопределённый


@dataclass
class QueryUnderstanding:
    """Результат анализа запроса."""
    original_query: str
    intent: QueryIntent = QueryIntent.UNKNOWN
    entities: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    aggregations: List[str] = field(default_factory=list)
    order_by: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    group_by: List[str] = field(default_factory=list)
    db_hints: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в dict."""
        return {
            "original_query": self.original_query,
            "intent": self.intent.value,
            "entities": self.entities,
            "tables": self.tables,
            "columns": self.columns,
            "filters": self.filters,
            "aggregations": self.aggregations,
            "order_by": self.order_by,
            "limit": self.limit,
            "group_by": self.group_by,
            "db_hints": self.db_hints,
            "confidence": self.confidence,
        }


class QueryUnderstandingAgent:
    """
    Query Understanding Agent для анализа запросов.
    
    Production features:
    - Intent classification
    - Entity extraction
    - Filter detection
    - Aggregation detection
    - Database routing hints
    """
    
    # Keywords для intent classification
    AGGREGATE_KEYWORDS = {
        "count", "sum", "avg", "average", "min", "maximum", "max", "minimum",
        "сколько", "подсчитай", "посчитай", "сумма", "средн", "минимум", "максимум"
    }
    
    RANKING_KEYWORDS = {
        "top", "best", "highest", "lowest", "first", "last",
        "лучш", "худш", "высш", "низш", "перв", "послед",
        "топ", "рейтинг"
    }
    
    TEMPORAL_KEYWORDS = {
        "year", "date", "time", "month", "day", "when",
        "год", "дата", "время", "месяц", "день", "когда"
    }
    
    COMPARISON_KEYWORDS = {
        "more", "less", "greater", "smaller", "higher", "lower", "than",
        "больше", "меньше", "выше", "ниже", "чем"
    }
    
    def __init__(self, schema_hints: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализировать agent.
        
        Args:
            schema_hints: Подсказки о схеме (таблицы, колонки).
        """
        self.schema_hints = schema_hints or {}
        self._table_names: Set[str] = set(schema_hints.get("tables", []))
        self._column_names: Set[str] = set(schema_hints.get("columns", []))
    
    def analyze(self, query: str) -> QueryUnderstanding:
        """
        Проанализировать запрос пользователя.
        
        Pipeline:
            1. Detect intent
            2. Extract entities
            3. Detect filters
            4. Detect aggregations
            5. Extract ORDER BY / LIMIT
            6. Generate DB routing hints
        
        Args:
            query: Запрос пользователя.
        
        Returns:
            QueryUnderstanding результат.
        """
        result = QueryUnderstanding(original_query=query)
        
        # Step 1: Intent classification
        result.intent = self._classify_intent(query)
        
        # Step 2: Entity extraction
        result.entities = self._extract_entities(query)
        
        # Step 3: Table detection
        result.tables = self._detect_tables(query)
        
        # Step 4: Filter detection
        result.filters = self._detect_filters(query)
        
        # Step 5: Aggregation detection
        result.aggregations = self._detect_aggregations(query)
        
        # Step 6: ORDER BY / LIMIT
        result.order_by = self._extract_order_by(query)
        result.limit = self._extract_limit(query)
        result.group_by = self._extract_group_by(query)
        
        # Step 7: DB routing hints
        result.db_hints = self._generate_db_hints(query, result)
        
        # Step 8: Confidence score
        result.confidence = self._calculate_confidence(result)
        
        logger.info(f"Query analyzed: intent={result.intent.value}, entities={len(result.entities)}")
        return result
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Классифицировать интент запроса."""
        query_lower = query.lower()
        
        # Check for aggregations
        if any(kw in query_lower for kw in self.AGGREGATE_KEYWORDS):
            return QueryIntent.AGGREGATE
        
        # Check for ranking
        if any(kw in query_lower for kw in self.RANKING_KEYWORDS):
            return QueryIntent.RANKING
        
        # Check for temporal
        if any(kw in query_lower for kw in self.TEMPORAL_KEYWORDS):
            return QueryIntent.TEMPORAL
        
        # Check for comparison
        if any(kw in query_lower for kw in self.COMPARISON_KEYWORDS):
            return QueryIntent.COMPARISON
        
        # Check for explicit GROUP BY indicators
        if "group" in query_lower or "by" in query_lower or "each" in query_lower:
            return QueryIntent.GROUP
        
        # Check for JOIN indicators
        if "join" in query_lower or "between" in query_lower or "across" in query_lower:
            return QueryIntent.JOIN
        
        # Check for filter indicators
        if "where" in query_lower or "with" in query_lower or "that" in query_lower:
            return QueryIntent.FILTER
        
        # Default to SELECT
        return QueryIntent.SELECT
    
    def _extract_entities(self, query: str) -> List[str]:
        """Извлечь entity mentions из запроса."""
        entities = []
        
        # Extract potential entity names (capitalized words, quoted strings)
        # Movies, albums, artists, etc.
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
        for match in quoted:
            entity = match[0] or match[1]
            if entity:
                entities.append(entity)
        
        # Extract capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b([A-Z][a-z]+)\b', query)
        for word in capitalized:
            if len(word) > 2 and word.lower() not in ("select", "from", "where", "the", "and", "with"):
                entities.append(word)
        
        return list(set(entities))
    
    def _detect_tables(self, query: str) -> List[str]:
        """Обнаружить упомянутые таблицы."""
        if not self._table_names:
            return []
        
        query_lower = query.lower()
        detected = []
        
        for table in self._table_names:
            # Check if table name appears in query
            if table.lower() in query_lower:
                detected.append(table)
        
        # Heuristic: detect from entities
        for entity in self._extract_entities(query):
            entity_lower = entity.lower()
            for table in self._table_names:
                if table.lower() in entity_lower or entity_lower in table.lower():
                    if table not in detected:
                        detected.append(table)
        
        return detected
    
    def _detect_filters(self, query: str) -> List[Dict[str, Any]]:
        """Обнаружить фильтры в запросе."""
        filters = []
        query_lower = query.lower()
        
        # Numeric filters: "above 7", "more than 100", "less than 50"
        numeric_patterns = [
            r'(?:above|over|more than|greater than)\s+(\d+(?:\.\d+)?)',
            r'(?:below|under|less than|fewer than)\s+(\d+(?:\.\d+)?)',
            r'(?:выше|более|больше)\s+(\d+(?:\.\d+)?)',
            r'(?:ниже|менее|меньше)\s+(\d+(?:\.\d+)?)',
        ]
        
        for pattern in numeric_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                filters.append({
                    "type": "numeric",
                    "value": float(match),
                    "operator": ">" if any(kw in query_lower for kw in ["above", "over", "more", "greater", "выше", "более", "больше"]) else "<",
                })
        
        # Year filters: "in 2020", "from 1990"
        year_patterns = [
            r'(?:in|from|year)\s*(\d{4})',
            r'(\d{4})\s*(?:год|year)',
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                filters.append({
                    "type": "year",
                    "value": int(match),
                    "operator": "=",
                })
        
        # Text filters: "directed by Spielberg"
        text_patterns = [
            r'(?:directed|created|written|produced)\s+by\s+(\w+(?:\s+\w+)?)',
            r'(?:режиссёр|режиссер|создан)\s+(?:кем|кто)\s*(\w+(?:\s+\w+)?)',
        ]
        
        for pattern in text_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                filters.append({
                    "type": "text",
                    "value": match.strip(),
                    "operator": "=",
                })
        
        return filters
    
    def _detect_aggregations(self, query: str) -> List[str]:
        """Обнаружить агрегации."""
        aggregations = []
        query_lower = query.lower()
        
        agg_map = {
            "count": "COUNT",
            "sum": "SUM",
            "avg": "AVG",
            "average": "AVG",
            "min": "MIN",
            "minimum": "MIN",
            "max": "MAX",
            "maximum": "MAX",
            "сколько": "COUNT",
            "подсчитай": "COUNT",
            "посчитай": "COUNT",
            "сумма": "SUM",
            "средн": "AVG",
            "минимум": "MIN",
            "максимум": "MAX",
        }
        
        for keyword, sql_func in agg_map.items():
            if keyword in query_lower:
                if sql_func not in aggregations:
                    aggregations.append(sql_func)
        
        return aggregations
    
    def _extract_order_by(self, query: str) -> Optional[Dict[str, Any]]:
        """Извлечь ORDER BY информацию."""
        query_lower = query.lower()
        
        # DESC indicators
        desc_keywords = ["top", "highest", "best", "descending", "топ", "лучш", "высш"]
        asc_keywords = ["lowest", "ascending", "худш", "низш"]
        
        order_column = None
        order_direction = "DESC"
        
        for kw in desc_keywords:
            if kw in query_lower:
                order_direction = "DESC"
                break
        
        for kw in asc_keywords:
            if kw in query_lower:
                order_direction = "ASC"
                break
        
        # Detect order column from context
        if "rating" in query_lower:
            order_column = "rating"
        elif "price" in query_lower:
            order_column = "price"
        elif "year" in query_lower:
            order_column = "year"
        elif "count" in query_lower or "сколько" in query_lower:
            order_column = "count"
        
        if order_column:
            return {"column": order_column, "direction": order_direction}
        
        return None
    
    def _extract_limit(self, query: str) -> Optional[int]:
        """Извлечь LIMIT из запроса."""
        query_lower = query.lower()
        
        # "top 10", "first 5", "top N"
        patterns = [
            r'top\s+(\d+)',
            r'first\s+(\d+)',
            r'last\s+(\d+)',
            r'топ\s+(\d+)',
            r'первые?\s+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_group_by(self, query: str) -> List[str]:
        """Извлечь GROUP BY колонки."""
        groups = []
        query_lower = query.lower()
        
        # "by genre", "by year", "by director"
        patterns = [
            r'by\s+(\w+)',
            r'per\s+(\w+)',
            r'each\s+(\w+)',
            r'для\s+каждого\s+(\w+)',
            r'по\s+(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query_lower)
            groups.extend(matches)
        
        return list(set(groups))
    
    def _generate_db_hints(self, query: str, understanding: QueryUnderstanding) -> List[str]:
        """Сгенерировать подсказки для выбора БД."""
        hints = []
        
        # Movie-related keywords
        movie_keywords = ["movie", "film", "фильм", "кино", "director", "режиссёр", "actor", "актёр"]
        if any(kw in query.lower() for kw in movie_keywords):
            hints.append("movie_db")
        
        # Music-related keywords
        music_keywords = ["music", "song", "album", "artist", "музык", "песн", "альбом", "исполнитель"]
        if any(kw in query.lower() for kw in music_keywords):
            hints.append("music_db")
        
        # User-related keywords
        user_keywords = ["user", "customer", "client", "пользователь", "клиент"]
        if any(kw in query.lower() for kw in user_keywords):
            hints.append("user_db")
        
        # Rating-related
        if "rating" in query.lower() or "рейтинг" in query.lower():
            hints.append("rating_db")
        
        return hints if hints else ["all"]
    
    def _calculate_confidence(self, understanding: QueryUnderstanding) -> float:
        """Вычислить confidence score анализа."""
        confidence = 1.0
        
        # Lower confidence if no tables detected
        if not understanding.tables:
            confidence -= 0.2
        
        # Lower confidence if unknown intent
        if understanding.intent == QueryIntent.UNKNOWN:
            confidence -= 0.1
        
        # Lower confidence if no entities for specific query
        if understanding.entities and not understanding.tables:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def get_routing_recommendation(
        self,
        understanding: QueryUnderstanding,
        available_dbs: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Получить рекомендацию для роутинга по БД.
        
        Args:
            understanding: Результат анализа запроса.
            available_dbs: Список доступных БД.
        
        Returns:
            Список рекомендованных БД.
        """
        recommended = []
        
        # Use db_hints from analysis
        for hint in understanding.db_hints:
            if hint == "all":
                return [db["db_name"] for db in available_dbs]
            
            for db in available_dbs:
                if hint.lower() in db["db_name"].lower():
                    if db["db_name"] not in recommended:
                        recommended.append(db["db_name"])
        
        # Fallback: use detected tables
        if not recommended:
            for table in understanding.tables:
                for db in available_dbs:
                    if table in db.get("tables", []):
                        if db["db_name"] not in recommended:
                            recommended.append(db["db_name"])
        
        return recommended if recommended else [db["db_name"] for db in available_dbs]
