# src\retrieval\graph_db.py
"""
Graph DB на основе Neo4j для хранения связей между таблицами.

Production features:
- Уникальные Column nodes (db_name + table_name + column_name)
- Индексы для производительности
- Optimized graph traversal (shortestPath)
- Join path discovery для SQL generation

Graph Schema:
    (Database {name})-[:HAS_TABLE]->(Table {db_name, name})
    (Table)-[:HAS_COLUMN]->(Column {db_name, table_name, name})
    (Table)-[:FOREIGN_KEY {from_column, to_column}]->(Table)
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import GraphDatabase, Session
from neo4j.exceptions import CypherSyntaxError, ServiceUnavailable

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ForeignKey:
    """Foreign key связь между таблицами."""
    from_db: str
    from_table: str
    from_column: str
    to_db: str
    to_table: str
    to_column: str

    def to_dict(self) -> Dict[str, str]:
        """Конвертировать в dict."""
        return {
            "from_db": self.from_db,
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_db": self.to_db,
            "to_table": self.to_table,
            "to_column": self.to_column,
        }


class Neo4jGraphDB:
    """
    Graph DB на основе Neo4j для хранения связей между таблицами.
    
    Production features:
    - Уникальные constraints для Table (db_name + name)
    - FIX: Column nodes уникальны по (db_name, table_name, name)
    - Индексы для Column.name для быстрого поиска
    - Optimized traversal через shortestPath
    """

    # FIX: Добавлен индекс для Column
    INDEXES = [
        "CREATE INDEX IF NOT EXISTS FOR (d:Database) ON (d.name)",
        "CREATE INDEX IF NOT EXISTS FOR (t:Table) ON (t.name)",
        "CREATE INDEX IF NOT EXISTS FOR (t:Table) ON (t.db_name)",
        "CREATE INDEX IF NOT EXISTS FOR (c:Column) ON (c.name)",  # FIX: индекс для Column
        "CREATE INDEX IF NOT EXISTS FOR (c:Column) ON (c.db_name)",
    ]

    # FIX: Constraint для Column включает db_name
    CONSTRAINTS = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Database) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Table) REQUIRE (t.db_name, t.name) IS UNIQUE",
        # FIX: Column уникален по комбинации db_name, table_name, name
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Column) REQUIRE (c.db_name, c.table_name, c.name) IS UNIQUE",
    ]

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.uri = uri or settings.get("neo4j_uri", "bolt://localhost:7687")
        self.username = username or settings.get("neo4j_username", "neo4j")
        self.password = password or settings.get("neo4j_password", "password")
        self.database = database or settings.get("neo4j_database", "neo4j")

        self._driver = None
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._cache_max_size = 500

        self._connect()
        self._init_schema()
        logger.info(f"Neo4jGraphDB initialized: {self.uri}")

    def _connect(self) -> None:
        """Установить соединение с Neo4j."""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=50,
            )
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j")
        except ServiceUnavailable as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise

    def _init_schema(self) -> None:
        """Создать индексы и constraints."""
        with self._driver.session(database=self.database) as session:
            for constraint in self.CONSTRAINTS:
                try:
                    session.run(constraint)
                except CypherSyntaxError as e:
                    logger.debug(f"Constraint already exists or syntax error: {e}")
            
            for index in self.INDEXES:
                try:
                    session.run(index)
                except CypherSyntaxError as e:
                    logger.debug(f"Index already exists or syntax error: {e}")
        
        logger.info("Neo4j schema initialized with constraints and indexes")

    def close(self) -> None:
        """Закрыть соединение с Neo4j."""
        if self._driver:
            self._driver.close()
        logger.info("Neo4j connection closed")

    def add_schema_batch(self, db_name: str, tables: List[Dict[str, Any]]) -> int:
        """
        Добавить схему базы данных батчем.
        
        FIX: Column nodes уникальны по (db_name, table_name, name)
        FIX: Foreign keys обрабатываются для каждой таблицы
        
        Args:
            db_name: Имя базы данных.
            tables: Список таблиц с колонками и FK.
        
        Returns:
            Количество созданных узлов.
        """
        if not tables:
            return 0

        with self._driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                nodes_count = 0

                # Создаём узел Database
                tx.run(
                    "MERGE (d:Database {name: $db_name}) SET d.updated_at = datetime()",
                    db_name=db_name,
                )
                nodes_count += 1

                # FIX: Обрабатываем каждую таблицу и её foreign_keys
                for table in tables:
                    column_types_json = json.dumps(table.get("column_types", {})) if table.get("column_types") else None

                    # Создаём узел Table (уникален по db_name + name)
                    tx.run(
                        """
                        MATCH (d:Database {name: $db_name})
                        MERGE (t:Table {db_name: $db_name, name: $table_name})
                        SET t.columns = $columns, 
                            t.column_types = $column_types,
                            t.primary_key = $primary_key, 
                            t.row_count = $row_count,
                            t.updated_at = datetime()
                        MERGE (d)-[:HAS_TABLE]->(t)
                        """,
                        db_name=db_name,
                        table_name=table["name"],
                        columns=table.get("columns", []),
                        column_types=column_types_json,
                        primary_key=table.get("primary_key"),
                        row_count=table.get("row_count"),
                    )
                    nodes_count += 1

                    # FIX: Создаём Column nodes с db_name для уникальности
                    for col_name, col_type in table.get("column_types", {}).items():
                        tx.run(
                            """
                            MATCH (t:Table {db_name: $db_name, name: $table_name})
                            MERGE (c:Column {
                                db_name: $db_name,
                                table_name: $table_name, 
                                name: $column_name
                            })
                            SET c.type = $column_type, c.updated_at = datetime()
                            MERGE (t)-[:HAS_COLUMN]->(c)
                            """,
                            db_name=db_name,
                            table_name=table["name"],
                            column_name=col_name,
                            column_type=col_type,
                        )
                        nodes_count += 1

                    # FIX: Создаём FOREIGN_KEY relationships для каждой таблицы
                    # foreign_keys берётся из текущей таблицы, а не последней
                    for fk in table.get("foreign_keys", []):
                        tx.run(
                            """
                            MATCH (from_t:Table {db_name: $db_name, name: $from_table})
                            MATCH (to_t:Table {db_name: $db_name, name: $to_table})
                            MERGE (from_t)-[r:FOREIGN_KEY {
                                from_column: $from_column, 
                                to_column: $to_column
                            }]->(to_t)
                            """,
                            db_name=db_name,
                            from_table=fk["from_table"],
                            to_table=fk["to_table"],
                            from_column=fk["from_column"],
                            to_column=fk["to_column"],
                        )

                tx.commit()
                logger.info(f"Batch added schema for {db_name}: {nodes_count} nodes")
                return nodes_count

    def find_join_path(
        self,
        tables: List[Tuple[str, str]],
        max_depth: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Найти путь JOIN между таблицами.
        
        OPTIMIZATION: использует shortestPath для производительности.
        
        Args:
            tables: Список (db_name, table_name).
            max_depth: Максимальная глубина обхода.
        
        Returns:
            Список путей с join conditions.
        """
        if len(tables) < 2:
            return []

        depth_range = min(max(1, max_depth), 5)

        with self._driver.session(database=self.database) as session:
            # OPTIMIZATION: используем shortestPath вместо произвольных путей
            query = f"""
                UNWIND $tables AS t
                MATCH (start:Table {{db_name: t[0], name: t[1]}})
                WITH start, $tables[1..] AS targets
                UNWIND range(0, size(targets)-1) AS i
                MATCH (end:Table {{db_name: targets[i][0], name: targets[i][1]}})
                WHERE start <> end
                MATCH path = shortestPath((start)-[r:FOREIGN_KEY*1..{depth_range}]-(end))
                RETURN
                    [node IN nodes(path) | {{db: node.db_name, table: node.name}}] AS tables,
                    [rel IN relationships(path) | {{
                        from: startNode(rel).name,
                        to: endNode(rel).name,
                        from_column: rel.from_column,
                        to_column: rel.to_column
                    }}] AS joins,
                    length(path) AS depth
                ORDER BY depth ASC
                LIMIT 10
                """

            result = session.run(query, tables=tables)
            paths = [
                {"tables": r["tables"], "joins": r["joins"], "depth": r["depth"]}
                for r in result
            ]

            logger.info(f"Found {len(paths)} join paths for {len(tables)} tables")
            return paths

    def find_related_tables(
        self,
        db_name: str,
        table_name: str,
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Найти связанные таблицы через foreign keys.
        
        OPTIMIZATION: ограничивает глубину и использует shortestPath.
        
        Args:
            db_name: Имя базы данных.
            table_name: Имя таблицы.
            max_depth: Максимальная глубина обхода.
        
        Returns:
            Список связанных таблиц с join conditions.
        """
        depth_range = min(max(1, max_depth), 3)  # OPTIMIZATION: снижено с 5 до 3

        with self._driver.session(database=self.database) as session:
            query = f"""
                MATCH (start:Table {{db_name: $db_name, name: $table_name}})
                MATCH path = (start)-[r:FOREIGN_KEY*1..{depth_range}]-(related:Table)
                WHERE related <> start
                RETURN DISTINCT
                    related.db_name AS db_name,
                    related.name AS table_name,
                    related.columns AS columns,
                    related.primary_key AS primary_key,
                    related.column_types AS column_types,
                    [rel IN relationships(path) | {{
                        from_column: rel.from_column, 
                        to_column: rel.to_column
                    }}] AS join_conditions,
                    length(path) AS depth
                ORDER BY depth ASC
                LIMIT 20
                """

            result = session.run(query, db_name=db_name, table_name=table_name)
            related = []
            for r in result:
                # Парсим column_types из JSON
                column_types = r["column_types"]
                if column_types and isinstance(column_types, str):
                    try:
                        column_types = json.loads(column_types)
                    except json.JSONDecodeError:
                        column_types = {}

                related.append({
                    "db_name": r["db_name"],
                    "table_name": r["table_name"],
                    "columns": r["columns"] or [],
                    "primary_key": r["primary_key"],
                    "column_types": column_types or {},
                    "join_conditions": r["join_conditions"] or [],
                    "depth": r["depth"],
                })

            logger.info(f"Found {len(related)} related tables for {db_name}.{table_name}")
            return related

    def get_table_schema(self, db_name: str, table_name: str) -> Optional[Dict[str, Any]]:
        """Получить схему таблицы."""
        with self._driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (t:Table {db_name: $db_name, name: $table_name})
                OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
                OPTIONAL MATCH (t)-[fk:FOREIGN_KEY]->(to_t:Table)
                RETURN
                    t.db_name AS db_name, 
                    t.name AS table_name,
                    t.columns AS columns, 
                    t.column_types AS column_types,
                    t.primary_key AS primary_key, 
                    t.row_count AS row_count,
                    collect(DISTINCT {{
                        column: c.name, 
                        type: c.type
                    }}) AS column_details,
                    collect(DISTINCT {{
                        from_column: fk.from_column, 
                        to_table: to_t.name, 
                        to_column: fk.to_column
                    }}) AS foreign_keys
                """,
                db_name=db_name,
                table_name=table_name,
            )

            record = result.single()
            if not record:
                return None

            column_types = record["column_types"]
            if column_types and isinstance(column_types, str):
                try:
                    column_types = json.loads(column_types)
                except json.JSONDecodeError:
                    column_types = {}

            return {
                "db_name": record["db_name"],
                "table_name": record["table_name"],
                "columns": record["columns"],
                "column_types": column_types,
                "primary_key": record["primary_key"],
                "row_count": record["row_count"],
                "column_details": record["column_details"],
                "foreign_keys": [
                    fk for fk in record["foreign_keys"] 
                    if fk["from_column"] is not None
                ],
            }

    def get_all_tables(self, db_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Получить все таблицы."""
        with self._driver.session(database=self.database) as session:
            query = "MATCH (t:Table)"
            params: Dict[str, Any] = {}

            if db_filter:
                query += " WHERE t.db_name IN $db_filter"
                params["db_filter"] = db_filter

            query += """
            RETURN t.db_name AS db_name, 
                   t.name AS table_name,
                   t.columns AS columns, 
                   t.column_types AS column_types,
                   t.primary_key AS primary_key, 
                   t.row_count AS row_count
            ORDER BY t.db_name, t.name
            """

            result = session.run(query, **params)
            tables = []
            for record in result:
                column_types = record["column_types"]
                if column_types and isinstance(column_types, str):
                    try:
                        column_types = json.loads(column_types)
                    except json.JSONDecodeError:
                        column_types = {}

                tables.append({
                    "db_name": record["db_name"],
                    "table_name": record["table_name"],
                    "columns": record["columns"],
                    "column_types": column_types,
                    "primary_key": record["primary_key"],
                    "row_count": record["row_count"],
                })

            logger.info(f"Retrieved {len(tables)} tables from Graph DB")
            return tables

    def delete_all(self) -> int:
        """Удалить все узлы и связи из графа."""
        with self._driver.session(database=self.database) as session:
            result = session.run("MATCH (n) DETACH DELETE n RETURN count(*) AS deleted")
            record = result.single()
            deleted = record["deleted"] if record else 0
            logger.info(f"Deleted all nodes from Graph DB: {deleted} nodes")
            return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику графа."""
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run("MATCH ()-[r]->() RETURN count(r) AS relationships")
                record = result.single()
                relationships = record["relationships"] if record and record["relationships"] is not None else 0

                result = session.run("MATCH (n) RETURN count(n) AS nodes")
                record = result.single()
                nodes = record["nodes"] if record and record["nodes"] is not None else 0

                return {
                    "nodes": nodes,
                    "relationships": relationships,
                    "tables": nodes - 1 if nodes > 0 else 0,
                    "foreign_keys": relationships,
                }
        except Exception as e:
            logger.debug(f"Failed to get graph stats: {e}")
            return {"nodes": 0, "relationships": 0, "tables": 0, "foreign_keys": 0}

    def get_join_hints(self, table_names: List[str]) -> List[Dict[str, Any]]:
        """
        Получить подсказки для JOIN между таблицами.
        
        Production feature: предварительно вычисленные join paths.
        
        Args:
            table_names: Список имён таблиц.
        
        Returns:
            Список join hints.
        """
        hints = []
        
        with self._driver.session(database=self.database) as session:
            for i, table1 in enumerate(table_names):
                for table2 in table_names[i+1:]:
                    result = session.run(
                        """
                        MATCH (t1:Table {name: $table1})
                        MATCH (t2:Table {name: $table2})
                        WHERE t1 <> t2
                        MATCH path = shortestPath((t1)-[:FOREIGN_KEY*1..3]-(t2))
                        RETURN
                            [node IN nodes(path) | node.name] AS tables,
                            [rel IN relationships(path) | {
                                from: startNode(rel).name,
                                to: endNode(rel).name,
                                from_column: rel.from_column,
                                to_column: rel.to_column
                            }] AS joins
                        """,
                        table1=table1,
                        table2=table2,
                    )
                    
                    record = result.single()
                    if record:
                        hints.append({
                            "tables": record["tables"],
                            "joins": record["joins"],
                        })
        
        return hints

    def _save_to_cache(self, key: str, value: List[Dict[str, Any]]) -> None:
        """Сохранить в кэш."""
        if len(self._cache) >= self._cache_max_size:
            keys_to_delete = list(self._cache.keys())[: self._cache_max_size // 2]
            for k in keys_to_delete:
                del self._cache[k]
        self._cache[key] = value

    def _invalidate_cache(self) -> None:
        """Очистить кэш."""
        self._cache.clear()
        logger.debug("Graph DB cache invalidated")
