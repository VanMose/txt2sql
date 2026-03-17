# src\agents\router_agent.py
"""
Router Agent для выбора релевантных баз данных с оптимизациями.

Оптимизации:
1. Кэширование результатов (hash-based + semantic)
2. Prefetch схем (параллельная загрузка)
3. Parallel execution (асинхронные операции)
4. Early exit для простых запросов

Использует комбинацию:
1. Vector DB (Qdrant) для семантического поиска
2. Graph DB (Neo4j) для анализа связей
3. LLM для финального ранжирования

Example:
    >>> from agents.router_agent import RouterAgent
    >>> router = RouterAgent(vector_db, graph_db)
    >>> selections = router.route("Show movies and albums", top_k_dbs=3)
"""
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..llm.inference import LLMService
from ..llm.llm_cache import SemanticCache
from ..llm.prompts import Prompts
from ..retrieval.graph_db import Neo4jGraphDB
from ..retrieval.vector_db import QdrantVectorDB, TableDocument
from ..utils.json_parser import parse_json

logger = logging.getLogger(__name__)


@dataclass
class DatabaseSelection:
    """
    Результат выбора базы данных.

    Attributes:
        db_name: Имя базы данных.
        db_path: Путь к базе данных.
        relevance_score: Оценка релевантности.
        reason: Причина выбора.
        tables: Список таблиц.
        confidence: Уровень уверенности.
        schema_prefetched: Флаг prefetch схемы.
    """
    db_name: str
    db_path: str
    relevance_score: float
    reason: str
    tables: List[str]
    confidence: float
    schema_prefetched: bool = False


class RouterAgent:
    """
    Агент маршрутизации запросов к релевантным базам данных.

    Оптимизации:
    - Hash-based кэширование результатов
    - Semantic cache для похожих запросов
    - Prefetch схем (параллельная загрузка)
    - Parallel execution

    Attributes:
        vector_db: Vector DB для поиска.
        graph_db: Graph DB для анализа связей.
        llm: LLM сервис для ранжирования.
        _cache: Кэш для результатов маршрутизации.
        _semantic_cache: Semantic cache для запросов.
        _schema_cache: Кэш загруженных схем.
    """

    def __init__(
        self,
        vector_db: QdrantVectorDB,
        graph_db: Neo4jGraphDB,
        llm: Optional[LLMService] = None,
        use_parallel: bool = True,
        use_semantic_cache: bool = True,
    ) -> None:
        """
        Инициализировать Router Agent.

        Args:
            vector_db: Vector DB для поиска.
            graph_db: Graph DB для анализа связей.
            llm: LLM сервис для ранжирования.
            use_parallel: Использовать параллельное выполнение.
            use_semantic_cache: Использовать semantic cache.
        """
        self.vector_db = vector_db
        self.graph_db = graph_db
        self.llm = llm or LLMService()

        self._cache: Dict[str, List[DatabaseSelection]] = {}
        self._cache_max_size = 200
        
        # Semantic cache
        self._use_semantic_cache = use_semantic_cache
        self._semantic_cache = SemanticCache() if use_semantic_cache else None
        
        # Schema cache для prefetch
        self._schema_cache: Dict[str, str] = {}
        
        # Parallel execution
        self._use_parallel = use_parallel
        self._executor = ThreadPoolExecutor(max_workers=4) if use_parallel else None

        logger.info(
            f"RouterAgent initialized: parallel={use_parallel}, "
            f"semantic_cache={use_semantic_cache}"
        )

    def _generate_cache_key(self, query: str, top_k_dbs: int, top_k_tables: int) -> str:
        """Сгенерировать ключ кэша."""
        key = f"{query}:{top_k_dbs}:{top_k_tables}"
        return hashlib.md5(key.encode()).hexdigest()

    def _is_simple_query(self, query: str) -> bool:
        """
        Проверить, является ли запрос простым.

        Простые запросы могут использовать только heuristic ranking без LLM.
        """
        simple_patterns = [
            "show all", "показать все", "покажи все", "select all",
            "count", "количество",
            "list all", "список всех",
            # 🔥 Добавлено: запросы с ключевыми словами типа данных
            "фильм", "кино", "movies", "film",
            "музык", "песн", "artist", "song", "album",
            "рейтинг", "rating",
        ]
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in simple_patterns)

    def route(
        self,
        query: str,
        top_k_dbs: int = 2,
        top_k_tables: int = 5,
        use_llm_ranking: bool = True,
    ) -> List[DatabaseSelection]:
        """
        Маршрутизировать запрос к релевантным базам данных.

        Args:
            query: Запрос пользователя.
            top_k_dbs: Максимальное количество выбираемых БД.
            top_k_tables: Количество таблиц для поиска в Vector DB.
            use_llm_ranking: Использовать LLM для финального ранжирования.

        Returns:
            Список выбранных баз данных.
        """
        start_time = time.time()
        cache_key = self._generate_cache_key(query, top_k_dbs, top_k_tables)

        # Проверка hash-based кэша
        if cache_key in self._cache:
            self._semantic_cache.set(query, str([s.db_name for s in self._cache[cache_key]]))
            logger.debug(f"Router cache HIT: {query[:50]}...")
            return self._cache[cache_key]

        # 🔥 ОТКЛЮЧЕНО ДЛЯ ОТЛАДКИ: Проверка semantic cache
        # if self._semantic_cache:
        #     cached_result, similarity = self._semantic_cache.get_similar(query)
        #     if cached_result is not None:
        #         logger.info(f"Semantic cache HIT: similarity={similarity:.4f}, result={cached_result}")
        #         # Возвращаем из обычного кэша по ключу результата
        #         for key, selections in self._cache.items():
        #             if str([s.db_name for s in selections]) == cached_result:
        #                 logger.info(f"Router cache HIT for semantic match: {[s.db_name for s in selections]}")
        #                 return selections
        #         # Если не нашли в кэше, продолжаем без кэша
        #         logger.warning(f"Semantic cache returned {cached_result} but not found in router cache")

        logger.info(f"🔍 Routing query: '{query[:100]}...' (use_llm_ranking={use_llm_ranking}, is_simple={self._is_simple_query(query)})")

        # Шаг 1: Vector search
        vector_results = self._vector_search(query, top_k_tables)
        logger.info(f"Vector search found {len(vector_results)} tables in {(time.time() - start_time) * 1000:.0f}ms")
        
        # 🔥 Детальное логирование vector search
        logger.info(f"📊 Vector results:")
        for doc, score in vector_results:
            logger.info(f"   - {doc.db_name}.{doc.table_name} (score={score:.4f})")

        if not vector_results:
            logger.warning("No tables found in vector search")
            return []

        # Шаг 2: Group by database
        db_groups = self._group_by_database(vector_results)
        logger.info(f"Grouped into {len(db_groups)} databases")

        # Шаг 3: Graph analysis + Prefetch схем (параллельно)
        if self._use_parallel:
            graph_enriched, prefetched_schemas = self._parallel_enrich(db_groups)
        else:
            graph_enriched = self._enrich_with_graph(db_groups)
            prefetched_schemas = {}

        # Шаг 4: LLM ranking или heuristic
        use_heuristic = not use_llm_ranking or self._is_simple_query(query)
        if use_heuristic:
            logger.debug("Using heuristic ranking (simple query or disabled)")
            selections = self._heuristic_ranking(query, graph_enriched, top_k_dbs)
        else:
            selections = self._llm_ranking(query, graph_enriched, top_k_dbs)

        # Добавляем флаг prefetch к результатам
        for selection in selections:
            if selection.db_name in prefetched_schemas:
                selection.schema_prefetched = True

        # Шаг 5: Кэширование
        self._save_to_cache(cache_key, selections)
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Selected {len(selections)} databases in {elapsed_ms:.0f}ms: "
            f"{[s.db_name for s in selections]}"
        )

        return selections

    def _vector_search(self, query: str, top_k: int) -> List[Tuple[TableDocument, float]]:
        """Поиск в Vector DB."""
        return self.vector_db.search_with_reranking(query, top_k=top_k)

    def _group_by_database(
        self,
        results: List[Tuple[TableDocument, float]],
    ) -> Dict[str, Dict[str, Any]]:
        """Сгруппировать результаты по базам данных."""
        db_groups: Dict[str, Dict[str, Any]] = {}

        for doc, score in results:
            db_name = doc.db_name
            if db_name not in db_groups:
                db_groups[db_name] = {
                    "db_name": db_name,
                    "db_path": doc.db_path,
                    "tables": [],
                    "scores": [],
                    "avg_score": 0.0,
                    "max_score": 0.0,
                    "row_counts": [],  # 🔥 Для подсчёта строк
                }

            db_groups[db_name]["tables"].append(doc.table_name)
            db_groups[db_name]["scores"].append(score)
            # Добавляем row_count если есть
            if doc.row_count:
                db_groups[db_name]["row_counts"].append(doc.row_count)

        for group in db_groups.values():
            scores = group["scores"]
            group["avg_score"] = sum(scores) / len(scores)
            group["max_score"] = max(scores)
            group["table_count"] = len(group["tables"])
            # 🔥 Подсчитываем общее количество строк
            group["total_row_count"] = sum(group.get("row_counts", []))

        return db_groups

    def _parallel_enrich(
        self,
        db_groups: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Параллельное обогащение данных и prefetch схем.

        Args:
            db_groups: Группы баз данных.

        Returns:
            Кортеж (enriched_data, prefetched_schemas).
        """
        enriched = []
        prefetched_schemas = {}

        def process_db(db_name: str, group: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
            """Обработать одну БД."""
            # Graph analysis
            related_tables = []
            join_paths = []

            for table_name in group["tables"]:
                try:
                    related = self.graph_db.find_related_tables(
                        db_name=db_name,
                        table_name=table_name,
                        max_depth=2
                    )
                    for rel in related:
                        if rel["table_name"] not in group["tables"]:
                            related_tables.append({
                                "table_name": rel["table_name"],
                                "depth": rel["depth"],
                                "join_conditions": rel["join_conditions"],
                            })

                    if len(group["tables"]) > 1:
                        paths = self.graph_db.find_join_path(
                            [(db_name, t) for t in group["tables"]],
                            max_depth=3
                        )
                        join_paths.extend(paths)

                except Exception as e:
                    logger.warning(f"Graph analysis failed for {db_name}.{table_name}: {e}")

            # Prefetch схемы
            schema = None
            try:
                schema = self._prefetch_schema(db_name, group["tables"])
            except Exception as e:
                logger.debug(f"Schema prefetch failed for {db_name}: {e}")

            enriched_data = {
                **group,
                "related_tables": related_tables,
                "join_paths": join_paths,
                "has_relations": len(related_tables) > 0,
                "join_complexity": len(join_paths),
            }

            return enriched_data, schema

        # Параллельное выполнение
        futures = {}
        for db_name, group in db_groups.items():
            future = self._executor.submit(process_db, db_name, group)
            futures[future] = db_name

        for future in as_completed(futures):
            db_name = futures[future]
            try:
                enriched_data, schema = future.result(timeout=5.0)
                enriched.append(enriched_data)
                if schema:
                    prefetched_schemas[db_name] = schema
            except Exception as e:
                logger.warning(f"Parallel processing failed for {db_name}: {e}")
                # Fallback: добавляем без обогащения
                enriched.append(db_groups[db_name])

        return enriched, prefetched_schemas

    def _prefetch_schema(self, db_name: str, tables: List[str]) -> str:
        """
        Prefetch схемы для БД.

        Args:
            db_name: Имя БД.
            tables: Список таблиц.

        Returns:
            Схема БД.
        """
        cache_key = f"{db_name}:{':'.join(sorted(tables))}"
        
        if cache_key in self._schema_cache:
            logger.debug(f"Schema cache HIT: {db_name}")
            return self._schema_cache[cache_key]

        # Загрузка схемы
        schema_parts = []
        try:
            db_tables = self.graph_db.get_all_tables(db_filter=[db_name])
            for table in db_tables:
                if table["table_name"] in tables or not tables:
                    schema_parts.append(
                        f"Table: {table['table_name']}\n"
                        f"Columns: {', '.join(table['columns'])}"
                    )
        except Exception as e:
            logger.warning(f"Schema load failed for {db_name}: {e}")
            return ""

        schema = "\n\n".join(schema_parts)
        self._schema_cache[cache_key] = schema
        return schema

    def _enrich_with_graph(self, db_groups: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Обогатить данные из Graph DB (синхронная версия)."""
        enriched: List[Dict[str, Any]] = []

        for db_name, group in db_groups.items():
            related_tables: List[Dict[str, Any]] = []
            join_paths: List[Any] = []

            for table_name in group["tables"]:
                try:
                    related = self.graph_db.find_related_tables(
                        db_name=db_name,
                        table_name=table_name,
                        max_depth=2
                    )
                    for rel in related:
                        if rel["table_name"] not in group["tables"]:
                            related_tables.append({
                                "table_name": rel["table_name"],
                                "depth": rel["depth"],
                                "join_conditions": rel["join_conditions"],
                            })

                    if len(group["tables"]) > 1:
                        paths = self.graph_db.find_join_path(
                            [(db_name, t) for t in group["tables"]],
                            max_depth=3
                        )
                        join_paths.extend(paths)

                except Exception as e:
                    logger.warning(f"Graph analysis failed for {db_name}.{table_name}: {e}")

            enriched.append({
                **group,
                "related_tables": related_tables,
                "join_paths": join_paths,
                "has_relations": len(related_tables) > 0,
                "join_complexity": len(join_paths),
            })

        return enriched

    def _llm_ranking(
        self,
        query: str,
        db_groups: List[Dict[str, Any]],
        top_k: int,
    ) -> List[DatabaseSelection]:
        """Ранжирование с помощью LLM."""
        prompt = Prompts.format_router(query=query, databases=db_groups)
        output = self.llm.generate(prompt, n=1, temperature=0.1)[0]

        logger.info(f"Router LLM output: {output[:300]}...")

        try:
            # Попытка извлечь JSON из вывода (модель может добавлять markdown)
            json_str = output.strip()
            
            # Удаляем markdown code blocks если есть
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            
            json_str = json_str.strip()
            
            # Попытка найти JSON в тексте
            if not json_str.startswith("{"):
                # Ищем первое { и последнее }
                start_idx = json_str.find("{")
                end_idx = json_str.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    json_str = json_str[start_idx:end_idx + 1]
            
            obj: Dict[str, Any] = parse_json(json_str)
            ranked_dbs = obj.get("ranked_databases", [])
            
            if not ranked_dbs:
                logger.warning("LLM returned empty ranked_databases")
                return self._heuristic_ranking(query, db_groups, top_k)
            
            logger.info(f"Successfully parsed {len(ranked_dbs)} databases from LLM: {[db.get('db_name') for db in ranked_dbs]}")
        except Exception as e:
            logger.warning(f"Failed to parse LLM ranking: {e}")
            logger.debug(f"Raw output: {output}")
            return self._heuristic_ranking(query, db_groups, top_k)

        selections: List[DatabaseSelection] = []
        for db in ranked_dbs[:top_k]:
            db_name = db.get("db_name", "")
            db_path = next((g["db_path"] for g in db_groups if g["db_name"] == db_name), "")

            selections.append(DatabaseSelection(
                db_name=db_name,
                db_path=db_path,
                relevance_score=db.get("relevance_score", 0.0),
                reason=db.get("reason", ""),
                tables=db.get("tables", []),
                confidence=db.get("confidence", 0.0),
            ))

        return selections

    def _heuristic_ranking(
        self,
        query: str,
        db_groups: List[Dict[str, Any]],
        top_k: int,
    ) -> List[DatabaseSelection]:
        """
        Эвристическое ранжирование с keyword matching.

        Для простых запросов типа "покажи все фильмы" выбирает БД по названию таблиц.
        """
        query_lower = query.lower()
        
        # Ключевые слова для определения типа данных (расширенный список)
        keyword_db_map = {
            # Фильмы
            'фильм': ['movie', 'film', 'rating', 'reviewer', 'director', 'actor'],
            'кино': ['movie', 'film'],
            'movies': ['movie', 'film'],
            'film': ['movie', 'film'],
            'режисс': ['director', 'movie'],
            'актер': ['actor', 'movie'],
            'rating': ['rating', 'movie', 'reviewer'],
            'рейтинг': ['rating', 'movie'],
            # Музыка
            'музык': ['music', 'song', 'artist', 'album', 'track'],
            'песн': ['song', 'track', 'music'],
            'artist': ['artist', 'music'],
            'song': ['song', 'music'],
            'album': ['album', 'music'],
            'альбом': ['album', 'music'],
            'трек': ['track', 'music'],
            # Общие
            'все': None,  # Любая БД
            'show all': None,
            'показать все': None,
            'покажи все': None,
        }
        
        # Определяем целевые таблицы по ключевым словам
        target_tables = set()
        for keyword, tables in keyword_db_map.items():
            if keyword in query_lower:
                if tables:
                    target_tables.update(tables)
                else:
                    # Общий запрос - используем первую БД
                    pass
                break
        
        logger.info(f"Heuristic ranking: query='{query_lower[:50]}', target_tables={target_tables}")

        scored: List[Tuple[Dict[str, Any], float, str]] = []

        for group in db_groups:
            score = group["avg_score"]

            # Бонус за количество таблиц
            score += min(group["table_count"] * 0.05, 0.2)

            # Бонус за связи
            if group.get("has_relations"):
                score += 0.1
            score += min(group.get("join_complexity", 0) * 0.03, 0.15)
            score += group["max_score"] * 0.1

            # 🔥 Бонус за количество строк в таблицах (данные есть!)
            total_rows = group.get("total_row_count", 0)
            if total_rows > 0:
                score += min(total_rows / 100, 0.5)  # +0.5 max за 50+ строк
                logger.debug(f"DB {group['db_name']}: {total_rows} rows → +{min(total_rows / 100, 0.5):.2f} score")

            # 🔥 Критическое исправление: бонус за совпадение таблиц с ключевыми словами
            if target_tables:
                for table_name in group["tables"]:
                    table_lower = table_name.lower()
                    for target in target_tables:
                        # Exact match (case-insensitive) - высший бонус
                        if target == table_lower:
                            score += 2.0  # 🔥 Exact match!
                            logger.info(f"🎯 EXACT match: '{target}' == '{table_name}' (db={group['db_name']})")
                            break
                        # Partial match - стандартный бонус
                        elif target in table_lower or table_lower in target:
                            score += 0.5  # 🔥 Partial match
                            logger.info(f"🎯 Partial match: '{target}' in '{table_name}' (db={group['db_name']})")
                            break

            reasons = []
            if group["table_count"] > 1:
                reasons.append(f"{group['table_count']} tables matched")
            if group.get("has_relations"):
                reasons.append("has table relations")
            if group.get("join_complexity", 0) > 0:
                reasons.append(f"{group['join_complexity']} join paths found")
            if target_tables and score > 1.0:
                reasons.append("keyword match")
            if total_rows > 0:
                reasons.append(f"{total_rows} rows in tables")

            scored.append((group, score, "; ".join(reasons) or "semantic match"))

        # Сортировка по убыванию score
        scored.sort(key=lambda x: x[1], reverse=True)

        selections: List[DatabaseSelection] = []
        for group, score, reason in scored[:top_k]:
            selections.append(DatabaseSelection(
                db_name=group["db_name"],
                db_path=group["db_path"],
                relevance_score=score,
                reason=reason,
                tables=group["tables"],
                confidence=min(score, 0.95),
            ))

        logger.info(f"Heuristic ranking: selected {[s.db_name for s in selections]} with scores {[s.relevance_score for s in selections]}")
        return selections

    def get_schema_for_selection(self, selections: List[DatabaseSelection]) -> str:
        """Получить схему для выбранных БД."""
        schema_parts = []

        for selection in selections:
            # Проверяем prefetch кэш
            cache_key = f"{selection.db_name}:{':'.join(sorted(selection.tables))}"
            if cache_key in self._schema_cache:
                schema_parts.append(
                    f"-- Database: {selection.db_name} ({selection.db_path})\n\n"
                    f"{self._schema_cache[cache_key]}"
                )
                continue

            # Загружаем схему
            tables = self.graph_db.get_all_tables(db_filter=[selection.db_name])

            if tables:
                schema_text = f"-- Database: {selection.db_name} ({selection.db_path})\n\n"
                for table in tables:
                    if table["table_name"] in selection.tables or not selection.tables:
                        schema_text += f"Table: {table['table_name']}\n"
                        schema_text += f"Columns: {', '.join(table['columns'])}\n"

                        if table.get("foreign_keys"):
                            fk_text = "; ".join(
                                f"{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
                                for fk in table["foreign_keys"]
                            )
                            schema_text += f"Foreign Keys: {fk_text}\n"
                        schema_text += "\n"

                schema_parts.append(schema_text)

        return "\n".join(schema_parts)

    def _save_to_cache(self, key: str, value: List[DatabaseSelection]) -> None:
        """Сохранить в кэш."""
        if len(self._cache) >= self._cache_max_size:
            keys_to_delete = list(self._cache.keys())[: self._cache_max_size // 2]
            for k in keys_to_delete:
                del self._cache[k]
        self._cache[key] = value

    def invalidate_cache(self) -> None:
        """Очистить кэш."""
        self._cache.clear()
        if self._semantic_cache:
            self._semantic_cache.clear()
        self._schema_cache.clear()
        logger.debug("RouterAgent cache invalidated")

    def close(self) -> None:
        """Закрыть executor."""
        if self._executor:
            self._executor.shutdown(wait=False)
            logger.debug("RouterAgent executor shut down")
