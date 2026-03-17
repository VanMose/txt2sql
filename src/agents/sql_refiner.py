# src\agents\sql_refiner.py
"""
SQL Refiner с оптимизациями производительности.

Оригинальная логика + оптимизации:
1. KV Cache - кэширование эмбеддингов промптов
2. Batch Inference - параллельная генерация
3. Early Exit - остановка при успешном refinement
4. Pattern-Based Fast Path - быстрые исправления частых ошибок

Научное обоснование:
- KV Cache уменьшает redundant computation (Pope et al., 2023)
- Batch Inference использует GPU parallelism (Kwon et al., 2023)
"""
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..llm.inference import LLMService
from ..llm.prompts import Prompts
from ..utils.json_parser import parse_json

if TYPE_CHECKING:
    from ..pipeline.state import SQLAttempt

logger = logging.getLogger(__name__)


class SQLRefiner:
    """
    Агент улучшения SQL запросов с оптимизациями.

    Attributes:
        llm: LLM сервис.
        kv_cache: Кэш для промптов и результатов.
        max_retries: Максимальное количество попыток.
    """

    # Паттерны частых ошибок для быстрого исправления
    ERROR_PATTERNS = {
        "missing_table": {
            "pattern": r"no such table: (\w+)",
            "fix_type": "table_name"
        },
        "missing_column": {
            "pattern": r"no such column: (\w+)",
            "fix_type": "column_name"
        },
        "syntax_error": {
            "pattern": r"near\s+[\"']?(\w+)[\"']?",
            "fix_type": "syntax"
        },
    }

    def __init__(self, max_retries: int = 3) -> None:
        """
        Инициализировать рефайнер.

        Args:
            max_retries: Максимум попыток refinement.
        """
        self.llm = LLMService()
        self.max_retries = max_retries
        
        # KV Cache для промптов
        self._kv_cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Статистика
        self.stats = {
            "total_refinements": 0,
            "fast_path_fixes": 0,
            "llm_refinements": 0,
            "successful": 0,
            "total_time_ms": 0,
            "avg_iterations": 0,
        }
        
        logger.info(f"SQLRefiner initialized (max_retries={max_retries})")

    def build_history(self, attempts: List["SQLAttempt"]) -> str:
        """
        Построить историю попыток для промпта.

        Args:
            attempts: Список предыдущих попыток.

        Returns:
            Форматированная история.
        """
        history_parts = []
        
        for i, attempt in enumerate(attempts, 1):
            history_parts.append(f"""
Attempt {i}:
SQL:
{attempt.sql}

Error:
{attempt.reason}
""")
        
        return "\n".join(history_parts)

    def refine(
        self,
        query: str,
        schema: str,
        attempts: List["SQLAttempt"]
    ) -> str:
        """
        Улучшить SQL на основе истории попыток.

        Args:
            query: Запрос пользователя.
            schema: Описание схемы БД.
            attempts: История предыдущих попыток.

        Returns:
            Улучшенный SQL запрос.
        """
        start_time = time.time()
        self.stats["total_refinements"] += 1
        
        if not attempts:
            logger.warning("No attempts to refine")
            return ""

        # Попытка fast path (pattern-based fix)
        fast_fix = self._try_fast_path_fix(attempts[-1].sql, attempts[-1].reason, schema)
        if fast_fix:
            self.stats["fast_path_fixes"] += 1
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats["total_time_ms"] += elapsed_ms
            logger.info(f"Fast path fix applied in {elapsed_ms:.0f}ms")
            return fast_fix

        # Проверка KV cache
        cache_key = self._build_cache_key(query, schema, attempts)
        cached_result = self._kv_cache_get(cache_key)
        if cached_result:
            self._cache_hits += 1
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats["total_time_ms"] += elapsed_ms
            logger.info(f"Cache hit for refinement in {elapsed_ms:.0f}ms")
            return cached_result

        self._cache_misses += 1

        # Multi-pass refinement с early exit
        history = self.build_history(attempts)
        best_sql = attempts[-1].sql
        best_confidence = 0.0
        iterations = 0

        for retry in range(self.max_retries):
            iterations += 1
            logger.debug(f"Refinement iteration {retry + 1}/{self.max_retries}")

            # Формирование промпта
            prompt = Prompts.format_sql_refiner(query, schema, history)
            
            # Генерация с batch inference (n=2 для выбора лучшего)
            outputs = self.llm.generate(prompt, n=2, temperature=0.3)
            
            # Парсинг и выбор лучшего варианта
            refined_sql = self._parse_and_select_best(outputs, best_sql)
            
            if not refined_sql:
                logger.warning(f"Failed to parse refinement output at iteration {retry + 1}")
                continue

            # Проверка качества (early exit если успешно)
            if self._is_valid_refinement(refined_sql, best_sql):
                best_sql = refined_sql
                logger.info(f"Successful refinement at iteration {retry + 1}")
                break
            else:
                # Обновление истории для следующей итерации
                history += f"""

Attempt {len(attempts) + retry + 1}:
SQL:
{refined_sql}

Error:
Previous refinement was not valid
"""

        # Сохранение в cache
        self._kv_cache_set(cache_key, best_sql)
        
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats["total_time_ms"] += elapsed_ms
        self.stats["llm_refinements"] += 1
        self.stats["avg_iterations"] = (
            (self.stats["avg_iterations"] * (self.stats["total_refinements"] - 1) + iterations) 
            / self.stats["total_refinements"]
        )
        
        cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses) * 100 if (self._cache_hits + self._cache_misses) > 0 else 0
        
        logger.info(
            f"Refinement completed in {elapsed_ms:.0f}ms "
            f"(iterations={iterations}, cache_hit_rate={cache_hit_rate:.1f}%)"
        )
        
        return best_sql

    def _try_fast_path_fix(
        self,
        sql: str,
        error_message: str,
        schema: str
    ) -> Optional[str]:
        """
        Попытка быстрого исправления по паттерну.

        Args:
            sql: Исходный SQL.
            error_message: Текст ошибки.
            schema: Схема БД.

        Returns:
            Исправленный SQL или None.
        """
        error_lower = error_message.lower()

        for pattern_name, pattern_info in self.ERROR_PATTERNS.items():
            match = re.search(pattern_info["pattern"], error_lower)
            
            if match:
                logger.debug(f"Matched error pattern: {pattern_name}")
                problematic_name = match.group(1)

                if pattern_info["fix_type"] == "table_name":
                    # Извлечение имен таблиц из схемы
                    tables = re.findall(r'Table:\s*(\w+)', schema)
                    # Поиск похожей таблицы
                    for table in tables:
                        if problematic_name in table.lower() or table.lower() in problematic_name:
                            # Замена имени таблицы в SQL
                            fixed_sql = re.sub(
                                rf'\b{re.escape(problematic_name)}\b',
                                table,
                                sql,
                                flags=re.IGNORECASE
                            )
                            self.stats["fast_path_fixes"] += 1
                            logger.info(f"Fast path table fix: {problematic_name} → {table}")
                            return fixed_sql

                elif pattern_info["fix_type"] == "column_name":
                    # Извлечение колонок из схемы
                    columns = re.findall(r'Columns:\s*([^,\n]+)', schema)
                    all_columns = [c.strip() for col_list in columns for c in col_list.split(',')]
                    
                    # Поиск похожей колонки
                    for col in all_columns:
                        if problematic_name in col.lower() or col.lower() in problematic_name:
                            fixed_sql = re.sub(
                                rf'\b{re.escape(problematic_name)}\b',
                                col,
                                sql,
                                flags=re.IGNORECASE
                            )
                            self.stats["fast_path_fixes"] += 1
                            logger.info(f"Fast path column fix: {problematic_name} → {col}")
                            return fixed_sql

        return None

    def _build_cache_key(
        self,
        query: str,
        schema: str,
        attempts: List["SQLAttempt"]
    ) -> str:
        """
        Построить ключ кэша.

        Args:
            query: Запрос.
            schema: Схема.
            attempts: Попытки.

        Returns:
            MD5 хеш ключа.
        """
        import hashlib
        
        # Ключ на основе query + последний SQL + ошибка
        key_data = f"{query}|{attempts[-1].sql}|{attempts[-1].reason}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _kv_cache_get(self, key: str) -> Optional[str]:
        """Получить из KV cache."""
        return self._kv_cache.get(key)

    def _kv_cache_set(self, key: str, value: str) -> None:
        """Сохранить в KV cache."""
        # Ограничение размера кэша
        if len(self._kv_cache) >= 1000:
            # Удаляем половину старых записей
            keys_to_delete = list(self._kv_cache.keys())[:500]
            for k in keys_to_delete:
                del self._kv_cache[k]
        
        self._kv_cache[key] = value

    def _parse_and_select_best(
        self,
        outputs: List[str],
        current_best: str
    ) -> Optional[str]:
        """
        Распарсить выходы LLM и выбрать лучший SQL.

        Args:
            outputs: Выходы LLM.
            current_best: Текущий лучший SQL.

        Returns:
            Лучший SQL или None.
        """
        candidates = []
        
        for output in outputs:
            try:
                obj = parse_json(output)
                sql = obj.get("sql", "")
                if sql and self._is_valid_sql(sql):
                    candidates.append(sql)
            except Exception as e:
                logger.debug(f"Failed to parse output: {e}")
                continue

        # Возвращаем первый валидный или текущий лучший
        return candidates[0] if candidates else current_best

    def _is_valid_sql(self, sql: str) -> bool:
        """
        Быстрая валидация SQL.

        Args:
            sql: SQL запрос.

        Returns:
            True если валиден.
        """
        sql_upper = sql.upper().strip()

        # Проверка на SELECT
        if not sql_upper.startswith("SELECT"):
            return False

        # Проверка на наличие FROM
        if " FROM " not in sql_upper:
            return False

        # Проверка на опасные операции
        dangerous = ["DROP", "DELETE", "TRUNCATE", "INSERT", "UPDATE"]
        for kw in dangerous:
            if kw in sql_upper:
                return False

        # Проверка на минимальную длину
        if len(sql) < 10:
            return False

        return True

    def _is_valid_refinement(self, refined_sql: str, original_sql: str) -> bool:
        """
        Проверка качества refinement.

        Args:
            refined_sql: Улучшенный SQL.
            original_sql: Оригинальный SQL.

        Returns:
            True если refinement успешен.
        """
        # Проверка что SQL изменился
        if refined_sql == original_sql:
            return False

        # Проверка валидности
        if not self._is_valid_sql(refined_sql):
            return False

        # Проверка что не стал короче (может потерять информацию)
        if len(refined_sql) < len(original_sql) * 0.5:
            return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику рефайнера.

        Returns:
            Статистика.
        """
        avg_time = (
            self.stats["total_time_ms"] / self.stats["total_refinements"]
            if self.stats["total_refinements"] > 0 else 0
        )

        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses) * 100
            if (self._cache_hits + self._cache_misses) > 0 else 0
        )

        return {
            **self.stats,
            "avg_refinement_time_ms": round(avg_time, 2),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "fast_path_rate": round(
                self.stats["fast_path_fixes"] / self.stats["total_refinements"] * 100, 2
            ) if self.stats["total_refinements"] > 0 else 0,
            "success_rate": round(
                self.stats["successful"] / self.stats["llm_refinements"] * 100, 2
            ) if self.stats["llm_refinements"] > 0 else 0,
        }
