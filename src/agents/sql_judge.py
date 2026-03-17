# src\agents\sql_judge.py
"""
Агент оценки качества SQL запросов с использованием LLM.

Оценивает:
- Соответствие SQL запросу пользователя
- Семантическая корректность
- Уверенность в правильности (confidence)
- Наличие ошибок

Использование:
    >>> from agents.sql_judge import SQLJudge
    >>> judge = SQLJudge()
    >>> confidence, reason = judge.evaluate("Show movies", "SELECT * FROM Movie")
"""
import logging
from typing import Any, Dict, Tuple

from ..llm.inference import LLMService
from ..llm.prompts import Prompts
from ..utils.json_parser import parse_json

logger = logging.getLogger(__name__)


class SQLJudge:
    """
    Агент оценки качества SQL запросов на основе LLM.
    
    Attributes:
        llm: LLM сервис для оценки.
    """

    def __init__(self) -> None:
        """Инициализировать judge."""
        self.llm = LLMService()

    def evaluate(self, query: str, sql: str) -> Tuple[float, str]:
        """
        Оценить качество SQL запроса через LLM.

        Args:
            query: Запрос пользователя.
            sql: SQL запрос для оценки.

        Returns:
            Кортеж (confidence, reason).
        """
        # 🔥 Программная проверка SELECT колонок
        select_check = self._check_select_columns(query, sql)
        if not select_check["valid"]:
            # Критическая ошибка - SELECT не возвращает то что нужно
            logger.warning(f"SELECT column mismatch: {select_check['reason']}")
            return 0.2, f"CRITICAL: {select_check['reason']}"
        
        try:
            # Формируем промпт для оценки
            prompt = Prompts.SQL_JUDGE_TEMPLATE.format(
                query=query,
                sql=sql,
            )
            
            logger.info(f"Evaluating SQL with LLM: query='{query[:50]}...', sql='{sql[:50]}...'")
            
            # Генерируем оценку через LLM
            outputs = self.llm.generate(
                prompt=prompt,
                n=1,
                temperature=0.1,  # Низкая температура для консистентности
            )
            
            if not outputs:
                logger.warning("LLM returned empty output for SQL evaluation")
                return 0.5, "LLM evaluation failed"
            
            # Парсим JSON ответ
            result = parse_json(outputs[0])
            
            if result is None:
                logger.warning(f"Failed to parse LLM JSON response: {outputs[0][:200]}")
                # Fallback: парсим confidence из текста
                confidence = self._extract_confidence_from_text(outputs[0])
                return confidence, "LLM response parse failed"
            
            # Извлекаем confidence и reason
            confidence = float(result.get("confidence", 0.5))
            error = result.get("error", False)
            reason = result.get("reason", "No reason provided")
            
            # Если error=true, снижаем confidence
            if error:
                confidence = min(confidence, 0.3)
                reason = f"Error detected: {reason}"
            
            logger.info(f"LLM judge: confidence={confidence:.2f}, reason={reason}")
            return confidence, reason
            
        except Exception as e:
            logger.error(f"SQL evaluation failed: {e}", exc_info=True)
            return 0.0, f"Evaluation error: {str(e)}"
    
    def _check_select_columns(self, query: str, sql: str) -> Dict[str, Any]:
        """
        Проверить что SELECT возвращает нужные колонки.
        
        Args:
            query: Запрос пользователя.
            sql: SQL запрос.
        
        Returns:
            Dict с {"valid": bool, "reason": str}
        """
        import re
        
        query_lower = query.lower()
        sql_upper = sql.upper()
        
        # Извлекаем SELECT часть
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return {"valid": True, "reason": "Cannot parse SELECT clause"}
        
        select_clause = select_match.group(1).lower()
        
        # 🔥 Проверка 1: Вопрос про rating → SELECT должен возвращать rating/stars
        if "rating" in query_lower or "рейтинг" in query_lower or "stars" in query_lower:
            # SELECT должен содержать rating, stars, score или类似
            rating_keywords = ["rating", "stars", "score", "r.stars", "r.rating", "avg", "average"]
            has_rating = any(kw in select_clause for kw in rating_keywords)
            
            if not has_rating:
                return {
                    "valid": False,
                    "reason": f"Question asks for rating but SELECT returns: {select_clause.strip()}"
                }
        
        # 🔥 Проверка 2: Вопрос про count → SELECT должен использовать COUNT()
        if "count" in query_lower or "сколько" in query_lower or "number" in query_lower:
            if "count(" not in select_clause:
                return {
                    "valid": False,
                    "reason": f"Question asks for count but SELECT doesn't use COUNT(): {select_clause.strip()}"
                }
        
        # 🔥 Проверка 3: Вопрос про название (title/name) → SELECT должен возвращать название
        if "title" in query_lower or "name" in query_lower or "название" in query_lower:
            if "what is the rating" not in query_lower:  # Не для rating вопросов
                name_keywords = ["title", "name", "t.title", "m.title", "a.name"]
                has_name = any(kw in select_clause for kw in name_keywords)
                
                # SELECT * тоже подходит для name вопросов
                if not has_name and select_clause.strip() != "*":
                    return {
                        "valid": False,
                        "reason": f"Question asks for title/name but SELECT returns: {select_clause.strip()}"
                    }
        
        return {"valid": True, "reason": "SELECT columns match question intent"}
    
    def evaluate_with_error(
        self,
        query: str,
        sql: str,
        error_message: str,
    ) -> Tuple[float, str]:
        """
        Оценить SQL с известной ошибкой выполнения.

        Args:
            query: Запрос пользователя.
            sql: SQL запрос.
            error_message: Сообщение об ошибке.

        Returns:
            Кортеж (confidence, reason).
        """
        # Модифицируем промпт для учёта ошибки
        prompt_with_error = f"""You are an expert SQL verifier.

## Task
Evaluate whether the SQL query is correct given the execution error.

## Question
{query}

## SQL Query
{sql}

## Execution Error
{error_message}

## Output Format
{{"confidence": 0.0-1.0, "error": true, "reason": "explanation"}}

## Evaluation (JSON)
"""
        
        try:
            outputs = self.llm.generate(
                prompt=prompt_with_error,
                n=1,
                temperature=0.1,
            )
            
            if not outputs:
                return 0.0, f"Execution error: {error_message}"
            
            result = parse_json(outputs[0])
            
            if result is None:
                # Fallback: низкая уверенность из-за ошибки
                return 0.2, f"Execution error: {error_message}"
            
            confidence = float(result.get("confidence", 0.2))
            reason = result.get("reason", f"Execution error: {error_message}")
            
            # Ошибка выполнения всегда снижает confidence
            confidence = min(confidence, 0.4)
            
            logger.info(f"LLM judge (with error): confidence={confidence:.2f}, reason={reason}")
            return confidence, reason
            
        except Exception as e:
            logger.error(f"SQL evaluation with error failed: {e}")
            return 0.0, f"Execution error: {error_message}"
    
    def _extract_confidence_from_text(self, text: str) -> float:
        """
        Извлечь confidence из текстового ответа LLM.
        
        Fallback метод если JSON парсинг не удался.
        """
        import re
        
        # Ищем паттерны типа "confidence": 0.8 или confidence=0.8
        patterns = [
            r'"confidence"\s*[:=]\s*(\d+\.?\d*)',
            r'confidence\s*[:=]\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*/\s*1\.0',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        
        # Fallback: ищем числа от 0 до 1
        numbers = re.findall(r'\b(0?\.\d+|1\.0+)\b', text)
        if numbers:
            try:
                return max(float(n) for n in numbers)
            except ValueError:
                pass
        
        return 0.5  # Default если ничего не найдено
    
    def evaluate_batch(
        self,
        query: str,
        sql_candidates: list[str],
    ) -> list[Tuple[str, float, str]]:
        """
        Оценить несколько SQL кандидатов.

        Args:
            query: Запрос пользователя.
            sql_candidates: Список SQL запросов.

        Returns:
            Список кортежей (sql, confidence, reason).
        """
        results = []
        for sql in sql_candidates:
            confidence, reason = self.evaluate(query, sql)
            results.append((sql, confidence, reason))
        
        # Сортировка по confidence
        results.sort(key=lambda x: x[1], reverse=True)
        return results
