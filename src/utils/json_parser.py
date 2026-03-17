"""
Утилиты для парсинга JSON из LLM ответов.

Example:
    >>> from utils.json_parser import parse_json, safe_parse_json
    >>> obj = parse_json('{"key": "value"}')
    >>> obj = safe_parse_json('invalid json', default={})
"""
import json
import re
from typing import Any, Optional


def parse_json(text: str) -> Any:
    """
    Извлечь и распарсить JSON из текста.

    Args:
        text: Текст, содержащий JSON.

    Returns:
        Распарсенный JSON объект.

    Raises:
        ValueError: Если JSON не найден или некорректен.
    """
    # Пробуем найти JSON в markdown блоках
    json_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(json_block_pattern, text, re.DOTALL)

    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Ищем {...} с учётом вложенности
    start_idx = text.find('{')
    if start_idx != -1:
        brace_count = 0
        end_idx = start_idx

        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        if end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            # Очистка от trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)

            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                try:
                    # Замена одинарных кавычек на двойные
                    json_str_fixed = json_str.replace("'", '"')
                    return json.loads(json_str_fixed)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON: {e}") from e

    # Пробуем распарсить весь текст
    return json.loads(text)


def extract_json_from_markdown(text: str) -> str:
    """
    Извлечь JSON из markdown блока.

    Args:
        text: Текст с markdown JSON блоком.

    Returns:
        Извлеченный JSON строкой.
    """
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else text


def safe_parse_json(text: str, default: Optional[Any] = None) -> Any:
    """
    Безопасный парсинг JSON с возвратом default при ошибке.

    Args:
        text: Текст для парсинга.
        default: Значение по умолчанию при ошибке.

    Returns:
        Распарсенный JSON или default.
    """
    try:
        return parse_json(text)
    except Exception:
        return default
