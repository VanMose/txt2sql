#!/usr/bin/env python3
"""
Quick test script для проверки работоспособности пайплайна.

Для полноценного тестирования используйте pytest:
    pytest tests/ -v
"""
import sys
import logging
from pathlib import Path

# Добавляем src в path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str) -> None:
    """Вывод заголовка."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")


def print_test_result(name: str, passed: bool, details: str = "") -> None:
    """Вывод результата теста."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {name}")
    if details and not passed:
        print(f"       {details}")


def main():
    """Запуск быстрых тестов."""
    print_header("TEXT-TO-SQL PIPELINE - QUICK TESTS")
    print("Для полноценного тестирования используйте: pytest tests/ -v\n")

    passed = 0
    failed = 0

    # Test 1: Imports
    print_header("TEST 1: Проверка импортов")
    try:
        from src.config.settings import Settings, get_settings
        from src.pipeline.state import PipelineState, SQLAttempt
        from src.pipeline.text2sql_pipeline import Text2SQLPipeline
        from src.db.schema_loader import SchemaLoader
        from src.db.executor import SQLExecutor
        from src.retrieval.embedder import SchemaEmbedder
        from src.retrieval.schema_retriever import SchemaRetriever
        from src.agents.sql_generator import SQLGenerator
        from src.agents.sql_validator import SQLValidator
        from src.agents.sql_judge import SQLJudge
        from src.agents.sql_refiner import SQLRefiner
        from src.utils.json_parser import parse_json
        from src.utils.sql_parser import extract_tables, is_select_query
        print_test_result("All imports", True)
        passed += 1
    except Exception as e:
        print_test_result("All imports", False, str(e))
        failed += 1
        print("\n⚠️  Критическая ошибка импортов. Дальнейшие тесты невозможны.")
        return

    # Test 2: Settings
    print_header("TEST 2: Проверка настроек")
    try:
        settings = get_settings()
        print(f"  Model: {settings.llm_model}")
        print(f"  N Samples: {settings.n_samples}")
        print(f"  Temperature: {settings.temperature}")
        print_test_result("Settings loaded", True)
        passed += 1
    except Exception as e:
        print_test_result("Settings loaded", False, str(e))
        failed += 1

    # Test 3: SQL Parser
    print_header("TEST 3: Проверка SQL Parser")
    try:
        sql = "SELECT * FROM users"
        tables = extract_tables(sql)
        is_select = is_select_query(sql)
        print(f"  Tables in query: {tables}")
        print(f"  Is SELECT: {is_select}")
        test_passed = "users" in tables and is_select
        print_test_result("SQL Parser", test_passed)
        if test_passed:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        print_test_result("SQL Parser", False, str(e))
        failed += 1

    # Test 4: JSON Parser
    print_header("TEST 4: Проверка JSON Parser")
    try:
        json_str = '{"sql": "SELECT * FROM users", "confidence": 0.9}'
        result = parse_json(json_str)
        test_passed = result.get("sql") == "SELECT * FROM users"
        print_test_result("JSON Parser", test_passed)
        if test_passed:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        print_test_result("JSON Parser", False, str(e))
        failed += 1

    # Test 5: SchemaLoader (если есть БД)
    print_header("TEST 5: Проверка SchemaLoader")
    db_path = project_root / "data" / "movie_1" / "movie_1.sqlite"
    if db_path.exists():
        try:
            loader = SchemaLoader(str(db_path))
            tables = loader.get_tables()
            print(f"  Tables found: {len(tables)}")
            for table in tables[:3]:
                schema = loader.get_table_schema(table)
                print(f"    - {table}: {len(schema['columns'])} columns")
            print_test_result("SchemaLoader", True)
            passed += 1
        except Exception as e:
            print_test_result("SchemaLoader", False, str(e))
            failed += 1
    else:
        print(f"  ⚠️  Database not found: {db_path}")
        print_test_result("SchemaLoader", False, "Database file not found")
        failed += 1

    # Test 6: SQLExecutor (если есть БД)
    print_header("TEST 6: Проверка SQLExecutor")
    if db_path.exists():
        try:
            executor = SQLExecutor(str(db_path))
            ok, result = executor.execute("SELECT 1")
            test_passed = ok and result == [(1,)]
            print(f"  Test query result: {result}")
            print_test_result("SQLExecutor", test_passed)
            if test_passed:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print_test_result("SQLExecutor", False, str(e))
            failed += 1
    else:
        print(f"  ⚠️  Database not found")
        print_test_result("SQLExecutor", False, "Database file not found")
        failed += 1

    # Test 7: SchemaEmbedder
    print_header("TEST 7: Проверка SchemaEmbedder")
    try:
        embedder = SchemaEmbedder()
        text = "Table: users, Columns: id, name, email"
        embedding = embedder.embed(text)
        test_passed = len(embedding) == 384
        print(f"  Embedding shape: {len(embedding)}")
        print_test_result("SchemaEmbedder", test_passed)
        if test_passed:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        print_test_result("SchemaEmbedder", False, str(e))
        failed += 1

    # Test 8: SQLValidator
    print_header("TEST 8: Проверка SQLValidator")
    try:
        validator = SQLValidator()
        test_cases = [
            ("SELECT * FROM users", True),
            ("INSERT INTO users VALUES (1)", False),
            ("INVALID SQL", False),
        ]
        all_passed = True
        for sql, expected in test_cases:
            result = validator.validate(sql)
            if result != expected:
                all_passed = False
                print(f"  ❌ validate('{sql}') = {result}, expected {expected}")
            else:
                print(f"  ✓ validate('{sql}') = {result}")
        print_test_result("SQLValidator", all_passed)
        if all_passed:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        print_test_result("SQLValidator", False, str(e))
        failed += 1

    # Test 9: Pipeline Initialization
    print_header("TEST 9: Проверка инициализации Pipeline")
    if db_path.exists():
        try:
            pipeline = Text2SQLPipeline(db_path=str(db_path))
            print(f"  DB Path: {db_path}")
            print(f"  Components: schema_loader, executor, embedder, generator, validator, judge, refiner")
            print_test_result("Pipeline Init", True)
            passed += 1
        except Exception as e:
            print_test_result("Pipeline Init", False, str(e))
            failed += 1
    else:
        print(f"  ⚠️  Database not found")
        print_test_result("Pipeline Init", False, "Database file not found")
        failed += 1

    # Test 10: Mock LLM Generation
    print_header("TEST 10: Проверка генерации SQL (Mock)")
    try:
        generator = SQLGenerator()
        schema = "Table: users, Columns: id, name, email"
        candidates = generator.generate("Show all users", schema, n=2)
        test_passed = len(candidates) > 0 and all(isinstance(c, str) for c in candidates)
        print(f"  Generated {len(candidates)} candidates")
        for i, sql in enumerate(candidates[:2], 1):
            print(f"    {i}. {sql[:60]}...")
        print_test_result("SQL Generator (Mock)", test_passed)
        if test_passed:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        print_test_result("SQL Generator (Mock)", False, str(e))
        failed += 1

    # Summary
    print_header("SUMMARY")
    print(f"  Passed: {passed}/{passed + failed}")
    print(f"  Failed: {failed}/{passed + failed}")
    print()

    if failed == 0:
        print("🎉 Все тесты пройдены!")
        print("\nДля полноценного тестирования запустите:")
        print("  pytest tests/ -v")
    else:
        print(f"⚠️  {failed} тест(а) не пройдены")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
