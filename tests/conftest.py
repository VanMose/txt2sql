"""
Pytest fixtures для тестов Text-to-SQL Pipeline.
"""
import os
import tempfile
import sqlite3
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def test_db_path() -> str:
    """Создать тестовую SQLite базу данных."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product TEXT,
            amount REAL,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price REAL,
            category TEXT
        )
    """)

    cursor.executemany(
        "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
        [
            ("Alice", "alice@example.com", 25),
            ("Bob", "bob@example.com", 30),
            ("Charlie", "charlie@example.com", 35),
        ]
    )

    cursor.executemany(
        "INSERT INTO orders (user_id, product, amount, created_at) VALUES (?, ?, ?, ?)",
        [
            (1, "Laptop", 999.99, "2024-01-15"),
            (1, "Mouse", 29.99, "2024-01-16"),
            (2, "Keyboard", 79.99, "2024-01-17"),
            (3, "Monitor", 299.99, "2024-01-18"),
        ]
    )

    cursor.executemany(
        "INSERT INTO products (name, price, category) VALUES (?, ?, ?)",
        [
            ("Laptop", 999.99, "Electronics"),
            ("Mouse", 29.99, "Electronics"),
            ("Keyboard", 79.99, "Electronics"),
            ("Monitor", 299.99, "Electronics"),
            ("Desk", 199.99, "Furniture"),
        ]
    )

    conn.commit()
    conn.close()

    yield db_path

    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def temp_sqlite_db() -> Generator[str, None, None]:
    """Создать временную SQLite БД для отдельных тестов."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
    cursor.execute("INSERT INTO test (value) VALUES ('test_value')")
    conn.commit()
    conn.close()

    yield db_path

    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def sample_queries() -> dict:
    """Примеры запросов для тестирования."""
    return {
        "simple": "Показать всех пользователей",
        "count": "Посчитать количество пользователей",
        "filter": "Найти пользователей старше 30 лет",
        "join": "Показать заказы с именами пользователей",
        "aggregate": "Найти общую сумму всех заказов",
    }


@pytest.fixture
def sample_sql_queries() -> dict:
    """Примеры SQL запросов для тестирования."""
    return {
        "select_all": "SELECT * FROM users",
        "select_columns": "SELECT id, name FROM users",
        "where": "SELECT * FROM users WHERE age > 30",
        "count": "SELECT COUNT(*) FROM users",
        "join": "SELECT u.name, o.product FROM users u JOIN orders o ON u.id = o.user_id",
        "aggregate": "SELECT SUM(amount) FROM orders",
        "group_by": "SELECT user_id, SUM(amount) FROM orders GROUP BY user_id",
        "invalid": "INVALID SQL QUERY",
        "non_select": "INSERT INTO users (name) VALUES ('Test')",
    }


@pytest.fixture(autouse=True)
def setup_env():
    """Настроить переменные окружения для тестов."""
    original_env = os.environ.copy()

    os.environ["TEXT2SQL_N_SAMPLES"] = "2"
    os.environ["TEXT2SQL_TEMPERATURE"] = "0.1"
    os.environ["TEXT2SQL_CONFIDENCE_THRESHOLD"] = "0.3"

    yield

    os.environ.clear()
    os.environ.update(original_env)
