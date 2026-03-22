# src\services\preflight.py
"""
Preflight checks for Text-to-SQL Pipeline.

Provides fail-fast diagnostics with human-readable error messages
before the pipeline starts. Avoids deep stack traces.

Checks:
- Ollama: Service running and model available
- Neo4j: Connection accessible
- Qdrant: API reachable
- Database files: Exist and readable

Example:
    >>> from services.preflight import PreflightChecker
    >>> checker = PreflightChecker()
    >>> checker.run_all_checks()
    >>> checker.print_report()
"""
import logging
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Результат проверки."""
    name: str
    success: bool
    message: str
    details: Optional[str] = None
    duration_ms: float = 0.0
    error: Optional[str] = None


class PreflightChecker:
    """
    Preflight checks for pipeline services.

    Features:
    - Human-readable error messages
    - Fail-fast behavior
    - No deep stack traces
    - Clear remediation steps
    """

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_local_path: Optional[str] = None,
        use_local_qdrant: bool = True,
        db_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Инициализировать checker.

        Args:
            ollama_url: URL Ollama сервера.
            neo4j_uri: URI Neo4j сервера.
            neo4j_username: Имя пользователя Neo4j.
            neo4j_password: Пароль Neo4j.
            qdrant_url: URL Qdrant сервера.
            qdrant_local_path: Путь к локальному Qdrant.
            use_local_qdrant: Использовать локальный Qdrant.
            db_paths: Пути к базам данных.
        """
        settings = get_settings()

        self.ollama_url = ollama_url or settings.ollama_base_url
        self.neo4j_uri = neo4j_uri or settings.neo4j_uri
        self.neo4j_username = neo4j_username or settings.neo4j_username
        self.neo4j_password = neo4j_password or settings.neo4j_password
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.qdrant_local_path = qdrant_local_path or settings.qdrant_local_path
        self.use_local_qdrant = use_local_qdrant
        self.db_paths = db_paths or settings.get("db_paths", [])

        self.results: List[CheckResult] = []
        self._critical_failures: List[str] = []

    def _check_ollama_service(self) -> CheckResult:
        """Проверить доступность Ollama сервиса."""
        start = time.time()

        try:
            # Check if service is running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()

            # Check if required model is available
            settings = get_settings()
            required_model = settings.llm_model

            models_data = response.json()
            available_models = [m["name"] for m in models_data.get("models", [])]

            # Check for exact match or prefix match
            model_found = any(
                required_model == m or m.startswith(required_model.split(":")[0])
                for m in available_models
            )

            duration = (time.time() - start) * 1000

            if model_found:
                return CheckResult(
                    name="Ollama Service",
                    success=True,
                    message=f"Ollama is running with model '{required_model}'",
                    details=f"Available models: {', '.join(available_models)}",
                    duration_ms=duration,
                )
            else:
                return CheckResult(
                    name="Ollama Model",
                    success=False,
                    message=f"Required model '{required_model}' not found",
                    details=f"Available models: {', '.join(available_models)}",
                    duration_ms=duration,
                    error=f"Model '{required_model}' is not available. Run: ollama pull {required_model}",
                )

        except requests.exceptions.ConnectionError as e:
            duration = (time.time() - start) * 1000
            return CheckResult(
                name="Ollama Service",
                success=False,
                message="Ollama service is not running",
                details=f"Cannot connect to {self.ollama_url}",
                duration_ms=duration,
                error="Start Ollama service or install it from https://ollama.ai",
            )
        except requests.exceptions.Timeout:
            duration = (time.time() - start) * 1000
            return CheckResult(
                name="Ollama Service",
                success=False,
                message="Ollama service timeout",
                details=f"Timeout connecting to {self.ollama_url}",
                duration_ms=duration,
                error="Ollama service is not responding. Restart the service.",
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return CheckResult(
                name="Ollama Service",
                success=False,
                message="Ollama service error",
                details=str(e),
                duration_ms=duration,
                error=f"Unexpected error: {str(e)}",
            )

    def _check_neo4j_connection(self) -> CheckResult:
        """Проверить подключение к Neo4j."""
        start = time.time()

        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password),
            )

            # Test connection
            driver.verify_connectivity()

            # Simple query to verify access
            with driver.session() as session:
                session.run("RETURN 1")

            driver.close()
            duration = (time.time() - start) * 1000

            return CheckResult(
                name="Neo4j Database",
                success=True,
                message=f"Neo4j is accessible at {self.neo4j_uri}",
                duration_ms=duration,
            )

        except ImportError:
            duration = (time.time() - start) * 1000
            return CheckResult(
                name="Neo4j Driver",
                success=False,
                message="Neo4j driver not installed",
                details="neo4j package not found",
                duration_ms=duration,
                error="Install Neo4j driver: pip install neo4j",
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            error_msg = str(e)

            # Parse common errors
            if "Failed to establish connection" in error_msg or "Connection refused" in error_msg:
                return CheckResult(
                    name="Neo4j Database",
                    success=False,
                    message="Neo4j is not running",
                    details=f"Cannot connect to {self.neo4j_uri}",
                    duration_ms=duration,
                    error="Start Neo4j: docker-compose up -d or check neo4j service",
                )
            elif "Authentication failed" in error_msg or "Invalid credentials" in error_msg:
                return CheckResult(
                    name="Neo4j Authentication",
                    success=False,
                    message="Neo4j authentication failed",
                    details=f"Username: {self.neo4j_username}",
                    duration_ms=duration,
                    error="Check credentials in .env file (TEXT2SQL_NEO4J_USERNAME/PASSWORD)",
                )
            else:
                return CheckResult(
                    name="Neo4j Database",
                    success=False,
                    message="Neo4j connection error",
                    details=error_msg[:200],
                    duration_ms=duration,
                    error=f"Connection error: {error_msg[:100]}",
                )

    def _check_qdrant_connection(self) -> CheckResult:
        """Проверить подключение к Qdrant."""
        start = time.time()

        if self.use_local_qdrant:
            # Check local Qdrant storage
            qdrant_path = Path(self.qdrant_local_path)

            if not qdrant_path.exists():
                # This is OK - will be created on first indexing
                duration = (time.time() - start) * 1000
                return CheckResult(
                    name="Qdrant Local Storage",
                    success=True,
                    message=f"Local Qdrant storage will be created at {self.qdrant_local_path}",
                    duration_ms=duration,
                )

            if not qdrant_path.is_dir():
                duration = (time.time() - start) * 1000
                return CheckResult(
                    name="Qdrant Local Storage",
                    success=False,
                    message=f"Qdrant path exists but is not a directory: {self.qdrant_local_path}",
                    duration_ms=duration,
                    error="Remove the file or change the path in .env",
                )

            # Check if directory is writable
            try:
                test_file = qdrant_path / ".write_test"
                test_file.touch()
                test_file.unlink()
                duration = (time.time() - start) * 1000
                return CheckResult(
                    name="Qdrant Local Storage",
                    success=True,
                    message=f"Local Qdrant storage is writable at {self.qdrant_local_path}",
                    duration_ms=duration,
                )
            except Exception as e:
                duration = (time.time() - start) * 1000
                return CheckResult(
                    name="Qdrant Local Storage",
                    success=False,
                    message="Qdrant storage is not writable",
                    details=str(e),
                    duration_ms=duration,
                    error="Check permissions or change path in .env",
                )
        else:
            # Check remote Qdrant
            try:
                response = requests.get(f"{self.qdrant_url}/", timeout=5)
                response.raise_for_status()

                duration = (time.time() - start) * 1000
                return CheckResult(
                    name="Qdrant API",
                    success=True,
                    message=f"Qdrant API is accessible at {self.qdrant_url}",
                    duration_ms=duration,
                )

            except requests.exceptions.ConnectionError:
                duration = (time.time() - start) * 1000
                return CheckResult(
                    name="Qdrant API",
                    success=False,
                    message="Qdrant API is not accessible",
                    details=f"Cannot connect to {self.qdrant_url}",
                    duration_ms=duration,
                    error="Start Qdrant: docker-compose up -d or check URL in .env",
                )
            except requests.exceptions.Timeout:
                duration = (time.time() - start) * 1000
                return CheckResult(
                    name="Qdrant API",
                    success=False,
                    message="Qdrant API timeout",
                    details=f"Timeout connecting to {self.qdrant_url}",
                    duration_ms=duration,
                    error="Qdrant service is not responding",
                )
            except Exception as e:
                duration = (time.time() - start) * 1000
                return CheckResult(
                    name="Qdrant API",
                    success=False,
                    message="Qdrant API error",
                    details=str(e)[:200],
                    duration_ms=duration,
                    error=f"API error: {str(e)[:100]}",
                )

    def _check_database_files(self) -> CheckResult:
        """Проверить наличие файлов баз данных."""
        start = time.time()

        if not self.db_paths:
            return CheckResult(
                name="Database Files",
                success=False,
                message="No database files configured",
                details="db_paths is empty",
                duration_ms=(time.time() - start) * 1000,
                error="Add SQLite databases to data/ folder or set TEXT2SQL_DB_PATHS in .env",
            )

        missing = []
        accessible = []

        for db_path in self.db_paths:
            path = Path(db_path)
            if path.exists():
                accessible.append(str(path.name))
            else:
                missing.append(str(path))

        duration = (time.time() - start) * 1000

        if missing:
            return CheckResult(
                name="Database Files",
                success=False,
                message=f"{len(missing)} database file(s) not found",
                details=f"Missing: {', '.join(missing)}",
                duration_ms=duration,
                error="Check database paths in .env or add files to data/ folder",
            )

        return CheckResult(
            name="Database Files",
            success=True,
            message=f"{len(accessible)} database file(s) found",
            details=f"Accessible: {', '.join(accessible)}",
            duration_ms=duration,
        )

    def run_all_checks(self) -> List[CheckResult]:
        """Выполнить все проверки."""
        logger.info("Running preflight checks...")

        self.results = []

        # Run checks in order of importance
        self.results.append(self._check_ollama_service())
        self.results.append(self._check_neo4j_connection())
        self.results.append(self._check_qdrant_connection())
        self.results.append(self._check_database_files())

        # Collect critical failures
        self._critical_failures = [
            r.name for r in self.results if not r.success
        ]

        # Log results
        for result in self.results:
            status = "✅" if result.success else "❌"
            logger.info(f"{status} {result.name}: {result.message}")

        return self.results

    def has_critical_failures(self) -> bool:
        """Проверить наличие критических ошибок."""
        return len(self._critical_failures) > 0

    def get_critical_failures(self) -> List[str]:
        """Получить список критических ошибок."""
        return self._critical_failures.copy()

    def print_report(self) -> None:
        """Вывести отчет о проверках."""
        print("\n" + "=" * 60)
        print("🔍 PREFLIGHT CHECK REPORT")
        print("=" * 60)

        for result in self.results:
            status = "✅" if result.success else "❌"
            print(f"\n{status} {result.name}")
            print(f"   {result.message}")

            if result.details:
                print(f"   Details: {result.details}")

            if result.error:
                print(f"   🔧 Fix: {result.error}")

            if result.duration_ms > 0:
                print(f"   ⏱️  Duration: {result.duration_ms:.0f}ms")

        print("\n" + "=" * 60)

        if self._critical_failures:
            print(f"❌ CRITICAL FAILURES: {len(self._critical_failures)}")
            for name in self._critical_failures:
                print(f"   - {name}")
            print("\n🛑 Pipeline cannot start. Fix the issues above.")
        else:
            print("✅ ALL CHECKS PASSED")
            print("🚀 Pipeline is ready to start")

        print("=" * 60 + "\n")

    def fail_fast(self) -> None:
        """
        Вызвать исключение при критических ошибках.

        Raises:
            RuntimeError: Если есть критические ошибки.
        """
        if self.has_critical_failures():
            errors = "\n".join([
                f"  - {r.name}: {r.error}"
                for r in self.results
                if not r.success and r.error
            ])
            raise RuntimeError(
                f"Preflight checks failed:\n{errors}\n\n"
                f"Run preflight checker for details: "
                f"from services.preflight import PreflightChecker; PreflightChecker().run_all_checks(); PreflightChecker().print_report()"
            )


def run_preflight(fail_on_error: bool = True) -> PreflightChecker:
    """
    Запустить preflight проверки.

    Args:
        fail_on_error: Вызвать исключение при ошибках.

    Returns:
        PreflightChecker с результатами.

    Raises:
        RuntimeError: Если fail_on_error=True и есть ошибки.
    """
    checker = PreflightChecker()
    checker.run_all_checks()
    checker.print_report()

    if fail_on_error:
        checker.fail_fast()

    return checker
