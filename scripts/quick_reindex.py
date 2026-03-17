# scripts/quick_reindex.py
"""
Быстрая переиндексация с обновлением row_count.

Использование:
    conda activate llm_env
    python scripts/quick_reindex.py
"""
import logging
import sys
from pathlib import Path

# Добавляем project root в path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import get_settings
from src.services.pipeline_service import PipelineService, DatabaseDiscoveryService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Переиндексировать базы данных с row_count."""
    settings = get_settings()
    data_dir = Path(settings.base_dir) / "data"
    
    # Обнаружить базы данных
    db_paths = DatabaseDiscoveryService.discover(str(data_dir))
    
    if not db_paths:
        logger.error(f"No databases found in {data_dir}")
        return
    
    logger.info(f"Found {len(db_paths)} databases")
    
    # Инициализировать сервис
    service = PipelineService(
        db_paths=db_paths,
        use_local_qdrant=settings.qdrant_use_local,
        qdrant_local_path=settings.qdrant_local_path,
    )
    
    if not service.initialize(warmup_model=False):
        logger.error("Failed to initialize pipeline")
        return
    
    # Переиндексация
    logger.info("Starting reindexing with row_count...")
    stats = service.recreate_vector_db_collection()
    
    logger.info(f"\n✅ Reindexing complete:")
    logger.info(f"  - Vector DB: {stats['vector_db']['points_count']} points")
    logger.info(f"  - Graph DB: {stats['graph_db']['nodes']} nodes")
    logger.info(f"  - Elapsed: {stats['elapsed_ms']:.0f}ms")
    logger.info(f"\n🎯 row_count added for better DB selection!")
    
    service.close()
    logger.info("Done")


if __name__ == "__main__":
    main()
