# scripts/reindex_full.py
"""
Полная переиндексация с русскими синонимами.

Использование:
    conda activate llm_env
    python scripts/reindex_full.py
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
    """Полная переиндексация баз данных."""
    settings = get_settings()
    data_dir = Path(settings.base_dir) / "data"
    
    # Обнаружить базы данных
    db_paths = DatabaseDiscoveryService.discover(str(data_dir))
    
    if not db_paths:
        logger.error(f"No databases found in {data_dir}")
        return
    
    logger.info(f"Found {len(db_paths)} databases:")
    for db in db_paths:
        logger.info(f"  - {db}")
    
    # Инициализировать сервис
    service = PipelineService(
        db_paths=db_paths,
        use_local_qdrant=settings.qdrant_use_local,
        qdrant_local_path=settings.qdrant_local_path,
    )
    
    if not service.initialize(warmup_model=False):
        logger.error("Failed to initialize pipeline")
        return
    
    # Полная переиндексация
    logger.info("Starting FULL reindexing with Russian synonyms...")
    stats = service.recreate_vector_db_collection()
    
    logger.info(f"\n✅ Reindexing complete:")
    logger.info(f"  - Vector DB: {stats['vector_db']['points_count']} points")
    logger.info(f"  - Graph DB: {stats['graph_db']['nodes']} nodes, {stats['graph_db']['relationships']} relationships")
    logger.info(f"  - Elapsed: {stats['elapsed_ms']:.0f}ms")
    logger.info(f"\n🎯 Russian synonyms added for better search!")
    
    service.close()
    logger.info("Done")


if __name__ == "__main__":
    main()
