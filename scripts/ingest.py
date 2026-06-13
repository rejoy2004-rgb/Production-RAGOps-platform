import argparse
from pathlib import Path
from src.config.config import settings
from src.ingestion.pipeline import IngestionPipeline
from src.utils.logger import setup_logger

logger = setup_logger("ingest_script")

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant & BM25 index.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(settings.data_dir / "sample_docs"),
        help="Directory containing documents to ingest."
    )
    
    args = parser.parse_args()
    input_path = Path(args.input_dir)
    
    logger.info("Initializing Ingestion Pipeline...")
    pipeline = IngestionPipeline()
    
    try:
        pipeline.run(input_path)
        logger.info("Ingestion complete.")
    except Exception as e:
        logger.critical(f"Ingestion script failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
