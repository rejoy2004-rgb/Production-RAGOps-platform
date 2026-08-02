#Coordinates the ingestion pipeline for the RAGOps platform, including document loading, chunking, embedding, Qdrant upsert, and BM25 index building.
import os
import pickle
import string
import uuid
from pathlib import Path
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

from src.config.config import settings
from src.ingestion.loader import DocumentLoader, Document
from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import Embedder
from src.utils.logger import setup_logger

logger = setup_logger("pipeline")

def tokenize_text(text: str) -> List[str]:
    """Tokenizes text by lowercasing and removing punctuation."""
    clean_text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return clean_text.split()

class IngestionPipeline:
    """Coordinates document loading, chunking, embedding, Qdrant upsert, and BM25 building."""
    
    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()
        self.embedder = Embedder()
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )

    def setup_qdrant_collection(self) -> None:
        """Ensures the Qdrant collection exists."""
        logger.info(f"Checking Qdrant collection: {settings.collection_name}")
        try:
            collections = self.qdrant_client.get_collections().collections
            exists = any(c.name == settings.collection_name for c in collections)
            
            if not exists:
                # Dynamically get embedding dimension by embedding a test string
                sample_embedding = self.embedder.embed_query("test")
                dim = len(sample_embedding)
                logger.info(f"Creating collection {settings.collection_name} with detected dim={dim}")
                self.qdrant_client.create_collection(
                    collection_name=settings.collection_name,
                    vectors_config=VectorParams(
                        size=dim,
                        distance=Distance.COSINE
                    )
                )
            else:
                logger.info(f"Collection {settings.collection_name} already exists.")
        except Exception as e:
            logger.error(f"Error checking/creating Qdrant collection: {e}")
            raise e

    def run(self, input_dir: Path) -> None:
        """Runs the complete ingestion pipeline over files in the input directory."""
        logger.info(f"Starting ingestion pipeline from: {input_dir}")
        self.setup_qdrant_collection()
        
        # 1. Load documents
        all_docs = []
        if not input_dir.exists():
            logger.warning(f"Input directory {input_dir} does not exist.")
            return
            
        for file_name in os.listdir(input_dir):
            file_path = input_dir / file_name
            if file_path.is_file():
                docs = self.loader.load_document(file_path)
                all_docs.extend(docs)
                
        if not all_docs:
            logger.warning("No documents loaded from the input directory.")
            return
            
        # 2. Chunk documents
        chunks = self.chunker.chunk_documents(all_docs)
        if not chunks:
            logger.warning("No chunks generated from documents.")
            return
            
        # 3. Generate embeddings and upload to Qdrant
        logger.info("Generating embeddings and uploading to Qdrant...")
        points = []
        texts = [c.page_content for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.metadata.get("chunk_id", str(uuid.uuid4()))
            # Convert string ID to a valid UUID namespace DNS representation for Qdrant
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            
            points.append(PointStruct(
                id=point_id,
                vector=embeddings[idx],
                payload={
                    "page_content": chunk.page_content,
                    **chunk.metadata
                }
            ))
            
        try:
            self.qdrant_client.upsert(
                collection_name=settings.collection_name,
                points=points
            )
            logger.info(f"Successfully uploaded {len(points)} points to Qdrant.")
        except Exception as e:
            logger.error(f"Error uploading points to Qdrant: {e}")
            raise e
            
        # 4. Build and persist BM25 index
        logger.info("Building and persisting BM25 index...")
        tokenized_corpus = [tokenize_text(c.page_content) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Serialize both BM25 instance and corresponding chunks to align indices
        bm25_data = {
            "bm25_instance": bm25,
            "chunks": chunks
        }
        
        try:
            os.makedirs(settings.bm25_index_path.parent, exist_ok=True)
            with open(settings.bm25_index_path, "wb") as f:
                pickle.dump(bm25_data, f)
            logger.info(f"Successfully serialized BM25 index to {settings.bm25_index_path}")
        except Exception as e:
            logger.error(f"Error serializing BM25 index: {e}")
            raise e
            
        logger.info("Ingestion pipeline completed successfully.")
