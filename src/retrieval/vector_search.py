from typing import List, Dict, Any
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from src.config.config import settings
from src.ingestion.embedder import Embedder
from src.utils.logger import setup_logger

logger = setup_logger("vector_search")

class ScoredDocument(BaseModel):
    """Document representation with a retrieval relevance score."""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float

# Try to import LangSmith traceable decorator for tracing
try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type=None, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

class VectorSearcher:
    """Handles vector similarity queries against Qdrant vector database."""
    
    def __init__(self):
        self.embedder = Embedder()
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )

    @traceable(name="VectorSearcher.search", run_type="retriever")
    def search(self, query: str, top_k: int = 5) -> List[ScoredDocument]:
        """Performs vector search in the Qdrant collection."""
        logger.info(f"Performing vector search for query: '{query}' (top_k={top_k})")
        try:
            # Generate query embedding
            query_vector = self.embedder.embed_query(query)
            
            # Query Qdrant
            results = self.qdrant_client.search(
                collection_name=settings.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            
            scored_docs = []
            for hit in results:
                payload = hit.payload or {}
                # Extract page content
                content = payload.pop("page_content", "")
                scored_docs.append(ScoredDocument(
                    page_content=content,
                    metadata=payload,
                    score=float(hit.score)
                ))
                
            logger.info(f"Vector search returned {len(scored_docs)} matches.")
            return scored_docs
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
