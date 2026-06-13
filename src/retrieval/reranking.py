from typing import List
from src.config.config import settings
from src.retrieval.vector_search import ScoredDocument
from src.utils.logger import setup_logger

logger = setup_logger("reranking")

# Check if CrossEncoder dependencies are installed
try:
    from sentence_transformers import CrossEncoder
    HAS_RERANKER = True
except ImportError:
    logger.warning("sentence-transformers not installed. Reranker will run in pass-through mode.")
    HAS_RERANKER = False

# Try to import LangSmith traceable decorator for tracing
try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type=None, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

class Reranker:
    """Reranks retrieved candidate documents using a Cross-Encoder model if available, otherwise falls back to pass-through."""
    
    def __init__(self, model_name: str = settings.rerank_model):
        self.model = None
        if HAS_RERANKER:
            logger.info(f"Initializing reranking CrossEncoder: {model_name}")
            try:
                self.model = CrossEncoder(model_name, device="cpu")
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder model: {e}. Falling back to pass-through mode.")
                self.model = None

    @traceable(name="Reranker.rerank", run_type="retriever")
    def rerank(self, query: str, documents: List[ScoredDocument], top_n: int = 3) -> List[ScoredDocument]:
        """Calculates relevance scores of query-document pairs and returns the top_n results."""
        if not documents:
            logger.info("No documents provided for reranking.")
            return []
            
        if not self.model:
            logger.info("Reranker running in pass-through fallback. Keeping original ranking.")
            # Chunks are already sorted by RRF score from hybrid search
            return documents[:top_n]
            
        logger.info(f"Reranking {len(documents)} documents for query: '{query}' (top_n={top_n})")
        
        # Prepare inputs for the cross-encoder: List of (query, page_content) pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Compute scores
        try:
            scores = self.model.predict(pairs)
            
            # Map scores to documents
            for idx, score in enumerate(scores):
                documents[idx].score = float(score)
                
            # Sort documents based on new scores in descending order
            documents.sort(key=lambda x: x.score, reverse=True)
            reranked_docs = documents[:top_n]
            
            logger.info(f"Reranking completed. Top score: {reranked_docs[0].score if reranked_docs else 'N/A'}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback to original results if model scoring fails
            return documents[:top_n]
