import pickle
from typing import List
from src.config.config import settings
from src.retrieval.vector_search import ScoredDocument
from src.ingestion.pipeline import tokenize_text
from src.utils.logger import setup_logger

logger = setup_logger("bm25_search")

# Try to import LangSmith traceable decorator for tracing
try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type=None, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

class BM25Searcher:
    """Handles keyword-based search using the rank-bm25 index."""
    
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self._last_mtime = None
        self._load_index()

    def _load_index(self) -> None:
        """Loads the serialized BM25 index from disk."""
        path = settings.bm25_index_path
        if not path.exists():
            logger.warning(f"BM25 index file not found at {path}. Dynamic search will be disabled until documents are ingested.")
            self.bm25 = None
            self.chunks = []
            self._last_mtime = None
            return
            
        try:
            mtime = path.stat().st_mtime
            if self.bm25 is not None and self._last_mtime == mtime:
                return  # Index is already up-to-date
                
            logger.info(f"Loading BM25 index from {path}")
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25_instance"]
                self.chunks = data["chunks"]
            self._last_mtime = mtime
            logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks.")
        except Exception as e:
            logger.error(f"Failed to load BM25 index from {path}: {e}")

    @traceable(name="BM25Searcher.search", run_type="retriever")
    def search(self, query: str, top_k: int = 5) -> List[ScoredDocument]:
        """Performs BM25 search and returns ranked chunks."""
        # Reload index dynamically in case ingestion occurred in another thread/process
        self._load_index()
        
        if not self.bm25 or not self.chunks:
            logger.warning("BM25 index is not loaded. Returning empty results.")
            return []
            
        logger.info(f"Performing BM25 search for query: '{query}' (top_k={top_k})")
        
        tokenized_query = tokenize_text(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Zip chunks with their scores
        scored_chunks = []
        for idx, score in enumerate(scores):
            # Only consider documents that match at least one token (score > 0)
            if score > 0.0:
                chunk = self.chunks[idx]
                scored_chunks.append(
                    ScoredDocument(
                        page_content=chunk.page_content,
                        metadata=chunk.metadata,
                        score=float(score)
                    )
                )
                
        # Sort descending by score
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        results = scored_chunks[:top_k]
        
        logger.info(f"BM25 search returned {len(results)} matches.")
        return results
