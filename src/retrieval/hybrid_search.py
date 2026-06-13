from typing import List, Dict
from src.retrieval.vector_search import VectorSearcher, ScoredDocument
from src.retrieval.bm25_search import BM25Searcher
from src.utils.logger import setup_logger

logger = setup_logger("hybrid_search")

# Try to import LangSmith traceable decorator for tracing
try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type=None, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

class HybridSearcher:
    """Combines vector search and BM25 search using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self):
        self.vector_searcher = VectorSearcher()
        self.bm25_searcher = BM25Searcher()

    @traceable(name="HybridSearcher.search", run_type="retriever")
    def search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[ScoredDocument]:
        """Runs vector and BM25 searches, and merges them using RRF."""
        logger.info(f"Performing Hybrid search for query: '{query}' (top_k={top_k})")
        
        # 1. Retrieve top-k from both vector and BM25 search
        vector_results = self.vector_searcher.search(query, top_k=top_k)
        bm25_results = self.bm25_searcher.search(query, top_k=top_k)
        
        # 2. Merge and Deduplicate using Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, ScoredDocument] = {}
        
        # Score Vector Results
        for rank, doc in enumerate(vector_results):
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                continue
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        # Score BM25 Results
        for rank, doc in enumerate(bm25_results):
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                continue
            # If not already mapped, add it
            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        # 3. Sort merged docs by RRF score descending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        hybrid_results = []
        for chunk_id in sorted_chunk_ids[:top_k]:
            doc = doc_map[chunk_id]
            # Update score to RRF score
            doc.score = rrf_scores[chunk_id]
            hybrid_results.append(doc)
            
        logger.info(f"Hybrid search merged and returned {len(hybrid_results)} candidates.")
        return hybrid_results
