from unittest.mock import MagicMock, patch
import pytest

from src.retrieval.vector_search import VectorSearcher, ScoredDocument
from src.retrieval.bm25_search import BM25Searcher
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.query_decomposition import QueryDecomposer
from src.retrieval.reranking import Reranker

@patch('src.retrieval.vector_search.QdrantClient')
@patch('src.retrieval.vector_search.Embedder')
def test_vector_searcher(mock_embedder_class, mock_qdrant_class):
    """Verifies that vector search parses Qdrant return payload and score correctly."""
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2]
    mock_embedder_class.return_value = mock_embedder
    
    mock_qdrant = MagicMock()
    # Mock hit structure
    mock_hit = MagicMock()
    mock_hit.score = 0.88
    mock_hit.payload = {"page_content": "Found text", "chunk_id": "c1", "source": "s1"}
    mock_qdrant.search.return_value = [mock_hit]
    mock_qdrant_class.return_value = mock_qdrant
    
    searcher = VectorSearcher()
    results = searcher.search("test query", top_k=1)
    
    assert len(results) == 1
    assert results[0].page_content == "Found text"
    assert results[0].score == 0.88
    assert results[0].metadata["chunk_id"] == "c1"

def test_bm25_searcher_no_index():
    """Verifies BM25 searcher handles missing index file gracefully."""
    with patch('src.retrieval.bm25_search.settings.bm25_index_path') as mock_path:
        mock_path.exists.return_value = False
        searcher = BM25Searcher()
        results = searcher.search("query")
        assert results == []

@patch('src.retrieval.bm25_search.BM25Searcher._load_index')
def test_bm25_searcher_with_index(mock_load):
    """Verifies BM25 search tokenization and scoring mapping."""
    searcher = BM25Searcher()
    # Mock loaded index
    searcher.bm25 = MagicMock()
    # scores for 2 chunks
    searcher.bm25.get_scores.return_value = [0.0, 5.5]
    
    chunk1 = MagicMock()
    chunk1.page_content = "ignored"
    chunk1.metadata = {"chunk_id": "c1"}
    
    chunk2 = MagicMock()
    chunk2.page_content = "relevant terms"
    chunk2.metadata = {"chunk_id": "c2"}
    
    searcher.chunks = [chunk1, chunk2]
    
    results = searcher.search("terms", top_k=2)
    assert len(results) == 1
    assert results[0].page_content == "relevant terms"
    assert results[0].score == 5.5
    assert results[0].metadata["chunk_id"] == "c2"

@patch('src.retrieval.hybrid_search.VectorSearcher')
@patch('src.retrieval.hybrid_search.BM25Searcher')
def test_hybrid_searcher(mock_bm25_class, mock_vector_class):
    """Verifies RRF (Reciprocal Rank Fusion) ranking logic."""
    mock_vector = MagicMock()
    doc_v1 = ScoredDocument(page_content="text1", metadata={"chunk_id": "c1"}, score=0.9)
    doc_v2 = ScoredDocument(page_content="text2", metadata={"chunk_id": "c2"}, score=0.8)
    mock_vector.search.return_value = [doc_v1, doc_v2]
    mock_vector_class.return_value = mock_vector
    
    mock_bm25 = MagicMock()
    # BM25 retrieves c2 first, c3 second
    doc_b2 = ScoredDocument(page_content="text2", metadata={"chunk_id": "c2"}, score=10.0)
    doc_b3 = ScoredDocument(page_content="text3", metadata={"chunk_id": "c3"}, score=5.0)
    mock_bm25.search.return_value = [doc_b2, doc_b3]
    mock_bm25_class.return_value = mock_bm25
    
    searcher = HybridSearcher()
    # Under RRF with k=60:
    # c1: rank 1 in vector (1/61) = 0.01639
    # c2: rank 2 in vector (1/62) + rank 1 in BM25 (1/61) = 0.01613 + 0.01639 = 0.0325
    # c3: rank 2 in BM25 (1/62) = 0.01613
    # c2 should be ranked 1st overall
    results = searcher.search("query", top_k=3, rrf_k=60)
    
    assert len(results) == 3
    assert results[0].metadata["chunk_id"] == "c2"
    assert results[1].metadata["chunk_id"] == "c1" or results[1].metadata["chunk_id"] == "c3"

@patch('src.retrieval.query_decomposition.LLMService')
def test_query_decomposer(mock_llm_class):
    """Verifies that decomposer splits query by lines and strips markdown numbering/bullets."""
    mock_llm = MagicMock()
    mock_llm.call_llm.return_value = "1. refund policy\n- cancellation process\n3) trial membership"
    mock_llm_class.return_value = mock_llm
    
    decomposer = QueryDecomposer()
    sub_queries = decomposer.decompose("complex question")
    
    assert len(sub_queries) == 3
    assert sub_queries[0] == "refund policy"
    assert sub_queries[1] == "cancellation process"
    assert sub_queries[2] == "trial membership"

@patch('src.retrieval.reranking.CrossEncoder')
def test_reranker(mock_cross_encoder_class):
    """Verifies reranker score reassignment and sorting."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9] # Scores for doc1 and doc2 respectively
    mock_cross_encoder_class.return_value = mock_model
    
    doc1 = ScoredDocument(page_content="bad context", metadata={"chunk_id": "c1"}, score=0.5)
    doc2 = ScoredDocument(page_content="good context", metadata={"chunk_id": "c2"}, score=0.4)
    
    reranker = Reranker()
    results = reranker.rerank("query", [doc1, doc2], top_n=2)
    
    # doc2 should be sorted first since its predict score is 0.9 > 0.1
    assert len(results) == 2
    assert results[0].metadata["chunk_id"] == "c2"
    assert results[0].score == 0.9
    assert results[1].metadata["chunk_id"] == "c1"
    assert results[1].score == 0.1
