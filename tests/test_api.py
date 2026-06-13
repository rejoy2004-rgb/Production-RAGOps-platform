from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from src.api.app import app
from src.retrieval.vector_search import ScoredDocument

client = TestClient(app)

def test_health_endpoint():
    """Verifies GET /health endpoint behavior."""
    with patch('src.api.routes.hybrid_searcher.vector_searcher.qdrant_client.get_collections') as mock_check:
        mock_check.return_value = MagicMock()
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "qdrant": "connected"}

def test_metrics_endpoint():
    """Verifies GET /metrics endpoint contains standard prometheus formatting."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rag_request_count_total" in response.text

@patch('src.api.routes.decomposer.decompose')
@patch('src.api.routes.hybrid_searcher.search')
@patch('src.api.routes.reranker.rerank')
@patch('src.api.routes.llm_service.generate_answer')
@patch('src.api.routes.evaluator.evaluate_faithfulness')
@patch('src.api.routes.evaluator.evaluate_context_precision')
def test_query_endpoint(
    mock_precision, mock_faithfulness, mock_generate, mock_rerank, mock_search, mock_decompose
):
    """Verifies POST /query decomposes question, retrieves docs, and generates response."""
    # Setup mocks
    mock_decompose.return_value = ["subquery1"]
    
    mock_doc = ScoredDocument(page_content="mocked chunk context", metadata={"chunk_id": "c1"}, score=0.9)
    mock_search.return_value = [mock_doc]
    mock_rerank.return_value = [mock_doc]
    
    mock_generate.return_value = "This is the generated answer."
    
    mock_faithfulness.return_value = 1.0
    mock_precision.return_value = 1.0
    
    payload = {"question": "What is the refund policy?"}
    
    # We use uvicorn/fastapi environment
    response = client.post("/query", json=payload)
    
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["answer"] == "This is the generated answer."
    assert len(res_data["contexts"]) == 1
    assert res_data["contexts"][0] == "mocked chunk context"
    
    # Assert proper calls
    from src.config.config import settings
    mock_decompose.assert_called_once_with("What is the refund policy?", model=settings.default_model)
    mock_search.assert_called_once_with("subquery1", top_k=5)
    mock_rerank.assert_called_once_with("What is the refund policy?", [mock_doc], top_n=3)
