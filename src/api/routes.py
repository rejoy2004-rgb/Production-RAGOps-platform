import time
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from src.config.config import settings
from src.retrieval.query_decomposition import QueryDecomposer
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranking import Reranker
from src.generation.llm import LLMService
from src.evaluation.evaluator import RAGEvaluator
from src.monitoring.metrics import (
    rag_request_count_total,
    rag_request_latency_seconds,
    rag_faithfulness_score,
    rag_context_precision_score,
    rag_context_recall_score
)
from src.utils.logger import setup_logger

logger = setup_logger("routes")
router = APIRouter()

# Instantiate services
decomposer = QueryDecomposer()
hybrid_searcher = HybridSearcher()
reranker = Reranker()
llm_service = LLMService()
evaluator = RAGEvaluator()

class QueryRequest(BaseModel):
    question: str
    model: Optional[str] = settings.default_model
    decompose: bool = True

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, background_tasks: BackgroundTasks):
    """Processes RAG queries: Decomposes, retrieves hybrid results, rerank, and generates answers."""
    start_time = time.time()
    model = request.model or settings.default_model
    question = request.question
    
    logger.info(f"Received query request: '{question}' using model '{model}', decompose={request.decompose}")
    
    try:
        # 1. Query Decomposition (optional)
        if request.decompose:
            sub_queries = decomposer.decompose(question, model=model)
        else:
            sub_queries = [question]
        
        # 2. Retrieve for each sub-query
        all_retrieved_docs = []
        for sq in sub_queries:
            sq_results = hybrid_searcher.search(sq, top_k=5)
            all_retrieved_docs.extend(sq_results)
            
        # Deduplicate results across sub-queries by chunk_id, keeping the highest score
        chunk_map = {}
        for doc in all_retrieved_docs:
            cid = doc.metadata.get("chunk_id")
            if not cid:
                continue
            if cid not in chunk_map or doc.score > chunk_map[cid].score:
                chunk_map[cid] = doc
                
        deduped_docs = list(chunk_map.values())
        # Sort by retrieval score descending so top matches from any sub-query are prioritized in pass-through
        deduped_docs.sort(key=lambda x: x.score, reverse=True)
                
        # 3. Rerank candidates relative to the original question
        reranked_docs = reranker.rerank(question, deduped_docs, top_n=3)
        contexts = [doc.page_content for doc in reranked_docs]
        
        # 4. Answer Generation
        context_str = "\n\n".join(contexts) if contexts else "No relevant context found."
        answer = llm_service.generate_answer(question, context_str, model=model)
        
        # Record Latency and count metric
        latency = time.time() - start_time
        rag_request_latency_seconds.labels(model=model).observe(latency)
        rag_request_count_total.labels(status="success", model=model).inc()
        
        # 5. Run Online LLM Evaluation (only if enabled)
        if settings.enable_online_evaluation:
            background_tasks.add_task(
                run_online_evaluation, question, contexts, answer, model
            )
        
        return QueryResponse(
            answer=answer,
            contexts=contexts
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        rag_request_count_total.labels(status="error", model=model).inc()
        raise HTTPException(status_code=500, detail=str(e))

def run_online_evaluation(question: str, contexts: List[str], answer: str, model: str):
    """Asynchronously evaluates Faithfulness and Context Precision to update gauges."""
    logger.info("Starting online background evaluation...")
    try:
        # Evaluate Faithfulness
        faithfulness = evaluator.evaluate_faithfulness(question, contexts, answer, model=model)
        rag_faithfulness_score.labels(model=model).set(faithfulness)
        
        # Evaluate Context Precision
        precision = evaluator.evaluate_context_precision(question, contexts, model=model)
        rag_context_precision_score.labels(model=model).set(precision)
        
        logger.info("Online evaluation completed and Prometheus gauges updated.")
    except Exception as e:
        logger.error(f"Failed to run online evaluation metrics: {e}")

@router.get("/health")
async def health_endpoint():
    """Checks platform health and verifies Qdrant connection."""
    try:
        # Check Qdrant status
        hybrid_searcher.vector_searcher.qdrant_client.get_collections()
        return {"status": "healthy", "qdrant": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "qdrant": f"disconnected: {str(e)}"}

@router.get("/metrics")
async def metrics_endpoint():
    """Exposes Prometheus metrics, updating evaluation gauges from offline results if available."""
    try:
        path = settings.eval_results_path
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                summary = data.get("summary", {})
                
                avg_faithfulness = summary.get("average_faithfulness")
                avg_precision = summary.get("average_context_precision")
                avg_recall = summary.get("average_context_recall")
                model_used = summary.get("model_used", settings.default_model)
                
                if avg_faithfulness is not None:
                    rag_faithfulness_score.labels(model=model_used).set(avg_faithfulness)
                if avg_precision is not None:
                    rag_context_precision_score.labels(model=model_used).set(avg_precision)
                if avg_recall is not None:
                    rag_context_recall_score.labels(model=model_used).set(avg_recall)
                    
    except Exception as e:
        logger.error(f"Failed to update evaluation metrics from file: {e}")
        
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
