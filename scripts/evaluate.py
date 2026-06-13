import json
import argparse
from typing import List, Dict, Any
from src.config.config import settings
from src.retrieval.query_decomposition import QueryDecomposer
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranking import Reranker
from src.generation.llm import LLMService
from src.evaluation.evaluator import RAGEvaluator
from src.utils.logger import setup_logger

logger = setup_logger("evaluate_script")

def load_golden_set(path) -> List[Dict[str, str]]:
    """Loads the golden dataset Q&A pairs."""
    logger.info(f"Loading golden dataset from {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results(path, results: Dict[str, Any]) -> None:
    """Saves evaluation results to JSON."""
    logger.info(f"Saving evaluation results to {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG platform using golden test set.")
    parser.add_argument(
        "--model",
        type=str,
        default=settings.default_model,
        help="LLM model to use for generation and evaluation."
    )
    
    args = parser.parse_args()
    model = args.model
    
    # Initialize components
    decomposer = QueryDecomposer()
    hybrid_searcher = HybridSearcher()
    reranker = Reranker()
    llm_service = LLMService()
    evaluator = RAGEvaluator()
    
    golden_set = load_golden_set(settings.golden_test_set_path)
    
    results = []
    total_faithfulness = 0.0
    total_precision = 0.0
    total_recall = 0.0
    
    logger.info(f"Starting evaluation on {len(golden_set)} test cases using model {model}...")
    
    for idx, item in enumerate(golden_set):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        logger.info(f"[{idx+1}/{len(golden_set)}] Evaluating query: '{question}'")
        
        try:
            # 1. Decompose
            sub_queries = decomposer.decompose(question, model=model)
            
            # 2. Hybrid Retrieve
            all_docs = []
            for sq in sub_queries:
                all_docs.extend(hybrid_searcher.search(sq, top_k=5))
                
            # Deduplicate
            seen = set()
            deduped = []
            for doc in all_docs:
                cid = doc.metadata.get("chunk_id")
                if cid and cid not in seen:
                    seen.add(cid)
                    deduped.append(doc)
                    
            # 3. Rerank
            reranked = reranker.rerank(question, deduped, top_n=3)
            contexts = [doc.page_content for doc in reranked]
            
            # 4. Generate Answer
            context_str = "\n\n".join(contexts) if contexts else "No relevant context found."
            answer = llm_service.generate_answer(question, context_str, model=model)
            
            # 5. Evaluate
            metrics = evaluator.evaluate_all(
                question=question,
                contexts=contexts,
                answer=answer,
                ground_truth=ground_truth,
                model=model
            )
            
            total_faithfulness += metrics["faithfulness"]
            total_precision += metrics["context_precision"]
            total_recall += metrics.get("context_recall", 0.0)
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": answer,
                "retrieved_contexts": contexts,
                "metrics": metrics
            })
            
        except Exception as e:
            logger.error(f"Failed to evaluate test case {idx+1}: {e}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "error": str(e),
                "metrics": {"faithfulness": 0.0, "context_precision": 0.0, "context_recall": 0.0}
            })

    # Compute averages
    count = len(golden_set)
    summary = {
        "average_faithfulness": round(total_faithfulness / count, 4) if count else 0.0,
        "average_context_precision": round(total_precision / count, 4) if count else 0.0,
        "average_context_recall": round(total_recall / count, 4) if count else 0.0,
        "total_questions": count,
        "model_used": model
    }
    
    report = {
        "summary": summary,
        "results": results
    }
    
    save_results(settings.eval_results_path, report)
    logger.info("--- EVALUATION REPORT ---")
    logger.info(json.dumps(summary, indent=2))
    
    # Check for regression against thresholds
    logger.info("Checking regression thresholds...")
    regression_detected = False
    
    if summary["average_faithfulness"] < settings.min_faithfulness:
        logger.error(
            f"REGRESSION DETECTED: average_faithfulness ({summary['average_faithfulness']}) "
            f"is below threshold ({settings.min_faithfulness})"
        )
        regression_detected = True
        
    if summary["average_context_precision"] < settings.min_context_precision:
        logger.error(
            f"REGRESSION DETECTED: average_context_precision ({summary['average_context_precision']}) "
            f"is below threshold ({settings.min_context_precision})"
        )
        regression_detected = True
        
    if summary["average_context_recall"] < settings.min_context_recall:
        logger.error(
            f"REGRESSION DETECTED: average_context_recall ({summary['average_context_recall']}) "
            f"is below threshold ({settings.min_context_recall})"
        )
        regression_detected = True
        
    if regression_detected:
        logger.critical("Regression check failed! Terminating execution with code 1.")
        exit(1)
        
    logger.info("All regression threshold checks passed successfully.")
    logger.info("Evaluation runner finished.")

if __name__ == "__main__":
    main()
