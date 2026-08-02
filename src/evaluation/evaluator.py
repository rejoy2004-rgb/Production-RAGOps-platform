#Checks the faithfulness of generated answers, context precision, and context recall using LLMs as judges. Provides a structured evaluation framework for RAG systems.
import json
import re
from typing import List, Dict, Any, Optional
from src.generation.llm import LLMService
from src.config.config import settings
from src.utils.logger import setup_logger

logger = setup_logger("evaluator")

class RAGEvaluator:
    """LLM-as-a-judge evaluation framework for Faithfulness, Context Precision, and Context Recall."""
    
    def __init__(self):
        self.llm_service = LLMService()

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Cleans and parses JSON from the LLM response, handling markdown blocks if present."""
        cleaned = text.strip()
        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            # extract content between first ```json and last ```
            match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", cleaned)
            if match:
                cleaned = match.group(1).strip()
                
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM evaluation response as JSON: {text}. Error: {e}")
            # Try to regex extract a score if JSON parsing fails
            score_match = re.search(r'"score"\s*:\s*(0\.\d+|1\.0|0|1)', cleaned)
            if score_match:
                try:
                    return {"score": float(score_match.group(1)), "reasoning": "Extracted via regex fallback."}
                except ValueError:
                    pass
            return {"score": 0.0, "reasoning": "Failed to parse JSON."}

    def evaluate_faithfulness(self, question: str, contexts: List[str], answer: str, model: Optional[str] = None) -> float:
        """Evaluates how faithful the generated answer is to the retrieved contexts."""
        if not contexts or not answer:
            return 0.0
            
        context_str = "\n---\n".join(contexts)
        
        system_prompt = (
            "You are a strict RAG evaluation judge. Your task is to evaluate the FAITHFULNESS of a generated answer. "
            "Faithfulness measures if all facts in the generated answer are directly supported by the context. "
            "If the answer contains any statements or assumptions not present in the context, the score must be low. "
            "Provide a JSON response containing step-by-step reasoning and a score between 0.0 (completely unfaithful) and 1.0 (perfectly faithful).\n"
            "Format:\n"
            "{\n"
            "  \"reasoning\": \"Step-by-step analysis of claims in answer and checking if they exist in the context\",\n"
            "  \"score\": 0.85\n"
            "}\n"
            "Output only the JSON block."
        )
        
        prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Generated Answer: {answer}\n\n"
            f"Evaluate Faithfulness:"
        )
        
        try:
            res = self.llm_service.call_llm(prompt=prompt, system_prompt=system_prompt, model=model)
            parsed = self._parse_json_response(res)
            score = float(parsed.get("score", 0.0))
            logger.info(f"Faithfulness score: {score}. Reasoning: {parsed.get('reasoning')}")
            return score
        except Exception as e:
            logger.error(f"Error evaluating faithfulness: {e}")
            return 0.0

    def evaluate_context_precision(self, question: str, contexts: List[str], model: Optional[str] = None) -> float:
        """Evaluates if the retrieved contexts are highly relevant to the query."""
        if not contexts:
            return 0.0
            
        system_prompt = (
            "You are a strict RAG evaluation judge. Your task is to evaluate CONTEXT PRECISION. "
            "Context Precision measures what fraction of the retrieved context chunks contain information directly relevant to answering the question. "
            "Provide a JSON response containing step-by-step reasoning checking each chunk, and a score between 0.0 (none relevant) and 1.0 (all relevant).\n"
            "Format:\n"
            "{\n"
            "  \"reasoning\": \"Checking chunk 1, chunk 2... and details of their relevance\",\n"
            "  \"score\": 1.0\n"
            "}\n"
            "Output only the JSON block."
        )
        
        chunks_str = ""
        for i, c in enumerate(contexts):
            chunks_str += f"Chunk {i+1}: {c}\n\n"
            
        prompt = (
            f"Question: {question}\n\n"
            f"Retrieved Chunks:\n{chunks_str}"
            f"Evaluate Context Precision:"
        )
        
        try:
            res = self.llm_service.call_llm(prompt=prompt, system_prompt=system_prompt, model=model)
            parsed = self._parse_json_response(res)
            score = float(parsed.get("score", 0.0))
            logger.info(f"Context Precision score: {score}. Reasoning: {parsed.get('reasoning')}")
            return score
        except Exception as e:
            logger.error(f"Error evaluating context precision: {e}")
            return 0.0

    def evaluate_context_recall(self, question: str, contexts: List[str], ground_truth: str, model: Optional[str] = None) -> float:
        """Evaluates if the retrieved context contains all necessary facts present in the ground truth answer."""
        if not contexts or not ground_truth:
            return 0.0
            
        context_str = "\n---\n".join(contexts)
        
        system_prompt = (
            "You are a strict RAG evaluation judge. Your task is to evaluate CONTEXT RECALL. "
            "Context Recall measures if all critical facts/claims present in the Ground Truth Answer are successfully captured in the retrieved Context. "
            "Provide a JSON response containing step-by-step reasoning checking ground truth claims against the retrieved context, and a score between 0.0 (none recalled) and 1.0 (all recalled).\n"
            "Format:\n"
            "{\n"
            "  \"reasoning\": \"List claims in ground truth and check if they are in the context\",\n"
            "  \"score\": 0.9\n"
            "}\n"
            "Output only the JSON block."
        )
        
        prompt = (
            f"Question: {question}\n\n"
            f"Retrieved Context:\n{context_str}\n\n"
            f"Ground Truth Answer: {ground_truth}\n\n"
            f"Evaluate Context Recall:"
        )
        
        try:
            res = self.llm_service.call_llm(prompt=prompt, system_prompt=system_prompt, model=model)
            parsed = self._parse_json_response(res)
            score = float(parsed.get("score", 0.0))
            logger.info(f"Context Recall score: {score}. Reasoning: {parsed.get('reasoning')}")
            return score
        except Exception as e:
            logger.error(f"Error evaluating context recall: {e}")
            return 0.0

    def evaluate_all(self, question: str, contexts: List[str], answer: str, ground_truth: Optional[str] = None, model: Optional[str] = None) -> Dict[str, float]:
        """Calculates all RAG metrics. Context Recall is calculated only if ground truth is supplied."""
        metrics = {
            "faithfulness": self.evaluate_faithfulness(question, contexts, answer, model),
            "context_precision": self.evaluate_context_precision(question, contexts, model)
        }
        
        if ground_truth:
            metrics["context_recall"] = self.evaluate_context_recall(question, contexts, ground_truth, model)
            
        return metrics
