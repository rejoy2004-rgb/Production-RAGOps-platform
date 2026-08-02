#Breaks down complex queries into simpler sub-queries using LLMs for better retrieval performance
import re
from typing import List, Optional
from src.generation.llm import LLMService
from src.utils.logger import setup_logger

logger = setup_logger("query_decomposition")

# Try to import LangSmith traceable decorator for tracing
try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type=None, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

class QueryDecomposer:
    """Uses LLM to decompose a complex query into simpler sub-queries."""
    
    def __init__(self):
        self.llm_service = LLMService()

    @traceable(name="QueryDecomposer.decompose", run_type="chain")
    def decompose(self, question: str, model: Optional[str] = None) -> List[str]:
        """Decomposes a question into sub-queries. Returns original question if simple or if decomposition fails."""
        logger.info(f"Attempting to decompose question: '{question}'")
        
        system_prompt = (
            "You are a query decomposition assistant. Your task is to break down a complex search question into "
            "simple search terms or sub-queries that can be run independently to gather complete information. "
            "Output the sub-queries one per line. Do not write any introduction, numbering, or explanation. "
            "If the question is already simple and cannot be broken down further, return only the original question."
        )
        
        prompt = f"Question to decompose: \"{question}\"\n\nSub-queries:"
        
        try:
            response = self.llm_service.call_llm(prompt=prompt, system_prompt=system_prompt, model=model)
            
            sub_queries = []
            for line in response.split("\n"):
                line = line.strip()
                # Remove common list bullet markers like -, *, •
                line = line.lstrip("-*•").strip()
                # Remove common numbering formats like "1. ", "2) "
                line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                
                # Strip wrapping quotes if any
                line = line.strip('"\'')
                
                if line:
                    sub_queries.append(line)
            
            # Fallback to original question if parsing returned empty or the response was trivial
            if not sub_queries:
                logger.info("Decomposition returned empty list. Using original question.")
                return [question]
                
            logger.info(f"Decomposed queries: {sub_queries}")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Error decomposing query: {e}. Falling back to original question.")
            return [question]
