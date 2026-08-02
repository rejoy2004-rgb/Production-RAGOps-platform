#converts text into embeddings using local or API-based methods
import httpx
from typing import List, Optional
from src.config.config import settings
from src.utils.logger import setup_logger

logger = setup_logger("embedder")

# Check if local embedding dependencies are available
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HAS_LOCAL = True
except ImportError:
    logger.warning("langchain-huggingface/torch not installed. Using API-based embeddings fallback.")
    HAS_LOCAL = False

class Embedder:
    """Wrapper class for generating embeddings locally or via free APIs."""
    
    def __init__(self, model_name: str = settings.embedding_model):
        self.model_name = model_name
        self.embeddings = None
        
        if HAS_LOCAL:
            logger.info(f"Initializing local HuggingFace embeddings: {model_name}")
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"}
                )
            except Exception as e:
                logger.error(f"Failed to initialize local embeddings: {e}. Falling back to API mode.")
                self.embeddings = None

    def _embed_via_gemini_api(self, text: str) -> List[float]:
        """Calls Google Gemini API for free embeddings."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={settings.gemini_api_key}"
        payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]}
        }
        with httpx.Client(timeout=15.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()["embedding"]["values"]

    def _embed_via_hf_inference_api(self, text: str) -> List[float]:
        """Calls Hugging Face serverless Inference API for free embeddings."""
        # Use a default model name or format it
        model = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
        url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/{model}"
        
        headers = {}
        # If the user has a HF token in environment, we can optionally use it to raise rate limits
        hf_token = settings.openrouter_api_key  # Fallback to key checks if applicable, or run unauthenticated
        
        payload = {"inputs": [text]}
        with httpx.Client(timeout=15.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            res_json = response.json()
            # Feature extraction API returns a list of floats (embedding) for the input text
            # Depending on API return shape, handle single nested list
            if isinstance(res_json, list) and len(res_json) > 0:
                if isinstance(res_json[0], list):
                    return res_json[0]
                return res_json
            raise ValueError(f"Unexpected HF API response: {res_json}")

    def embed_query(self, text: str) -> List[float]:
        """Generates embedding for a query string."""
        if self.embeddings:
            return self.embeddings.embed_query(text)
            
        # API Fallback Modes
        if settings.gemini_api_key:
            try:
                return self._embed_via_gemini_api(text)
            except Exception as e:
                logger.error(f"Gemini API embedding failed: {e}. Trying Hugging Face Inference API...")
                
        # Hugging Face serverless API fallback
        try:
            return self._embed_via_hf_inference_api(text)
        except Exception as e:
            logger.critical(f"All embedding methods failed. Error: {e}")
            raise e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for multiple document chunks."""
        if self.embeddings:
            return self.embeddings.embed_documents(texts)
            
        # Loop over texts for API generation
        logger.info(f"Generating embeddings for {len(texts)} chunks via API...")
        embeddings = []
        for t in texts:
            embeddings.append(self.embed_query(t))
        return embeddings
