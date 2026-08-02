#Stores System Behaviour and Configuration Settings for the RAGOps Platform
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Settings
    app_name: str = "RAGOps Platform"
    environment: str = "development"
    
    # API Keys
    openrouter_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # LangSmith Tracing
    langchain_tracing_v2: str = "false"
    langchain_api_key: Optional[str] = None
    langchain_project: str = "ragops-platform"
    
    # Regression Thresholds
    min_faithfulness: float = 0.80
    min_context_precision: float = 0.35
    min_context_recall: float = 0.50
    
    # Qdrant Database Settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "ragops_collection"
    
    # Embeddings and Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    vector_dimension: int = 384
    
    # LLM Settings
    default_model: str = "meta-llama/llama-3.2-3b-instruct:free"
    enable_online_evaluation: bool = False
    
    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = base_dir / "data"
    bm25_index_path: Path = data_dir / "bm25_index.pkl"
    eval_results_path: Path = data_dir / "eval_results.json"
    golden_test_set_path: Path = data_dir / "golden_test_set.json"
    
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Load configuration singleton
settings = Settings()

# Export LangSmith settings to environment variables if tracing is enabled
if settings.langchain_tracing_v2.lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    if settings.langchain_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

# If running inside Docker, map localhost Qdrant url to service container name 'qdrant'
if os.path.exists("/.dockerenv") and "localhost" in settings.qdrant_url:
    settings.qdrant_url = settings.qdrant_url.replace("localhost", "qdrant")

# Create data directories if they don't exist
os.makedirs(settings.data_dir, exist_ok=True)
os.makedirs(settings.data_dir / "sample_docs", exist_ok=True)
