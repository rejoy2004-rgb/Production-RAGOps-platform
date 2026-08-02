#Creation of Web Application using FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config.config import settings
from src.api.routes import router
from src.utils.logger import setup_logger

logger = setup_logger("app")

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="Production-Ready RAGOps Platform with Hybrid Search, Monitoring, and Evaluation",
        version="1.0.0"
    )
    
    # Configure CORS for developer ease
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount routes
    app.include_router(router)
    
    logger.info("FastAPI App initialized successfully.")
    return app

app = create_app()
