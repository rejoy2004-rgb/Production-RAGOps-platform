#Contains shared utilities for logging and monitoring across the RAG system, ensuring consistent logging format and metrics collection.
import logging
import sys

def setup_logger(name: str) -> logging.Logger:
    """Sets up a structured console logger."""
    # Reconfigure streams to use utf-8 with replacement for unrecognized characters to prevent UnicodeEncodeError on Windows terminals
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
                
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
