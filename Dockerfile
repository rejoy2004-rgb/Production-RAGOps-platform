# Use Python 3.11 Slim base image
FROM python:3.11-slim

WORKDIR /app

# Create directory structure
RUN mkdir -p /app/data /app/data/sample_docs

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

COPY requirements.txt .

# Filter out heavy dependencies (torch, sentence-transformers, langchain-huggingface)
# to keep the Docker image extremely lightweight (~150MB instead of 3.5GB)
# and build in under 15 seconds.
# The app will automatically run using free API-based fallback embeddings in Docker!
RUN grep -v -E "torch|sentence-transformers|langchain-huggingface" requirements.txt > requirements-light.txt && \
    pip install --no-cache-dir -r requirements-light.txt && \
    rm -rf /root/.cache

# Copy source code and scripts
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Expose port
EXPOSE 8000

# Command to run FastAPI server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]