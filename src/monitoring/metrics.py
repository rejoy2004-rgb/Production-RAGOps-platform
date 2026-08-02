#Answers what is happening in the RAG system when a user visits an endpoint, including metrics collection for Prometheus monitoring.
from prometheus_client import Counter, Histogram, Gauge

# 1. Total RAG Request Count
# Labels: status (success, error), model (openai/gpt-4o-mini, etc.)
rag_request_count_total = Counter(
    "rag_request_count_total",
    "Total count of RAG requests processed",
    ["status", "model"]
)

# 2. RAG Request Latency in Seconds
# Labels: model
rag_request_latency_seconds = Histogram(
    "rag_request_latency_seconds",
    "RAG query execution latency in seconds",
    ["model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, float("inf"))
)

# 3. RAG Faithfulness Score
# Labels: model
rag_faithfulness_score = Gauge(
    "rag_faithfulness_score",
    "Faithfulness score of the generated answer relative to retrieved context (0.0 to 1.0)",
    ["model"]
)

# 4. RAG Context Precision Score
# Labels: model
rag_context_precision_score = Gauge(
    "rag_context_precision_score",
    "Precision score of the retrieved context relative to the query (0.0 to 1.0)",
    ["model"]
)

# 5. RAG Context Recall Score
# Labels: model
rag_context_recall_score = Gauge(
    "rag_context_recall_score",
    "Recall score of the retrieved context relative to the ground truth answer (0.0 to 1.0)",
    ["model"]
)
