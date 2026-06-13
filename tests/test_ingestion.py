import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.ingestion.loader import DocumentLoader, Document
from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import Embedder
from src.ingestion.pipeline import IngestionPipeline

def test_document_loader_txt():
    """Verifies that the TXT file loader works correctly."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write("This is a cancellation policy document text.")
        tmp_name = tmp.name
        
    try:
        docs = DocumentLoader.load_txt(Path(tmp_name))
        assert len(docs) == 1
        assert docs[0].page_content == "This is a cancellation policy document text."
        assert docs[0].metadata["source"] == os.path.basename(tmp_name)
        assert docs[0].metadata["page_number"] == 1
    finally:
        os.remove(tmp_name)

def test_document_loader_csv():
    """Verifies that the CSV file loader maps rows to text formatted key-values."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as tmp:
        tmp.write("Header1,Header2\nValue1,Value2\nValue3,Value4")
        tmp_name = tmp.name
        
    try:
        docs = DocumentLoader.load_csv(Path(tmp_name))
        assert len(docs) == 2
        assert docs[0].page_content == "Header1: Value1, Header2: Value2"
        assert docs[0].metadata["row_index"] == 0
        assert docs[1].page_content == "Header1: Value3, Header2: Value4"
    finally:
        os.remove(tmp_name)

def test_document_chunker():
    """Verifies document splitter and metadata creation."""
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
    doc = Document(
        page_content="abcdefghij", # 10 chars
        metadata={"source": "test.txt", "page_number": 1}
    )
    
    chunks = chunker.chunk_documents([doc])
    # Should chunk page content and add chunk_id
    assert len(chunks) >= 1
    assert chunks[0].metadata["chunk_id"] == "test.txt_p1_c0"
    assert chunks[0].metadata["source"] == "test.txt"

@patch('src.ingestion.embedder.HuggingFaceEmbeddings')
def test_embedder(mock_hf_embeddings):
    """Verifies embedder wrapper maps correctly to underlying library calls."""
    mock_instance = MagicMock()
    mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_hf_embeddings.return_value = mock_instance
    
    embedder = Embedder()
    vec = embedder.embed_query("test query")
    assert vec == [0.1, 0.2, 0.3]
    
    vecs = embedder.embed_documents(["doc text"])
    assert vecs == [[0.1, 0.2, 0.3]]
