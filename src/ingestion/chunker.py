#chunks documents and generates metadata tags including chunk_id
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.ingestion.loader import Document
from src.utils.logger import setup_logger

logger = setup_logger("chunker")

class DocumentChunker:
    """Chunks documents and generates metadata tags including chunk_id."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of documents into chunks and adds metadata."""
        logger.info(f"Chunking {len(documents)} document pages/rows")
        chunked_docs = []
        
        # Track chunk counts per source file to ensure unique IDs
        source_chunk_counts = {}
        
        for doc in documents:
            chunks = self.splitter.split_text(doc.page_content)
            source = doc.metadata.get("source", "unknown")
            page_number = doc.metadata.get("page_number", 1)
            
            if source not in source_chunk_counts:
                source_chunk_counts[source] = 0
                
            for chunk_text in chunks:
                idx = source_chunk_counts[source]
                source_chunk_counts[source] += 1
                
                # Format a unique structured chunk_id
                chunk_id = f"{source}_p{page_number}_c{idx}"
                metadata = {
                    "source": source,
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                }
                # Keep other metadata fields if present (e.g. csv row index)
                for k, v in doc.metadata.items():
                    if k not in metadata:
                        metadata[k] = v
                        
                chunked_docs.append(Document(
                    page_content=chunk_text,
                    metadata=metadata
                ))
                
        logger.info(f"Generated {len(chunked_docs)} chunks from original documents")
        return chunked_docs
