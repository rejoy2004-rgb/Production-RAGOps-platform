import os
import csv
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from pypdf import PdfReader
from src.utils.logger import setup_logger

logger = setup_logger("loader")

class Document(BaseModel):
    """Standardized Document representation for ingestion."""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentLoader:
    """Loader utility to read documents from multiple file formats."""
    
    @staticmethod
    def load_txt(file_path: Path) -> List[Document]:
        """Loads a plaintext file."""
        logger.info(f"Loading TXT file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return [Document(
                page_content=content,
                metadata={
                    "source": os.path.basename(file_path),
                    "page_number": 1
                }
            )]
        except Exception as e:
            logger.error(f"Error loading TXT file {file_path}: {e}")
            raise e

    @staticmethod
    def load_pdf(file_path: Path) -> List[Document]:
        """Loads a PDF file page by page."""
        logger.info(f"Loading PDF file: {file_path}")
        documents = []
        try:
            reader = PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(file_path),
                        "page_number": idx + 1
                    }
                ))
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            raise e

    @staticmethod
    def load_csv(file_path: Path) -> List[Document]:
        """Loads a CSV file. Each row is converted to text representation."""
        logger.info(f"Loading CSV file: {file_path}")
        documents = []
        try:
            with open(file_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    # Convert row to key:value format
                    row_content = ", ".join([f"{k}: {v}" for k, v in row.items() if v])
                    documents.append(Document(
                        page_content=row_content,
                        metadata={
                            "source": os.path.basename(file_path),
                            "page_number": 1,
                            "row_index": idx
                        }
                    ))
            return documents
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise e

    def load_document(self, file_path: Path) -> List[Document]:
        """Auto-detects format and loads the document."""
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return self.load_txt(file_path)
        elif suffix == ".pdf":
            return self.load_pdf(file_path)
        elif suffix == ".csv":
            return self.load_csv(file_path)
        else:
            logger.warning(f"Unsupported file type: {suffix} for file {file_path}")
            return []
