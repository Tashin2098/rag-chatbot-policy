"""
Document Ingestion Module
- Parses PDF/TXT
- Chunks text with overlap
"""
import os
from typing import List, Tuple
from PyPDF2 import PdfReader

class DocumentIngestor:
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_pdf(self, filepath: str) -> str:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            pt = page.extract_text() or ""
            text += pt + "\n"
        return text

    def load_txt(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(words), step):
            block = " ".join(words[i:i + self.chunk_size]).strip()
            if len(block) > 50:
                chunks.append(block)
        return chunks

    def process_document(self, filepath: str) -> Tuple[List[str], str]:
        filename = os.path.basename(filepath)
        if filepath.endswith(".pdf"):
            text = self.load_pdf(filepath)
        elif filepath.endswith(".txt"):
            text = self.load_txt(filepath)
        else:
            return [], filename
        chunks = self.chunk_text(text)
        return chunks, filename
