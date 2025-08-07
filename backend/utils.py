"""
PERSON 1 TASK: Data Parsing + Embedding Upload
- PDF text extraction and cleaning
- Text chunking for optimal retrieval
- Generate embeddings and upload to Pinecone
"""

import os
import fitz  # PyMuPDF
import pdfplumber
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_text_with_pdfplumber(self, pdf_path: str) -> str:
        """Alternative PDF extraction using pdfplumber"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove special characters but keep important punctuation
        import re
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Remove very short lines (likely headers/footers)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(cleaned_lines)
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for embedding"""
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        return embeddings
    
    def process_document(self, pdf_path: str) -> Dict:
        """Complete document processing pipeline"""
        print(f"Processing document: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            text = self.extract_text_with_pdfplumber(pdf_path)
        
        if not text:
            raise ValueError(f"Could not extract text from {pdf_path}")
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Chunk text
        chunks = self.chunk_text(cleaned_text)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        return {
            "document_path": pdf_path,
            "chunks": chunks,
            "embeddings": embeddings,
            "chunk_count": len(chunks)
        }
    
    def process_directory(self, data_dir: str) -> List[Dict]:
        """Process all PDFs in a directory"""
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        processed_docs = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(data_dir, pdf_file)
            try:
                doc_data = self.process_document(pdf_path)
                processed_docs.append(doc_data)
            except Exception as e:
                print(f"Failed to process {pdf_file}: {e}")
        
        return processed_docs

# PERSON 1: Add your upload to Pinecone function here
def upload_to_pinecone(processed_docs: List[Dict]):
    """
    TODO: Implement Pinecone upload
    This should be coordinated with Person 2's pinecone_client.py
    """
    from pinecone_client import PineconeClient
    
    client = PineconeClient()
    for doc in processed_docs:
        client.upload_embeddings(
            embeddings=doc["embeddings"],
            chunks=doc["chunks"],
            metadata={"document": doc["document_path"]}
        )

if __name__ == "__main__":
    # Test the document processor
    processor = DocumentProcessor()
    
    # Process all documents in data directory
    data_dir = "../data"
    if os.path.exists(data_dir):
        processed_docs = processor.process_directory(data_dir)
        print(f"Processed {len(processed_docs)} documents")
        
        # Upload to Pinecone (uncomment when ready)
        # upload_to_pinecone(processed_docs)
    else:
        print(f"Data directory {data_dir} not found. Add PDF files to test.")
