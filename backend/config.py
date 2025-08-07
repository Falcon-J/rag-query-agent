import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "hackrx-rag-index")

# LLM Configuration
LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

# RAG Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5
MAX_CONTEXT_LENGTH = 4000

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
