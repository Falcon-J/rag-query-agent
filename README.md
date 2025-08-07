# HackRx RAG Query Agent

A 4-member team project for building a Retrieval-Augmented Generation (RAG) system that accepts queries, retrieves relevant documents from Pinecone, and generates answers using LLMs.

## Project Structure

```
hackrx-rag-query-agent/
├── backend/
│   ├── main.py               # FastAPI API entrypoint
│   ├── rag.py                # Handles embedding, retrieval, generation
│   ├── pinecone_client.py    # Connect to Pinecone and query
│   ├── llm_client.py         # Query LLM endpoint (Groq or Together.ai)
│   ├── utils.py              # For text cleaning, PDF parsing, etc.
│   └── config.py             # Loads environment variables
├── data/                     # Raw PDF documents
├── .env.example              # Environment variables template
├── requirements.txt
└── README.md
```

## Team Task Division

### Person 1: Data Parsing + Embedding Upload

**Files:** `utils.py`, `data/` directory, embedding upload script

- PDF text extraction and cleaning
- Text chunking for optimal retrieval
- Generate embeddings and upload to Pinecone
- Data preprocessing pipeline

### Person 2: Pinecone Vector Search + Filtering

**Files:** `pinecone_client.py`

- Pinecone connection and configuration
- Vector similarity search implementation
- Result filtering and ranking
- Query optimization

### Person 3: LLM Client + JSON Output

**Files:** `llm_client.py`, response formatting

- OpenRouter/Groq API integration
- Prompt engineering for RAG
- Response parsing and JSON formatting
- Error handling for LLM calls

### Person 4: FastAPI Integration + API Routes

**Files:** `main.py`, `rag.py`, final integration

- FastAPI application setup
- `/query` endpoint implementation
- Request/response models
- Integration of all components
- Submission API integration

## Quick Start

1. Copy `.env.example` to `.env` and fill in your API keys
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `uvicorn backend.main:app --reload`
4. Test the API at `http://localhost:8000/docs`

## API Endpoints

- `GET /` - Health check
- `POST /query` - Submit query and get RAG response
- `POST /upload` - Upload and process new documents (optional)

## Environment Variables

See `.env.example` for required configuration.
"# rag-query-agent" 
