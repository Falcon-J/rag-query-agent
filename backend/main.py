from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncio
from datetime import datetime
import uvicorn

# Assume these are your project's custom modules
from rag import RAGPipeline
from utils import DocumentProcessor
from config import API_HOST, API_PORT

# Initialize FastAPI app
app = FastAPI(
    title="HackRx RAG Query Agent",
    description="Retrieval-Augmented Generation API for document querying",
    version="1.0.0"
)

# Add CORS middleware for a permissive policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline (assuming it's a synchronous or async-compatible class)
rag_pipeline = RAGPipeline()

# Pydantic models for API requests and responses

# Main query endpoint models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask", min_length=1, max_length=1000)
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    score_threshold: Optional[float] = Field(0.7, description="Minimum similarity score", ge=0.0, le=1.0)

class QueryResponse(BaseModel):
    query: str
    answer: str
    status: str
    metadata: Dict
    timestamp: str

# Batch query endpoint models
class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., description="List of questions to ask", max_items=10)
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve per query", ge=1, le=20)

class BatchQueryResponse(BaseModel):
    results: List[QueryResponse]
    total_queries: int
    timestamp: str

# Document upload models
class UploadResponse(BaseModel):
    message: str
    processed_documents: int
    timestamp: str

# Health check model
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# HackRx submission webhook models
class HackRxSubmissionRequest(BaseModel):
    query: str = Field(..., description="The query from HackRx evaluation system")
    request_id: str = Field(..., description="Unique ID for this request")
    team_id: str = "your-team-id" # Update with your actual team ID

class HackRxSubmissionResponse(BaseModel):
    request_id: str
    team_id: str
    answer: str
    justification: str
    status: str

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to ensure the service is running.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using the RAG pipeline.
    
    1. Searches for relevant documents.
    2. Generates a context-aware answer using an LLM.
    3. Returns a structured JSON response with metadata.
    """
    try:
        result = await rag_pipeline.query(
            user_query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )
        
        result["timestamp"] = datetime.now().isoformat()
        
        # Ensure the result dictionary matches the QueryResponse model
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/query/batch", response_model=BatchQueryResponse)
async def batch_query_documents(request: BatchQueryRequest):
    """
    Process multiple queries in a batch.
    """
    try:
        results = await rag_pipeline.batch_query(request.queries)
        
        timestamp = datetime.now().isoformat()
        for result in results:
            result["timestamp"] = timestamp
        
        return BatchQueryResponse(
            results=[QueryResponse(**result) for result in results],
            total_queries=len(results),
            timestamp=timestamp
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch queries: {str(e)}"
        )

@app.post("/upload", response_model=UploadResponse)
async def upload_documents(background_tasks: BackgroundTasks):
    """
    Upload and process new documents from the data directory.
    This process runs in the background to avoid blocking the API.
    """
    try:
        def process_and_upload():
            processor = DocumentProcessor()
            processed_docs = processor.process_directory("../data")
            
            # Assuming a utility function exists to handle vector DB upload
            from utils import upload_to_pinecone
            upload_to_pinecone(processed_docs)
            
            return len(processed_docs)
        
        background_tasks.add_task(process_and_upload)
        
        return UploadResponse(
            message="Document processing started in background",
            processed_documents=0,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading documents: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Get system statistics, such as Pinecone index details."""
    try:
        # Assuming RAGPipeline has a Pinecone client for stats
        stats = rag_pipeline.pinecone_client.get_index_stats()
        return {
            "pinecone_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )

@app.post("/hackrx/submit", response_model=HackRxSubmissionResponse)
async def hackrx_submission_webhook(request: HackRxSubmissionRequest):
    """
    Webhook endpoint, when deploy we will submit url: with path /hackrx/submit
    """
    try:
        result = await rag_pipeline.query(
            user_query=request.query,
            top_k=5, 
            score_threshold=0.7
        )

        answer = result.get("response", {}).get("answer", "No answer found.")
        justification = result.get("response", {}).get("rationale", "No rationale provided.")

        return HackRxSubmissionResponse(
            request_id=request.request_id,
            team_id=request.team_id,
            answer=answer,
            justification=justification,
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing HackRx submission: {str(e)}"
        )

#pending: it returns a Json as given in HackrXSubmissionResponse so need to modify llm response to fit the payload format.

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "timestamp": datetime.now().isoformat()}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    print("Starting HackRx RAG Query Agent...")
    print(f"API Documentation: http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(
        app, 
        host=API_HOST, 
        port=API_PORT,
        reload=True
    )