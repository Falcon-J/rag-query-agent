"""
PERSON 4 TASK: FastAPI Integration + API Routes
- FastAPI application setup
- /query endpoint implementation  
- Request/response models
- Integration of all components
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncio
from datetime import datetime
import uvicorn

from rag import RAGPipeline
from utils import DocumentProcessor
from config import API_HOST, API_PORT

# Initialize FastAPI app
app = FastAPI(
    title="HackRx RAG Query Agent",
    description="Retrieval-Augmented Generation API for document querying",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask", min_length=1, max_length=1000)
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    score_threshold: Optional[float] = Field(0.7, description="Minimum similarity score", ge=0.0, le=1.0)

class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., description="List of questions to ask", max_items=10)
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve per query", ge=1, le=20)

class QueryResponse(BaseModel):
    query: str
    answer: str
    status: str
    metadata: Dict
    timestamp: str

class BatchQueryResponse(BaseModel):
    results: List[QueryResponse]
    total_queries: int
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class UploadResponse(BaseModel):
    message: str
    processed_documents: int
    timestamp: str

# API Endpoints

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using RAG pipeline
    
    This endpoint:
    1. Searches for relevant documents in Pinecone
    2. Generates context-aware answer using LLM
    3. Returns structured response with metadata
    """
    try:
        # Process query through RAG pipeline
        result = await rag_pipeline.query(
            user_query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )
        
        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/query/batch", response_model=BatchQueryResponse)
async def batch_query_documents(request: BatchQueryRequest):
    """
    Process multiple queries in batch
    """
    try:
        # Process all queries
        results = await rag_pipeline.batch_query(request.queries)
        
        # Add timestamps to each result
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
    Upload and process new documents
    This endpoint processes PDFs in the data directory
    """
    try:
        def process_documents():
            processor = DocumentProcessor()
            processed_docs = processor.process_directory("../data")
            
            # Upload to Pinecone
            from utils import upload_to_pinecone
            upload_to_pinecone(processed_docs)
            
            return len(processed_docs)
        
        # Run document processing in background
        background_tasks.add_task(process_documents)
        
        return UploadResponse(
            message="Document processing started in background",
            processed_documents=0,  # Will be updated after processing
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading documents: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
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

# PERSON 4: Add submission endpoint for HackRx
@app.post("/submit")
async def submit_to_hackrx(query_request: QueryRequest):
    """
    Submit query result to HackRx evaluation API
    Modify this endpoint based on HackRx submission requirements
    """
    try:
        # Process query
        result = await rag_pipeline.query(
            user_query=query_request.query,
            top_k=query_request.top_k,
            score_threshold=query_request.score_threshold
        )
        
        # Format for HackRx submission
        submission_payload = {
            "team_id": "your-team-id",  # Update with actual team ID
            "query": query_request.query,
            "answer": result["response"]["answer"],
            "timestamp": datetime.now().isoformat()
        }
        
        # TODO: Submit to HackRx API
        # response = await submit_to_hackrx_api(submission_payload)
        
        return {
            "status": "submitted",
            "query": query_request.query,
            "answer": result["response"]["answer"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting to HackRx: {str(e)}"
        )

# Error handlers
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
