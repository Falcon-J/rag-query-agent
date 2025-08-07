"""
PERSON 2 TASK: Pinecone Vector Search + Filtering
- Pinecone connection and configuration
- Vector similarity search implementation
- Result filtering and ranking
"""

import pinecone
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from config import (
    PINECONE_API_KEY, 
    PINECONE_ENV, 
    PINECONE_INDEX,
    EMBEDDING_MODEL,
    TOP_K_RESULTS
)

class PineconeClient:
    def __init__(self):
        # Initialize Pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV
        )
        
        self.index_name = PINECONE_INDEX
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Create index if it doesn't exist
        self._create_index_if_not_exists()
        
        # Get index reference
        self.index = pinecone.Index(self.index_name)
    
    def _create_index_if_not_exists(self):
        """Create Pinecone index if it doesn't exist"""
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=384,  # all-MiniLM-L6-v2 embedding dimension
                metric="cosine"
            )
            print(f"Created Pinecone index: {self.index_name}")
        else:
            print(f"Using existing Pinecone index: {self.index_name}")
    
    def upload_embeddings(self, embeddings: np.ndarray, chunks: List[str], metadata: Dict):
        """Upload embeddings to Pinecone with metadata"""
        vectors = []
        
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            vector_id = f"{metadata.get('document', 'doc')}_{i}"
            
            vector_data = {
                "id": vector_id,
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk,
                    "document": metadata.get("document", ""),
                    "chunk_index": i
                }
            }
            vectors.append(vector_data)
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"Uploaded {len(vectors)} vectors to Pinecone")
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query"""
        embedding = self.embedding_model.encode([query])[0]
        return embedding.tolist()
    
    def search_similar(self, query: str, top_k: int = TOP_K_RESULTS, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents in Pinecone"""
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Search in Pinecone
        search_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        formatted_results = []
        for match in search_results.matches:
            result = {
                "id": match.id,
                "score": float(match.score),
                "text": match.metadata.get("text", ""),
                "document": match.metadata.get("document", ""),
                "chunk_index": match.metadata.get("chunk_index", 0)
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def search_with_filters(self, query: str, document_filter: Optional[str] = None, 
                          score_threshold: float = 0.7, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """Search with additional filtering options"""
        
        # Build filter dictionary
        filter_dict = {}
        if document_filter:
            filter_dict["document"] = {"$eq": document_filter}
        
        # Get initial results
        results = self.search_similar(query, top_k=top_k*2, filter_dict=filter_dict)
        
        # Filter by score threshold
        filtered_results = [r for r in results if r["score"] >= score_threshold]
        
        # Return top k results
        return filtered_results[:top_k]
    
    def get_context_from_results(self, results: List[Dict], max_length: int = 4000) -> str:
        """Combine search results into context for LLM"""
        context_parts = []
        current_length = 0
        
        for result in results:
            text = result["text"]
            if current_length + len(text) <= max_length:
                context_parts.append(f"[Score: {result['score']:.3f}] {text}")
                current_length += len(text)
            else:
                break
        
        return "\n\n".join(context_parts)
    
    def delete_document(self, document_path: str):
        """Delete all vectors for a specific document"""
        # Get all vector IDs for this document
        filter_dict = {"document": {"$eq": document_path}}
        
        # Query to get IDs (without vectors to save bandwidth)
        results = self.index.query(
            vector=[0] * 384,  # Dummy vector
            top_k=10000,  # Large number to get all
            include_metadata=True,
            filter=filter_dict
        )
        
        # Extract IDs and delete
        ids_to_delete = [match.id for match in results.matches]
        if ids_to_delete:
            self.index.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} vectors for document: {document_path}")
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the Pinecone index"""
        return self.index.describe_index_stats()

# PERSON 2: Add any additional search optimization functions here

def optimize_search_results(results: List[Dict], query: str) -> List[Dict]:
    """
    Additional result optimization (semantic similarity, etc.)
    """
    # TODO: Implement additional ranking logic
    # - Semantic similarity scoring
    # - Query-specific filtering
    # - Result clustering/deduplication
    
    return results

if __name__ == "__main__":
    # Test the Pinecone client
    client = PineconeClient()
    
    # Test search
    test_query = "What is machine learning?"
    results = client.search_similar(test_query)
    
    print(f"Search results for: '{test_query}'")
    for result in results:
        print(f"Score: {result['score']:.3f} | Text: {result['text'][:100]}...")
    
    # Print index stats
    stats = client.get_index_stats()
    print(f"\nIndex stats: {stats}")
