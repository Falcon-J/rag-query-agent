"""
RAG Pipeline Integration
Coordinates between Pinecone search and LLM generation
"""

from typing import Dict, List
from pinecone_client import PineconeClient
from llm_client import LLMClient
from config import MAX_CONTEXT_LENGTH
import asyncio

class RAGPipeline:
    def __init__(self):
        self.pinecone_client = PineconeClient()
        self.llm_client = LLMClient()
    
    async def query(self, user_query: str, top_k: int = 5, score_threshold: float = 0.7) -> Dict:
        """Complete RAG pipeline: retrieve + generate"""
        
        # Step 1: Search for relevant documents
        search_results = self.pinecone_client.search_with_filters(
            query=user_query,
            score_threshold=score_threshold,
            top_k=top_k
        )
        
        if not search_results:
            return {
                "query": user_query,
                "answer": "I couldn't find any relevant information to answer your question.",
                "status": "no_results",
                "metadata": {"sources_used": 0}
            }
        
        # Step 2: Prepare context for LLM
        context = self.pinecone_client.get_context_from_results(
            search_results, 
            max_length=MAX_CONTEXT_LENGTH
        )
        
        # Step 3: Generate answer using LLM
        llm_response = await self.llm_client.generate_answer(user_query, context)
        
        # Step 4: Format final response
        final_response = self.llm_client.format_api_response(llm_response, search_results)
        
        return final_response
    
    async def batch_query(self, queries: List[str]) -> List[Dict]:
        """Process multiple queries"""
        tasks = [self.query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append({
                    "query": queries[i],
                    "answer": f"Error processing query: {str(result)}",
                    "status": "error"
                })
            else:
                formatted_results.append(result)
        
        return formatted_results
