"""
PERSON 3 TASK: LLM Client + JSON Output
- OpenRouter/Groq API integration
- Prompt engineering for RAG
- Response parsing and JSON formatting
"""

import httpx
import json
from typing import Dict, List, Optional
import asyncio
from config import LLM_API_URL, LLM_API_KEY, LLM_MODEL

class LLMClient:
    def __init__(self):
        self.api_url = LLM_API_URL
        self.api_key = LLM_API_KEY
        self.model = LLM_MODEL
        self.base_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def create_rag_prompt(self, query: str, context: str) -> str:
        """Create a well-structured prompt for RAG"""
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context above
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question."
- Be concise but comprehensive
- Cite specific parts of the context when relevant
- Format your response in clear, readable sentences

ANSWER:"""
        
        return prompt
    
    async def call_llm(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Make async call to LLM API"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.api_url}/chat/completions",
                    headers=self.base_headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
                
            except httpx.HTTPStatusError as e:
                print(f"HTTP error: {e}")
                return f"Error: Failed to get response from LLM (HTTP {e.response.status_code})"
            except Exception as e:
                print(f"LLM API error: {e}")
                return f"Error: {str(e)}"
    
    async def generate_answer(self, query: str, context: str) -> Dict:
        """Generate answer with metadata"""
        # Create RAG prompt
        prompt = self.create_rag_prompt(query, context)
        
        # Get LLM response
        answer = await self.call_llm(prompt)
        
        # Format response
        response = {
            "query": query,
            "answer": answer,
            "context_used": len(context),
            "model": self.model,
            "status": "success" if not answer.startswith("Error:") else "error"
        }
        
        return response
    
    def create_multi_step_prompt(self, query: str, context: str) -> str:
        """Advanced prompt for complex queries"""
        prompt = f"""You are an expert analyst. Please analyze the following question and context carefully.

CONTEXT:
{context}

QUESTION: {query}

Please provide a structured response with:
1. **Direct Answer**: A clear, concise answer to the question
2. **Supporting Evidence**: Key facts from the context that support your answer
3. **Confidence Level**: How confident you are in this answer (High/Medium/Low)
4. **Additional Notes**: Any relevant context or limitations

Format your response clearly with these sections."""
        
        return prompt
    
    async def generate_structured_answer(self, query: str, context: str) -> Dict:
        """Generate a more structured response"""
        prompt = self.create_multi_step_prompt(query, context)
        answer = await self.call_llm(prompt, max_tokens=800)
        
        # Try to parse structured response
        response = {
            "query": query,
            "answer": answer,
            "structured": self._parse_structured_response(answer),
            "context_length": len(context),
            "model": self.model
        }
        
        return response
    
    def _parse_structured_response(self, answer: str) -> Dict:
        """Attempt to parse structured response sections"""
        sections = {
            "direct_answer": "",
            "supporting_evidence": "",
            "confidence_level": "",
            "additional_notes": ""
        }
        
        # Simple parsing - can be improved
        lines = answer.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if "Direct Answer" in line or "1." in line:
                current_section = "direct_answer"
            elif "Supporting Evidence" in line or "2." in line:
                current_section = "supporting_evidence"
            elif "Confidence Level" in line or "3." in line:
                current_section = "confidence_level"
            elif "Additional Notes" in line or "4." in line:
                current_section = "additional_notes"
            elif current_section and line:
                sections[current_section] += line + " "
        
        return sections
    
    async def batch_generate(self, queries: List[str], context: str) -> List[Dict]:
        """Generate answers for multiple queries"""
        tasks = [self.generate_answer(query, context) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append({
                    "query": queries[i],
                    "answer": f"Error: {str(result)}",
                    "status": "error"
                })
            else:
                formatted_results.append(result)
        
        return formatted_results
    
    def format_api_response(self, llm_response: Dict, search_results: List[Dict]) -> Dict:
        """Format the final API response"""
        return {
            "response": llm_response,
            "metadata": {
                "sources_used": len(search_results),
                "top_score": max([r["score"] for r in search_results]) if search_results else 0,
                "retrieval_results": [
                    {
                        "document": r["document"],
                        "score": r["score"],
                        "snippet": r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"]
                    }
                    for r in search_results[:3]  # Top 3 sources
                ]
            },
            "timestamp": "",  # Add timestamp in main.py
        }

# PERSON 3: Add any additional LLM optimization functions here

async def optimize_prompt_for_domain(query: str, context: str, domain: str = "general") -> str:
    """
    Domain-specific prompt optimization
    """
    # TODO: Implement domain-specific prompts
    # - Technical documentation
    # - Legal documents
    # - Medical literature
    # - Academic papers
    
    return f"[{domain.upper()}] {query}"

if __name__ == "__main__":
    # Test the LLM client
    async def test_llm():
        client = LLMClient()
        
        test_query = "What is machine learning?"
        test_context = "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data."
        
        result = await client.generate_answer(test_query, test_context)
        print("Generated Response:")
        print(json.dumps(result, indent=2))
        
        # Test structured response
        structured_result = await client.generate_structured_answer(test_query, test_context)
        print("\nStructured Response:")
        print(json.dumps(structured_result, indent=2))
    
    # Run test
    asyncio.run(test_llm())
