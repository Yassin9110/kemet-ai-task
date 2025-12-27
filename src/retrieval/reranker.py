# src/retrieval/reranker.py
"""
Reranker to improve search results using Cohere's rerank model.
Takes initial results and reorders them by relevance.
"""

from src.config import settings
from src.core.logging import get_logger
from src.providers import get_provider

logger = get_logger(__name__, settings.log_level)


class Reranker:
    """
    Reranks search results for better relevance.
    
    Why rerank?
    - Initial search is fast but approximate
    - Reranking is slower but more accurate
    - We rerank only the top results (best of both worlds)
    
    Example:
        reranker = Reranker()
        better_results = reranker.rerank(query, initial_results, top_k=5)
    """
    
    def __init__(self):
        """Initialize reranker with Cohere provider."""
        self.provider = get_provider("cohere")
        logger.info("Reranker initialized")
    
    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = None
    ) -> list[dict]:
        """
        Rerank search results by relevance.
        
        Args:
            query: The search query
            results: Initial search results (must have 'payload' with 'content')
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked results (best first)
        """
        top_k = top_k or settings.rerank_top_k
        
        # Handle empty results
        if not results:
            return []
        
        # Handle case where we have fewer results than requested
        if len(results) <= top_k:
            return results
        
        logger.info(f"Reranking {len(results)} results to top {top_k}")
        
        # Step 1: Extract text content for reranking
        documents = [r["payload"]["content"] for r in results]
        
        # Step 2: Call Cohere reranker
        rerank_results = self.provider.rerank(query, documents)
        
        # Step 3: Reorder original results based on reranking
        reranked = []
        for item in rerank_results[:top_k]:
            original_index = item["index"]
            original_result = results[original_index].copy()
            original_result["rerank_score"] = item["score"]
            reranked.append(original_result)
        
        logger.info(f"Reranking complete: {len(reranked)} results")
        
        return reranked