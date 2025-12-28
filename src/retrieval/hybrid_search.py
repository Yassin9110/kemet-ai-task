# src/retrieval/hybrid_search.py
"""
Hybrid search combining dense and sparse retrieval.
Uses RRF (Reciprocal Rank Fusion) to merge results.
"""

from src.config import settings
# from src.core.logging import  get_logger
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore

#logger = get_#logger(__name__, settings.log_level)


# class HybridSearcher:
#     """
#     Combines dense (semantic) and sparse (keyword) search.
    
#     Why hybrid?
#     - Dense: Good at understanding meaning ("car" matches "automobile")
#     - Sparse: Good at exact matches ("COVID-19" matches "COVID-19")
#     - Together: Best of both worlds!
    
#     Example:
#         searcher = HybridSearcher()
#         results = searcher.search("What is machine learning?", top_k=10)
#     """
    
#     def __init__(
#         self,
#         embedder: EmbeddingGenerator = None,
#         vector_store: VectorStore = None
#     ):
#         """
#         Initialize hybrid searcher.
        
#         Args:
#             embedder: Embedding generator (created if not provided)
#             vector_store: Vector store (created if not provided)
#         """
#         self.embedder = embedder or EmbeddingGenerator()
#         self.vector_store = vector_store or VectorStore()
        
#         #logger.info("HybridSearcher initialized")
    
#     def search(
#         self,
#         query: str,
#         top_k: int = None,
#         dense_weight: float = 0.5
#     ) -> list[dict]:
#         """
#         Perform hybrid search.
        
#         Args:
#             query: Search query text
#             top_k: Number of results to return
#             dense_weight: Weight for dense results (0-1)
#                          0 = only sparse, 1 = only dense, 0.5 = equal
            
#         Returns:
#             List of results sorted by combined score
#         """
#         top_k = top_k or settings.top_k_retrieval
        
#         #logger.info(f"Hybrid search: '{query[:50]}...' (top_k={top_k})")
        
#         # Step 1: Create query embeddings
#         dense_query = self.embedder.embed_dense_query(query)
#         sparse_query = self.embedder.embed_sparse_query(query)
        
#         # Step 2: Search with both methods
#         dense_results = self.vector_store.search_dense(dense_query, top_k=top_k)
#         sparse_results = self.vector_store.search_sparse(sparse_query, top_k=top_k)
        
#         #logger.debug(f"Dense results: {len(dense_results)}, Sparse results: {len(sparse_results)}")
        
#         # Step 3: Combine using RRF
#         combined = self._rrf_fusion(
#             dense_results=dense_results,
#             sparse_results=sparse_results,
#             dense_weight=dense_weight,
#             top_k=top_k
#         )
        
#         #logger.info(f"Hybrid search returned {len(combined)} results")
        
#         return combined
    
#     def _rrf_fusion(
#         self,
#         dense_results: list[dict],
#         sparse_results: list[dict],
#         dense_weight: float,
#         top_k: int
#     ) -> list[dict]:
#         """
#         Combine results using Reciprocal Rank Fusion.
        
#         RRF Formula: score = weight / (k + rank)
#         Where k=60 is a constant that reduces the impact of high ranks.
        
#         Args:
#             dense_results: Results from dense search
#             sparse_results: Results from sparse search
#             dense_weight: Weight for dense results
#             top_k: Number of results to return
            
#         Returns:
#             Combined and sorted results
#         """
#         k = 60  # RRF constant
#         scores = {}  # id -> {"score": float, "payload": dict}
#         sparse_weight = 1 - dense_weight
        
#         # Score dense results
#         for rank, result in enumerate(dense_results, start=1):
#             doc_id = result["id"]
#             rrf_score = dense_weight / (k + rank)
            
#             if doc_id not in scores:
#                 scores[doc_id] = {
#                     "score": 0,
#                     "payload": result["payload"]
#                 }
#             scores[doc_id]["score"] += rrf_score
        
#         # Score sparse results
#         for rank, result in enumerate(sparse_results, start=1):
#             doc_id = result["id"]
#             rrf_score = sparse_weight / (k + rank)
            
#             if doc_id not in scores:
#                 scores[doc_id] = {
#                     "score": 0,
#                     "payload": result["payload"]
#                 }
#             scores[doc_id]["score"] += rrf_score
        
#         # Sort by score and return top_k
#         sorted_results = sorted(
#             [{"id": k, **v} for k, v in scores.items()],
#             key=lambda x: x["score"],
#             reverse=True
#         )
        
#         return sorted_results[:top_k]



class HybridSearcher:
    """
    Combines dense (semantic) and sparse (keyword) search using Qdrant's native Fusion.
    """
    
    def __init__(
        self,
        embedder: EmbeddingGenerator = None,
        vector_store: VectorStore = None
    ):
        self.embedder = embedder or EmbeddingGenerator()
        self.vector_store = vector_store or VectorStore()
    
    def search(
        self,
        query: str,
        top_k: int = None,
        dense_weight: float = 0.5  # Note: Native RRF weighs rank sets equally
    ) -> list[dict]:
        """
        Perform hybrid search using native Qdrant RRF.
        """
        # Fallback to settings if top_k is not provided
        search_limit = top_k or settings.top_k_retrieval
        
        # Step 1: Create query embeddings
        dense_query = self.embedder.embed_dense_query(query)
        sparse_query = self.embedder.embed_sparse_query(query)
        
        # Step 2: Search using native fusion
        # The fusion happens inside the VectorStore now
        combined = self.vector_store.search_hybrid(
            dense_query=dense_query,
            sparse_query=sparse_query,
            top_k=search_limit
        )
        
        return combined