# src/retrieval/embeddings.py
"""
Embedding generator for dense and sparse vectors.
- Dense: Cohere embed-v4.0 (semantic meaning)
- Sparse: BM25 via FastEmbed (keyword matching)
"""

from fastembed import SparseTextEmbedding
from src.config import settings
# from src.core.logging import  get_logger
from src.llmproviders import get_provider

#logger = get_#logger(__name__, settings.log_level)


class EmbeddingGenerator:
    """
    Creates embeddings for text.
    
    Two types of embeddings:
    1. Dense: Captures meaning (good for "What is AI?" matching "Artificial Intelligence")
    2. Sparse: Captures keywords (good for exact term matching)
    
    Example:
        embedder = EmbeddingGenerator()
        dense = embedder.embed_dense(["Hello world"])
        sparse = embedder.embed_sparse(["Hello world"])
    """
    
    def __init__(self):
        """Initialize embedding models."""
        # Dense embeddings from Cohere
        self.provider = get_provider("cohere")
        
        # Sparse embeddings from FastEmbed (BM25)
        self.sparse_model = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )
        
        #logger.info("EmbeddingGenerator initialized")
    
    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """
        Create dense embeddings for documents.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = self.provider.embed(texts)
        #logger.debug(f"Created {len(embeddings)} dense embeddings")
        
        return embeddings
    
    def embed_dense_query(self, query: str) -> list[float]:
        """
        Create dense embedding for a search query.
        
        Note: Queries are embedded differently than documents!
        
        Args:
            query: Search query text
            
        Returns:
            Single embedding vector
        """
        embedding = self.provider.embed_query(query)
        #logger.debug("Created query dense embedding")
        
        return embedding
    
    def embed_sparse(self, texts: list[str]) -> list[dict]:
        """
        Create sparse embeddings (BM25) for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sparse vectors (each has 'indices' and 'values')
        """
        if not texts:
            return []
        
        # FastEmbed returns a generator, convert to list
        embeddings = list(self.sparse_model.embed(texts))
        
        # Convert to dict format for Qdrant
        sparse_vectors = []
        for emb in embeddings:
            sparse_vectors.append({
                "indices": emb.indices.tolist(),
                "values": emb.values.tolist()
            })
        
        #logger.debug(f"Created {len(sparse_vectors)} sparse embeddings")
        
        return sparse_vectors
    
    def embed_sparse_query(self, query: str) -> dict:
        """
        Create sparse embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Sparse vector dict with 'indices' and 'values'
        """
        embeddings = list(self.sparse_model.query_embed(query))
        
        if embeddings:
            emb = embeddings[0]
            return {
                "indices": emb.indices.tolist(),
                "values": emb.values.tolist()
            }
        
        return {"indices": [], "values": []}