# src/retrieval/vector_store.py
"""
Qdrant vector store for storing and searching document chunks.
Supports both dense and sparse vectors for hybrid search.
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config import settings
# from src.core.logging import  get_logger
from src.core.models import DocumentChunk

#logger = get_#logger(__name__, settings.log_level)


# class VectorStore:
#     """
#     Manages vector storage in Qdrant.
    
#     Example:
#         store = VectorStore()
#         store.add_chunks(chunks, dense_embeddings, sparse_embeddings)
#         results = store.search_dense(query_embedding, top_k=10)
#     """
    
#     # Embedding dimensions (Cohere embed-v4.0)
#     DENSE_DIM = 1536

#     def __init__(self, qdrant_client: QdrantClient = None):
#         """Initialize Qdrant client and collection."""
#         # Create local Qdrant client
#         self.client = qdrant_client
#         self.collection_name = settings.qdrant_collection_name
        
#         # Create collection if it doesn't exist
#         self._ensure_collection()
        
#         #logger.info(f"VectorStore initialized: {self.collection_name}")
    
#     def _ensure_collection(self):
#         """Create collection if it doesn't exist."""
#         collections = self.client.get_collections().collections
#         exists = any(c.name == self.collection_name for c in collections)
        
#         if not exists:
#             self.client.create_collection(
#                 collection_name=self.collection_name,
#                 vectors_config={
#                     # Dense vectors
#                     "dense": models.VectorParams(
#                         size=self.DENSE_DIM,
#                         distance=models.Distance.COSINE
#                     )
#                 },
#                 sparse_vectors_config={
#                     # Sparse vectors (BM25)
#                     "sparse": models.SparseVectorParams()
#                 }
#             )
#             #logger.info(f"Created collection: {self.collection_name}")
    
#     def add_chunks(
#         self,
#         chunks: list[DocumentChunk],
#         dense_embeddings: list[list[float]],
#         sparse_embeddings: list[dict]
#     ) -> int:
#         """
#         Add chunks to the vector store.
        
#         Args:
#             chunks: List of document chunks
#             dense_embeddings: Dense vectors for each chunk
#             sparse_embeddings: Sparse vectors for each chunk
            
#         Returns:
#             Number of chunks added
#         """
#         if not chunks:
#             return 0
        
#         # Build points for Qdrant
#         points = []
#         print("\n\n in add chunsks vector store \n\n")
#         for i, chunk in enumerate(chunks):
#             # Create payload (metadata)
#             payload = {
#                 "chunk_id": chunk.chunk_id,
#                 "content": chunk.content,
#                 "document_name": chunk.metadata.document_name,
#                 "page_number": chunk.metadata.page_number,
#                 "language": chunk.metadata.language,
#                 "file_type": chunk.metadata.file_type,
#                 "chunk_index": chunk.chunk_index
#             }
            
#             # Create point
#             point = models.PointStruct(
#                 id=i + self._get_current_count(),  # Unique ID
#                 vector={
#                     "dense": dense_embeddings[i],
#                     "sparse": models.SparseVector(
#                         indices=sparse_embeddings[i]["indices"],
#                         values=sparse_embeddings[i]["values"]
#                     )
#                 },
#                 payload=payload
#             )
#             points.append(point)
#         print("\nfinished the loop in vector stor function \n")
#         # Upload to Qdrant
#         self.client.upsert(
#             collection_name=self.collection_name,
#             points=points
#         )
        
#         #logger.info(f"Added {len(points)} chunks to vector store")
        
#         return len(points)
    
#     def search_dense(
#         self,
#         query_embedding: list[float],
#         top_k: int = 10
#     ) -> list[dict]:
#         """
#         Search using dense (semantic) vectors.
        
#         Args:
#             query_embedding: Query dense vector
#             top_k: Number of results to return
            
#         Returns:
#             List of results with 'payload' and 'score'
#         """
#         results = self.client.query_points(
#             collection_name=self.collection_name,
#             query=models.NamedVector(
#                 name="dense",
#                 vector=query_embedding
#             ),
#             limit=top_k
#         )
        
#         return self._format_results(results)
    
#     def search_sparse(
#         self,
#         query_sparse: dict,
#         top_k: int = 10
#     ) -> list[dict]:
#         """
#         Search using sparse (keyword) vectors.
        
#         Args:
#             query_sparse: Query sparse vector
#             top_k: Number of results to return
            
#         Returns:
#             List of results with 'payload' and 'score'
#         """
#         results = self.client.search(
#             collection_name=self.collection_name,
#             query_vector=models.NamedSparseVector(
#                 name="sparse",
#                 vector=models.SparseVector(
#                     indices=query_sparse["indices"],
#                     values=query_sparse["values"]
#                 )
#             ),
#             limit=top_k
#         )
        
#         return self._format_results(results)
    
#     def _format_results(self, results) -> list[dict]:
#         """Convert Qdrant results to simple dict format."""
#         formatted = []
        
#         for result in results:
#             formatted.append({
#                 "id": result.id,
#                 "score": result.score,
#                 "payload": result.payload
#             })
        
#         return formatted
    
#     def _get_current_count(self) -> int:
#         """Get current number of points in collection."""
#         info = self.client.get_collection(self.collection_name)
#         return info.points_count
    
#     def clear(self):
#         """Delete all data in the collection."""
#         self.client.delete_collection(self.collection_name)
#         self._ensure_collection()
#         #logger.info("Vector store cleared")
    
#     def get_stats(self) -> dict:
#         """Get collection statistics."""
#         info = self.client.get_collection(self.collection_name)
#         return {
#             "total_chunks": info.points_count,
#             "collection_name": self.collection_name
#         }
    






from qdrant_client import models

class VectorStore:
    DENSE_DIM = 1536

    def __init__(self, qdrant_client: QdrantClient = None):
        self.client = qdrant_client
        self.collection_name = settings.qdrant_collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        """Standard collection setup with named vectors."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.DENSE_DIM,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        dense_embeddings: list[list[float]],
        sparse_embeddings: list[dict]
    ) -> int:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of document chunks
            dense_embeddings: Dense vectors for each chunk
            sparse_embeddings: Sparse vectors for each chunk
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Build points for Qdrant
        points = []
        print("\n\n in add chunsks vector store \n\n")
        for i, chunk in enumerate(chunks):
            # Create payload (metadata)
            payload = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "document_name": chunk.metadata.document_name,
                "page_number": chunk.metadata.page_number,
                "language": chunk.metadata.language,
                "file_type": chunk.metadata.file_type,
                "chunk_index": chunk.chunk_index
            }
            
            # Create point
            point = models.PointStruct(
                id=i + self._get_current_count(),  # Unique ID
                vector={
                    "dense": dense_embeddings[i],
                    "sparse": models.SparseVector(
                        indices=sparse_embeddings[i]["indices"],
                        values=sparse_embeddings[i]["values"]
                    )
                },
                payload=payload
            )
            points.append(point)
        print("\nfinished the loop in vector stor function \n")
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        #logger.info(f"Added {len(points)} chunks to vector store")
        
        return len(points)

    def search_hybrid(
        self,
        dense_query: list[float],
        sparse_query: dict,
        top_k: int = 10
    ) -> list[dict]:
        """
        Executes a single-call hybrid search using RRF Fusion.
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # Dense Prefetch
                models.Prefetch(
                    query=dense_query,        # just a list of floats
                    using="dense",            # name of the dense vector field
                    limit=top_k,
                ),
                # Sparse Prefetch
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query["indices"],
                        values=sparse_query["values"],
                    ),
                    using="sparse",           # name of the sparse vector field
                    limit=top_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )
        
        return self._format_results(results.points)

    def _format_results(self, results) -> list[dict]:
        """Output format remains identical to your original code."""
        formatted = []
        for result in results:
            formatted.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })
        return formatted
    
    def _get_current_count(self) -> int:
        """Get current number of points in collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count
    
    def clear(self):
        """Delete all data in the collection."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        #logger.info("Vector store cleared")
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "total_chunks": info.points_count,
            "collection_name": self.collection_name
        }