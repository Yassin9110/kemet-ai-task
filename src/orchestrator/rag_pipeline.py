# src/orchestrator/rag_pipeline.py
"""
Main RAG Orchestrator that ties everything together.
This is the "brain" of the system - it coordinates all components.
"""

import time

from qdrant_client import QdrantClient
from src.config import settings
# from src.core.logging import  get_logger
from src.core.models import (
    DocumentChunk,
    GenerationResult,
    IngestionResult,
    ChatMessage,
    ChatRole,
    Language
)
from src.core.language import LanguageDetector
from src.ingestion import IngestionPipeline
from src.retrieval import EmbeddingGenerator, VectorStore, HybridSearcher, Reranker
from src.generation import ResponseGenerator

#logger = get_#logger(__name__, settings.log_level)


class RAGOrchestrator:
    """
    Main orchestrator for the RAG system.
    
    Handles:
    - Document ingestion (upload -> parse -> chunk -> embed -> store)
    - Query processing (question -> retrieve -> rerank -> generate)
    - Conversation management (multi-turn chat)
    
    Example:
        rag = RAGOrchestrator()
        
        # Upload a document
        result = rag.ingest_document("doc.pdf", file_bytes)
        
        # Ask questions
        answer = rag.query("What is AI?", chat_history=[])
    """
    
    def __init__(self):
        """Initialize all components."""
        #logger.info("Initializing RAG Orchestrator...")
        self.qdrant_client = QdrantClient(path=settings.qdrant_path)
        # Core components
        self.ingestion = IngestionPipeline()
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore(self.qdrant_client)
        self.searcher = HybridSearcher(self.embedder, self.vector_store)
        self.reranker = Reranker()
        self.generator = ResponseGenerator()
        self.language_detector = LanguageDetector()
        
        
        #logger.info("RAG Orchestrator ready!")
    
    # ==================
    # DOCUMENT INGESTION
    # ==================
    
    def ingest_document(
        self,
        filename: str,
        file_bytes: bytes
    ) -> IngestionResult:
        """
        Ingest a document into the system.
        
        Flow: Parse -> Chunk -> Embed -> Store
        
        Args:
            filename: Name of the file
            file_bytes: Raw file content
            
        Returns:
            IngestionResult with status and stats
        """
        #logger.info(f"Starting ingestion: {filename}")
        start_time = time.time()
        
        # Step 1: Validate file
        is_valid, error = self.ingestion.validate_file(filename, len(file_bytes))
        if not is_valid:
            #logger.error(f"Validation failed: {error}")
            return IngestionResult(
                document_name=filename,
                total_chunks=0,
                language=Language.UNKNOWN,
                success=False,
                error_message=error,
                processing_time_ms=0
            )
        
        # Step 2: Parse and chunk
        chunks, result = self.ingestion.ingest(filename, file_bytes)
        
        if not result.success:
            return result
        
        # Step 3: Create embeddings
        #logger.info("Creating embeddings...")
        texts = [chunk.content for chunk in chunks]
        
        dense_embeddings = self.embedder.embed_dense(texts)
        sparse_embeddings = self.embedder.embed_sparse(texts)
        
        embedding_time = (time.time() - start_time) * 1000
        #logger.log_embedding(len(chunks), embedding_time)
        
        # Step 4: Store in vector database
        #logger.info("Storing in vector database...")
        self.vector_store.add_chunks(chunks, dense_embeddings, sparse_embeddings)
        
        # Update processing time
        total_time = (time.time() - start_time) * 1000
        result.processing_time_ms = total_time
        
        #logger.info(f"Ingestion complete: {len(chunks)} chunks in {total_time:.0f}ms")
        
        return result
    
    # ==================
    # QUERY PROCESSING
    # ==================
    
    def query(
        self,
        question: str,
        chat_history: list[dict] = None
    ) -> GenerationResult:
        """
        Process a user question and generate an answer.
        
        Flow: Detect Language -> Search -> Rerank -> Generate
        
        Args:
            question: User's question
            chat_history: Previous conversation (list of {"role": str, "content": str})
            
        Returns:
            GenerationResult with answer and sources
        """
        chat_history = chat_history or []
        start_time = time.time()
        
        #logger.info(f"Processing query: {question[:50]}...")
        
        # Step 1: Detect language
        language = self.language_detector.detect(question)
        lang_code = language.value  # "en" or "ar"
        #logger.log_language_detected(question, lang_code)
        
        # Step 2: Check if we have any documents
        stats = self.vector_store.get_stats()
        if stats["total_chunks"] == 0:
            #logger.warning("No documents in vector store")
            return self._no_documents_response(lang_code, start_time)
        
        # Step 3: Search for relevant chunks
        #logger.info("Searching for relevant chunks...")
        search_results = self.searcher.search(
            query=question,
            top_k=settings.top_k_retrieval
        )
        
        retrieval_time = (time.time() - start_time) * 1000
        #logger.log_retrieval(question, len(search_results), retrieval_time)
        
        # Step 4: Rerank results
        if search_results:
            #logger.info("Reranking results...")
            reranked_results = self.reranker.rerank(
                query=question,
                results=search_results,
                top_k=settings.rerank_top_k
            )
            #logger.log_reranking(len(search_results), len(reranked_results), 0)
        else:
            reranked_results = []
        
        # Step 5: Generate answer
        #logger.info("Generating answer...")
        result = self.generator.generate(
            query=question,
            results=reranked_results,
            language=lang_code,
            chat_history=chat_history
        )
        
        total_time = (time.time() - start_time) * 1000
        #logger.info(f"Query complete in {total_time:.0f}ms")
        
        return result
    
    # ==================
    # UTILITY METHODS
    # ==================
    
    def clear_documents(self) -> None:
        """
        Clear all documents from the vector store.
        Use this when starting a new session.
        """
        #logger.info("Clearing all documents...")
        self.vector_store.clear()
        #logger.info("Documents cleared")
    
    def get_stats(self) -> dict:
        """
        Get current system statistics.
        
        Returns:
            Dictionary with stats like total_chunks
        """
        return self.vector_store.get_stats()
    
    def _no_documents_response(
        self,
        language: str,
        start_time: float
    ) -> GenerationResult:
        """Create response when no documents are uploaded."""
        if language == "ar":
            message = "لم يتم رفع أي مستندات بعد. يرجى رفع مستند أولاً."
        else:
            message = "No documents have been uploaded yet. Please upload a document first."
        
        return GenerationResult(
            answer=message,
            sources=[],
            language=Language(language),
            has_answer=False,
            generation_time_ms=(time.time() - start_time) * 1000
        )