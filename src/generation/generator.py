# src/generation/generator.py
"""
Response generator that creates answers using the LLM.
Combines context retrieval with answer generation.
"""

import time
from src.config import settings
# from src.core.logging import  get_logger
from src.core.models import GenerationResult, RetrievedChunk, DocumentChunk, DocumentMetadata, Language
from src.llmproviders import get_provider
from .prompts import get_system_prompt, get_no_answer_message
from .citations import CitationFormatter

#logger = get_#logger(__name__, settings.log_level)


class ResponseGenerator:
    """
    Generates answers using retrieved context and LLM.
    
    Example:
        generator = ResponseGenerator()
        result = generator.generate(
            query="What is AI?",
            results=search_results,
            language="en",
            chat_history=[]
        )
        print(result.answer)
    """
    
    def __init__(self):
        """Initialize the generator."""
        self.provider = get_provider("gemini")
        self.citation_formatter = CitationFormatter()
        
        #logger.info("ResponseGenerator initialized")
    
    def generate(
        self,
        query: str,
        results: list[dict],
        language: str,
        chat_history: list[dict] = None
    ) -> GenerationResult:
        """
        Generate an answer based on retrieved results.
        
        Args:
            query: User's question
            results: Retrieved search results
            language: Response language ("en" or "ar")
            chat_history: Previous conversation messages
            
        Returns:
            GenerationResult with answer and sources
        """
        start_time = time.time()
        chat_history = chat_history or []
        
        # Handle no results
        if not results:
            return self._create_no_answer_result(language, start_time)
        
        # Step 1: Format context with citations
        if language == "ar":
            context, sources = self.citation_formatter.format_context_arabic(results)
        else:
            context, sources = self.citation_formatter.format_context(results)
        
        # Step 2: Build system prompt
        system_prompt = get_system_prompt(language, context)
        
        # Step 3: Generate answer
        # try:
        answer = self.provider.generate_text(
            prompt=query,
            context=system_prompt,
            chat_history=chat_history
        )
        
        # Step 4: Add sources to answer
        full_answer = self.citation_formatter.add_sources_to_answer(
            answer=answer,
            sources=sources,
            language=language
        )
        
        # Calculate time
        generation_time = (time.time() - start_time) * 1000
        
        # Create source chunks for result
        source_chunks = self._convert_to_retrieved_chunks(results)
        
        #logger.log_generation(language, len(full_answer), generation_time)
        
        return GenerationResult(
            answer=full_answer,
            sources=source_chunks,
            language=Language(language),
            has_answer=True,
            generation_time_ms=generation_time
        )
            
        # except Exception as e:
        #     #logger.error(f"Generation failed: {str(e)}")
        #     return self._create_error_result(language, str(e), start_time)
    
    def _create_no_answer_result(
        self,
        language: str,
        start_time: float
    ) -> GenerationResult:
        """Create result when no context is available."""
        answer = get_no_answer_message(language)
        generation_time = (time.time() - start_time) * 1000
        
        return GenerationResult(
            answer=answer,
            sources=[],
            language=Language(language),
            has_answer=False,
            generation_time_ms=generation_time
        )
    
    def _create_error_result(
        self,
        language: str,
        error: str,
        start_time: float
    ) -> GenerationResult:
        """Create result when an error occurs."""
        if language == "ar":
            answer = f"حدث خطأ أثناء إنشاء الإجابة: {error}"
        else:
            answer = f"An error occurred while generating the answer: {error}"
        
        generation_time = (time.time() - start_time) * 1000
        
        return GenerationResult(
            answer=answer,
            sources=[],
            language=Language(language),
            has_answer=False,
            generation_time_ms=generation_time
        )
    
    def _convert_to_retrieved_chunks(
        self,
        results: list[dict]
    ) -> list[RetrievedChunk]:
        """Convert search results to RetrievedChunk objects."""
        chunks = []
        
        for i, result in enumerate(results):
            payload = result["payload"]
            
            # Create metadata
            metadata = DocumentMetadata(
                document_name=payload.get("document_name", "Unknown"),
                page_number=payload.get("page_number"),
                language=Language(payload.get("language", "en")),
                file_type=payload.get("file_type", "unknown")
            )
            
            # Create chunk
            chunk = DocumentChunk(
                chunk_id=payload.get("chunk_id", ""),
                content=payload.get("content", ""),
                metadata=metadata,
                chunk_index=payload.get("chunk_index", 0)
            )
            
            # Create retrieved chunk
            retrieved = RetrievedChunk(
                chunk=chunk,
                score=result.get("rerank_score", result.get("score", 0.0)),
                rank=i + 1
            )
            chunks.append(retrieved)
        
        return chunks