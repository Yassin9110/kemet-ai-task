# src/generation/citations.py
"""
Citation formatter for adding source references to answers.
Creates inline citations like [1], [2] with source details.
"""

from src.core.logging import get_logger
from src.config import settings

logger = get_logger(__name__, settings.log_level)


class CitationFormatter:
    """
    Formats citations for RAG responses.
    
    Example:
        formatter = CitationFormatter()
        context, sources = formatter.format_context(results)
        # context = "[1] Some text...\n[2] More text..."
        # sources = "Sources:\n[1] document.pdf, Page 1\n..."
    """
    
    def format_context(self, results: list[dict]) -> tuple[str, str]:
        """
        Format retrieved results into context with citations.
        
        Args:
            results: List of search results with 'payload'
            
        Returns:
            Tuple of (formatted_context, sources_text)
        """
        if not results:
            return "", ""
        
        context_parts = []
        sources_parts = []
        
        for i, result in enumerate(results, start=1):
            payload = result["payload"]
            content = payload.get("content", "")
            doc_name = payload.get("document_name", "Unknown")
            page_num = payload.get("page_number")
            
            # Format context with citation number
            context_parts.append(f"[{i}] {content}")
            
            # Format source reference
            if page_num:
                sources_parts.append(f"[{i}] {doc_name}, Page {page_num}")
            else:
                sources_parts.append(f"[{i}] {doc_name}")
        
        # Join all parts
        context = "\n\n".join(context_parts)
        sources = "Sources:\n" + "\n".join(sources_parts)
        
        logger.debug(f"Formatted {len(results)} citations")
        
        return context, sources
    
    def format_context_arabic(self, results: list[dict]) -> tuple[str, str]:
        """
        Format retrieved results for Arabic responses.
        
        Args:
            results: List of search results with 'payload'
            
        Returns:
            Tuple of (formatted_context, sources_text)
        """
        if not results:
            return "", ""
        
        context_parts = []
        sources_parts = []
        
        for i, result in enumerate(results, start=1):
            payload = result["payload"]
            content = payload.get("content", "")
            doc_name = payload.get("document_name", "Unknown")
            page_num = payload.get("page_number")
            
            # Format context with citation number
            context_parts.append(f"[{i}] {content}")
            
            # Format source reference in Arabic
            if page_num:
                sources_parts.append(f"[{i}] {doc_name}، صفحة {page_num}")
            else:
                sources_parts.append(f"[{i}] {doc_name}")
        
        # Join all parts
        context = "\n\n".join(context_parts)
        sources = "المصادر:\n" + "\n".join(sources_parts)
        
        logger.debug(f"Formatted {len(results)} Arabic citations")
        
        return context, sources
    
    def add_sources_to_answer(
        self,
        answer: str,
        sources: str,
        language: str
    ) -> str:
        """
        Add sources section to the answer.
        
        Args:
            answer: The generated answer
            sources: The formatted sources text
            language: "en" or "ar"
            
        Returns:
            Answer with sources appended
        """
        if not sources:
            return answer
        
        # Add separator and sources
        separator = "\n\n---\n\n"
        return f"{answer}{separator}{sources}"