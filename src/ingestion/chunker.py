# src/ingestion/chunker.py
"""
Text chunker that splits documents into smaller pieces.
Uses recursive splitting to respect natural text boundaries.
"""

from src.config import settings
from src.core.logging import get_logger
from src.core.models import DocumentChunk, DocumentMetadata, Language
from src.core.language import LanguageDetector

logger = get_logger(__name__, settings.log_level)


class TextChunker:
    """
    Splits text into overlapping chunks.
    
    Example:
        chunker = TextChunker(chunk_size=512, overlap=60)
        chunks = chunker.chunk(text, metadata)
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum characters per chunk (default from settings)
            overlap: Overlap between chunks (default from settings)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
        self.language_detector = LanguageDetector()
        
        logger.info(f"TextChunker initialized: size={self.chunk_size}, overlap={self.overlap}")
    
    def chunk(
        self,
        text: str,
        document_name: str,
        file_type: str,
        page_number: int = None,
        total_pages: int = None
    ) -> list[DocumentChunk]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: The text to split
            document_name: Name of the source document
            file_type: Type of file (pdf, txt)
            page_number: Page number if applicable
            total_pages: Total pages in document
            
        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Step 1: Split text into pieces
        text_pieces = self._split_text(text)
        
        # Step 2: Detect language from first chunk
        language = self._detect_language(text)
        
        # Step 3: Create chunk objects
        chunks = []
        for i, piece in enumerate(text_pieces):
            # Create metadata
            metadata = DocumentMetadata(
                document_name=document_name,
                page_number=page_number,
                total_pages=total_pages,
                language=language,
                file_type=file_type
            )
            
            # Create chunk
            chunk = DocumentChunk(
                content=piece,
                metadata=metadata,
                chunk_index=i
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from text")
        
        return chunks
    
    def chunk_document(
        self,
        page_texts: list[str],
        document_name: str,
        file_type: str
    ) -> list[DocumentChunk]:
        """
        Chunk an entire document with page awareness.
        
        Args:
            page_texts: List of text, one per page
            document_name: Name of the document
            file_type: Type of file (pdf, txt)
            
        Returns:
            List of all chunks from the document
        """
        all_chunks = []
        total_pages = len(page_texts)
        
        for page_num, page_text in enumerate(page_texts, start=1):
            page_chunks = self.chunk(
                text=page_text,
                document_name=document_name,
                file_type=file_type,
                page_number=page_num if file_type == "pdf" else None,
                total_pages=total_pages if file_type == "pdf" else None
            )
            all_chunks.extend(page_chunks)
        
        # Re-index all chunks globally
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
        
        logger.info(f"Document chunked: {len(all_chunks)} total chunks from {total_pages} pages")
        
        return all_chunks
    
    def _split_text(self, text: str) -> list[str]:
        """
        Split text into chunks using recursive splitting.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Separators to try, in order of preference
        separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "، ",    # Arabic comma
            " ",     # Words
            ""       # Characters (last resort)
        ]
        
        return self._recursive_split(text, separators)
    
    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using available separators.
        
        Args:
            text: Text to split
            separators: List of separators to try
            
        Returns:
            List of text chunks
        """
        # Base case: text is small enough
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        # Try each separator
        for separator in separators:
            if separator and separator in text:
                return self._split_with_separator(text, separator, separators)
        
        # No separator found, force split by characters
        return self._force_split(text)
    
    def _split_with_separator(
        self,
        text: str,
        separator: str,
        separators: list[str]
    ) -> list[str]:
        """
        Split text using a specific separator.
        
        Args:
            text: Text to split
            separator: Separator to use
            separators: Remaining separators for recursion
            
        Returns:
            List of text chunks
        """
        parts = text.split(separator)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            # Try adding this part to current chunk
            test_chunk = current_chunk + separator + part if current_chunk else part
            
            if len(test_chunk) <= self.chunk_size:
                # Fits, add to current chunk
                current_chunk = test_chunk
            else:
                # Doesn't fit
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Check if part itself is too big
                if len(part) > self.chunk_size:
                    # Recursively split with next separator
                    remaining_separators = separators[separators.index(separator) + 1:]
                    sub_chunks = self._recursive_split(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Add overlap between chunks
        return self._add_overlap(chunks)
    
    def _force_split(self, text: str) -> list[str]:
        """
        Force split text by character count.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - self.overlap  # Move back by overlap amount
        
        return [c for c in chunks if c]  # Remove empty chunks
    
    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """
        Add overlap between chunks for better context.
        
        Args:
            chunks: List of chunks without overlap
            
        Returns:
            List of chunks with overlap added
        """
        if len(chunks) <= 1 or self.overlap == 0:
            return chunks
        
        overlapped = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: no prefix overlap
                overlapped.append(chunk)
            else:
                # Add end of previous chunk as prefix
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.overlap:] if len(prev_chunk) > self.overlap else prev_chunk
                new_chunk = overlap_text + " " + chunk
                
                # Trim if too long
                if len(new_chunk) > self.chunk_size:
                    new_chunk = new_chunk[:self.chunk_size]
                
                overlapped.append(new_chunk)
        
        return overlapped
    
    def _detect_language(self, text: str) -> Language:
        """
        Detect the language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected Language enum value
        """
        # Use first 500 characters for detection
        sample = text[:500]
        return self.language_detector.detect(sample)