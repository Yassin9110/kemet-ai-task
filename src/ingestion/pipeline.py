# src/ingestion/pipeline.py
"""
Ingestion pipeline that orchestrates the entire document processing flow.
Parse -> Chunk -> Embed -> Store
"""

import time
from src.config import settings
# from src.core.logging import  get_logger
from src.core.models import DocumentChunk, IngestionResult, Language
from src.core.language import LanguageDetector
from .parser import DocumentParser
from .chunker import TextChunker

#logger = get_#logger(__name__, settings.log_level)


class IngestionPipeline:
    """
    Orchestrates document ingestion: parsing, chunking, and preparation for storage.
    
    Example:
        pipeline = IngestionPipeline()
        chunks, result = pipeline.ingest("doc.pdf", file_bytes)
    """
    
    def __init__(self):
        """Initialize pipeline components."""
        self.parser = DocumentParser()
        self.chunker = TextChunker()
        self.language_detector = LanguageDetector()
        
        #logger.info("IngestionPipeline initialized")
    
    def ingest(
        self,
        filename: str,
        file_bytes: bytes
    ) -> tuple[list[DocumentChunk], IngestionResult]:
        """
        Ingest a document: parse and chunk it.
        
        Args:
            filename: Name of the file
            file_bytes: Raw file content
            
        Returns:
            Tuple of (list of chunks, ingestion result)
        """
        start_time = time.time()
        
        # try:
        # Log start
        #logger.log_ingestion_start(filename, len(file_bytes))
        
        # Step 1: Parse the document
        #logger.info("Step 1: Parsing document...")
        parsed = self.parser.parse(filename, file_bytes)
        
        #logger.log_parsing(filename, parsed.get("pages"))
        
        # Step 2: Chunk the document
        #logger.info("Step 2: Chunking document...")
        chunks = self.chunker.chunk_document(
            page_texts=parsed["page_texts"],
            document_name=filename,
            file_type=parsed["file_type"]
        )
        
        # Calculate average chunk size
        if chunks:
            avg_size = sum(len(c.content) for c in chunks) / len(chunks)
            #logger.log_chunking(len(chunks), avg_size)
        
        # Step 3: Detect primary language
        primary_language = self._detect_primary_language(chunks)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Create result
        result = IngestionResult(
            document_name=filename,
            total_chunks=len(chunks),
            total_pages=parsed.get("pages"),
            language=primary_language,
            success=True,
            processing_time_ms=processing_time
        )
        
        #logger.log_ingestion_complete(filename, len(chunks), processing_time)
        
        return chunks, result
            
        # except Exception as e:
        #     processing_time = (time.time() - start_time) * 1000
            
        #     #logger.error(f"Ingestion failed for {filename}: {str(e)}")
            
        #     result = IngestionResult(
        #         document_name=filename,
        #         total_chunks=0,
        #         total_pages=None,
        #         language=Language.UNKNOWN,
        #         success=False,
        #         error_message=str(e),
        #         processing_time_ms=processing_time
        #     )
            
        #     return [], result
    
    def validate_file(self, filename: str, file_size: int) -> tuple[bool, str]:
        """
        Validate a file before processing.
        
        Args:
            filename: Name of the file
            file_size: Size in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        if file_size > settings.max_file_size_bytes:
            max_mb = settings.max_file_size_mb
            actual_mb = file_size / (1024 * 1024)
            return False, f"File too large: {actual_mb:.1f}MB (max: {max_mb}MB)"
        
        # Check file type
        allowed_types = ["pdf", "txt"]
        file_ext = filename.split(".")[-1].lower()
        
        if file_ext not in allowed_types:
            return False, f"Unsupported file type: {file_ext}. Allowed: {allowed_types}"
        
        # Check filename
        if not filename or len(filename) < 5:  # minimum: "a.txt"
            return False, "Invalid filename"
        
        return True, ""
    
    def _detect_primary_language(self, chunks: list[DocumentChunk]) -> Language:
        """
        Detect the primary language from chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Most common language in chunks
        """
        if not chunks:
            return Language.UNKNOWN
        
        # Count languages
        lang_counts = {
            Language.ENGLISH: 0,
            Language.ARABIC: 0,
            Language.UNKNOWN: 0
        }
        
        # Sample some chunks (not all, for speed)
        sample_size = min(5, len(chunks))
        for chunk in chunks[:sample_size]:
            lang = self.language_detector.detect(chunk.content)
            lang_counts[lang] += 1
        
        # Return most common
        return max(lang_counts, key=lang_counts.get)