# src/core/logging.py
"""
Logging configuration for the Multilingual RAG System.
Supports per-upload session logging and structured output.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import re


class RAGLogger:
    """
    Custom logger for the RAG system.
    Creates per-upload session log files and console output.
    """
    
    # Class-level storage for loggers
    _loggers: dict[str, logging.Logger] = {}
    _log_dir: Path = Path("logs")
    
    def __init__(
        self,
        name: str,
        log_level: str = "INFO",
        log_dir: Optional[Path] = None
    ):
        """
        Initialize the RAG logger.
        
        Args:
            name: Logger name (usually module name)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_dir: Directory for log files
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        if log_dir:
            RAGLogger._log_dir = log_dir
        
        # Ensure log directory exists
        RAGLogger._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Get or create logger
        self.logger = self._get_or_create_logger()
    
    def _get_or_create_logger(self) -> logging.Logger:
        """Get existing logger or create a new one."""
        if self.name in RAGLogger._loggers:
            return RAGLogger._loggers[self.name]
        
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Prevent duplicate handlers
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # Store logger
        RAGLogger._loggers[self.name] = logger
        
        return logger
    
    def add_file_handler(self, filename: str) -> Path:
        """
        Add a file handler for a specific upload session.
        
        Args:
            filename: Original filename being processed
            
        Returns:
            Path to the log file
        """
        # Sanitize filename
        safe_filename = self._sanitize_filename(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"upload_{timestamp}_{safe_filename}.log"
        log_path = RAGLogger._log_dir / log_filename
        
        # Create file handler
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Log file created: {log_path}")
        
        return log_path
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for use in log file names."""
        # Remove extension
        name = Path(filename).stem
        # Replace non-alphanumeric characters with underscore
        safe_name = re.sub(r"[^\w\-]", "_", name)
        # Limit length
        return safe_name[:50]
    
    # -----------------
    # Logging Methods
    # -----------------
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, **kwargs)
    
    # -----------------
    # Contextual Logging
    # -----------------
    def log_ingestion_start(self, filename: str, file_size: int) -> None:
        """Log the start of document ingestion."""
        self.info(f"Starting ingestion: {filename} ({file_size / 1024:.2f} KB)")
    
    def log_ingestion_complete(
        self,
        filename: str,
        chunks: int,
        duration_ms: float
    ) -> None:
        """Log successful ingestion completion."""
        self.info(
            f"Ingestion complete: {filename} | "
            f"Chunks: {chunks} | "
            f"Duration: {duration_ms:.2f}ms"
        )
    
    def log_parsing(self, filename: str, pages: Optional[int] = None) -> None:
        """Log document parsing."""
        if pages:
            self.info(f"Parsed document: {filename} | Pages: {pages}")
        else:
            self.info(f"Parsed document: {filename}")
    
    def log_chunking(self, total_chunks: int, avg_size: float) -> None:
        """Log chunking results."""
        self.info(f"Chunking complete | Total: {total_chunks} | Avg size: {avg_size:.1f} chars")
    
    def log_embedding(self, num_chunks: int, duration_ms: float) -> None:
        """Log embedding generation."""
        self.info(f"Embeddings generated | Chunks: {num_chunks} | Duration: {duration_ms:.2f}ms")
    
    def log_retrieval(
        self,
        query: str,
        num_results: int,
        duration_ms: float
    ) -> None:
        """Log retrieval operation."""
        truncated_query = query[:50] + "..." if len(query) > 50 else query
        self.info(
            f"Retrieval complete | Query: '{truncated_query}' | "
            f"Results: {num_results} | Duration: {duration_ms:.2f}ms"
        )
    
    def log_reranking(
        self,
        before_count: int,
        after_count: int,
        duration_ms: float
    ) -> None:
        """Log reranking operation."""
        self.info(
            f"Reranking complete | Before: {before_count} | "
            f"After: {after_count} | Duration: {duration_ms:.2f}ms"
        )
    
    def log_generation(
        self,
        language: str,
        answer_length: int,
        duration_ms: float
    ) -> None:
        """Log answer generation."""
        self.info(
            f"Generation complete | Language: {language} | "
            f"Answer length: {answer_length} chars | Duration: {duration_ms:.2f}ms"
        )
    
    def log_language_detected(self, text_sample: str, language: str) -> None:
        """Log language detection."""
        truncated = text_sample[:30] + "..." if len(text_sample) > 30 else text_sample
        self.debug(f"Language detected: {language} | Sample: '{truncated}'")


def get_logger(name: str, log_level: str = "INFO") -> RAGLogger:
    """
    Factory function to get a RAG logger.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level
        
    Returns:
        RAGLogger instance
    """
    return RAGLogger(name=name, log_level=log_level)