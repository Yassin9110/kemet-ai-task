# src/config/settings.py
"""
Pydantic Settings for Multilingual RAG System.
Loads configuration from environment variables and .env file.
"""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # -----------------
    # API Keys
    # -----------------
    cohere_api_key: str = Field(
        ...,
        description="Cohere API key for embeddings, reranking, and generation"
    )
    llama_cloud_api_key: str = Field(
        ...,
        description="LlamaCloud API key for LlamaParse PDF parsing"
    )
    
    # -----------------
    # Qdrant Settings
    # -----------------
    qdrant_path: str = Field(
        default="./data/qdrant_storage",
        description="Path for local Qdrant persistent storage"
    )
    qdrant_collection_name: str = Field(
        default="multilingual_rag",
        description="Name of the Qdrant collection"
    )
    
    # -----------------
    # Chunking Settings
    # -----------------
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=2000,
        description="Maximum chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=60,
        ge=0,
        le=500,
        description="Overlap between chunks in tokens"
    )
    
    # -----------------
    # Retrieval Settings
    # -----------------
    top_k_retrieval: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of chunks to retrieve"
    )
    rerank_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks after reranking"
    )
    
    # -----------------
    # Model Settings
    # -----------------
    cohere_generation_model: str = Field(
        default="command-a-03-2025",
        description="Cohere model for text generation"
    )
    cohere_embed_model: str = Field(
        default="embed-v4.0",
        description="Cohere model for dense embeddings"
    )
    cohere_rerank_model: str = Field(
        default="rerank-v3.5",
        description="Cohere model for reranking"
    )
    
    # -----------------
    # App Settings
    # -----------------
    max_file_size_mb: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum upload file size in MB"
    )
    max_conversation_turns: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum conversation turns to keep in context"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    # -----------------
    # Computed Properties
    # -----------------
    @property
    def max_file_size_bytes(self) -> int:
        """Return max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def qdrant_path_resolved(self) -> Path:
        """Return resolved Qdrant storage path."""
        path = Path(self.qdrant_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        # Note: chunk_size may not be available during validation
        # Full validation happens at runtime
        return v
    
    def validate_settings(self) -> None:
        """Run runtime validations."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.rerank_top_k > self.top_k_retrieval:
            raise ValueError(
                f"rerank_top_k ({self.rerank_top_k}) must be <= "
                f"top_k_retrieval ({self.top_k_retrieval})"
            )


# Global settings instance
settings = Settings()