# src/core/models.py
"""
Pydantic data models for the Multilingual RAG System.
Defines schemas for documents, chunks, chat messages, and results.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    ARABIC = "ar"
    UNKNOWN = "unknown"


class ChatRole(str, Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class DocumentMetadata(BaseModel):
    """Metadata associated with a document."""
    
    document_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique document identifier"
    )
    document_name: str = Field(
        ...,
        description="Original filename"
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Page number (for PDFs)"
    )
    total_pages: Optional[int] = Field(
        default=None,
        description="Total pages in document"
    )
    language: Language = Field(
        default=Language.UNKNOWN,
        description="Detected language of the content"
    )
    file_type: str = Field(
        ...,
        description="File extension (pdf, txt)"
    )
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of ingestion"
    )
    
    class Config:
        use_enum_values = True


class DocumentChunk(BaseModel):
    """A chunk of document text with metadata."""
    
    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique chunk identifier"
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Text content of the chunk"
    )
    metadata: DocumentMetadata = Field(
        ...,
        description="Associated metadata"
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Index of chunk within the document"
    )
    start_char: Optional[int] = Field(
        default=None,
        description="Starting character position in original text"
    )
    end_char: Optional[int] = Field(
        default=None,
        description="Ending character position in original text"
    )
    
    def __str__(self) -> str:
        return f"Chunk({self.chunk_id[:8]}..., page={self.metadata.page_number})"


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store with relevance score."""
    
    chunk: DocumentChunk = Field(
        ...,
        description="The retrieved document chunk"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score (0-1)"
    )
    rank: int = Field(
        ...,
        ge=1,
        description="Rank in retrieval results"
    )
    retrieval_method: str = Field(
        default="hybrid",
        description="Method used for retrieval (dense, sparse, hybrid)"
    )
    
    class Config:
        frozen = False


class ChatMessage(BaseModel):
    """A single chat message."""
    
    role: ChatRole = Field(
        ...,
        description="Role of the message sender"
    )
    content: str = Field(
        ...,
        description="Message content"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Message timestamp"
    )
    language: Optional[Language] = Field(
        default=None,
        description="Detected language of the message"
    )
    
    class Config:
        use_enum_values = True


class GenerationResult(BaseModel):
    """Result of answer generation."""
    
    answer: str = Field(
        ...,
        description="Generated answer text"
    )
    sources: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Source chunks used for generation"
    )
    language: Language = Field(
        ...,
        description="Language of the response"
    )
    has_answer: bool = Field(
        default=True,
        description="Whether an answer was found in the documents"
    )
    generation_time_ms: Optional[float] = Field(
        default=None,
        description="Time taken for generation in milliseconds"
    )
    
    class Config:
        use_enum_values = True


class IngestionResult(BaseModel):
    """Result of document ingestion."""
    
    document_name: str = Field(
        ...,
        description="Name of the ingested document"
    )
    total_chunks: int = Field(
        ...,
        ge=0,
        description="Total number of chunks created"
    )
    total_pages: Optional[int] = Field(
        default=None,
        description="Total pages processed (for PDFs)"
    )
    language: Language = Field(
        ...,
        description="Detected primary language"
    )
    success: bool = Field(
        default=True,
        description="Whether ingestion was successful"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if ingestion failed"
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )
    
    class Config:
        use_enum_values = True


class ConversationContext(BaseModel):
    """Stores conversation context for multi-turn chat."""
    
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session identifier"
    )
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="List of chat messages"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Session creation timestamp"
    )
    
    def add_message(self, role: ChatRole, content: str, language: Optional[Language] = None) -> None:
        """Add a message to the conversation."""
        self.messages.append(
            ChatMessage(role=role, content=content, language=language)
        )
    
    def get_recent_messages(self, max_turns: int) -> list[ChatMessage]:
        """Get the most recent messages up to max_turns."""
        # Each turn = 1 user + 1 assistant message
        max_messages = max_turns * 2
        return self.messages[-max_messages:] if self.messages else []
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.utcnow()