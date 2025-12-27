# src/providers/base.py
"""
Base class that defines what any LLM provider must do.
Think of this as a "contract" - any provider must have these methods.
"""

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Any provider (Cohere, OpenAI, etc.) must implement these methods.
    This makes it easy to swap providers in the future.
    """
    
    @abstractmethod
    def generate(self, prompt: str, context: str, chat_history: list[dict]) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: The user's question
            context: Retrieved document chunks as context
            chat_history: Previous conversation messages
            
        Returns:
            Generated answer as a string
        """
        pass
    
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Create embeddings (vector representations) for texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        pass
    
    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        """
        Rerank documents by relevance to the query.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            
        Returns:
            List of dicts with 'index' and 'score' keys, sorted by relevance
        """
        pass