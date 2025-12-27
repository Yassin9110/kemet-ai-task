# src/providers/cohere_provider.py
"""
Cohere implementation of the LLM provider.
Handles text generation, embeddings, and reranking using Cohere's API.
"""

import cohere
from .base import BaseLLMProvider
from src.config import settings
from src.core.logging import get_logger

# Create logger for this module
logger = get_logger(__name__, settings.log_level)


class CohereProvider(BaseLLMProvider):
    """
    Cohere API provider for the RAG system.
    
    Uses:
        - command-a-03-2025 for text generation
        - embed-v4.0 for embeddings
        - rerank-v3.5 for reranking
    """
    
    def __init__(self):
        """Initialize the Cohere client."""
        self.client = cohere.Client(api_key=settings.cohere_api_key)
        
        # Store model names from settings
        self.generation_model = settings.cohere_generation_model
        self.embed_model = settings.cohere_embed_model
        self.rerank_model = settings.cohere_rerank_model
        
        logger.info(f"CohereProvider initialized with model: {self.generation_model}")
    
    def generate(self, prompt: str, context: str, chat_history: list[dict]) -> str:
        """
        Generate a response using Cohere's chat API.
        
        Args:
            prompt: The user's question
            context: Retrieved document chunks as context
            chat_history: Previous conversation messages
            
        Returns:
            Generated answer as a string
        """
        try:
            # Build the system message with context
            system_message = self._build_system_message(context)
            
            # Convert chat history to Cohere format
            cohere_history = self._format_chat_history(chat_history)
            
            # Call Cohere API
            response = self.client.chat(
                model=self.generation_model,
                message=prompt,
                system=system_message,
                chat_history=cohere_history,
                temperature=0.3,  # Lower = more focused answers
            )
            
            answer = response.text
            logger.info(f"Generated response with {len(answer)} characters")
            
            return answer
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Create embeddings using Cohere's embed API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        # Handle empty input
        if not texts:
            return []
        
        try:
            response = self.client.embed(
                texts=texts,
                model=self.embed_model,
                input_type="search_document",  # For documents being stored
            )
            
            embeddings = response.embeddings
            logger.debug(f"Created {len(embeddings)} embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> list[float]:
        """
        Create embedding for a search query.
        
        Note: Queries use different input_type than documents.
        
        Args:
            query: The search query
            
        Returns:
            Single embedding vector
        """
        try:
            response = self.client.embed(
                texts=[query],
                model=self.embed_model,
                input_type="search_query",  # Different from documents!
            )
            
            return response.embeddings[0]
            
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            raise
    
    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        """
        Rerank documents by relevance using Cohere's rerank API.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            
        Returns:
            List of dicts with 'index' and 'score', sorted by relevance
        """
        # Handle empty input
        if not documents:
            return []
        
        try:
            response = self.client.rerank(
                model=self.rerank_model,
                query=query,
                documents=documents,
                top_n=len(documents),  # Return all, we'll filter later
            )
            
            # Convert to simple dict format
            results = []
            for item in response.results:
                results.append({
                    "index": item.index,
                    "score": item.relevance_score
                })
            
            logger.debug(f"Reranked {len(documents)} documents")
            
            return results
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            raise
    
    def _build_system_message(self, context: str) -> str:
        """
        Build the system message with context for the LLM.
        
        Args:
            context: Retrieved document chunks
            
        Returns:
            Formatted system message
        """
        return f"""You are a helpful assistant that answers questions based on the provided context.

INSTRUCTIONS:
- Answer the question using ONLY the information in the context below
- If the answer is not in the context, say "I don't have information about that in the documents"
- Respond in the SAME LANGUAGE as the user's question
- Be clear and concise
- If citing sources, mention the page numbers

CONTEXT:
{context}"""
    
    def _format_chat_history(self, chat_history: list[dict]) -> list[dict]:
        """
        Convert chat history to Cohere's expected format.
        
        Args:
            chat_history: List of messages with 'role' and 'content'
            
        Returns:
            Formatted history for Cohere API
        """
        cohere_history = []
        
        for message in chat_history:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Cohere uses "USER" and "CHATBOT" (uppercase)
            if role == "user":
                cohere_history.append({
                    "role": "USER",
                    "message": content
                })
            elif role == "assistant":
                cohere_history.append({
                    "role": "CHATBOT", 
                    "message": content
                })
        
        return cohere_history