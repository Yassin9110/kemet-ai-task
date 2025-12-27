# src/providers/factory.py
"""
Factory for creating LLM providers.
Makes it easy to get the right provider without knowing the details.
"""

from .base import BaseLLMProvider
from .cohere_provider import CohereProvider


def get_provider(provider_name: str = "cohere") -> BaseLLMProvider:
    """
    Get an LLM provider by name.
    
    Args:
        provider_name: Name of the provider ("cohere", etc.)
        
    Returns:
        An instance of the requested provider
        
    Raises:
        ValueError: If provider name is not supported
        
    Example:
        provider = get_provider("cohere")
        answer = provider.generate("What is AI?", context, history)
    """
    # Convert to lowercase for flexibility
    provider_name = provider_name.lower()
    
    # Map of available providers
    providers = {
        "cohere": CohereProvider,
    }
    
    # Check if provider exists
    if provider_name not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(
            f"Unknown provider: '{provider_name}'. "
            f"Available providers: {available}"
        )
    
    # Create and return the provider
    return providers[provider_name]()