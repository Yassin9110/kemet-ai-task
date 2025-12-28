# src/providers/factory.py
from .cohere_provider import CohereProvider
from .gemini_provider import GeminiProvider # New import
from .base import BaseLLMProvider

def get_provider(provider_name: str = "gemini") -> BaseLLMProvider: # Defaulted to gemini
    provider_name = provider_name.lower()
    
    providers = {
        "cohere": CohereProvider,
        "gemini": GeminiProvider, # Registered
    }
    
    if provider_name not in providers:
        # ... error handling ...
        pass
    
    return providers[provider_name]()