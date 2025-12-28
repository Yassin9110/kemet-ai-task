# src/providers/gemini_provider.py
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from .base import BaseLLMProvider
from src.config import settings

class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API provider for the RAG system.
    """

    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.generation_model_name = settings.gemini_generation_model
        self.embed_model_name = settings.gemini_embed_model

        # Highly permissive safety settings for RAG context
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    # ------------------------------------------------------------------
    # TEXT GENERATION
    # ------------------------------------------------------------------
    def generate(self, prompt: str, context: str, chat_history: list[dict]) -> str:
        """Standard method defined in BaseLLMProvider."""
        return self.generate_text(prompt, context, chat_history)

    def generate_text(self, prompt: str, context: str, chat_history: list[dict]) -> str:
        """Method called by ResponseGenerator."""
        # Initialize model with the context as system_instruction
        model = genai.GenerativeModel(
            model_name=self.generation_model_name,
            system_instruction=context, # Injected from ResponseGenerator
            generation_config={"temperature": 0.3}
        )

        # Convert history: 'assistant' -> 'model'
        formatted_history = self._format_chat_history(chat_history)
        
        # Start chat session
        chat = model.start_chat(history=formatted_history)
        
        response = chat.send_message(
            prompt, 
            safety_settings=self.safety_settings
        )

        return response.text

    # ------------------------------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Gemini API limits batching to 100 texts per call
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = genai.embed_content(
                model=self.embed_model_name,
                content=batch,
                task_type="RETRIEVAL_DOCUMENT"
            )
            all_embeddings.extend(result['embedding'])

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        result = genai.embed_content(
            model=self.embed_model_name,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        return result['embedding']

    # ------------------------------------------------------------------
    # RERANK (Passthrough)
    # ------------------------------------------------------------------
    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        """Skips reranking and returns original indices with dummy scores."""
        return [
            {"index": i, "score": 1.0 - (i * 0.001)} 
            for i in range(len(documents))
        ]

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _format_chat_history(self, chat_history: list[dict]) -> list[dict]:
        """Maps 'assistant' role to Gemini's 'model' role."""
        gemini_history = []
        for msg in chat_history:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append({
                "role": role,
                "parts": [msg["content"]]
            })
        return gemini_history