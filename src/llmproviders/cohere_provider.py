# src/providers/cohere_provider.py

import cohere
from .base import BaseLLMProvider
from src.config import settings


class CohereProvider(BaseLLMProvider):
    """
    Cohere API provider for the RAG system (Cohere V2).
    """

    def __init__(self):
        self.client = cohere.Client(api_key=settings.cohere_api_key)

        self.generation_model = settings.cohere_generation_model
        self.embed_model = settings.cohere_embed_model
        self.rerank_model = settings.cohere_rerank_model

    # ------------------------------------------------------------------
    # TEXT GENERATION (V2)
    # ------------------------------------------------------------------
    def generate(self, prompt: str, context: str, chat_history: list[dict]) -> str:
        system_message = self._build_system_message(context)

        messages = [
            {"role": "system", "content": system_message},
            *self._format_chat_history(chat_history),
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat(
            model=self.generation_model,
            message=messages,
            temperature=0.3,
        )

        return response.message.content[0].text

    # ------------------------------------------------------------------
    # EMBEDDINGS (V2)
    # ------------------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embed(
            model=self.embed_model,
            texts=texts,
            input_type="search_document",
        )

        return response.embeddings

    def embed_query(self, query: str) -> list[float]:
        response = self.client.embed(
            model=self.embed_model,
            texts=[query],
            input_type="search_query",
        )

        return response.embeddings[0]

    # ------------------------------------------------------------------
    # RERANK (V2)
    # ------------------------------------------------------------------
    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        if not documents:
            return []

        response = self.client.rerank(
            model=self.rerank_model,
            query=query,
            documents=documents,
            top_n=len(documents),
        )

        return [
            {
                "index": item.index,
                "score": item.score,
            }
            for item in response.results
        ]

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _build_system_message(self, context: str) -> str:
        return f"""You are a helpful assistant that answers questions based on the provided context.

INSTRUCTIONS:
- Answer ONLY using the context
- If not found, say: "I don't have information about that in the documents"
- Respond in the SAME LANGUAGE as the user
- Be concise

CONTEXT:
{context}
"""

    def _format_chat_history(self, chat_history: list[dict]) -> list[dict]:
        messages = []

        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content")

            if role in {"user", "assistant"}:
                messages.append({
                    "role": role,
                    "content": content
                })

        return messages




