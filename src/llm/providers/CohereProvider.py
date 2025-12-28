from llm.LLMInterface import LLMInterface
from llm.LLMEnums import LLMEnums, CohereEnums, DocumentTypeEnum
import cohere
import logging
from src.config import settings

class CohereProvider(LLMInterface):
    def __init__(self, default_max_input_tokens: int = 10000, default_max_output_tokens: int = 10000, temperature: float = 0.7):
        self.client = cohere.Client(settings.cohere_api_key)
        self.generation_model = settings.cohere_generation_model
        self.embed_model = settings.cohere_embed_model
        self.rerank_model = settings.cohere_rerank_model

        self.default_max_input_tokens = default_max_input_tokens
        self.default_max_output_tokens = default_max_output_tokens
        self.temperature = temperature

    
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
    
    def process_text(self, text: str):
        return text[:self.default_max_input_tokens].strip()

    def generate_text(self, prompt: str, **kwargs):
        if not self.generation_model:
            self.logger.error("Generation model not set.")
            return None
        if not self.client:
            self.logger.error("Cohere client not initialized.")
            return None
        
        chat_history = kwargs.get('chat_history', [])

        chat_history.append(self.construct_prompt(prompt, CohereEnums.USER.value))

        
        response = self.client.chat(
            model = self.generation_model,
            messages = chat_history,
            max_tokens = kwargs.get('max_output_tokens', self.default_max_output_tokens),
            temperature = kwargs.get('temperature', self.temperature)
        )

        if not response:
            self.logger.error("Invalid response from generation API.")
            return None
        
        return response.message.content[0].text
    

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
        

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "content": self.process_text(prompt)}
