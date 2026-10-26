"""Lazy Gemini embedding client for offline experiments (no calls at import time)."""

from google import genai
from google.genai import types

MODEL = "gemini-embedding-001"
TASK_TYPE = "CLASSIFICATION"


class EmbeddingClient:
    def __init__(self):
        # The SDK reads GEMINI_API_KEY or GOOGLE_API_KEY from the environment.
        self.client = genai.Client()

    def embed(self, question: str) -> list[float]:  # Embed only raw question text.
        result = self.client.models.embed_content(
            model=MODEL,
            contents=question,
            config=types.EmbedContentConfig(task_type=TASK_TYPE),
        )
        if not result.embeddings or not result.embeddings[0].values:
            raise ValueError("Empty embedding response")
        return result.embeddings[0].values

    def close(self) -> None:
        self.client.close()
