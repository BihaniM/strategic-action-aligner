from __future__ import annotations

import requests
from src.config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL


class EmbeddingEngine:
    def __init__(self) -> None:
        self.ollama_base_url = OLLAMA_BASE_URL
        self.ollama_embedding_model = OLLAMA_EMBEDDING_MODEL

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned_texts = [text if isinstance(text, str) else str(text) for text in texts]
        embeddings: list[list[float]] = []
        for text in cleaned_texts:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={"model": self.ollama_embedding_model, "prompt": text},
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=600,
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["embedding"]
            embeddings.append(embedding)
        return embeddings
