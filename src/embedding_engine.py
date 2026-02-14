from __future__ import annotations

import importlib
import os

import requests
from openai import OpenAI
from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    SENTENCE_TRANSFORMER_MODEL,
)


class EmbeddingEngine:
    def __init__(self) -> None:
        self.provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
        self.sentence_transformer_model = os.getenv(
            "SENTENCE_TRANSFORMER_MODEL", SENTENCE_TRANSFORMER_MODEL
        )
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL).rstrip("/")
        self.ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", OLLAMA_EMBEDDING_MODEL)
        self.client = None
        self.model = None

        if self.provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not configured.")
            self.client = OpenAI(api_key=self.openai_api_key)
        elif self.provider == "sentence-transformers":
            try:
                sentence_transformers_module = importlib.import_module("sentence_transformers")
                sentence_transformer_class = getattr(sentence_transformers_module, "SentenceTransformer")
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed. Install it with: pip install sentence-transformers"
                ) from exc
            self.model = sentence_transformer_class(self.sentence_transformer_model)
        elif self.provider == "ollama":
            pass
        else:
            raise ValueError(
                "Unsupported EMBEDDING_PROVIDER. Use 'openai', 'sentence-transformers', or 'ollama'."
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned_texts = [text if isinstance(text, str) else str(text) for text in texts]

        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.openai_embedding_model,
                input=cleaned_texts,
            )
            return [item.embedding for item in response.data]

        if self.provider == "ollama":
            embeddings: list[list[float]] = []
            for text in cleaned_texts:
                response = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={"model": self.ollama_embedding_model, "prompt": text},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding")
                if not embedding:
                    raise ValueError("Ollama embedding response missing 'embedding' field.")
                embeddings.append(embedding)
            return embeddings

        embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True)
        return embeddings.tolist()
