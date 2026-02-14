from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.hf_client import HFClient


class EmbeddingEngine:
    def __init__(self) -> None:
        self.client = HFClient()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned_texts = [text if isinstance(text, str) else str(text) for text in texts]
        return self.client.embed_texts(cleaned_texts)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    return EmbeddingEngine().embed_texts(texts)


def cosine_similarity_matrix(a_embeddings: list[list[float]], b_embeddings: list[list[float]]) -> np.ndarray:
    a = np.array(a_embeddings, dtype=float)
    b = np.array(b_embeddings, dtype=float)
    return cosine_similarity(a, b)
