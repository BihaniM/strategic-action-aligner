from __future__ import annotations

import numpy as np
import pandas as pd

from src.embedding_engine import EmbeddingEngine
from src.hf_client import HFClient
from src.vector_store import VectorStore


class AlignmentAnalyzer:
    def __init__(self) -> None:
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.hf_client = HFClient()

    def index_strategic_goals(self, strategic_df: pd.DataFrame) -> None:
        ids = strategic_df["goal_id"].astype(str).tolist()
        texts = strategic_df["goal_text"].astype(str).tolist()
        embeddings = self.embedding_engine.embed_texts(texts)
        metadatas = [{"goal_id": goal_id} for goal_id in ids]
        self.vector_store.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    def _llm_assess_alignment(self, action_text: str, goal_text: str, similarity_score: float) -> tuple[str, str]:
        system_prompt = (
            "Assess whether an action is aligned with a strategic goal. "
            "Return strict JSON with keys label and rationale. "
            "Label must be one of Aligned, Partially Aligned, Not Aligned."
        )
        payload = {
            "action_text": action_text,
            "goal_text": goal_text,
            "similarity_score": similarity_score,
        }
        parsed = self.hf_client.generate_json(system_prompt=system_prompt, user_payload=payload)
        return parsed.get("label", "Partially Aligned"), parsed.get(
            "rationale", "No rationale provided."
        )

    def analyze(self, action_df: pd.DataFrame, top_k: int = 1) -> pd.DataFrame:
        action_ids = action_df["action_id"].astype(str).tolist()
        action_texts = action_df["action_text"].astype(str).tolist()
        action_embeddings = self.embedding_engine.embed_texts(action_texts)

        query_results = self.vector_store.query(query_embeddings=action_embeddings, n_results=top_k)
        matched_goal_ids = [hits[0].get("goal_id", "") if hits else "" for hits in query_results.get("metadatas", [])]
        matched_goal_texts = [hits[0] if hits else "" for hits in query_results.get("documents", [])]
        distances = [hits[0] if hits else 1.0 for hits in query_results.get("distances", [])]
        similarity_scores = [float(1.0 / (1.0 + max(d, 0.0))) for d in distances]

        labels = []
        rationales = []
        for action_text, goal_text, score in zip(action_texts, matched_goal_texts, similarity_scores):
            label, rationale = self._llm_assess_alignment(action_text, goal_text, score)
            labels.append(label)
            rationales.append(rationale)

        result = pd.DataFrame(
            {
                "action_id": action_ids,
                "action_text": action_texts,
                "matched_goal_id": matched_goal_ids,
                "matched_goal_text": matched_goal_texts,
                "similarity_score": np.round(similarity_scores, 4),
                "llm_alignment_label": labels,
                "llm_rationale": rationales,
            }
        )
        return result
