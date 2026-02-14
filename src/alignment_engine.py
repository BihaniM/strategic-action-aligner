from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import build_chunk_dataframe, build_chunk_dataframe_from_dataframes
from src.embedding_engine import EmbeddingEngine


@dataclass
class AlignmentArtifacts:
    chunks_df: pd.DataFrame
    strategic_df: pd.DataFrame
    action_df: pd.DataFrame
    strategic_embeddings: np.ndarray
    action_embeddings: np.ndarray
    similarity_matrix: np.ndarray


def build_alignment_artifacts(
    strategic_csv_path: str = "data/input/strategic_plan.csv",
    action_csv_path: str = "data/input/action_plan.csv",
    embedder: EmbeddingEngine | None = None,
) -> AlignmentArtifacts:
    chunk_df = build_chunk_dataframe(strategic_csv_path, action_csv_path)
    return _build_alignment_artifacts_from_chunks(chunk_df, embedder=embedder)


def build_alignment_artifacts_from_dataframes(
    strategic_df: pd.DataFrame,
    action_df: pd.DataFrame,
    embedder: EmbeddingEngine | None = None,
) -> AlignmentArtifacts:
    chunk_df = build_chunk_dataframe_from_dataframes(strategic_df, action_df)
    return _build_alignment_artifacts_from_chunks(chunk_df, embedder=embedder)


def _build_alignment_artifacts_from_chunks(
    chunk_df: pd.DataFrame,
    embedder: EmbeddingEngine | None = None,
) -> AlignmentArtifacts:
    strategic_chunks = chunk_df[chunk_df["document_type"] == "strategic"].reset_index(drop=True)
    action_chunks = chunk_df[chunk_df["document_type"] == "action"].reset_index(drop=True)

    if strategic_chunks.empty or action_chunks.empty:
        raise ValueError("Strategic and Action chunk sets must both be non-empty.")

    engine = embedder or EmbeddingEngine()
    strategic_embeddings = np.array(engine.embed_texts(strategic_chunks["text"].tolist()), dtype=float)
    action_embeddings = np.array(engine.embed_texts(action_chunks["text"].tolist()), dtype=float)

    similarity_matrix = cosine_similarity(strategic_embeddings, action_embeddings)

    return AlignmentArtifacts(
        chunks_df=chunk_df,
        strategic_df=strategic_chunks,
        action_df=action_chunks,
        strategic_embeddings=strategic_embeddings,
        action_embeddings=action_embeddings,
        similarity_matrix=similarity_matrix,
    )


def calculate_overall_alignment_percentage(artifacts: AlignmentArtifacts) -> float:
    max_similarities = artifacts.similarity_matrix.max(axis=1)
    return float(np.clip(np.mean(max_similarities) * 100.0, 0.0, 100.0))


def build_strategy_alignment_table(artifacts: AlignmentArtifacts, top_k: int = 3) -> pd.DataFrame:
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    rows: list[dict[str, Any]] = []
    action_texts = artifacts.action_df["text"].tolist()
    action_ids = artifacts.action_df["chunk_id"].tolist()

    for idx, strategy_row in artifacts.strategic_df.iterrows():
        sims = artifacts.similarity_matrix[idx]
        top_indices = np.argsort(sims)[::-1][:top_k]
        matched_actions = [action_texts[i] for i in top_indices]
        matched_action_ids = [action_ids[i] for i in top_indices]
        score = float(np.max(sims[top_indices])) if len(top_indices) else 0.0

        rows.append(
            {
                "strategy": strategy_row["text"],
                "matched_actions": " || ".join(matched_actions),
                "similarity_score": round(score, 4),
                "strategy_chunk_id": strategy_row["chunk_id"],
                "matched_action_chunk_ids": "||".join(matched_action_ids),
                "section_name": strategy_row["section_name"],
            }
        )

    return pd.DataFrame(rows)


def get_low_alignment_pairs(strategy_alignment_df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    return strategy_alignment_df[strategy_alignment_df["similarity_score"] < threshold].reset_index(drop=True)
