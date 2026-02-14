from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

from src.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL
from src.embedding_engine import EmbeddingEngine


@dataclass
class ImprovementResult:
    strategy: str
    baseline_actions: str
    baseline_similarity_score: float
    improved_similarity_score: float
    improved: bool
    suggestions: dict[str, Any]


def suggest_improvements_for_pair(
    strategy_text: str,
    matched_actions_text: str,
    similarity_score: float,
) -> dict[str, Any]:
    # LLM response generation
    prompt = {
        "task": "Improve action plan alignment with strategic objective.",
        "strategy": strategy_text,
        "matched_actions": matched_actions_text,
        "similarity_score": similarity_score,
        "instructions": [
            "Suggest missing actions",
            "Suggest improved KPIs",
            "Suggest timeline or scope changes",
            "Return valid JSON only",
        ],
        "required_output_schema": {
            "missing_actions": ["string"],
            "improved_kpis": ["string"],
            "timeline_or_scope_changes": ["string"],
            "revised_action_plan_summary": "string",
        },
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_CHAT_MODEL,
            "format": "json",
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a strategic planning expert."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
        },
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    content = data["message"]["content"]
    parsed = json.loads(content)
    parsed["generation_mode"] = "ollama"
    parsed.setdefault("missing_actions", [])
    parsed.setdefault("improved_kpis", [])
    parsed.setdefault("timeline_or_scope_changes", [])
    parsed.setdefault("revised_action_plan_summary", "")
    return parsed


def _suggestion_text_for_embedding(suggestions: dict[str, Any]) -> str:
    pieces = []
    pieces.extend(suggestions.get("missing_actions", []))
    pieces.extend(suggestions.get("improved_kpis", []))
    pieces.extend(suggestions.get("timeline_or_scope_changes", []))
    summary = suggestions.get("revised_action_plan_summary", "")
    if summary:
        pieces.append(summary)
    text = "\n".join(str(part) for part in pieces if str(part).strip())
    return text


def run_improvement_agent_loop(
    low_alignment_df: pd.DataFrame,
    similarity_threshold: float = 0.6,
    max_iterations: int = 3,
    embedder: EmbeddingEngine | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    # Improvement loop
    if low_alignment_df.empty:
        return low_alignment_df.copy(), []

    engine = embedder or EmbeddingEngine()
    working = low_alignment_df.copy().reset_index(drop=True)
    history: list[dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        improved_any = False

        for idx, row in working.iterrows():
            baseline_score = float(row["similarity_score"])
            if baseline_score >= similarity_threshold:
                continue

            suggestions = suggest_improvements_for_pair(
                strategy_text=str(row["strategy"]),
                matched_actions_text=str(row["matched_actions"]),
                similarity_score=baseline_score,
            )

            strategy_embedding = np.array(engine.embed_texts([str(row["strategy"])]), dtype=float)
            suggestion_text = _suggestion_text_for_embedding(suggestions)
            if suggestion_text == "":
                suggestion_text = str(row["matched_actions"])
            suggestion_embedding = np.array(engine.embed_texts([suggestion_text]), dtype=float)
            improved_score = float(cosine_similarity(strategy_embedding, suggestion_embedding)[0][0])

            improved = improved_score > baseline_score
            if improved:
                improved_any = True
                working.at[idx, "similarity_score"] = round(improved_score, 4)
                working.at[idx, "matched_actions"] = suggestion_text

            history.append(
                {
                    "iteration": iteration,
                    "strategy_chunk_id": row.get("strategy_chunk_id", ""),
                    "strategy": row["strategy"],
                    "baseline_similarity_score": round(baseline_score, 4),
                    "improved_similarity_score": round(improved_score, 4),
                    "improved": improved,
                    "suggestions": suggestions,
                }
            )

        if not improved_any:
            break

    return working, history
