from __future__ import annotations

import json
from typing import Any

import pandas as pd
import requests

from src.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


def _ollama_json(system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_CHAT_MODEL,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        },
        headers={"ngrok-skip-browser-warning": "true"},
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    content = data["message"]["content"]
    return json.loads(content)


def _agentic_reasoning(strategy: str, matched_actions: str, similarity_score: float) -> dict[str, Any]:
    system_prompt = "You are an agentic strategic planner with diagnose-propose-critique reasoning. Return strict JSON."
    payload = {
        "task": "Perform diagnose-propose-critique reasoning for strategic alignment.",
        "strategy": strategy,
        "matched_actions": matched_actions,
        "similarity_score": similarity_score,
        "required_output_schema": {
            "diagnosis": {
                "root_causes": ["string"],
                "capability_gaps": ["string"],
                "risk_level": "Low|Medium|High",
                "diagnostic_summary": "string",
            },
            "proposal": {
                "recommended_actions": ["string"],
                "recommended_kpis": ["string"],
                "timeline_scope_adjustments": ["string"],
                "expected_alignment_delta": "float",
                "proposal_summary": "string",
            },
            "critique": {
                "approved": "boolean",
                "issues": ["string"],
                "confidence": "float",
            },
        },
    }
    return _ollama_json(system_prompt, payload)


def run_agentic_reasoning_layer(
    low_alignment_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if low_alignment_df.empty:
        return low_alignment_df.copy(), []

    recommendation_rows: list[dict[str, Any]] = []
    reasoning_log: list[dict[str, Any]] = []

    for row in low_alignment_df.itertuples(index=False):
        strategy = str(row.strategy)
        matched_actions = str(row.matched_actions)
        similarity_score = float(row.similarity_score)
        strategy_chunk_id = str(getattr(row, "strategy_chunk_id", ""))

        reasoning_result = _agentic_reasoning(strategy, matched_actions, similarity_score)
        diagnosis = reasoning_result.get("diagnosis", {})
        proposal = reasoning_result.get("proposal", {})
        critique = reasoning_result.get("critique", {})

        recommendation_rows.append(
            {
                "strategy_chunk_id": strategy_chunk_id,
                "strategy": strategy,
                "baseline_similarity_score": similarity_score,
                "recommended_actions": " || ".join(proposal.get("recommended_actions", [])),
                "recommended_kpis": " || ".join(proposal.get("recommended_kpis", [])),
                "timeline_scope_adjustments": " || ".join(
                    proposal.get("timeline_scope_adjustments", [])
                ),
                "expected_alignment_delta": float(proposal.get("expected_alignment_delta", 0.0)),
                "critic_approved": bool(critique.get("approved", False)),
                "critic_confidence": float(critique.get("confidence", 0.0)),
            }
        )

        reasoning_log.append(
            {
                "strategy_chunk_id": strategy_chunk_id,
                "strategy": strategy,
                "diagnosis": diagnosis,
                "final_proposal": proposal,
                "final_critique": critique,
            }
        )

    recommendations_df = pd.DataFrame(recommendation_rows)
    return recommendations_df, reasoning_log
