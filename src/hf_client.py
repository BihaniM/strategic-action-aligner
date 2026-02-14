from __future__ import annotations

import json
from typing import Any

import numpy as np
import requests

from src.config import HF_API_BASE, HF_CHAT_MODEL, HF_EMBEDDING_MODEL, HF_TOKEN


class HFClient:
    def __init__(self) -> None:
        self.api_base = HF_API_BASE
        self.token = HF_TOKEN
        self.embedding_model = HF_EMBEDDING_MODEL
        self.chat_model = HF_CHAT_MODEL

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _model_url(self, model_name: str) -> str:
        return f"{self.api_base}/{model_name}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            response = requests.post(
                self._model_url(self.embedding_model),
                headers=self._headers(),
                json={
                    "inputs": text,
                    "options": {"wait_for_model": True},
                },
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            vectors.append(self._normalize_embedding_payload(payload))
        return vectors

    def _normalize_embedding_payload(self, payload: Any) -> list[float]:
        if isinstance(payload, list) and payload and isinstance(payload[0], (int, float)):
            return [float(value) for value in payload]

        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            matrix = np.array(payload, dtype=float)
            if matrix.ndim == 2:
                return matrix.mean(axis=0).tolist()

        raise ValueError("Unexpected Hugging Face embedding payload format.")

    def generate_json(self, system_prompt: str, user_payload: dict[str, Any], max_new_tokens: int = 700) -> dict[str, Any]:
        prompt = (
            f"{system_prompt}\n"
            "Return strict JSON only.\n"
            f"Input:\n{json.dumps(user_payload, ensure_ascii=False)}"
        )

        response = requests.post(
            self._model_url(self.chat_model),
            headers=self._headers(),
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                    "return_full_text": False,
                },
                "options": {"wait_for_model": True},
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        text = self._extract_generated_text(payload)
        return self._extract_json_object(text)

    def _extract_generated_text(self, payload: Any) -> str:
        if isinstance(payload, list) and payload and "generated_text" in payload[0]:
            return str(payload[0]["generated_text"])
        if isinstance(payload, dict) and "generated_text" in payload:
            return str(payload["generated_text"])
        raise ValueError("Unexpected Hugging Face generation payload format.")

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in Hugging Face model output.")
        return json.loads(text[start : end + 1])
