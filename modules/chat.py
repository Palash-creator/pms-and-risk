"""Chat provider adapters."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

import httpx

LOGGER = logging.getLogger(__name__)


def build_system_prompt(summary: Dict[str, Any], portfolio: Dict[str, Any]) -> str:
    context = {
        "portfolio": portfolio,
        "summary": summary,
        "instruction": "Answer ONLY using provided context; if information is missing, direct the user to the page that contains it.",
    }
    return json.dumps(context, indent=2, default=str)


class ChatProviderError(RuntimeError):
    """Raised when provider execution fails."""


def call_groq(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    if not api_key:
        raise ChatProviderError("Groq API key missing")
    payload = {
        "model": model or "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = httpx.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=40)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except httpx.HTTPError as exc:  # noqa: BLE001
        LOGGER.exception("Groq call failed")
        raise ChatProviderError(str(exc)) from exc


def call_gemini(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    if not api_key:
        raise ChatProviderError("Gemini API key missing")
    model_name = model or "gemini-pro"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": system_prompt},
                    {"text": user_prompt},
                ]
            }
        ],
    }
    try:
        response = httpx.post(url, json=payload, timeout=40)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "No response"
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in parts]
        return "\n".join(filter(None, texts)) or "No response"
    except httpx.HTTPError as exc:  # noqa: BLE001
        LOGGER.exception("Gemini call failed")
        raise ChatProviderError(str(exc)) from exc
