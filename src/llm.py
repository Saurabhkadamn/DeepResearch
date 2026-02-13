"""
Deep Research - LLM Client
OpenRouter API calls with dual model support.
"""

import json
import httpx
from typing import AsyncIterator

from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL_FAST


async def call_llm(
    prompt: str,
    model: str = "",
    system_prompt: str = "You are a helpful research assistant. Always respond in valid JSON.",
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """Call LLM via OpenRouter. Returns full response text."""
    if not model:
        model = LLM_MODEL_FAST

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Deep Research",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def call_llm_streaming(
    prompt: str,
    model: str = "",
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.3,
    max_tokens: int = 8192,
) -> AsyncIterator[str]:
    """Stream LLM response via OpenRouter. Yields text chunks."""
    if not model:
        model = LLM_MODEL_FAST

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Deep Research",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        content = data["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


async def call_llm_simple(
    messages: list[dict],
    model: str = "",
    temperature: float = 0.3,
) -> str:
    """Simple chat completion with message list. Used for normal chat mode."""
    if not model:
        model = LLM_MODEL_FAST

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Deep Research",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def parse_llm_json(response: str) -> dict:
    """Parse JSON from LLM response. Handles markdown code blocks."""
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse LLM JSON: {e}")
        print(f"[DEBUG] Raw: {text[:300]}")
        return {}
