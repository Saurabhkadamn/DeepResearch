"""
Deep Research - LLM Client
OpenRouter API calls with dual model support + reasoning model.
"""

import json
import httpx
from typing import AsyncIterator

from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL_FAST,
    LLM_MODEL_REASONING,
    REASONING_BUDGET_TOKENS,
)


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


async def call_llm_thinking(
    prompt: str,
    model: str = "",
    system_prompt: str = "You are an expert research strategist. Think deeply before answering.",
    budget_tokens: int = REASONING_BUDGET_TOKENS,
) -> tuple[str, str]:
    """
    Call a reasoning model via OpenRouter's unified reasoning API.

    OpenRouter normalises reasoning across all providers with one param:
        request:  "reasoning": {"effort": "high"}   (or "max_tokens": N)
        response: message["reasoning_details"]       (list of thinking blocks)

    Supported models (set LLM_MODEL_REASONING in .env):
        moonshotai/kimi-k2.5          ← recommended, cheapest, great quality
        moonshotai/kimi-k2-thinking   ← dedicated reasoning variant
        deepseek/deepseek-r1          ← budget option
        anthropic/claude-sonnet-4-5:thinking
        openai/o3, openai/o4-mini

    Returns:
        (thinking_trace, final_answer)
        thinking_trace: raw chain-of-thought (empty string if not exposed)
        final_answer:   the final response text
    """
    if not model:
        model = LLM_MODEL_REASONING

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Deep Research",
    }

    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 16000,
        # ✅ OpenRouter unified reasoning param — works for ALL reasoning models
        # (Kimi K2.5, Kimi K2 Thinking, DeepSeek R1, Claude, OpenAI o-series)
        "reasoning": {
            "effort": "high",       # "low" | "medium" | "high"
            # alternative: "max_tokens": budget_tokens
        },
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    message = data["choices"][0]["message"]
    thinking_trace = ""
    final_answer   = message.get("content", "") or ""

    # ── Priority 1: OpenRouter unified reasoning_details array ─────────
    # Returned by Kimi K2.5, Kimi K2 Thinking, Claude, OpenAI o-series
    reasoning_details = message.get("reasoning_details", [])
    if reasoning_details:
        thinking_trace = "\n".join(
            block.get("thinking", "")
            for block in reasoning_details
            if block.get("type") == "thinking"
        )

    # ── Priority 2: DeepSeek-style <think> tags in content string ──────
    # Fallback for older DeepSeek R1 responses not yet normalised by OpenRouter
    if not thinking_trace and "<think>" in final_answer and "</think>" in final_answer:
        s = final_answer.index("<think>") + len("<think>")
        e = final_answer.index("</think>")
        thinking_trace = final_answer[s:e].strip()
        final_answer   = final_answer[e + len("</think>"):].strip()

    # ── Priority 3: Claude content-block list (direct Anthropic SDK style) ─
    # Shouldn't happen via OpenRouter but kept as safety net
    if not thinking_trace and isinstance(message.get("content"), list):
        text_parts = []
        for block in message["content"]:
            if block.get("type") == "thinking":
                thinking_trace = block.get("thinking", "")
            elif block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        if text_parts:
            final_answer = "\n".join(text_parts)

    # ── Priority 4: raw reasoning_content field ────────────────────────
    if not thinking_trace and message.get("reasoning_content"):
        thinking_trace = message["reasoning_content"]

    return thinking_trace, final_answer


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