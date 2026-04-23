"""
Multi-provider LLM abstraction layer.

Each provider implements the same async interface so agents are decoupled
from the underlying model vendor.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from config import LLMProvider, Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """Common interface for all LLM providers."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
    ) -> str:
        """Generate a completion from the model."""
        ...

    async def generate_json(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> dict:
        """Generate and parse a JSON response."""
        json_instruction = (
            "\n\nYou MUST respond with valid JSON only. "
            "No markdown fences, no extra text."
        )
        result = await self.generate(
            messages=messages,
            system_prompt=system_prompt + json_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Strip any markdown code fences the model might add
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        return json.loads(cleaned)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseLLMProvider):
    async def generate(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
    ) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        kwargs = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseLLMProvider):
    async def generate(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
    ) -> str:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self.api_key)

        # Anthropic uses a separate system parameter
        response = await client.messages.create(
            model=self.model,
            system=system_prompt or "You are a helpful assistant.",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content[0].text


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GoogleProvider(BaseLLMProvider):
    async def generate(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
    ) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        # Convert messages to Gemini content format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])],
            ))

        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = await client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text or ""


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------

class OllamaProvider(BaseLLMProvider):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model, api_key)
        self.base_url = base_url

    async def generate(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
    ) -> str:
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_provider(
    provider: LLMProvider,
    model: str,
    api_key: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> BaseLLMProvider:
    """Create an LLM provider instance from configuration."""
    if provider == LLMProvider.OPENAI:
        return OpenAIProvider(model=model, api_key=api_key)
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicProvider(model=model, api_key=api_key)
    elif provider == LLMProvider.GOOGLE:
        return GoogleProvider(model=model, api_key=api_key)
    elif provider == LLMProvider.OLLAMA:
        base_url = "http://localhost:11434"
        if settings:
            base_url = settings.ollama_url
        return OllamaProvider(model=model, base_url=base_url)
    else:
        raise ValueError(f"Unknown provider: {provider}")
