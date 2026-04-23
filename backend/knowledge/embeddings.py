"""
Embedding provider abstraction for vector-based retrieval.

Generates dense vector embeddings from text so concept nodes can be
stored with an ``embedding`` property and retrieved via Neo4j's native
vector index (cosine similarity).

Supported backends:
  - OpenAI  (text-embedding-3-small, text-embedding-ada-002, etc.)
  - Ollama  (nomic-embed-text, mxbai-embed-large, etc.)
  - NoOp    (returns None — graceful fallback when no provider is configured)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseEmbeddingProvider(ABC):
    """Common interface for embedding providers."""

    def __init__(self, model: str, dimensions: int = 1536, **kwargs):
        self.model = model
        self.dimensions = dimensions

    @abstractmethod
    async def embed(self, text: str) -> Optional[list[float]]:
        """Generate a vector embedding for *text*. Returns None on failure."""
        ...


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str, api_key: str, dimensions: int = 1536, **kwargs):
        super().__init__(model, dimensions)
        self.api_key = api_key

    async def embed(self, text: str) -> Optional[list[float]]:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)
            response = await client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"OpenAI embedding failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 dimensions: int = 768, **kwargs):
        super().__init__(model, dimensions)
        self.base_url = base_url

    async def embed(self, text: str) -> Optional[list[float]]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": text},
                )
                if resp.status_code != 200:
                    logger.warning(f"Ollama embed error {resp.status_code}: {resp.text[:200]}")
                    return None
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if embeddings and len(embeddings) > 0:
                    return embeddings[0]
                return None
        except Exception as e:
            logger.warning(f"Ollama embedding failed: {e}")
            return None


# ---------------------------------------------------------------------------
# NoOp (fallback)
# ---------------------------------------------------------------------------

class NoOpEmbeddingProvider(BaseEmbeddingProvider):
    """Returns None for all embed calls — used when no provider is configured."""

    async def embed(self, text: str) -> Optional[list[float]]:
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_embedding_provider(settings) -> BaseEmbeddingProvider:
    """
    Create an embedding provider from application settings.

    Falls back to NoOp if the provider is not recognized or API keys
    are missing, ensuring the system can always start.
    """
    from config import LLMProvider

    provider_type = settings.embedding_provider
    model = settings.embedding_model
    api_key = settings.embedding_api_key
    dimensions = settings.embedding_dimensions

    if provider_type == LLMProvider.OPENAI and api_key:
        logger.info(f"Embedding provider: OpenAI ({model}, {dimensions}d)")
        return OpenAIEmbeddingProvider(
            model=model, api_key=api_key, dimensions=dimensions,
        )
    elif provider_type == LLMProvider.OLLAMA:
        ollama_url = settings.ollama_url
        logger.info(f"Embedding provider: Ollama ({model}, {dimensions}d) at {ollama_url}")
        return OllamaEmbeddingProvider(
            model=model, base_url=ollama_url, dimensions=dimensions,
        )
    else:
        if provider_type != LLMProvider.OLLAMA and not api_key:
            logger.warning(
                f"Embedding provider '{provider_type.value}' configured but no API key set. "
                f"Falling back to NoOp — vector search will be disabled."
            )
        else:
            logger.warning(
                f"Unsupported embedding provider '{provider_type.value}'. "
                f"Falling back to NoOp — vector search will be disabled."
            )
        return NoOpEmbeddingProvider(model="none", dimensions=dimensions)
