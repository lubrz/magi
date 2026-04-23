"""
TRIAD configuration — loaded from environment variables / .env file.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class Neo4jMode(str, Enum):
    LOCAL = "local"
    AURA = "aura"


class AgentConfig(BaseSettings):
    """Configuration for a single agent's LLM connection."""

    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = "llama3.2"
    api_key: Optional[str] = None


class Settings(BaseSettings):
    """Global application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Neo4j ---
    neo4j_mode: Neo4jMode = Field(default=Neo4jMode.LOCAL, alias="NEO4J_MODE")
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="triad2026", alias="NEO4J_PASSWORD")

    # --- Embedding ---
    embedding_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI, alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    embedding_api_key: Optional[str] = Field(default=None, alias="EMBEDDING_API_KEY")

    # --- Ollama (shared) ---
    ollama_url: str = Field(default="http://localhost:11434", alias="OLLAMA_URL")

    # --- Agent configs (loaded from prefixed env vars) ---
    axiom_provider: LLMProvider = Field(default=LLMProvider.OLLAMA, alias="AXIOM_PROVIDER")
    axiom_model: str = Field(default="llama3.2", alias="AXIOM_MODEL")
    axiom_api_key: Optional[str] = Field(default=None, alias="AXIOM_API_KEY")

    prism_provider: LLMProvider = Field(default=LLMProvider.OLLAMA, alias="PRISM_PROVIDER")
    prism_model: str = Field(default="llama3.2", alias="PRISM_MODEL")
    prism_api_key: Optional[str] = Field(default=None, alias="PRISM_API_KEY")

    forge_provider: LLMProvider = Field(default=LLMProvider.OLLAMA, alias="FORGE_PROVIDER")
    forge_model: str = Field(default="llama3.2", alias="FORGE_MODEL")
    forge_api_key: Optional[str] = Field(default=None, alias="FORGE_API_KEY")

    # --- Deliberation ---
    max_rounds: int = Field(default=3, alias="MAX_ROUNDS")
    consensus_threshold: float = Field(default=0.7, alias="CONSENSUS_THRESHOLD")
    consensus_min_votes: int = Field(default=2, alias="CONSENSUS_MIN_VOTES")

    # --- Server ---
    backend_host: str = Field(default="0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(default=8080, alias="BACKEND_PORT")

    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Return an AgentConfig for the given agent name."""
        name = agent_name.lower()
        return AgentConfig(
            provider=getattr(self, f"{name}_provider"),
            model=getattr(self, f"{name}_model"),
            api_key=getattr(self, f"{name}_api_key"),
        )

    @property
    def neo4j_is_aura(self) -> bool:
        return self.neo4j_mode == Neo4jMode.AURA


settings = Settings()
