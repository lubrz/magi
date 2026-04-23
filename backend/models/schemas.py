"""
Pydantic models for API requests, responses, and internal data structures.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentName(str, Enum):
    AXIOM = "axiom"
    PRISM = "prism"
    FORGE = "forge"


class ConsensusStatus(str, Enum):
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    DEADLOCK = "deadlock"
    PENDING = "pending"


class DeliberationStatus(str, Enum):
    RUNNING = "running"
    CONSENSUS = "consensus"
    DEADLOCK = "deadlock"
    USER_DECIDED = "user_decided"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

class GraphSource(BaseModel):
    """A source document from the knowledge graph."""
    title: str
    url: Optional[str] = None
    source_type: str = "document"
    relevance: float = 0.0


class GraphContext(BaseModel):
    """Context retrieved from an agent's knowledge graph."""
    concepts: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    sources: list[GraphSource] = Field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Agent Responses
# ---------------------------------------------------------------------------

class AgentPosition(BaseModel):
    """A single agent's position on the question."""
    agent: AgentName
    position: str = Field(description="The agent's answer/stance")
    reasoning: str = Field(description="Detailed reasoning behind the position")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    sources: list[GraphSource] = Field(default_factory=list)
    round_number: int = 1


class AgentCritique(BaseModel):
    """An agent's critique of another agent's position."""
    critic: AgentName
    target: AgentName
    agreement: float = Field(ge=0.0, le=1.0, description="How much the critic agrees 0-1")
    critique: str = Field(description="The critique text")
    revised_confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Deliberation
# ---------------------------------------------------------------------------

class RoundResult(BaseModel):
    """Result of a single deliberation round."""
    round_number: int
    positions: list[AgentPosition]
    critiques: list[AgentCritique] = Field(default_factory=list)
    consensus_status: ConsensusStatus = ConsensusStatus.PENDING


class ConsensusResult(BaseModel):
    """Final consensus evaluation."""
    status: ConsensusStatus
    agreeing_agents: list[AgentName] = Field(default_factory=list)
    dissenting_agents: list[AgentName] = Field(default_factory=list)
    unified_position: Optional[str] = None
    confidence: float = 0.0


class DeliberationResult(BaseModel):
    """Complete result of a deliberation process."""
    id: str
    question: str
    status: DeliberationStatus
    rounds: list[RoundResult] = Field(default_factory=list)
    consensus: Optional[ConsensusResult] = None
    final_answer: Optional[str] = None
    selected_by: Optional[str] = None  # "system" or "user"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# API Request / Response
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Request to submit a question to TRIAD."""
    question: str = Field(min_length=3, max_length=2000)
    max_rounds: Optional[int] = Field(default=None, ge=1, le=10)


class UserVoteRequest(BaseModel):
    """User selects an agent's position during deadlock."""
    deliberation_id: str
    selected_agent: AgentName


class HealthResponse(BaseModel):
    """System health check response."""
    status: str = "ok"
    neo4j: str = "unknown"
    agents: dict[str, str] = Field(default_factory=dict)
    graph_stats: dict[str, int] = Field(default_factory=dict)


class GraphStatsResponse(BaseModel):
    """Knowledge graph statistics."""
    total_concepts: int = 0
    total_relationships: int = 0
    total_sources: int = 0
    per_agent: dict[str, dict[str, int]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# WebSocket Events
# ---------------------------------------------------------------------------

class WSEventType(str, Enum):
    SYSTEM_STATUS = "system_status"
    ROUND_START = "round_start"
    AGENT_THINKING = "agent_thinking"
    AGENT_RESPONSE = "agent_response"
    AGENT_CRITIQUE = "agent_critique"
    CONSENSUS_CHECK = "consensus_check"
    CONSENSUS_REACHED = "consensus_reached"
    DEADLOCK = "deadlock"
    USER_VOTE_REQUIRED = "user_vote_required"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


class WSEvent(BaseModel):
    """WebSocket event sent to the frontend."""
    type: WSEventType
    data: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
