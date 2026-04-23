"""
Base Agent class — shared logic for all three TRIAD agents.
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.llm_providers import BaseLLMProvider
from models.schemas import (
    AgentCritique,
    AgentName,
    AgentPosition,
    GraphContext,
    GraphSource,
)

logger = logging.getLogger(__name__)

# JSON schema that agents must follow for structured responses
POSITION_JSON_SCHEMA = """\
{
  "position": "<your clear, concise answer/stance>",
  "reasoning": "<detailed reasoning with evidence from the provided context>",
  "confidence": <float 0.0-1.0>,
  "sources_used": ["<source title 1>", "<source title 2>"]
}"""

CRITIQUE_JSON_SCHEMA = """\
{
  "agreement": <float 0.0-1.0 — how much you agree with this position>,
  "critique": "<your analysis of strengths and weaknesses>",
  "revised_confidence": <float 0.0-1.0 — your updated confidence in YOUR OWN position after seeing this>
}"""


# ---------------------------------------------------------------------------
# Safe value helpers — prevent Pydantic validation errors from LLM output
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.5) -> float:
    """
    Safely parse a float value from LLM output.

    Handles None, strings, out-of-range values, and non-numeric garbage.
    Always returns a float clamped to [0.0, 1.0].
    """
    if value is None:
        return default
    try:
        f = float(value)
        return min(1.0, max(0.0, f))
    except (ValueError, TypeError):
        logger.debug(f"Could not parse float from {value!r}, using default {default}")
        return default


def _safe_str(value, default: str = "") -> str:
    """
    Safely coerce a value to a non-empty string.

    Handles None, lists, dicts, and other non-string types that LLMs
    sometimes produce instead of a plain string.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value if value.strip() else default
    # If the LLM returned a list or dict, serialise it
    try:
        import json
        return json.dumps(value)
    except Exception:
        return str(value) or default


class BaseAgent:
    """
    Base class for TRIAD agents.

    Each agent has:
    - A unique name and system prompt defining its perspective
    - An LLM provider for generation
    - Access to a labeled subgraph in Neo4j for context retrieval
    """

    name: AgentName
    label: str  # Neo4j node label, e.g. "AxiomConcept"
    system_prompt: str

    def __init__(
        self,
        llm: BaseLLMProvider,
        graph_manager=None,
    ):
        self.llm = llm
        self.graph = graph_manager

    async def retrieve_context(self, question: str) -> GraphContext:
        """Retrieve relevant context from this agent's knowledge graph."""
        if self.graph is None:
            return GraphContext()

        try:
            return await self.graph.retrieve(
                question=question,
                label=self.label,
            )
        except Exception as e:
            logger.warning(f"[{self.name}] Knowledge graph retrieval failed: {e}")
            return GraphContext()

    async def respond(
        self,
        question: str,
        context: Optional[GraphContext] = None,
        round_number: int = 1,
        previous_positions: Optional[list[AgentPosition]] = None,
    ) -> AgentPosition:
        """
        Generate a position on the given question.

        In round 1, the agent responds based on its own knowledge graph.
        In later rounds, it also considers other agents' positions.
        """
        if context is None:
            context = await self.retrieve_context(question)

        # Build the user message
        parts = [f"**Question:** {question}"]

        # Add knowledge graph context
        if context.raw_text:
            parts.append(
                f"\n**Relevant knowledge from your domain:**\n{context.raw_text}"
            )

        # In later rounds, show other agents' positions
        if previous_positions:
            parts.append("\n**Other agents' positions from the previous round:**")
            for pos in previous_positions:
                if pos.agent != self.name:
                    parts.append(
                        f"\n- **{pos.agent.value.upper()}** (confidence: {pos.confidence:.2f}):\n"
                        f"  Position: {pos.position}\n"
                        f"  Reasoning: {pos.reasoning}"
                    )
            parts.append(
                "\nConsider their perspectives. You may update your position, "
                "strengthen it, or respectfully disagree with evidence."
            )

        # Add explicit instructions for source-grounded, independent reasoning
        parts.append(
            "\n**IMPORTANT:**\n- Your reasoning MUST cite at least one identified source from the provided context above.\n- You may not agree or disagree solely because another agent said so; your analysis must be independent and grounded in evidence.\n- If no relevant source exists for your position, state this explicitly and explain your reasoning."
        )

        parts.append(
            f"\n\nRespond with this exact JSON format:\n{POSITION_JSON_SCHEMA}"
        )

        user_message = "\n".join(parts)

        try:
            result = await self.llm.generate_json(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=self.system_prompt,
                temperature=0.7 if round_number == 1 else 0.5,
            )

            # Map source titles to GraphSource objects
            sources_used = result.get("sources_used", [])
            if not isinstance(sources_used, list):
                sources_used = []
            sources = []
            for title in sources_used:
                title_str = _safe_str(title)
                if not title_str:
                    continue
                # Try to match against context sources
                matched = next(
                    (s for s in context.sources if s.title == title_str), None
                )
                if matched:
                    sources.append(matched)
                else:
                    sources.append(GraphSource(title=title_str))

            return AgentPosition(
                agent=self.name,
                position=_safe_str(result.get("position"), "No position generated"),
                reasoning=_safe_str(result.get("reasoning"), "No reasoning provided"),
                confidence=_safe_float(result.get("confidence"), 0.5),
                sources=sources,
                round_number=round_number,
            )
        except Exception as e:
            logger.error(f"[{self.name}] Failed to generate response: {e}")
            return AgentPosition(
                agent=self.name,
                position=f"Error generating response: {str(e)[:200]}",
                reasoning="Agent encountered an error during generation.",
                confidence=0.0,
                sources=[],
                round_number=round_number,
            )

    async def critique(
        self,
        question: str,
        target_position: AgentPosition,
    ) -> AgentCritique:
        """Critique another agent's position."""
        user_message = (
            f"**Question:** {question}\n\n"
            f"**{target_position.agent.value.upper()}'s position** "
            f"(confidence: {target_position.confidence:.2f}):\n"
            f"Position: {target_position.position}\n"
            f"Reasoning: {target_position.reasoning}\n\n"
            f"Analyze this position from your perspective. "
            f"Identify strengths, weaknesses, and any overlooked factors.\n\n"
            f"Respond with this exact JSON format:\n{CRITIQUE_JSON_SCHEMA}"
        )

        try:
            result = await self.llm.generate_json(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=self.system_prompt,
                temperature=0.4,
            )

            return AgentCritique(
                critic=self.name,
                target=target_position.agent,
                agreement=_safe_float(result.get("agreement"), 0.5),
                critique=_safe_str(result.get("critique"), "No critique provided"),
                revised_confidence=_safe_float(result.get("revised_confidence"), 0.5),
            )
        except Exception as e:
            logger.error(f"[{self.name}] Failed to critique: {e}")
            return AgentCritique(
                critic=self.name,
                target=target_position.agent,
                agreement=0.5,
                critique=f"Error during critique: {str(e)[:200]}",
                revised_confidence=0.5,
            )
