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
            sources = []
            for title in sources_used:
                # Try to match against context sources
                matched = next(
                    (s for s in context.sources if s.title == title), None
                )
                if matched:
                    sources.append(matched)
                else:
                    sources.append(GraphSource(title=title))

            return AgentPosition(
                agent=self.name,
                position=result.get("position", "No position generated"),
                reasoning=result.get("reasoning", "No reasoning provided"),
                confidence=min(1.0, max(0.0, float(result.get("confidence", 0.5)))),
                sources=sources,
                round_number=round_number,
            )
        except Exception as e:
            logger.error(f"[{self.name}] Failed to generate response: {e}")
            return AgentPosition(
                agent=self.name,
                position=f"Error generating response: {str(e)}",
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
                agreement=min(1.0, max(0.0, float(result.get("agreement", 0.5)))),
                critique=result.get("critique", "No critique provided"),
                revised_confidence=min(
                    1.0, max(0.0, float(result.get("revised_confidence", 0.5)))
                ),
            )
        except Exception as e:
            logger.error(f"[{self.name}] Failed to critique: {e}")
            return AgentCritique(
                critic=self.name,
                target=target_position.agent,
                agreement=0.5,
                critique=f"Error during critique: {str(e)}",
                revised_confidence=0.5,
            )
