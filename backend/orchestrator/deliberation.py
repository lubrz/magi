"""
Deliberation orchestrator — manages multi-round agent debate.

Runs up to N rounds of:
  1. All agents generate positions in parallel
  2. Agents critique each other's positions
  3. Consensus is evaluated
  4. If no consensus, agents see critiques and iterate
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import AsyncGenerator, Callable, Optional

from agents.base import BaseAgent
from models.schemas import (
    AgentName,
    AgentPosition,
    ConsensusStatus,
    DeliberationResult,
    DeliberationStatus,
    RoundResult,
    WSEvent,
    WSEventType,
)
from orchestrator.consensus import evaluate_consensus

logger = logging.getLogger(__name__)


class Deliberation:
    """
    Manages a single deliberation session across multiple rounds.
    """

    def __init__(
        self,
        agents: dict[AgentName, BaseAgent],
        arbiter: BaseAgent,
        max_rounds: int = 3,
        consensus_threshold: float = 0.7,
        consensus_min_votes: int = 2,
    ):
        self.agents = {k: v for k, v in agents.items() if k != AgentName.ARBITER}
        self.arbiter = arbiter
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.consensus_min_votes = consensus_min_votes
        self.id = str(uuid.uuid4())[:8]

    async def run(
        self,
        question: str,
        on_event: Optional[Callable[[WSEvent], None]] = None,
    ) -> DeliberationResult:
        """
        Run the full deliberation loop.

        Args:
            question: The question to deliberate on.
            on_event: Optional callback for real-time event streaming.

        Returns:
            Complete deliberation result.
        """
        start_time = time.time()
        rounds: list[RoundResult] = []
        previous_positions: Optional[list[AgentPosition]] = None

        self._emit(on_event, WSEventType.SYSTEM_STATUS, {
            "message": f"Deliberation {self.id} started",
            "question": question,
            "max_rounds": self.max_rounds,
        })

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"[{self.id}] Starting round {round_num}/{self.max_rounds}")

            self._emit(on_event, WSEventType.ROUND_START, {
                "round": round_num,
                "max_rounds": self.max_rounds,
            })

            # --- Phase 1: All agents generate positions in parallel ---
            positions = await self._gather_positions(
                question, round_num, previous_positions, on_event
            )

            # --- Phase 2: Agents critique each other ---
            critiques = []
            if round_num <= self.max_rounds:
                critiques = await self._gather_critiques(
                    question, positions, on_event
                )

            # --- Phase 3: Evaluate consensus ---
            self._emit(on_event, WSEventType.CONSENSUS_CHECK, {
                "round": round_num,
            })

            # Arbiter evaluates consensus
            consensus = await self.arbiter.evaluate_consensus(
                question=question,
                positions=positions,
                critiques=critiques
            )

            round_result = RoundResult(
                round_number=round_num,
                positions=positions,
                critiques=critiques,
                consensus_status=consensus.status,
            )
            rounds.append(round_result)

            # --- Check if we have consensus ---
            if consensus.status in (
                ConsensusStatus.UNANIMOUS,
                ConsensusStatus.MAJORITY,
            ):
                logger.info(
                    f"[{self.id}] Consensus reached in round {round_num}: "
                    f"{consensus.status.value}"
                )

                self._emit(on_event, WSEventType.CONSENSUS_REACHED, {
                    "status": consensus.status.value,
                    "agreeing": [a.value for a in consensus.agreeing_agents],
                    "dissenting": [a.value for a in consensus.dissenting_agents],
                    "position": consensus.unified_position,
                    "confidence": consensus.confidence,
                })

                duration = time.time() - start_time
                return DeliberationResult(
                    id=self.id,
                    question=question,
                    status=DeliberationStatus.CONSENSUS,
                    rounds=rounds,
                    consensus=consensus,
                    final_answer=consensus.unified_position,
                    selected_by="system",
                    duration_seconds=round(duration, 2),
                )

            # No consensus — prepare for next round
            logger.info(f"[{self.id}] No consensus in round {round_num}")
            previous_positions = positions

        # --- Deadlock: max rounds exceeded ---
        logger.info(f"[{self.id}] Deadlock after {self.max_rounds} rounds")

        self._emit(on_event, WSEventType.DEADLOCK, {
            "rounds_completed": self.max_rounds,
            "positions": [
                {
                    "agent": p.agent.value,
                    "position": p.position,
                    "confidence": p.confidence,
                    "reasoning": p.reasoning,
                }
                for p in rounds[-1].positions
            ],
        })

        final_consensus = await self.arbiter.evaluate_consensus(
            question=question,
            positions=rounds[-1].positions,
            critiques=rounds[-1].critiques
        )

        duration = time.time() - start_time
        return DeliberationResult(
            id=self.id,
            question=question,
            status=DeliberationStatus.DEADLOCK,
            rounds=rounds,
            consensus=final_consensus,
            final_answer=None,
            selected_by=None,
            duration_seconds=round(duration, 2),
        )

    async def _gather_positions(
        self,
        question: str,
        round_number: int,
        previous_positions: Optional[list[AgentPosition]],
        on_event: Optional[Callable],
    ) -> list[AgentPosition]:
        """Gather positions from all agents in parallel."""

        async def _get_position(agent: BaseAgent) -> AgentPosition:
            self._emit(on_event, WSEventType.AGENT_THINKING, {
                "agent": agent.name.value,
                "round": round_number,
            })

            # Retrieve context and emit what was found
            context = await agent.retrieve_context(question)

            self._emit(on_event, WSEventType.CONTEXT_RETRIEVED, {
                "agent": agent.name.value,
                "round": round_number,
                "concepts_found": len(context.concepts),
                "concept_names": context.concepts[:5],
                "sources_found": len(context.sources),
                "source_names": [s.title for s in context.sources[:5]],
                "has_context": bool(context.raw_text),
            })

            logger.info(
                f"[{self.id}] {agent.name.value.upper()} retrieved "
                f"{len(context.concepts)} concept(s), "
                f"{len(context.sources)} source(s) from graph"
            )

            # Loop until Arbiter accepts or max retries
            max_retries = 2
            for attempt in range(max_retries):
                position = await agent.respond(
                    question=question,
                    context=context,
                    round_number=round_number,
                    previous_positions=previous_positions,
                )
                
                # Arbiter review
                review = await self.arbiter.review_position(
                    question=question,
                    position=position,
                    other_positions=previous_positions or []
                )
                
                if review["approved"]:
                    self._emit(on_event, WSEventType.SYSTEM_STATUS, {
                        "message": f"ARBITER APPROVED {agent.name.value.upper()}'S POSITION"
                    })
                    break
                
                # If rejected, we retry
                logger.warning(f"[{self.id}] Arbiter rejected {agent.name.value}'s position: {review['reason']}. Retrying...")
                
                self._emit(on_event, WSEventType.SYSTEM_STATUS, {
                    "message": f"ARBITER REJECTED {agent.name.value.upper()}'S POSITION: {review['reason']}"
                })
                
                agent.system_prompt += f"\n\nARBITER FEEDBACK: {review['feedback']} - DO NOT COPY OTHERS AND ALIGN WITH YOUR PROFILE."

            self._emit(on_event, WSEventType.AGENT_RESPONSE, {
                "agent": position.agent.value,
                "position": position.position,
                "reasoning": position.reasoning,
                "confidence": position.confidence,
                "sources": [s.title for s in position.sources],
                "round": round_number,
            })

            return position

        # Run all agents in parallel
        tasks = [_get_position(agent) for agent in self.agents.values()]
        positions = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_positions = []
        for pos in positions:
            if isinstance(pos, Exception):
                logger.error(f"Agent error: {pos}")
            else:
                valid_positions.append(pos)

        return valid_positions

    async def _gather_critiques(
        self,
        question: str,
        positions: list[AgentPosition],
        on_event: Optional[Callable],
    ) -> list:
        """Each agent critiques the other agents' positions."""

        async def _critique(
            agent: BaseAgent, target: AgentPosition
        ):
            max_retries = 2
            for attempt in range(max_retries):
                critique = await agent.critique(question, target)
                
                # We can reuse review_position but adapt it for critique by faking a position
                review = await self.arbiter.review_position(
                    question=question,
                    position=AgentPosition(
                        agent=agent.name,
                        position=critique.critique,
                        reasoning=f"Agreement: {critique.agreement}, Target: {target.agent.value}",
                        confidence=critique.revised_confidence,
                        sources=[],
                        round_number=0
                    ),
                    other_positions=[target]
                )
                
                if review["approved"]:
                    self._emit(on_event, WSEventType.SYSTEM_STATUS, {
                        "message": f"ARBITER APPROVED {agent.name.value.upper()}'S CRITIQUE"
                    })
                    break
                    
                logger.warning(f"[{self.id}] Arbiter rejected {agent.name.value}'s critique: {review['reason']}. Retrying...")
                self._emit(on_event, WSEventType.SYSTEM_STATUS, {
                    "message": f"ARBITER REJECTED {agent.name.value.upper()}'S CRITIQUE: {review['reason']}"
                })
                
                agent.system_prompt += f"\n\nARBITER FEEDBACK ON CRITIQUE: {review['feedback']}"

            self._emit(on_event, WSEventType.AGENT_CRITIQUE, {
                "critic": critique.critic.value,
                "target": critique.target.value,
                "agreement": critique.agreement,
                "critique": critique.critique,
                "revised_confidence": critique.revised_confidence,
            })

            return critique

        tasks = []
        for agent in self.agents.values():
            for pos in positions:
                if pos.agent != agent.name:
                    tasks.append(_critique(agent, pos))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Critique error: {r}")
            else:
                valid.append(r)

        return valid

    @staticmethod
    def _emit(
        callback: Optional[Callable],
        event_type: WSEventType,
        data: dict,
    ) -> None:
        """Emit an event if a callback is registered."""
        if callback:
            try:
                callback(WSEvent(type=event_type, data=data))
            except Exception as e:
                logger.warning(f"Event emission failed: {e}")
