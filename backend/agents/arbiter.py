"""
Arbiter (Judge) Agent — evaluates outputs for profile alignment and originality.
"""

from __future__ import annotations

import logging
import json
from typing import Optional

from agents.base import BaseAgent
from models.schemas import AgentName, AgentPosition, ConsensusResult, ConsensusStatus

logger = logging.getLogger(__name__)

REVIEW_JSON_SCHEMA = """\
{
  "approved": <boolean>,
  "reason": "<explanation of why it is approved or rejected>",
  "feedback": "<instructions for the agent to fix their response if rejected>"
}"""

CONSENSUS_JSON_SCHEMA = """\
{
  "status": "<unanimous, majority, or deadlock>",
  "agreeing_agents": ["<agent name 1>", "<agent name 2>"],
  "dissenting_agents": ["<agent name>"],
  "unified_position": "<a merged position statement if consensus reached, else null>",
  "confidence": <float 0.0-1.0>
}"""

class ArbiterAgent(BaseAgent):
    """
    Arbiter agent ensures other agents align with their cognitive profiles
    and do not plagiarize. It evaluates consensus.
    """
    name = AgentName.ARBITER
    label = "ArbiterConcept" 
    system_prompt = "You are the strict ARBITER in a multi-agent system."

    async def review_position(
        self,
        question: str,
        position: AgentPosition,
        other_positions: list[AgentPosition]
    ) -> dict:
        """
        Review an agent's position for profile alignment and originality.
        Returns a dict with 'approved', 'reason', 'feedback'.
        """
        other_text = "\n".join(
            f"- {p.agent.value.upper()}: {p.position}" 
            for p in other_positions if p.agent != position.agent
        )
        
        user_message = (
            f"**Question:** {question}\n\n"
            f"**Agent being reviewed:** {position.agent.value.upper()}\n"
            f"**Agent's Profile:**\n"
            f"- AXIOM: Science, empirical data, rigorous analysis.\n"
            f"- PRISM: Culture, sociology, ethics, philosophy.\n"
            f"- FORGE: Engineering, pragmatism, implementation.\n\n"
            f"**Agent's Position:** {position.position}\n"
            f"**Agent's Reasoning:** {position.reasoning}\n\n"
            f"**Other Agents' Positions (for plagiarism check):**\n{other_text or 'None yet.'}\n\n"
            f"Evaluate if the agent is aligned with its profile and NOT copying other agents.\n"
            f"Respond with this exact JSON format:\n{REVIEW_JSON_SCHEMA}"
        )
        
        try:
            result = await self.llm.generate_json(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=self.system_prompt,
                temperature=0.2,
            )
            return {
                "approved": bool(result.get("approved", True)),
                "reason": str(result.get("reason", "")),
                "feedback": str(result.get("feedback", ""))
            }
        except Exception as e:
            logger.error(f"[{self.name}] Failed to review position: {e}")
            return {"approved": True, "reason": "Error during review", "feedback": ""}

    async def evaluate_consensus(
        self,
        question: str,
        positions: list[AgentPosition],
        critiques: list,
        round_number: int = 1,
        max_rounds: int = 3
    ) -> ConsensusResult:
        """
        Arbiter evaluates if consensus is reached based on positions, critiques, and agent confidence.
        If consensus is not unanimous or majority, and not all rounds are completed, signals to continue deliberation.
        """
        pos_text = "\n".join(
            f"- {p.agent.value.upper()} (conf: {p.confidence}): {p.position}" 
            for p in positions
        )
        crit_text = "\n".join(
            f"- {c.critic.value.upper()} on {c.target.value.upper()}: Agreement={c.agreement}, Critique={c.critique}"
            for c in critiques
        )
        avg_conf = sum(p.confidence for p in positions) / len(positions) if positions else 0.0
        user_message = (
            f"**Question:** {question}\n\n"
            f"**Final Positions:**\n{pos_text}\n\n"
            f"**Critiques:**\n{crit_text}\n\n"
            f"**Average agent confidence:** {avg_conf:.2f}\n\n"
            f"Based on this data, determine if there is a consensus.\n"
            f"Status must be one of: unanimous, majority, deadlock.\n"
            f"If consensus is not reached and this is not the final round (round {round_number} of {max_rounds}), signal that deliberation should continue.\n"
            f"Respond with this exact JSON format:\n{CONSENSUS_JSON_SCHEMA}"
        )
        try:
            result = await self.llm.generate_json(
                messages=[{"role": "user", "content": user_message}],
                system_prompt="You are the ARBITER. Evaluate consensus objectively, considering agent confidence and critiques. If consensus is not reached and more rounds remain, deliberation must continue.",
                temperature=0.1,
            )
            status_str = str(result.get("status", "deadlock")).lower()
            try:
                status = ConsensusStatus(status_str)
            except ValueError:
                status = ConsensusStatus.DEADLOCK
            agreeing = [AgentName(a) for a in result.get("agreeing_agents", []) if a in [n.value for n in AgentName]]
            dissenting = [AgentName(a) for a in result.get("dissenting_agents", []) if a in [n.value for n in AgentName]]
            # Add a flag to indicate if deliberation should continue
            should_continue = False
            if status == ConsensusStatus.DEADLOCK and round_number < max_rounds:
                should_continue = True
            consensus = ConsensusResult(
                status=status,
                agreeing_agents=agreeing,
                dissenting_agents=dissenting,
                unified_position=result.get("unified_position"),
                confidence=float(result.get("confidence", 0.0))
            )
            consensus.should_continue = should_continue  # Attach flag for orchestrator
            return consensus
        except Exception as e:
            logger.error(f"[{self.name}] Failed to evaluate consensus: {e}")
            consensus = ConsensusResult(status=ConsensusStatus.DEADLOCK)
            consensus.should_continue = round_number < max_rounds
            return consensus
