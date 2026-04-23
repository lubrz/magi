"""
Consensus evaluation — determines if the three agents agree.

Uses a combination of:
  1. Position similarity (keyword overlap heuristic)
  2. Confidence-weighted voting
  3. Mutual agreement scores from critiques
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from models.schemas import (
    AgentCritique,
    AgentName,
    AgentPosition,
    ConsensusResult,
    ConsensusStatus,
)

logger = logging.getLogger(__name__)


def evaluate_consensus(
    positions: list[AgentPosition],
    critiques: list[AgentCritique],
    threshold: float = 0.7,
    min_votes: int = 2,
) -> ConsensusResult:
    """
    Evaluate whether the agents have reached consensus.

    Strategy:
      1. Compute pairwise similarity between positions
      2. Build an agreement graph from critique scores
      3. Determine if a majority cluster exists with sufficient confidence
    """
    if len(positions) < 2:
        return ConsensusResult(status=ConsensusStatus.PENDING)

    agents = [p.agent for p in positions]

    # --- Step 1: Pairwise position similarity ---
    sim_matrix = {}
    for i, p1 in enumerate(positions):
        for j, p2 in enumerate(positions):
            if i < j:
                sim = _text_similarity(p1.position, p2.position)
                sim_matrix[(p1.agent, p2.agent)] = sim

    # --- Step 2: Agreement from critiques ---
    critique_agreement = {}
    for c in critiques:
        critique_agreement[(c.critic, c.target)] = c.agreement

    # --- Step 3: Build combined agreement scores ---
    # For each pair, combine text similarity and critique agreement
    pair_scores = {}
    for pair, sim in sim_matrix.items():
        # Check if we have critique scores for this pair
        crit_fwd = critique_agreement.get(pair, None)
        crit_rev = critique_agreement.get((pair[1], pair[0]), None)

        crit_scores = [s for s in [crit_fwd, crit_rev] if s is not None]
        if crit_scores:
            avg_crit = sum(crit_scores) / len(crit_scores)
            # Weight: 40% text similarity, 60% critique agreement
            combined = 0.4 * sim + 0.6 * avg_crit
        else:
            combined = sim

        pair_scores[pair] = combined

    # --- Step 4: Find agreement clusters ---
    # Check for unanimous agreement
    all_high = all(score >= threshold for score in pair_scores.values())
    if all_high:
        avg_confidence = sum(p.confidence for p in positions) / len(positions)
        return ConsensusResult(
            status=ConsensusStatus.UNANIMOUS,
            agreeing_agents=agents,
            dissenting_agents=[],
            unified_position=_merge_positions(positions),
            confidence=avg_confidence,
        )

    # Check for majority (2/3) agreement
    # Find the pair with highest agreement
    if pair_scores:
        best_pair = max(pair_scores, key=pair_scores.get)
        best_score = pair_scores[best_pair]

        if best_score >= threshold:
            agreeing = list(best_pair)
            dissenting = [a for a in agents if a not in agreeing]

            # Use the higher-confidence position from the agreeing pair
            agreeing_positions = [p for p in positions if p.agent in agreeing]
            best_pos = max(agreeing_positions, key=lambda p: p.confidence)

            avg_confidence = sum(p.confidence for p in agreeing_positions) / len(
                agreeing_positions
            )

            return ConsensusResult(
                status=ConsensusStatus.MAJORITY,
                agreeing_agents=agreeing,
                dissenting_agents=dissenting,
                unified_position=best_pos.position,
                confidence=avg_confidence,
            )

    # --- Deadlock ---
    return ConsensusResult(
        status=ConsensusStatus.DEADLOCK,
        agreeing_agents=[],
        dissenting_agents=agents,
        unified_position=None,
        confidence=0.0,
    )


def _text_similarity(text1: str, text2: str) -> float:
    """
    Simple keyword-overlap similarity between two texts.
    Returns 0.0–1.0.
    """
    words1 = set(_tokenize(text1))
    words2 = set(_tokenize(text2))

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    # Jaccard similarity
    return len(intersection) / len(union) if union else 0.0


def _tokenize(text: str) -> list[str]:
    """Extract meaningful tokens from text."""
    text = text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text)
    stop = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "has", "have", "been", "this",
        "that", "with", "from", "they", "will", "would", "could", "should",
        "their", "which", "about", "more", "also", "into", "some", "than",
    }
    return [w for w in words if w not in stop]


def _merge_positions(positions: list[AgentPosition]) -> str:
    """Create a merged position statement from agreeing agents."""
    if len(positions) == 1:
        return positions[0].position

    # Use the highest-confidence position as the base
    best = max(positions, key=lambda p: p.confidence)
    return best.position
