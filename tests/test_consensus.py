import pytest
from orchestrator.consensus import evaluate_consensus
from models.schemas import AgentPosition, AgentName, AgentCritique, ConsensusStatus

def test_unanimous_consensus():
    positions = [
        AgentPosition(agent=AgentName.AXIOM, position="Yes, it works.", reasoning="Logic.", confidence=0.9),
        AgentPosition(agent=AgentName.PRISM, position="Yes, it works.", reasoning="Pattern.", confidence=0.8),
        AgentPosition(agent=AgentName.FORGE, position="Yes, it works.", reasoning="Practice.", confidence=0.7),
    ]
    critiques = [
        AgentCritique(critic=AgentName.AXIOM, target=AgentName.PRISM, agreement=0.9, critique="Good", revised_confidence=0.9),
        AgentCritique(critic=AgentName.PRISM, target=AgentName.FORGE, agreement=0.9, critique="Good", revised_confidence=0.9),
    ]
    
    result = evaluate_consensus(positions, critiques)
    assert result.status == ConsensusStatus.UNANIMOUS
    assert len(result.agreeing_agents) == 3

def test_majority_consensus():
    positions = [
        AgentPosition(agent=AgentName.AXIOM, position="Yes.", reasoning="...", confidence=0.9),
        AgentPosition(agent=AgentName.PRISM, position="Yes.", reasoning="...", confidence=0.8),
        AgentPosition(agent=AgentName.FORGE, position="No.", reasoning="...", confidence=0.5),
    ]
    # Axiom and Prism agree
    critiques = [
        AgentCritique(critic=AgentName.AXIOM, target=AgentName.PRISM, agreement=0.9, critique="Agree", revised_confidence=0.9),
        AgentCritique(critic=AgentName.FORGE, target=AgentName.AXIOM, agreement=0.2, critique="Disagree", revised_confidence=0.5),
    ]
    
    result = evaluate_consensus(positions, critiques, threshold=0.7)
    assert result.status == ConsensusStatus.MAJORITY
    assert AgentName.AXIOM in result.agreeing_agents
    assert AgentName.PRISM in result.agreeing_agents

def test_deadlock():
    positions = [
        AgentPosition(agent=AgentName.AXIOM, position="Option A", reasoning="...", confidence=0.9),
        AgentPosition(agent=AgentName.PRISM, position="Option B", reasoning="...", confidence=0.9),
        AgentPosition(agent=AgentName.FORGE, position="Option C", reasoning="...", confidence=0.9),
    ]
    critiques = [
        AgentCritique(critic=AgentName.AXIOM, target=AgentName.PRISM, agreement=0.1, critique="Bad", revised_confidence=0.9),
        AgentCritique(critic=AgentName.PRISM, target=AgentName.FORGE, agreement=0.1, critique="Bad", revised_confidence=0.9),
    ]
    
    result = evaluate_consensus(positions, critiques, threshold=0.8)
    assert result.status == ConsensusStatus.DEADLOCK
