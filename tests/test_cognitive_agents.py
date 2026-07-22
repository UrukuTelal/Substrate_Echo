"""Tests for Cognitive Agents."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from substrate_echo.core.cognitive_agents import (
    AgentEcology, PerceptionAgent, MemoryAgent, PlanningAgent,
    CreativityAgent, EnvironmentAgent, AgentRole,
)


def test_agent_activation():
    agent = PerceptionAgent()
    assert not agent.is_active
    
    # Low awareness — should not activate
    low_state = np.full(16, 0.5)
    assert not agent.should_activate(low_state)
    
    # High awareness — should activate
    high_state = np.full(16, 0.5)
    high_state[0] = 0.9  # Awareness
    high_state[8] = 0.9  # Presence
    assert agent.should_activate(high_state)
    print("PASS: test_agent_activation")


def test_agent_ecology_tick():
    ecology = AgentEcology()
    
    # Neutral state — few agents active
    neutral = np.full(16, 0.5)
    responses = ecology.tick(neutral)
    
    # High awareness + presence — perception should activate
    state = np.full(16, 0.5)
    state[0] = 0.9  # Awareness
    state[8] = 0.9  # Presence
    responses = ecology.tick(state)
    
    active_roles = ecology.get_active_roles()
    assert AgentRole.PERCEPTION in active_roles
    print("PASS: test_agent_ecology_tick")


def test_agent_consensus():
    ecology = AgentEcology()
    
    # Set high values in all affinity pillars to cross activation thresholds
    state = np.full(16, 0.5)
    state[0] = 0.95   # Awareness
    state[1] = 0.95   # Willpower
    state[4] = 0.95   # Resistance
    state[7] = 0.95   # Relation
    state[8] = 0.95   # Presence
    state[10] = 0.95  # Memory
    state[11] = 0.95  # Attraction
    state[13] = 0.95  # Distortion
    state[14] = 0.95  # Flux
    state[15] = 0.95  # Depth
    
    responses = ecology.tick(state)
    assert len(responses) > 0, "No agents activated"
    
    consensus = ecology.get_consensus(responses)
    
    assert consensus is not None
    assert consensus.confidence > 0.0
    print("PASS: test_agent_consensus")


def test_all_agents_specialize():
    """Each agent should have different pillar affinities."""
    agents = [
        PerceptionAgent(), MemoryAgent(), PlanningAgent(),
        CreativityAgent(), EnvironmentAgent(),
    ]
    
    affinities = [set(a.pillar_affinity) for a in agents]
    # All affinities should be different
    for i in range(len(affinities)):
        for j in range(i + 1, len(affinities)):
            assert affinities[i] != affinities[j], \
                f"Agents {i} and {j} have same pillar affinity"
    print("PASS: test_all_agents_specialize")


def test_ecology_stats():
    ecology = AgentEcology()
    
    state = np.full(16, 0.95)
    ecology.tick(state)
    
    stats = ecology.stats()
    assert stats["total_agents"] == 5
    assert stats["active_agents"] >= 1
    print("PASS: test_ecology_stats")


if __name__ == "__main__":
    test_agent_activation()
    test_agent_ecology_tick()
    test_agent_consensus()
    test_all_agents_specialize()
    test_ecology_stats()
    print("\nAll cognitive agent tests passed!")
