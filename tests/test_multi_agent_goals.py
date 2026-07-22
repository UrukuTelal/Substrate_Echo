"""Tests for Multi-Agent Goal Inference."""
import numpy as np
import sys
sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.goal_tracker import GoalManager, GoalPhase
from substrate_echo.core.multi_agent_goals import (
    MultiAgentGoalInference, InteractionType, PairInteraction,
    MultiAgentGoalInferenceConfig,
)


def _make_gm_with_agents(agents: list[dict]) -> GoalManager:
    """Helper: create a GoalManager with pre-loaded agent positions."""
    gm = GoalManager()
    for a in agents:
        for i, pos in enumerate(a["positions"]):
            gm.update(
                entity_id=a["id"],
                position=np.array(pos),
                timestamp=float(i),
                social_intent=a.get("social_intent", 0.0),
            )
    return gm


# ── Convergence ──────────────────────────────────────────────────

def test_converging_agents():
    gm = _make_gm_with_agents([
        {"id": 1, "positions": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
        {"id": 2, "positions": [[5, 0, 0], [4, 0, 0], [3, 0, 0]]},
    ])
    inf = MultiAgentGoalInference()
    interactions = inf.infer(gm)
    types = [i.interaction_type for i in interactions]
    assert InteractionType.CONVERGING in types


# ── Diverging ────────────────────────────────────────────────────

def test_diverging_agents():
    gm = _make_gm_with_agents([
        {"id": 1, "positions": [[2, 0, 0], [1, 0, 0], [0, 0, 0]]},
        {"id": 2, "positions": [[3, 0, 0], [4, 0, 0], [5, 0, 0]]},
    ])
    inf = MultiAgentGoalInference()
    interactions = inf.infer(gm)
    # Should be diverging or avoiding
    types = [i.interaction_type for i in interactions]
    assert InteractionType.DIVERGING in types or InteractionType.AVOIDING in types


# ── Co-presence ──────────────────────────────────────────────────

def test_co_present():
    gm = _make_gm_with_agents([
        {"id": 1, "positions": [[1, 0, 0], [1, 0, 0], [1, 0, 0]]},
        {"id": 2, "positions": [[1.1, 0, 0], [1.1, 0, 0], [1.1, 0, 0]]},
    ])
    inf = MultiAgentGoalInference()
    interactions = inf.infer(gm)
    types = [i.interaction_type for i in interactions]
    assert InteractionType.CO_PRESENT in types


# ── Joint Goal ───────────────────────────────────────────────────

def test_joint_goal():
    gm = GoalManager()
    # Both heading toward same attractor at [10, 0, 0]
    for i in range(5):
        t = float(i)
        # Agent 1: heading right
        gm.update(1, np.array([float(i), 0.0, 0.0]), t)
        # Agent 2: heading right from different start
        gm.update(2, np.array([0.0, float(i), 0.0]), t)
    
    # Manually set attractors close together
    gm._agents[1].attractor = np.array([10.0, 0.0, 0.0])
    gm._agents[1].attractor_confidence = 0.8
    gm._agents[2].attractor = np.array([10.0, 1.0, 0.0])
    gm._agents[2].attractor_confidence = 0.8
    
    inf = MultiAgentGoalInference()
    interactions = inf.infer(gm)
    types = [i.interaction_type for i in interactions]
    assert InteractionType.JOINT_GOAL in types


# ── Following ────────────────────────────────────────────────────

def test_following():
    gm = _make_gm_with_agents([
        {"id": 1, "positions": [
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
            [3, 0, 0], [4, 0, 0], [5, 0, 0],
        ]},
        {"id": 2, "positions": [
            [0, 0, 0], [0.5, 0, 0], [1.5, 0, 0],
            [2.5, 0, 0], [3.5, 0, 0], [4.5, 0, 0],
        ]},
    ])
    inf = MultiAgentGoalInference()
    interactions = inf.infer(gm)
    types = [i.interaction_type for i in interactions]
    assert InteractionType.FOLLOWING in types


# ── No Interaction (far apart, random motion) ───────────────────

def test_no_interaction_far_apart():
    gm = _make_gm_with_agents([
        {"id": 1, "positions": [[0, 0, 0], [0.01, 0, 0], [0, 0, 0]]},
        {"id": 2, "positions": [[100, 0, 0], [100, 0, 0], [100, 0, 0]]},
    ])
    inf = MultiAgentGoalInference()
    interactions = inf.infer(gm)
    assert len(interactions) == 0


# ── Single Agent (no pairs) ──────────────────────────────────────

def test_single_agent():
    gm = _make_gm_with_agents([
        {"id": 1, "positions": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
    ])
    inf = MultiAgentGoalInference()
    interactions = inf.infer(gm)
    assert len(interactions) == 0


# ── PairInteraction to_dict ──────────────────────────────────────

def test_pair_interaction_to_dict():
    p = PairInteraction(
        entity_a=1, entity_b=2,
        interaction_type=InteractionType.CONVERGING,
        confidence=0.8,
        convergence_rate=-0.2,
        distance=3.0,
        shared_attractor=False,
    )
    d = p.to_dict()
    assert d["entity_a"] == 1
    assert d["interaction_type"] == "CONVERGING"


# ── Confidence Range ─────────────────────────────────────────────

def test_confidence_range():
    gm = _make_gm_with_agents([
        {"id": 1, "positions": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
        {"id": 2, "positions": [[5, 0, 0], [4, 0, 0], [3, 0, 0]]},
    ])
    inf = MultiAgentGoalInference()
    for pair in inf.infer(gm):
        assert 0.0 <= pair.confidence <= 1.0


# ── Reset ────────────────────────────────────────────────────────

def test_reset():
    inf = MultiAgentGoalInference()
    inf.reset()  # should not raise


# ── Three agents ─────────────────────────────────────────────────

def test_three_agents():
    gm = _make_gm_with_agents([
        {"id": 1, "positions": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
        {"id": 2, "positions": [[5, 0, 0], [4, 0, 0], [3, 0, 0]]},
        {"id": 3, "positions": [[0, 5, 0], [0, 4, 0], [0, 3, 0]]},
    ])
    inf = MultiAgentGoalInference()
    interactions = inf.infer(gm)
    # Should have at least 1-2 pairs detected
    assert len(interactions) >= 1


# ── Motion-based group detection ──────────────────────────────────

def test_same_group_same_direction():
    """Two agents moving in same direction at similar speed = same group."""
    gm = _make_gm_with_agents([
        {"id": 1, "positions": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
        {"id": 2, "positions": [[3, 0.5, 0], [4, 0.5, 0], [5, 0.5, 0]]},
    ])
    inf = MultiAgentGoalInference()
    interactions = inf.infer(gm)
    types = [p.interaction_type for p in interactions]
    # Same direction, close together = CO_PRESENT (same group)
    assert InteractionType.CO_PRESENT in types


def test_different_groups_different_directions():
    """Two groups moving in opposite directions = not same group."""
    gm = _make_gm_with_agents([
        {"id": 1, "positions": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
        {"id": 2, "positions": [[3, 0, 0], [2, 0, 0], [1, 0, 0]]},
    ])
    inf = MultiAgentGoalInference()
    interactions = inf.infer(gm)
    types = [p.interaction_type for p in interactions]
    # Opposite directions, approaching = CONVERGING or AVOIDING (both valid)
    assert InteractionType.CONVERGING in types or InteractionType.NONE in types or InteractionType.AVOIDING in types


def test_far_apart_different_groups():
    """Two agents far apart are never same group."""
    gm = _make_gm_with_agents([
        {"id": 1, "positions": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
        {"id": 2, "positions": [[20, 0, 0], [19, 0, 0], [18, 0, 0]]},
    ])
    inf = MultiAgentGoalInference()
    interactions = inf.infer(gm)
    # Far apart, converging
    for p in interactions:
        assert p.distance > 10.0
