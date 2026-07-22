"""Tests for Agent-to-Agent Perception."""
import numpy as np
import sys
sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.agent_perception import (
    AgentPerception, AgentPerceptionConfig, AgentPerceptionResult,
)


def _make_agent(agent_id, position, velocity=None, pillars=None,
                shadow=None, active=True):
    """Helper to create raw agent dict."""
    return {
        "id": agent_id,
        "position": list(position),
        "velocity": list(velocity or [0, 0, 0]),
        "pillars": list(pillars if pillars is not None else np.zeros(16)),
        "shadow_state": list(shadow if shadow is not None else np.zeros(16)),
        "active": active,
    }


# ── Basic Processing ─────────────────────────────────────────────

def test_single_agent():
    p = AgentPerception()
    agents = [_make_agent(1, [5, 0, 0])]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert len(results) == 1
    assert results[0].agent_id == 1


def test_inactive_agent_skipped():
    p = AgentPerception()
    agents = [_make_agent(1, [5, 0, 0], active=False)]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert len(results) == 0


def test_out_of_range_skipped():
    p = AgentPerception(AgentPerceptionConfig(view_distance=10.0))
    agents = [_make_agent(1, [100, 0, 0])]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert len(results) == 0


# ── Distance ─────────────────────────────────────────────────────

def test_distance_correct():
    p = AgentPerception()
    agents = [_make_agent(1, [3, 4, 0])]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert abs(results[0].distance - 5.0) < 0.01


def test_direction_unit_vector():
    p = AgentPerception()
    agents = [_make_agent(1, [10, 0, 0])]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    d = results[0].direction
    assert abs(np.linalg.norm(d) - 1.0) < 0.01
    assert abs(d[0] - 1.0) < 0.01


# ── Relative Velocity ────────────────────────────────────────────

def test_approaching():
    p = AgentPerception()
    agents = [_make_agent(1, [5, 0, 0], velocity=[-1, 0, 0])]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert results[0].relative_velocity < 0  # moving toward observer


def test_retreating():
    p = AgentPerception()
    agents = [_make_agent(1, [5, 0, 0], velocity=[1, 0, 0])]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert results[0].relative_velocity > 0  # moving away


def test_perpendicular():
    p = AgentPerception()
    agents = [_make_agent(1, [5, 0, 0], velocity=[0, 1, 0])]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert abs(results[0].relative_velocity) < 0.01


# ── Pillar Similarity ────────────────────────────────────────────

def test_identical_pillars():
    p = AgentPerception()
    psv = np.random.rand(16) * 100
    agents = [_make_agent(1, [5, 0, 0], pillars=psv)]
    results = p.process(np.zeros(3), psv, agents)
    assert results[0].pillar_similarity > 0.9


def test_opposite_pillars():
    p = AgentPerception()
    psv = np.ones(16) * 50
    agents = [_make_agent(1, [5, 0, 0], pillars=-psv)]
    results = p.process(np.zeros(3), psv, agents)
    assert results[0].pillar_similarity < 0.5


# ── Shadow Divergence ────────────────────────────────────────────

def test_shadow_same_as_pillars():
    p = AgentPerception()
    psv = np.ones(16) * 30
    agents = [_make_agent(1, [5, 0, 0], pillars=psv, shadow=psv)]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert results[0].shadow_divergence < 0.1


def test_shadow_different():
    p = AgentPerception()
    psv = np.ones(16) * 80
    shadow = np.ones(16) * 10
    agents = [_make_agent(1, [5, 0, 0], pillars=psv, shadow=shadow)]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert results[0].shadow_divergence > 0.5


# ── Dominant / Weakest Pillar ────────────────────────────────────

def test_pillar_dominance():
    p = AgentPerception()
    psv = np.ones(16) * 50.0
    psv[5] = 90.0  # Integrity is highest
    psv[3] = 10.0  # Influence is lowest
    agents = [_make_agent(1, [5, 0, 0], pillars=psv)]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert results[0].pillar_dominance == 5
    assert results[0].pillar_weakest == 3


# ── Threat ───────────────────────────────────────────────────────

def test_threat_close_fast():
    p = AgentPerception()
    agents = [_make_agent(1, [2, 0, 0], velocity=[-3, 0, 0])]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert results[0].threat_level > 0.3


def test_threat_far_slow():
    p = AgentPerception(AgentPerceptionConfig(view_distance=50.0))
    agents = [_make_agent(1, [40, 0, 0], velocity=[0.1, 0, 0])]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert results[0].threat_level < 0.3


# ── Social Signal ────────────────────────────────────────────────

def test_social_high_warmth():
    p = AgentPerception()
    psv = np.zeros(16)
    psv[9] = 90.0  # Warmth
    psv[7] = 80.0  # Relation
    psv[8] = 70.0  # Presence
    agents = [_make_agent(1, [5, 0, 0], pillars=psv)]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert results[0].social_signal > 0.5


def test_social_low_warmth():
    p = AgentPerception()
    psv = np.zeros(16)
    agents = [_make_agent(1, [5, 0, 0], pillars=psv)]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert results[0].social_signal < 0.1


# ── New Agent Tracking ───────────────────────────────────────────

def test_new_agent_flag():
    p = AgentPerception()
    agents = [_make_agent(1, [5, 0, 0])]
    r1 = p.process(np.zeros(3), np.zeros(16), agents)
    assert r1[0].is_new is True
    r2 = p.process(np.zeros(3), np.zeros(16), agents)
    assert r2[0].is_new is False


def test_frames_tracked():
    p = AgentPerception()
    agents = [_make_agent(1, [5, 0, 0])]
    for _ in range(5):
        results = p.process(np.zeros(3), np.zeros(16), agents)
    assert results[0].frames_tracked == 5


# ── Multiple Agents ──────────────────────────────────────────────

def test_multiple_agents():
    p = AgentPerception()
    agents = [
        _make_agent(1, [5, 0, 0]),
        _make_agent(2, [0, 5, 0]),
        _make_agent(3, [0, 0, 5]),
    ]
    results = p.process(np.zeros(3), np.zeros(16), agents)
    assert len(results) == 3
    ids = {r.agent_id for r in results}
    assert ids == {1, 2, 3}


# ── Reset ────────────────────────────────────────────────────────

def test_reset():
    p = AgentPerception()
    agents = [_make_agent(1, [5, 0, 0])]
    p.process(np.zeros(3), np.zeros(16), agents)
    p.reset()
    assert p.get_agent_summary(1) is None


# ── Summary ──────────────────────────────────────────────────────

def test_get_agent_summary():
    p = AgentPerception()
    agents = [_make_agent(1, [5, 0, 0])]
    p.process(np.zeros(3), np.zeros(16), agents)
    s = p.get_agent_summary(1)
    assert s is not None
    assert s["frames_tracked"] == 1
