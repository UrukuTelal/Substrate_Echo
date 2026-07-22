"""Tests for integration bridges and visualization."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from substrate_echo.integration.psv_bridge import PSVBridge, PILLAR_NAMES
from substrate_echo.integration.void_bridge import VoidBridge
from substrate_echo.integration.engine_bridge import EngineBridge
from substrate_echo.integration.council_bridge import CouncilBridge
from substrate_echo.visualization.field_renderer import FieldRenderer
from substrate_echo.visualization.world_view import WorldView
from substrate_echo.visualization.agent_view import AgentView
from substrate_echo.core.cognitive_agents import AgentEcology, AgentRole


# ── PSV Bridge Tests ──────────────────────────────────────────────

def test_psv_cosine_similarity():
    a = np.ones(16) * 0.5
    b = np.ones(16) * 0.5
    assert abs(PSVBridge.cosine_similarity(a, b) - 1.0) < 1e-10
    
    c = np.zeros(16)
    c[0] = 1.0
    d = np.zeros(16)
    d[8] = 1.0
    assert abs(PSVBridge.cosine_similarity(c, d)) < 1e-10
    print("PASS: test_psv_cosine_similarity")


def test_psv_coherence():
    uniform = np.ones(16) * 0.5
    assert abs(PSVBridge.coherence(uniform) - 1.0) < 1e-10
    
    varied = np.random.rand(16)
    assert PSVBridge.coherence(varied) < 1.0
    print("PASS: test_psv_coherence")


def test_psv_dominant_weakest():
    state = np.full(16, 0.5)
    state[3] = 0.9  # Influence dominant
    state[11] = 0.1  # Attraction weakest
    
    assert PSVBridge.dominant_pillar(state) == 3
    assert PSVBridge.weakest_pillar(state) == 11
    print("PASS: test_psv_dominant_weakest")


def test_psv_pillar_summary():
    state = np.array([0.1 * i for i in range(16)])
    summary = PSVBridge.pillar_summary(state)
    
    assert len(summary) == 16
    assert "Awareness" in summary
    assert "Depth" in summary
    print("PASS: test_psv_pillar_summary")


def test_psv_pillar_names():
    assert len(PILLAR_NAMES) == 16
    assert PILLAR_NAMES[0] == "Awareness"
    assert PILLAR_NAMES[15] == "Depth"
    print("PASS: test_psv_pillar_names")


# ── Engine Bridge Tests ───────────────────────────────────────────

def test_engine_entity_conversion():
    entity_data = {
        "uid": "abc123",
        "name": "TestEntity",
        "psv": [0.5] * 16,
        "position": [1.0, 2.0, 3.0],
        "entity_type": "agent",
        "mass": 2.0,
    }
    
    result = EngineBridge.engine_entity_to_dict(entity_data)
    
    assert result["object_id"] == "abc123"
    assert result["name"] == "TestEntity"
    assert np.allclose(result["psv"], 0.5)
    assert result["position"] == [1.0, 2.0, 3.0]
    print("PASS: test_engine_entity_conversion")


def test_engine_thought_palace_realms():
    for realm_id in range(5):
        state = EngineBridge.thought_palace_realm_to_state(realm_id)
        assert state.shape == (16,)
        assert np.all(state >= 0.0)
        assert np.all(state <= 1.0)
    print("PASS: test_engine_thought_palace_realms")


# ── Council Bridge Tests ──────────────────────────────────────────

def test_council_validate_good_action():
    action = {"action_type": "grasp", "target": "cup"}
    state = np.full(16, 0.5)  # balanced state
    
    result = CouncilBridge.validate_action(action, state)
    assert result["approved"]
    assert result["confidence"] > 0.5
    print("PASS: test_council_validate_good_action")


def test_council_validate_harmful_action():
    action = {"action_type": "destroy", "target": "everything"}
    state = np.full(16, 0.5)
    state[12] = 0.95  # high Harm pillar
    
    result = CouncilBridge.validate_action(action, state)
    assert not result["approved"]
    assert not result["checks"]["harm_check"]
    print("PASS: test_council_validate_harmful_action")


# ── Visualization Tests ───────────────────────────────────────────

def test_field_renderer_state():
    renderer = FieldRenderer()
    state = np.array([0.1 * i for i in range(16)])
    
    output = renderer.render_state(state, title="Test State")
    assert "Test State" in output
    assert "Awareness" in output
    assert "Depth" in output
    assert "█" in output
    print("PASS: test_field_renderer_state")


def test_field_renderer_trajectory():
    renderer = FieldRenderer()
    trajectory = [np.full(16, 0.3 + i * 0.05) for i in range(10)]
    
    output = renderer.render_trajectory(trajectory)
    assert "10 steps" in output
    assert "Awareness" in output
    assert "Depth" in output
    print("PASS: test_field_renderer_trajectory")


def test_world_view_empty():
    view = WorldView()
    output = view.render_objects({})
    assert "empty" in output
    print("PASS: test_world_view_empty")


def test_agent_view_ecology():
    view = AgentView()
    ecology = AgentEcology()
    
    output = view.render_agent_states(ecology)
    assert "PERCEPTION" in output
    assert "Energy" in output
    print("PASS: test_agent_view_ecology")


def test_agent_view_empty_responses():
    view = AgentView()
    output = view.render_responses([])
    assert "no responses" in output
    print("PASS: test_agent_view_empty_responses")


if __name__ == "__main__":
    print("=== PSV Bridge ===")
    test_psv_cosine_similarity()
    test_psv_coherence()
    test_psv_dominant_weakest()
    test_psv_pillar_summary()
    test_psv_pillar_names()
    
    print("\n=== Engine Bridge ===")
    test_engine_entity_conversion()
    test_engine_thought_palace_realms()
    
    print("\n=== Council Bridge ===")
    test_council_validate_good_action()
    test_council_validate_harmful_action()
    
    print("\n=== Visualization ===")
    test_field_renderer_state()
    test_field_renderer_trajectory()
    test_world_view_empty()
    test_agent_view_ecology()
    test_agent_view_empty_responses()
    
    print("\nAll integration + visualization tests passed!")
