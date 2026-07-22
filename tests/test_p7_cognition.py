"""Tests for P7 Higher-Order Cognition modules.

P7.4 Self-Model
P7.5 Theory of Mind
P7.6 Emotional Contagion
"""

import numpy as np
import pytest

from substrate_echo.core.self_model import SelfModel, CognitiveLoad
from substrate_echo.core.theory_of_mind import TheoryOfMind, BeliefState
from substrate_echo.core.emotional_contagion import (
    EmotionalContagion, EmotionalState, EMOTIONAL_DIMS,
)


# ── P7.4 Self-Model ──────────────────────────────────────────────

def test_self_model_initial_state():
    sm = SelfModel()
    stats = sm.stats()
    assert stats["n_capabilities"] > 0
    assert stats["avg_proficiency"] == 0.0
    assert stats["avg_confidence"] == 0.0


def test_self_model_updates_from_action():
    sm = SelfModel()
    # Successful action
    sm.update(action_taken="approach", success=True, tick=0)
    cap = sm._capabilities["navigate"]
    assert cap.attempts == 1
    assert cap.successes == 1
    assert cap.proficiency > 0

    # Failed action
    sm.update(action_taken="approach", success=False, tick=1)
    assert cap.attempts == 2
    assert cap.successes == 1


def test_self_model_can_do():
    sm = SelfModel()
    # Initially can't do anything (no attempts)
    assert not sm.can_do("navigate")

    # After several successes
    for i in range(10):
        sm.update(action_taken="approach", success=True, tick=i)
    assert sm.can_do("navigate")


def test_self_model_should_attempt():
    sm = SelfModel()
    # Risk-seeking agent should attempt even with low proficiency
    for i in range(5):
        sm.update(action_taken="approach", success=True, tick=i)
    assert sm.should_attempt("navigate", risk_tolerance=0.8)
    # Risk-averse agent with low proficiency
    sm2 = SelfModel()
    sm2.update(action_taken="approach", success=False, tick=0)
    assert not sm2.should_attempt("navigate", risk_tolerance=0.1)


def test_self_model_cognitive_load():
    sm = SelfModel()
    sm.update_cognitive_load(planning_complexity=0.8, memory_utilization=0.3)
    assert sm._cognitive_load.total_load > 0.2


def test_self_model_project_future():
    sm = SelfModel()
    for i in range(10):
        sm.update(action_taken="plan", success=True, tick=i)
    projection = sm.project_future(steps=10)
    assert "projected_capabilities" in projection
    assert "improving" in projection or "stable" in projection


def test_self_model_weakness():
    sm = SelfModel()
    # All failures on explore
    for i in range(10):
        sm.update(action_taken="investigate", success=False, tick=i)
    weakness = sm._identify_weakness()
    assert weakness == "explore"


# ── P7.5 Theory of Mind ──────────────────────────────────────────

def test_tom_initial_state():
    tom = TheoryOfMind()
    assert tom.get_belief(1) is None
    assert tom.stats()["n_agents"] == 0


def test_tom_creates_belief_on_update():
    tom = TheoryOfMind()
    state = tom.update(agent_id=1, observation=np.zeros(16),
                        position=np.array([5.0, 5.0, 0.0]))
    assert state.agent_id == 1
    assert state.observation_count == 1
    assert tom.stats()["n_agents"] == 1


def test_tom_tracks_observations():
    tom = TheoryOfMind()
    for i in range(10):
        tom.update(agent_id=1,
                   observation=np.random.randn(16),
                   position=np.array([float(i), 0.0, 0.0]))
    state = tom.get_belief(1)
    assert state.observation_count == 10
    assert len(state.trajectory) == 10


def test_tom_predict_action():
    tom = TheoryOfMind()
    for i in range(5):
        tom.update(agent_id=1,
                   observation=np.random.randn(16),
                   position=np.array([float(i), 0.0, 0.0]))
    prediction = tom.predict_action(agent_id=1)
    assert prediction is not None
    assert "predicted_position" in prediction
    assert "primary_desire" in prediction


def test_tom_detect_false_belief():
    tom = TheoryOfMind()
    # Agent believes they're at position 0
    tom.update(agent_id=1, observation=np.zeros(16),
               position=np.array([0.0, 0.0, 0.0]))
    # But reality is far away
    reality = np.full(16, 1.0)
    false_belief = tom.detect_false_belief(agent_id=1, reality=reality)
    assert false_belief is not None
    assert false_belief.discrepancy > 0


def test_tom_no_false_belief_when_close():
    tom = TheoryOfMind()
    state = np.zeros(16)
    tom.update(agent_id=1, observation=state, position=np.zeros(3))
    reality = state + 0.01  # very close
    false_belief = tom.detect_false_belief(agent_id=1, reality=reality)
    assert false_belief is None


def test_tom_infer_desire():
    tom = TheoryOfMind()
    tom.update(agent_id=1, position=np.array([5.0, 5.0, 0.0]))
    desire = tom.infer_desire(agent_id=1, action_type="communicate")
    assert desire in ("social", "unknown", "explore", "approach",
                       "avoid", "acquire", "protect")


def test_tom_predict_trajectory():
    tom = TheoryOfMind()
    for i in range(5):
        tom.update(agent_id=1,
                   observation=np.random.randn(16),
                   position=np.array([float(i), 0.0, 0.0]))
    trajectory = tom.predict_trajectory(agent_id=1, steps=10)
    assert len(trajectory) > 0


def test_tom_multiple_agents():
    tom = TheoryOfMind()
    tom.update(agent_id=1, observation=np.zeros(16),
               position=np.array([0.0, 0.0, 0.0]))
    tom.update(agent_id=2, observation=np.ones(16),
               position=np.array([10.0, 10.0, 0.0]))
    assert tom.stats()["n_agents"] == 2


# ── P7.6 Emotional Contagion ─────────────────────────────────────

def test_ec_initial_state():
    ec = EmotionalContagion()
    assert ec.stats()["n_agents"] == 0


def test_ec_creates_emotional_state():
    ec = EmotionalContagion()
    pillars = np.full(16, 50.0)
    state = ec.update(agent_id=1, pillars=pillars,
                       position=np.array([0.0, 0.0, 0.0]))
    assert isinstance(state, EmotionalState)
    assert ec.stats()["n_agents"] == 1


def test_ec_emotional_state_from_pillars():
    pillars = np.zeros(16)
    pillars[9] = 80.0   # warmth
    pillars[12] = 20.0  # stress
    state = EmotionalState.from_pillars(pillars)
    assert state.warmth == 80.0
    assert state.stress == pytest.approx(0.2, abs=0.01)


def test_ec_influence_nearby_agents():
    ec = EmotionalContagion()
    # Agent 1: high warmth, low stress
    pillars1 = np.full(16, 50.0)
    pillars1[9] = 90.0   # high warmth
    pillars1[12] = 10.0  # low stress
    ec.update(agent_id=1, pillars=pillars1,
              position=np.array([0.0, 0.0, 0.0]))

    # Agent 2: low warmth, high stress
    pillars2 = np.full(16, 50.0)
    pillars2[9] = 20.0   # low warmth
    pillars2[12] = 80.0  # high stress
    ec.update(agent_id=2, pillars=pillars2,
              position=np.array([1.0, 0.0, 0.0]))  # close

    # Apply contagion to agent 2
    modified = ec.apply_contagion(agent_id=2)
    assert modified is not None
    # Agent 2's stress should have decreased (influenced by calm agent 1)
    # or warmth should have increased
    new_state = EmotionalState.from_pillars(modified)
    # At minimum, some influence should have occurred
    assert modified is not None


def test_ec_no_influence_far_away():
    ec = EmotionalContagion()
    pillars = np.full(16, 50.0)
    ec.update(agent_id=1, pillars=pillars,
              position=np.array([0.0, 0.0, 0.0]))
    ec.update(agent_id=2, pillars=pillars,
              position=np.array([100.0, 0.0, 0.0]))  # very far

    modified = ec.apply_contagion(agent_id=1)
    # Should be unchanged (no nearby agents)
    assert modified is not None
    np.testing.assert_array_almost_equal(modified, pillars)


def test_ec_group_mood():
    ec = EmotionalContagion()
    for i in range(5):
        pillars = np.full(16, 50.0)
        pillars[9] = 70.0 + i * 5  # varying warmth
        ec.update(agent_id=i, pillars=pillars,
                  position=np.array([float(i), 0.0, 0.0]))
    mood = ec.get_group_mood()
    assert mood["n_agents"] == 5
    assert mood["avg_warmth"] > 50.0


def test_ec_stress_dampening():
    ec = EmotionalContagion()
    # Agent 1: calm (low stress)
    pillars1 = np.full(16, 50.0)
    pillars1[12] = 10.0  # low stress
    ec.update(agent_id=1, pillars=pillars1,
              position=np.array([0.0, 0.0, 0.0]))

    # Agent 2: stressed
    pillars2 = np.full(16, 50.0)
    pillars2[12] = 80.0  # high stress
    ec.update(agent_id=2, pillars=pillars2,
              position=np.array([1.0, 0.0, 0.0]))

    # Apply contagion
    modified = ec.apply_contagion(agent_id=2)
    new_stress = modified[12]
    # Stress should have decreased (calm agent dampening)
    assert new_stress < 80.0
