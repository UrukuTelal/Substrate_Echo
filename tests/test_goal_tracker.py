"""Tests for Goal State Machine."""
import numpy as np
import sys
sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.goal_tracker import (
    GoalPhase, GoalTransition, GoalManager, GoalManagerConfig,
    VelocityEstimate, TrajectoryPrediction, AgentGoalState,
)

# Config for tests that need velocity to decay fast
FAST_DECAY = GoalManagerConfig(velocity_smoothing=0.8)


# ── Velocity Estimation ──────────────────────────────────────────

def test_velocity_initial():
    gm = GoalManager()
    agent = gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    assert agent.velocity.speed == 0.0


def test_velocity_builds_over_time():
    gm = GoalManager()
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    gm.update(1, np.array([0.1, 0.0, 0.0]), 0.5)
    gm.update(1, np.array([0.2, 0.0, 0.0]), 1.0)
    
    agent = gm.get_state(1)
    assert agent.velocity.speed > 0.0
    assert np.dot(agent.velocity.direction, np.array([1.0, 0.0, 0.0])) > 0.9


def test_velocity_smoothed():
    gm = GoalManager(GoalManagerConfig(velocity_smoothing=0.1))
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    gm.update(1, np.array([1.0, 0.0, 0.0]), 1.0)  # fast!
    gm.update(1, np.array([1.0, 0.0, 0.0]), 2.0)  # stopped
    
    agent = gm.get_state(1)
    # Should be smoothed, not jumped to max
    assert agent.velocity.speed < 1.0


# ── Phase Transitions ────────────────────────────────────────────

def test_idle_initially():
    gm = GoalManager()
    agent = gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    assert agent.phase == GoalPhase.IDLE


def test_motion_start_transitions_to_exploring():
    gm = GoalManager()
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    gm.update(1, np.array([0.1, 0.0, 0.0]), 0.5)
    gm.update(1, np.array([0.2, 0.0, 0.0]), 1.0)
    
    agent = gm.get_state(1)
    assert agent.phase in (GoalPhase.EXPLORING, GoalPhase.APPROACHING)
    assert agent.last_transition in (
        GoalTransition.MOTION_START, GoalTransition.ATTRACTOR_DETECTED)


def test_stop_transitions_to_idle():
    gm = GoalManager(FAST_DECAY)
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    gm.update(1, np.array([0.5, 0.0, 0.0]), 0.5)
    # Stop — with alpha=0.8, velocity decays in ~5 ticks
    for i in range(6):
        gm.update(1, np.array([0.5, 0.0, 0.0]), 1.0 + 0.5 * i)
    
    agent = gm.get_state(1)
    assert agent.phase == GoalPhase.IDLE


def test_social_intent_triggers_communicating():
    gm = GoalManager()
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0, social_intent=0.8)
    
    agent = gm.get_state(1)
    assert agent.phase == GoalPhase.COMMUNICATING


def test_social_signal_lost_returns_idle():
    gm = GoalManager()
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0, social_intent=0.8)
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.5, social_intent=0.1)
    
    agent = gm.get_state(1)
    assert agent.phase == GoalPhase.IDLE


# ── Attractor Estimation ─────────────────────────────────────────

def test_attractor_estimated_for_moving_entity():
    gm = GoalManager()
    for i in range(5):
        gm.update(1, np.array([0.1 * i, 0.0, 0.0]), float(i))
    
    agent = gm.get_state(1)
    assert agent.attractor is not None
    # Should be ahead of current position
    assert agent.attractor[0] > agent.position[0]


def test_attractor_confidence_grows():
    gm = GoalManager()
    for i in range(10):
        gm.update(1, np.array([0.1 * i, 0.0, 0.0]), float(i))
    
    agent = gm.get_state(1)
    assert agent.attractor_confidence > 0.3


def test_attractor_no_motion_no_attractor():
    gm = GoalManager()
    gm.update(1, np.array([0.5, 0.5, 0.0]), 0.0)
    gm.update(1, np.array([0.5, 0.5, 0.0]), 1.0)
    
    agent = gm.get_state(1)
    assert agent.attractor is None or agent.attractor_confidence < 0.2


# ── Trajectory Prediction ────────────────────────────────────────

def test_prediction_has_positions():
    gm = GoalManager()
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    gm.update(1, np.array([0.1, 0.0, 0.0]), 0.5)
    gm.update(1, np.array([0.2, 0.0, 0.0]), 1.0)
    
    agent = gm.get_state(1)
    pred = agent.prediction
    assert pred is not None
    assert len(pred.positions) > 0
    assert pred.confidence > 0


def test_prediction_advances_forward():
    gm = GoalManager()
    for i in range(5):
        gm.update(1, np.array([0.1 * i, 0.0, 0.0]), float(i))
    
    agent = gm.get_state(1)
    pred = agent.prediction
    # Predicted positions should continue in same direction
    assert pred.positions[0][0] > agent.position[0]


def test_prediction_idle_slowdowns():
    gm = GoalManager(FAST_DECAY)
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    gm.update(1, np.array([0.5, 0.0, 0.0]), 0.5)
    # Stop — with alpha=0.8, velocity decays in ~5 ticks
    for i in range(6):
        gm.update(1, np.array([0.5, 0.0, 0.0]), 1.0 + 0.5 * i)
    
    agent = gm.get_state(1)
    pred = agent.prediction
    # Prediction should be empty or nearly stationary
    assert agent.phase == GoalPhase.IDLE
    if pred.positions:
        assert pred.positions[0][0] < 0.7  # barely moves


def test_prediction_approaching_moves_toward_attractor():
    gm = GoalManager()
    # Move toward x=5
    for i in range(10):
        gm.update(1, np.array([0.1 * i, 0.0, 0.0]), float(i))
    
    # Force attractor to be ahead
    agent = gm.get_state(1)
    agent.attractor = np.array([5.0, 0.0, 0.0])
    agent.phase = GoalPhase.APPROACHING
    
    pred = gm._predict_trajectory(agent)
    # First predicted position should be closer to attractor than current
    assert pred.positions is not None
    if pred.positions:
        current_dist = np.linalg.norm(agent.position - agent.attractor)
        pred_dist = np.linalg.norm(pred.positions[0] - agent.attractor)
        assert pred_dist < current_dist + 0.1  # approximately closer


# ── Multi-Agent Tracking ─────────────────────────────────────────

def test_multiple_agents_independent():
    gm = GoalManager()
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    gm.update(2, np.array([5.0, 5.0, 0.0]), 0.0)
    gm.update(1, np.array([0.1, 0.0, 0.0]), 1.0)
    gm.update(2, np.array([5.0, 4.0, 0.0]), 1.0)
    
    s1 = gm.get_state(1)
    s2 = gm.get_state(2)
    assert s1 is not None
    assert s2 is not None
    assert s1.entity_id == 1
    assert s2.entity_id == 2


def test_get_all_states():
    gm = GoalManager()
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    gm.update(2, np.array([1.0, 0.0, 0.0]), 0.0)
    
    all_states = gm.get_all_states()
    assert len(all_states) == 2
    assert 1 in all_states
    assert 2 in all_states


def test_remove_agent():
    gm = GoalManager()
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    gm.remove(1)
    assert gm.get_state(1) is None


# ── History Management ───────────────────────────────────────────

def test_history_bounded():
    gm = GoalManager(GoalManagerConfig(max_history=10))
    for i in range(20):
        gm.update(1, np.array([float(i), 0.0, 0.0]), float(i))
    
    agent = gm.get_state(1)
    assert len(agent.recent_positions) <= 10
    assert len(agent.recent_timestamps) <= 10


# ── Goal Description ─────────────────────────────────────────────

def test_goal_description_idle():
    gm = GoalManager(FAST_DECAY)
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    gm.update(1, np.array([0.5, 0.0, 0.0]), 0.5)
    # Stop — transition back to IDLE sets description
    for i in range(6):
        gm.update(1, np.array([0.5, 0.0, 0.0]), 1.0 + 0.5 * i)
    assert gm.get_state(1).estimated_goal_desc == "stationary"


def test_goal_description_exploring():
    gm = GoalManager()
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    gm.update(1, np.array([0.5, 0.0, 0.0]), 0.5)
    gm.update(1, np.array([1.0, 0.0, 0.0]), 1.0)
    agent = gm.get_state(1)
    assert "exploring" in agent.estimated_goal_desc or "approaching" in agent.estimated_goal_desc


def test_goal_description_communicating():
    gm = GoalManager()
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0, social_intent=0.8)
    agent = gm.get_state(1)
    assert "communicating" in agent.estimated_goal_desc


# ── Transition Counting ──────────────────────────────────────────

def test_transition_count_increases():
    gm = GoalManager(FAST_DECAY)
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    gm.update(1, np.array([0.5, 0.0, 0.0]), 0.5)
    # Stop — enough ticks for velocity to decay
    for i in range(6):
        gm.update(1, np.array([0.5, 0.0, 0.0]), 1.0 + 0.5 * i)
    
    agent = gm.get_state(1)
    assert agent.transition_count >= 2  # IDLE→EXPLORING, EXPLORING→IDLE


# ── Retreating ───────────────────────────────────────────────────

def test_retreat_on_reversal():
    gm = GoalManager()
    # Move right
    for i in range(8):
        gm.update(1, np.array([0.1 * i, 0.0, 0.0]), float(i))
    
    # Now reverse direction (away from attractor)
    for i in range(5):
        gm.update(1, np.array([0.8 - 0.1 * i, 0.0, 0.0]), float(8 + i))
    
    agent = gm.get_state(1)
    # Should be retreating or exploring (depends on timing)
    assert agent.phase in (GoalPhase.RETREATING, GoalPhase.EXPLORING)


# ── Edge Cases ───────────────────────────────────────────────────

def test_single_observation():
    gm = GoalManager()
    agent = gm.update(1, np.array([1.0, 2.0, 3.0]), 0.0)
    assert agent.phase == GoalPhase.IDLE
    assert agent.position is not None
    assert np.allclose(agent.position, [1.0, 2.0, 3.0])


def test_nonlinear_motion():
    gm = GoalManager()
    gm.update(1, np.array([0.0, 0.0, 0.0]), 0.0)
    gm.update(1, np.array([1.0, 0.0, 0.0]), 0.5)
    gm.update(1, np.array([1.0, 1.0, 0.0]), 1.0)
    
    agent = gm.get_state(1)
    assert agent.velocity.speed > 0


def test_prediction_empty_for_stationary():
    gm = GoalManager()
    gm.update(1, np.array([0.5, 0.5, 0.0]), 0.0)
    
    agent = gm.get_state(1)
    pred = agent.prediction
    assert pred is not None
    assert pred.confidence <= 0.2 or len(pred.positions) <= 1
