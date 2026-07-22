"""Tests for State Transition Manager — the common language layer."""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from substrate_echo.dynamics.state_transitions import (
    StateTransitionManager, StateTransition, TransitionCause,
    TransitionStatus, TransitionConstraint, TransitionCallback,
)


def test_basic_record():
    mgr = StateTransitionManager()
    
    src = np.full(16, 0.5)
    tgt = np.full(16, 0.6)
    
    t = StateTransition(
        source_state=src,
        target_state=tgt,
        cause=TransitionCause.FIELD_CHANGE,
        energy_cost=0.01,
    )
    
    result = mgr.record(t)
    assert result.status == TransitionStatus.ACCEPTED
    assert mgr._total_transitions == 1
    assert mgr._total_accepted == 1
    print("PASS: test_basic_record")


def test_reject_out_of_bounds():
    mgr = StateTransitionManager()
    
    src = np.full(16, 0.5)
    tgt = np.full(16, 1.5)  # out of bounds
    
    t = StateTransition(
        source_state=src,
        target_state=tgt,
        cause=TransitionCause.AGENT_ACTION,
    )
    
    result = mgr.record(t)
    assert result.status == TransitionStatus.REJECTED_BOUNDS
    assert mgr._total_rejected == 1
    print("PASS: test_reject_out_of_bounds")


def test_constraint_rejection():
    mgr = StateTransitionManager()
    
    # Add a constraint: energy must decrease
    def check_energy_decrease(t: StateTransition) -> tuple[bool, str]:
        if t.energy_cost > 0.5:
            return False, "energy cost too high"
        return True, ""
    
    mgr.add_constraint(TransitionConstraint(
        name="energy_bound",
        description="Energy cost must be reasonable",
        check_fn=check_energy_decrease,
    ))
    
    # Should pass
    t1 = StateTransition(
        source_state=np.full(16, 0.5),
        target_state=np.full(16, 0.5),
        cause=TransitionCause.FIELD_CHANGE,
        energy_cost=0.1,
    )
    result1 = mgr.record(t1)
    assert result1.status == TransitionStatus.ACCEPTED
    
    # Should fail
    t2 = StateTransition(
        source_state=np.full(16, 0.5),
        target_state=np.full(16, 0.5),
        cause=TransitionCause.FIELD_CHANGE,
        energy_cost=1.0,
    )
    result2 = mgr.record(t2)
    assert result2.status == TransitionStatus.REJECTED_ENERGY
    print("PASS: test_constraint_rejection")


def test_constraint_correction():
    mgr = StateTransitionManager()
    
    # Constraint that can auto-correct: clamp to [0, 1]
    def check_and_clamp(t: StateTransition) -> tuple[bool, str]:
        if np.any(t.target_state > 1.0) or np.any(t.target_state < 0.0):
            return False, "out of bounds"
        return True, ""
    
    def clamp_fn(t: StateTransition) -> np.ndarray:
        return np.clip(t.target_state, 0.0, 1.0)
    
    mgr.add_constraint(TransitionConstraint(
        name="bounds",
        description="State must be in [0, 1]",
        check_fn=check_and_clamp,
        correctable=True,
        correct_fn=clamp_fn,
    ))
    
    t = StateTransition(
        source_state=np.full(16, 0.5),
        target_state=np.full(16, 1.5),
        cause=TransitionCause.FIELD_CHANGE,
    )
    
    result = mgr.record(t)
    assert result.status == TransitionStatus.CORRECTED
    assert np.all(result.target_state <= 1.0)
    print("PASS: test_constraint_correction")


def test_callback_notification():
    mgr = StateTransitionManager()
    
    received = []
    def on_field_change(t: StateTransition):
        received.append(t.cause)
    
    mgr.add_callback(TransitionCallback(
        name="field_listener",
        filter_causes=[TransitionCause.FIELD_CHANGE],
        callback=on_field_change,
    ))
    
    # Field change — should notify
    mgr.record(StateTransition(
        source_state=np.full(16, 0.5),
        target_state=np.full(16, 0.5),
        cause=TransitionCause.FIELD_CHANGE,
    ))
    
    # Agent action — should NOT notify (filtered)
    mgr.record(StateTransition(
        source_state=np.full(16, 0.5),
        target_state=np.full(16, 0.5),
        cause=TransitionCause.AGENT_ACTION,
    ))
    
    assert len(received) == 1
    assert received[0] == TransitionCause.FIELD_CHANGE
    print("PASS: test_callback_notification")


def test_callback_all_causes():
    mgr = StateTransitionManager()
    
    received = []
    def on_any(t: StateTransition):
        received.append(t.cause)
    
    mgr.add_callback(TransitionCallback(
        name="universal",
        filter_causes=None,
        callback=on_any,
    ))
    
    for cause in [TransitionCause.FIELD_CHANGE, TransitionCause.AGENT_ACTION,
                  TransitionCause.MEMORY_UPDATE]:
        mgr.record(StateTransition(
            source_state=np.full(16, 0.5),
            target_state=np.full(16, 0.5),
            cause=cause,
        ))
    
    assert len(received) == 3
    print("PASS: test_callback_all_causes")


def test_recent_query():
    mgr = StateTransitionManager()
    
    for i in range(20):
        mgr.record(StateTransition(
            source_state=np.full(16, 0.5),
            target_state=np.full(16, 0.5 + i * 0.01),
            cause=TransitionCause.FIELD_CHANGE if i % 2 == 0 else TransitionCause.AGENT_ACTION,
        ))
    
    # Last 5
    recent = mgr.recent(n=5)
    assert len(recent) == 5
    
    # Last 5 field changes only
    recent_field = mgr.recent(n=5, causes=[TransitionCause.FIELD_CHANGE])
    assert all(t.cause == TransitionCause.FIELD_CHANGE for t in recent_field)
    print("PASS: test_recent_query")


def test_energy_tracking():
    mgr = StateTransitionManager()
    
    for i in range(5):
        mgr.record(StateTransition(
            source_state=np.full(16, 0.5),
            target_state=np.full(16, 0.5),
            cause=TransitionCause.FIELD_CHANGE,
            energy_cost=0.1,
        ))
    
    total = mgr.total_energy_cost()
    assert abs(total - 0.5) < 1e-10
    print("PASS: test_energy_tracking")


def test_state_trajectory():
    mgr = StateTransitionManager()
    
    states = []
    for i in range(10):
        s = np.full(16, 0.5 + i * 0.05)
        mgr.record(StateTransition(
            source_state=s - 0.05,
            target_state=s,
            cause=TransitionCause.FIELD_CHANGE,
        ))
        states.append(s)
    
    trajectory = mgr.state_trajectory(n=5)
    assert len(trajectory) == 5
    assert np.allclose(trajectory[-1], states[-1])
    print("PASS: test_state_trajectory")


def test_information_rate():
    mgr = StateTransitionManager()
    
    t0 = time.time()
    for i in range(10):
        mgr.record(StateTransition(
            source_state=np.full(16, 0.5),
            target_state=np.full(16, 0.5),
            cause=TransitionCause.FIELD_CHANGE,
            information_delta=0.1,
        ))
    
    rate = mgr.information_rate(window_seconds=60.0)
    assert rate > 0
    print("PASS: test_information_rate")


def test_cause_counts():
    mgr = StateTransitionManager()
    
    causes = [
        TransitionCause.FIELD_CHANGE,
        TransitionCause.FIELD_CHANGE,
        TransitionCause.AGENT_ACTION,
        TransitionCause.MEMORY_UPDATE,
        TransitionCause.MEMORY_UPDATE,
        TransitionCause.MEMORY_UPDATE,
    ]
    
    for cause in causes:
        mgr.record(StateTransition(
            source_state=np.full(16, 0.5),
            target_state=np.full(16, 0.5),
            cause=cause,
        ))
    
    stats = mgr.stats()
    assert stats["by_cause"]["FIELD_CHANGE"] == 2
    assert stats["by_cause"]["AGENT_ACTION"] == 1
    assert stats["by_cause"]["MEMORY_UPDATE"] == 3
    print("PASS: test_cause_counts")


def test_stats():
    mgr = StateTransitionManager()
    
    for i in range(10):
        mgr.record(StateTransition(
            source_state=np.full(16, 0.5),
            target_state=np.full(16, 0.5),
            cause=TransitionCause.FIELD_CHANGE,
            energy_cost=0.05,
            information_delta=0.01,
        ))
    
    stats = mgr.stats()
    assert stats["total_transitions"] == 10
    assert stats["accepted"] == 10
    assert stats["acceptance_rate"] == 1.0
    assert abs(stats["total_energy_cost"] - 0.5) < 1e-10
    print("PASS: test_stats")


def test_reset():
    mgr = StateTransitionManager()
    
    mgr.record(StateTransition(
        source_state=np.full(16, 0.5),
        target_state=np.full(16, 0.5),
        cause=TransitionCause.FIELD_CHANGE,
        energy_cost=0.1,
    ))
    
    mgr.reset()
    assert mgr._total_transitions == 0
    assert mgr._total_accepted == 0
    assert len(mgr.history) == 0
    print("PASS: test_reset")


def test_transitions_by_cause():
    mgr = StateTransitionManager()
    
    for cause in [TransitionCause.FIELD_CHANGE, TransitionCause.AGENT_ACTION,
                  TransitionCause.FIELD_CHANGE, TransitionCause.MEMORY_UPDATE,
                  TransitionCause.FIELD_CHANGE]:
        mgr.record(StateTransition(
            source_state=np.full(16, 0.5),
            target_state=np.full(16, 0.5),
            cause=cause,
        ))
    
    field_transitions = mgr.transitions_by_cause(TransitionCause.FIELD_CHANGE)
    assert len(field_transitions) == 3
    
    agent_transitions = mgr.transitions_by_cause(TransitionCause.AGENT_ACTION)
    assert len(agent_transitions) == 1
    print("PASS: test_transitions_by_cause")


def test_multiple_constraints():
    mgr = StateTransitionManager()
    
    # Constraint 1: energy bound
    mgr.add_constraint(TransitionConstraint(
        name="energy",
        description="energy check",
        check_fn=lambda t: (t.energy_cost < 1.0, "too high"),
    ))
    
    # Constraint 2: norm bound
    def check_norm(t):
        norm = np.linalg.norm(t.target_state)
        if norm > 5.0:
            return False, f"norm too high: {norm}"
        return True, ""
    
    mgr.add_constraint(TransitionConstraint(
        name="norm",
        description="norm check",
        check_fn=check_norm,
    ))
    
    # Passes both
    t1 = mgr.record(StateTransition(
        source_state=np.full(16, 0.5),
        target_state=np.full(16, 0.5),
        cause=TransitionCause.FIELD_CHANGE,
        energy_cost=0.1,
    ))
    assert t1.status == TransitionStatus.ACCEPTED
    
    # Fails energy
    t2 = mgr.record(StateTransition(
        source_state=np.full(16, 0.5),
        target_state=np.full(16, 0.5),
        cause=TransitionCause.FIELD_CHANGE,
        energy_cost=5.0,
    ))
    assert t2.status == TransitionStatus.REJECTED_ENERGY
    assert len(t2.validation_errors) == 1
    
    # Fails norm
    t3 = mgr.record(StateTransition(
        source_state=np.full(16, 0.5),
        target_state=np.full(16, 10.0),
        cause=TransitionCause.FIELD_CHANGE,
        energy_cost=0.1,
    ))
    assert t3.status == TransitionStatus.REJECTED_ENERGY  # energy check runs first
    print("PASS: test_multiple_constraints")


def test_history_bounded():
    mgr = StateTransitionManager(max_history=5)
    
    for i in range(10):
        mgr.record(StateTransition(
            source_state=np.full(16, 0.5),
            target_state=np.full(16, 0.5),
            cause=TransitionCause.FIELD_CHANGE,
        ))
    
    assert len(mgr.history) == 5
    print("PASS: test_history_bounded")


if __name__ == "__main__":
    test_basic_record()
    test_reject_out_of_bounds()
    test_constraint_rejection()
    test_constraint_correction()
    test_callback_notification()
    test_callback_all_causes()
    test_recent_query()
    test_energy_tracking()
    test_state_trajectory()
    test_information_rate()
    test_cause_counts()
    test_stats()
    test_reset()
    test_transitions_by_cause()
    test_multiple_constraints()
    test_history_bounded()
    print("\nAll state transition manager tests passed!")
