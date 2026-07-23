"""Tests for Executive Function (S10)."""
from __future__ import annotations
import pytest
import numpy as np
import time
from substrate_echo.kernel.executive import (
    GoalState, GoalStatus, GoalTier, GoalConflict,
    ExecutiveState, PriorityScorer, AttentionAllocator,
    GoalGenerator, ExecutiveFunction
)
from substrate_echo.kernel import Observation


def test_goal_priority_effective_active():
    g = GoalState(id=1, target=[0.5]*4, status=GoalStatus.ACTIVE, priority_score=0.8)
    assert g.effective_priority() == 0.8


def test_goal_priority_effective_paused():
    g = GoalState(id=1, target=[0.5]*4, status=GoalStatus.PAUSED, priority_score=0.8)
    assert g.effective_priority() == pytest.approx(0.08, abs=0.001)


def test_goal_priority_effective_completed():
    g = GoalState(id=1, target=[0.5]*4, status=GoalStatus.COMPLETED, priority_score=0.8)
    assert g.effective_priority() == 0.0


def test_priority_scorer_basic():
    scorer = PriorityScorer()
    g1 = GoalState(id=1, target=[0.5]*4, urgency=0.9, importance=0.9, tier=GoalTier.SAFETY)
    g2 = GoalState(id=2, target=[0.5]*4, urgency=0.3, importance=0.3, tier=GoalTier.IDLE)
    s1 = scorer.score(g1)
    s2 = scorer.score(g2)
    assert s1 > s2
    assert 0 <= s1 <= 1
    assert 0 <= s2 <= 1


def test_priority_scorer_rescore_all():
    scorer = PriorityScorer()
    goals = [
        GoalState(id=1, target=[0.5]*4, urgency=0.3, importance=0.3),
        GoalState(id=2, target=[0.5]*4, urgency=0.9, importance=0.9),
    ]
    scored = scorer.rescore_all(goals)
    assert scored[0].priority_score >= scored[1].priority_score
    assert scored[0].id == 2


def test_attention_allocator_basic():
    allocator = AttentionAllocator(max_attention=10.0)
    goals = [
        GoalState(id=1, target=[0.5]*4, status=GoalStatus.ACTIVE,
                  priority_score=0.9, attention_weight=0.8),
        GoalState(id=2, target=[0.5]*4, status=GoalStatus.COMPLETED,
                  priority_score=0.5, attention_weight=0.5),
    ]
    attention = allocator.allocate(goals)
    assert isinstance(attention, dict)
    assert allocator.global_focus() <= 1.0


def test_attention_allocator_prediction_error_boost():
    allocator = AttentionAllocator()
    goals = [
        GoalState(id=1, target=[0.5]*4, status=GoalStatus.ACTIVE,
                  priority_score=0.5, attention_weight=0.5),
    ]
    # First call with low error
    a1 = allocator.allocate(goals, prediction_errors={100: 0.1})
    val_low = a1.get(100, 0)
    # Second call with high error
    a2 = allocator.allocate(goals, prediction_errors={100: 0.9})
    val_high = a2.get(100, 0)
    # High error should get more attention than low error
    assert val_high > val_low


def test_goal_generator_safety_extreme_low():
    gen = GoalGenerator()
    obs = Observation(vector=[0.01] + [0.5]*15, embodiment_id="test")
    goals = gen.check(obs)
    assert len(goals) > 0
    assert goals[0].tier == GoalTier.SAFETY


def test_goal_generator_safety_normal():
    gen = GoalGenerator()
    obs = Observation(vector=[0.5]*16, embodiment_id="test")
    goals = gen.check(obs)
    assert len(goals) == 0


def test_executive_function_add_and_activate():
    ef = ExecutiveFunction()
    g = GoalState(id=0, target=[0.5]*4, description="test goal")
    gid = ef.add_goal(g)
    assert gid == g.id
    ef.activate_goal(gid)
    state = ef.tick()
    assert state.n_active >= 1


def test_executive_function_complete():
    ef = ExecutiveFunction()
    gid = ef.add_goal(GoalState(id=0, target=[0.5]*4, description="to complete"))
    ef.activate_goal(gid)
    ef.complete_goal(gid)
    state = ef.tick()
    assert state.n_completed == 1
    assert state.n_active == 0


def test_executive_function_fail():
    ef = ExecutiveFunction()
    gid = ef.add_goal(GoalState(id=0, target=[0.5]*4, description="to fail"))
    ef.activate_goal(gid)
    ef.fail_goal(gid)
    state = ef.tick()
    assert state.n_failed == 1


def test_executive_function_pause_resume():
    ef = ExecutiveFunction()
    gid = ef.add_goal(GoalState(id=0, target=[0.5]*4, description="pausable"))
    ef.activate_goal(gid)
    ef.pause_goal(gid)
    state = ef.tick()
    assert state.n_active == 0
    ef.resume_goal(gid)
    state = ef.tick()
    assert state.n_active == 1


def test_executive_function_auto_generates_goals():
    ef = ExecutiveFunction()
    obs = Observation(vector=[0.01] + [0.5]*15, embodiment_id="test")
    state = ef.tick(obs)
    assert state.n_goals > 0


def test_executive_function_get_goals_sorted():
    ef = ExecutiveFunction()
    ef.add_goal(GoalState(id=1, target=[0.5]*4, urgency=0.2))
    ef.add_goal(GoalState(id=2, target=[0.5]*4, urgency=0.9))
    goals = ef.get_goals()
    assert goals[0].priority_score >= goals[1].priority_score


def test_executive_state_serializable():
    es = ExecutiveState(
        active_goals=[], priority_weights={},
        attention_focus={}, conflicts=[], uncertainty=0.3,
        n_goals=5, n_active=2, n_completed=1, n_failed=0
    )
    assert es.n_goals == 5
    assert es.uncertainty == pytest.approx(0.3)
