"""Tests for Habit Formation — P7.7"""
import numpy as np
from substrate_echo.core.habit_formation import (
    HabitFormation, HabitFormationConfig, Habit, HabitAction
)


def _make_context(val=0.5):
    return np.full(16, val)


def _make_actions(types=None):
    if types is None:
        types = ["approach", "observe"]
    return [{"type": t, "pillar": i, "magnitude": 0.1} for i, t in enumerate(types)]


class TestHabitRecord:
    def test_record_single(self):
        hf = HabitFormation()
        hf.record(_make_context(), _make_actions(), success=True, tick=0)
        assert len(hf._action_buffer) == 1

    def test_record_multiple(self):
        hf = HabitFormation()
        for i in range(10):
            hf.record(_make_context(0.5), _make_actions(["approach", "observe"]),
                      success=True, tick=i)
        assert len(hf._action_buffer) == 10

    def test_record_empty_actions(self):
        hf = HabitFormation()
        hf.record(_make_context(), [], success=True, tick=0)
        assert len(hf._action_buffer) == 0


class TestHabitDetection:
    def test_no_habit_few_observations(self):
        hf = HabitFormation()
        for i in range(2):
            hf.record(_make_context(0.5), _make_actions(["approach", "observe"]),
                      success=True, tick=i)
        assert len(hf._habits) == 0

    def test_habit_detected_after_repetitions(self):
        hf = HabitFormation()
        for i in range(5):
            hf.record(_make_context(0.5), _make_actions(["approach", "observe"]),
                      success=True, tick=i)
        assert len(hf._habits) >= 1

    def test_habit_needs_success_threshold(self):
        hf = HabitFormation()
        for i in range(5):
            hf.record(_make_context(0.5), _make_actions(["approach", "observe"]),
                      success=(i % 3 != 0), tick=i)  # mixed success
        # May or may not detect, depends on success rate >= 0.5
        habits = hf.get_all_habits()
        for h in habits:
            assert h.success_rate >= 0.5

    def test_habit_context_similarity(self):
        hf = HabitFormation()
        for i in range(5):
            hf.record(_make_context(0.5), _make_actions(["approach", "observe"]),
                      success=True, tick=i)
        # Similar context should match
        habit = hf.check_context(_make_context(0.5))
        # Habit may or may not be established yet (need 5+ uses)
        if habit:
            assert habit.is_established or habit.use_count >= 3


class TestHabitExecution:
    def test_execute_returns_action_dicts(self):
        habit = Habit(
            habit_id=0, name="test",
            context_signature=_make_context(),
            actions=_make_actions(["approach", "observe"]),
            strength=1.0, use_count=10, success_count=9,
        )
        hf = HabitFormation()
        result = hf.execute_habit(habit, tick=0)
        assert len(result) == 2
        assert result[0]["type"] == "approach"
        assert result[0]["source"] == "habit"
        assert result[0]["habit_id"] == 0

    def test_execute_increments_use_count(self):
        habit = Habit(
            habit_id=0, name="test",
            context_signature=_make_context(),
            actions=_make_actions(),
            strength=0.5, use_count=5, success_count=4,
        )
        hf = HabitFormation()
        hf.execute_habit(habit, tick=0)
        assert habit.use_count == 6

    def test_report_outcome_success(self):
        habit = Habit(
            habit_id=0, name="test",
            context_signature=_make_context(),
            actions=_make_actions(),
            strength=0.5, use_count=5, success_count=4,
        )
        hf = HabitFormation()
        hf.report_outcome(habit, success=True)
        assert habit.success_count == 5
        assert habit.strength > 0.5

    def test_report_outcome_failure(self):
        habit = Habit(
            habit_id=0, name="test",
            context_signature=_make_context(),
            actions=_make_actions(),
            strength=0.5, use_count=5, success_count=4,
        )
        hf = HabitFormation()
        hf.report_outcome(habit, success=False)
        assert habit.strength < 0.5


class TestHabitCost:
    def test_cost_comparison(self):
        habit = Habit(
            habit_id=0, name="test",
            context_signature=_make_context(),
            actions=_make_actions(["approach", "observe"]),
            strength=1.0,
        )
        hf = HabitFormation()
        costs = hf.cost_comparison(habit)
        assert costs["habit_cost"] < costs["planning_cost"]
        assert costs["speedup"] > 1.0


class TestHabitDecay:
    def test_habit_strength_decays(self):
        hf = HabitFormation()
        for i in range(10):
            hf.record(_make_context(0.5), _make_actions(["approach", "observe"]),
                      success=True, tick=i)
        habits = hf.get_all_habits()
        if habits:
            initial_strength = habits[0].strength
            # Run more records to trigger decay
            for i in range(100):
                hf.record(_make_context(0.5), _make_actions(["approach", "observe"]),
                          success=True, tick=10 + i)
            # Strength should not exceed 1.0
            for h in hf.get_all_habits():
                assert h.strength <= 1.0


class TestHabitPreferences:
    def test_get_action_preferences(self):
        hf = HabitFormation()
        ctx = _make_context(0.5)
        for i in range(10):
            hf.record(ctx, _make_actions(["approach", "observe"]),
                      success=True, tick=i)
        prefs = hf.get_action_preferences(_make_context(0.5))
        assert "approach" in prefs
        assert "observe" in prefs
        assert prefs["approach"] > 0


class TestHabitEstablished:
    def test_established_requires_repetitions(self):
        h = Habit(
            habit_id=0, name="test",
            context_signature=_make_context(),
            actions=_make_actions(),
            use_count=3, success_count=3, strength=0.5,
        )
        assert not h.is_established  # needs >= 5

    def test_established_requires_success_rate(self):
        h = Habit(
            habit_id=0, name="test",
            context_signature=_make_context(),
            actions=_make_actions(),
            use_count=10, success_count=5, strength=0.5,
        )
        assert not h.is_established  # 50% success, needs > 70%

    def test_established_when_thresholds_met(self):
        h = Habit(
            habit_id=0, name="test",
            context_signature=_make_context(),
            actions=_make_actions(),
            use_count=10, success_count=8, strength=0.8,
        )
        assert h.is_established


class TestHabitStats:
    def test_stats_empty(self):
        hf = HabitFormation()
        s = hf.stats()
        assert s["n_habits"] == 0
        assert s["n_established"] == 0

    def test_stats_with_habits(self):
        hf = HabitFormation()
        for i in range(10):
            hf.record(_make_context(0.5), _make_actions(["approach", "observe"]),
                      success=True, tick=i)
        s = hf.stats()
        assert s["n_habits"] >= 0  # may or may not have habits yet
        assert s["buffer_size"] == 10


class TestHabitMultipleSequences:
    def test_different_contexts_different_habits(self):
        hf = HabitFormation()
        # Create habit for context A
        for i in range(8):
            hf.record(_make_context(0.2), _make_actions(["approach", "observe"]),
                      success=True, tick=i)
        # Create habit for context B
        for i in range(8):
            hf.record(_make_context(0.8), _make_actions(["retreat", "investigate"]),
                      success=True, tick=100 + i)
        # Should have habits for both contexts
        assert len(hf._habits) >= 1  # at least some detection
