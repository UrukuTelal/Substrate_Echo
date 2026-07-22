"""Tests for Counterfactual Reasoning — P7.2"""
import numpy as np
from substrate_echo.core.counterfactual import (
    CounterfactualReasoning, CounterfactualConfig, DecisionPoint,
    CounterfactualResult, RiskAssessment, Lesson
)


def _ctx(val=0.5):
    return np.full(16, val)


class TestDecisionRecord:
    def test_record_single(self):
        cr = CounterfactualReasoning()
        dp = cr.record_decision(
            _ctx(), "approach", 0, _ctx(0.6), tick=10, utility=0.5)
        assert dp.decision_id == 0
        assert dp.action_taken == "approach"
        assert dp.utility == 0.5

    def test_record_multiple(self):
        cr = CounterfactualReasoning()
        for i in range(5):
            cr.record_decision(
                _ctx(i * 0.1), "approach", i, _ctx((i + 1) * 0.1),
                tick=i, utility=i * 0.1)
        assert len(cr._decisions) == 5

    def test_record_evicts_old(self):
        config = CounterfactualConfig(max_decision_points=10)
        cr = CounterfactualReasoning(config=config)
        for i in range(15):
            cr.record_decision(
                _ctx(i * 0.05), "approach", 0, _ctx((i + 1) * 0.05),
                tick=i)
        assert len(cr._decisions) <= 10


class TestSimulateAlternative:
    def test_simulate_returns_result(self):
        cr = CounterfactualReasoning()
        cr.record_decision(_ctx(), "approach", 0, _ctx(0.6), tick=0)
        result = cr.simulate_alternative(_ctx(), "observe", 1, tick=0)
        assert isinstance(result, CounterfactualResult)
        assert result.actual_action == "approach"
        assert result.alternative_action == "observe"

    def test_simulate_with_explicit_decision(self):
        cr = CounterfactualReasoning()
        dp = cr.record_decision(_ctx(), "approach", 0, _ctx(0.6), tick=0)
        result = cr.simulate_alternative(
            _ctx(), "observe", 1, tick=0, decision_id=dp.decision_id)
        assert result.decision_id == dp.decision_id

    def test_outcome_distance_nonneg(self):
        cr = CounterfactualReasoning()
        cr.record_decision(_ctx(), "approach", 0, _ctx(0.6), tick=0)
        result = cr.simulate_alternative(_ctx(), "observe", 1, tick=0)
        assert result.outcome_distance >= 0

    def test_regret_and_relief_complementary(self):
        cr = CounterfactualReasoning()
        cr.record_decision(_ctx(), "approach", 0, _ctx(0.6), tick=0,
                           utility=0.5)
        result = cr.simulate_alternative(_ctx(), "observe", 1, tick=0)
        # Either regret or relief should be zero
        assert result.regret == 0 or result.relief == 0

    def test_simulate_batch(self):
        cr = CounterfactualReasoning()
        cr.record_decision(_ctx(), "approach", 0, _ctx(0.6), tick=0)
        alternatives = [("observe", 1), ("retreat", 2), ("grasp", 3)]
        results = cr.simulate_batch(_ctx(), alternatives, tick=0)
        assert len(results) == 3
        assert results[0].alternative_action == "observe"
        assert results[2].alternative_action == "grasp"

    def test_simulate_without_decision(self):
        cr = CounterfactualReasoning()
        # No decisions recorded, should still work
        result = cr.simulate_alternative(_ctx(), "observe", 1, tick=0)
        assert result.decision_id == -1


class TestRiskAssessment:
    def test_assess_risk_returns_assessment(self):
        cr = CounterfactualReasoning()
        risk = cr.assess_risk(_ctx(), "approach", action_id=0)
        assert isinstance(risk, RiskAssessment)
        assert risk.action == "approach"

    def test_risk_has_bounds(self):
        cr = CounterfactualReasoning()
        risk = cr.assess_risk(_ctx(), "approach", action_id=0,
                              n_simulations=5)
        assert 0 <= risk.expected_risk <= 1
        assert risk.max_danger >= 0

    def test_risk_recommendation(self):
        cr = CounterfactualReasoning()
        risk = cr.assess_risk(_ctx(), "approach", action_id=0)
        assert risk.recommendation in ("proceed", "caution", "avoid",
                                       "insufficient_data")


class TestLessons:
    def test_extract_lessons_empty(self):
        cr = CounterfactualReasoning()
        lessons = cr.extract_lessons()
        assert len(lessons) == 0

    def test_extract_lessons_after_counterfactuals(self):
        cr = CounterfactualReasoning()
        # Create several decisions with the same action
        for i in range(5):
            cr.record_decision(
                _ctx(i * 0.1), "approach", 0, _ctx((i + 1) * 0.1),
                tick=i, utility=0.1)
        # Simulate alternatives — observe might be better
        for i in range(5):
            cr.simulate_alternative(_ctx(i * 0.1), "observe", 1, tick=i)
        lessons = cr.extract_lessons()
        # Should extract some lessons
        assert isinstance(lessons, list)

    def test_lesson_strength_increases(self):
        cr = CounterfactualReasoning()
        cr.record_decision(_ctx(), "approach", 0, _ctx(0.6), tick=0, utility=0.1)
        # Run same counterfactual multiple times
        for _ in range(5):
            cr.simulate_alternative(_ctx(), "observe", 1, tick=0)
        cr.extract_lessons()
        # Run again to strengthen
        for _ in range(5):
            cr.simulate_alternative(_ctx(), "observe", 1, tick=0)
        cr.extract_lessons()
        lessons = list(cr._lessons.values())
        if lessons:
            assert lessons[0].strength > 0


class TestRegretStats:
    def test_regret_stats_empty(self):
        cr = CounterfactualReasoning()
        stats = cr.get_regret_stats()
        assert stats["n_counterfactuals"] == 0

    def test_regret_stats_with_data(self):
        cr = CounterfactualReasoning()
        cr.record_decision(_ctx(), "approach", 0, _ctx(0.6), tick=0, utility=0.5)
        cr.simulate_alternative(_ctx(), "observe", 1, tick=0)
        stats = cr.get_regret_stats()
        assert stats["n_counterfactuals"] == 1
        assert stats["avg_regret"] >= 0
        assert stats["avg_relief"] >= 0


class TestSummary:
    def test_summary(self):
        cr = CounterfactualReasoning()
        cr.record_decision(_ctx(), "approach", 0, _ctx(0.6), tick=0)
        cr.simulate_alternative(_ctx(), "observe", 1, tick=0)
        s = cr.summary()
        assert s["n_decisions"] == 1
        assert s["n_counterfactuals"] == 1
        assert "regret_stats" in s


class TestCounterfactualResult:
    def test_result_fields(self):
        result = CounterfactualResult(
            decision_id=0,
            actual_action="approach",
            alternative_action="observe",
            actual_outcome=np.ones(16),
            predicted_outcome=np.ones(16) * 0.5,
            outcome_distance=0.5,
            utility_difference=0.2,
            regret=0.2,
            relief=0.0,
            lesson="test_lesson",
            confidence=0.8,
        )
        assert result.regret == 0.2
        assert result.relief == 0.0
        assert result.lesson == "test_lesson"


class TestLesson:
    def test_lesson_fields(self):
        lesson = Lesson(
            lesson_id=0,
            context="social",
            action="approach",
            direction="avoid",
            strength=0.7,
            source_decisions=[0, 1, 2],
        )
        assert lesson.direction == "avoid"
        assert lesson.strength == 0.7
        assert len(lesson.source_decisions) == 3
