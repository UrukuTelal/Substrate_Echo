"""Counterfactual Reasoning — P7.2

Simulates alternate futures from branching decision points.

Core idea: the agent doesn't just remember what happened — it can
imagine what *would have happened* if it had acted differently. This
enables regret, relief, risk assessment, and strategy improvement.

Architecture:

    Actual trajectory:  state → action A → outcome Y
    Counterfactual:     state → action B → predicted outcome Y'

    comparison(Y, Y') → regret / relief / lesson

This module uses DynamicsMemory for simulation and Evaluator for
outcome comparison. It stores counterfactual results as structured
events and extracts actionable lessons.

Usage:
    cr = CounterfactualReasoning(dynamics_memory, evaluator)

    # At a decision point
    cr.record_decision(
        state=current_state,
        action_taken="approach",
        action_taken_id=0,
        outcome=actual_outcome,
        tick=t,
    )

    # Simulate alternatives
    result = cr.simulate_alternative(
        state=current_state,
        alternative_action="observe",
        alternative_action_id=1,
        tick=t,
    )

    # Extract lessons
    lessons = cr.extract_lessons()

    # Risk assessment
    risk = cr.assess_risk(current_state, candidate_action="approach")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
from collections import defaultdict


@dataclass
class DecisionPoint:
    """A recorded decision: state, action taken, and outcome."""
    decision_id: int
    tick: int
    state: np.ndarray
    action_taken: str
    action_taken_id: int
    outcome: np.ndarray
    pillar_values: Optional[np.ndarray] = None
    utility: float = 0.0  # utility of actual outcome


@dataclass
class CounterfactualResult:
    """Comparison between actual and counterfactual outcomes."""
    decision_id: int
    actual_action: str
    alternative_action: str
    actual_outcome: np.ndarray
    predicted_outcome: np.ndarray

    # Comparison metrics
    outcome_distance: float = 0.0    # how different the outcomes are
    utility_difference: float = 0.0  # alt_utility - actual_utility
    regret: float = 0.0              # positive = alternative was better
    relief: float = 0.0              # positive = actual was better
    risk_delta: float = 0.0          # how much riskier/safer the alternative was

    # Lesson extracted
    lesson: str = ""
    confidence: float = 0.0


@dataclass
class RiskAssessment:
    """Risk evaluation for a candidate action in a given state."""
    action: str
    state: np.ndarray
    predicted_utility: float = 0.0
    worst_case_utility: float = 0.0
    best_case_utility: float = 0.0
    expected_risk: float = 0.0     # 0-1, probability of bad outcome
    max_danger: float = 0.0       # worst possible damage
    recommendation: str = ""       # "proceed", "caution", "avoid"


@dataclass
class Lesson:
    """An extracted lesson from counterfactual analysis."""
    lesson_id: int
    context: str          # when this applies
    action: str           # what to do / not do
    direction: str        # "prefer" or "avoid"
    strength: float       # 0-1, how confident in this lesson
    source_decisions: list[int] = field(default_factory=list)
    uses: int = 0


@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual reasoning."""
    simulation_depth: int = 3        # how many steps to simulate ahead
    max_decision_points: int = 200
    max_alternatives: int = 5        # per decision point
    regret_threshold: float = 0.1    # min utility diff to count as regret
    risk_danger_threshold: float = 0.7
    lesson_min_strength: float = 0.3
    context_dim: int = 16


class CounterfactualReasoning:
    """Simulates alternate futures and extracts lessons.

    Records decision points, simulates what would have happened
    with different actions, compares outcomes, and extracts
    actionable lessons for future behavior.

    Usage:
        cr = CounterfactualReasoning(dynamics_memory, evaluator)

        # Record what happened
        cr.record_decision(state, "approach", 0, outcome, tick)

        # What if I had observed instead?
        result = cr.simulate_alternative(state, "observe", 1, tick)

        # Extract lessons from all counterfactuals
        lessons = cr.extract_lessons()
    """

    def __init__(self,
                 dynamics_memory=None,
                 evaluator=None,
                 config: Optional[CounterfactualConfig] = None):
        self.config = config or CounterfactualConfig()
        self._dm = dynamics_memory
        self._evaluator = evaluator

        self._decisions: dict[int, DecisionPoint] = {}
        self._counterfactuals: list[CounterfactualResult] = []
        self._lessons: dict[int, Lesson] = {}
        self._next_decision_id = 0
        self._next_lesson_id = 0

        # Index for fast lookup
        self._state_action_index: dict[tuple, list[int]] = defaultdict(list)

    def record_decision(self, state: np.ndarray, action_taken: str,
                        action_taken_id: int, outcome: np.ndarray,
                        tick: int,
                        pillar_values: Optional[np.ndarray] = None,
                        utility: float = 0.0) -> DecisionPoint:
        """Record a decision that was actually made.

        Args:
            state: decision state
            action_taken: name of action chosen
            action_taken_id: numeric action ID
            outcome: resulting state
            tick: current tick
            pillar_values: PSV at decision time
            utility: utility of the actual outcome

        Returns:
            The created DecisionPoint
        """
        dp_id = self._next_decision_id
        self._next_decision_id += 1

        dp = DecisionPoint(
            decision_id=dp_id,
            tick=tick,
            state=np.asarray(state, dtype=np.float64),
            action_taken=action_taken,
            action_taken_id=action_taken_id,
            outcome=np.asarray(outcome, dtype=np.float64),
            pillar_values=pillar_values,
            utility=utility,
        )

        self._decisions[dp_id] = dp

        # Index by (state_region, action) for fast lookup
        state_key = self._quantize_state(state)
        self._state_action_index[(state_key, action_taken)].append(dp_id)

        # Evict old decisions
        if len(self._decisions) > self.config.max_decision_points:
            self._evict_old()

        return dp

    def simulate_alternative(self, state: np.ndarray,
                             alternative_action: str,
                             alternative_action_id: int,
                             tick: int,
                             decision_id: Optional[int] = None,
                             action_magnitude: float = 0.1) -> CounterfactualResult:
        """Simulate what would happen with a different action.

        Uses DynamicsMemory to predict the trajectory from the
        alternative action, then compares to the actual outcome.

        Args:
            state: the decision state (same as actual decision)
            alternative_action: name of alternative action
            alternative_action_id: numeric action ID
            tick: current tick
            decision_id: link to the actual decision (if known)
            action_magnitude: strength of alternative action

        Returns:
            CounterfactualResult with comparison
        """
        state = np.asarray(state, dtype=np.float64)

        # Find matching actual decision if not provided
        if decision_id is None:
            decision_id = self._find_closest_decision(state, alternative_action)

        # Get actual outcome for comparison
        if decision_id is not None and decision_id in self._decisions:
            actual = self._decisions[decision_id]
            actual_outcome = actual.outcome
            actual_action = actual.action_taken
            actual_utility = actual.utility
        else:
            # No actual decision to compare to
            actual_outcome = state.copy()
            actual_action = "unknown"
            actual_utility = 0.0

        # Simulate alternative trajectory
        predicted_outcome = self._simulate_trajectory(
            state, alternative_action_id, action_magnitude, tick)

        # Compute comparison metrics
        outcome_distance = float(np.linalg.norm(predicted_outcome - actual_outcome))
        alt_utility = self._evaluate_state(predicted_outcome)
        utility_diff = alt_utility - actual_utility

        regret = max(0.0, utility_diff)
        relief = max(0.0, -utility_diff)

        # Risk delta: compare danger of outcomes
        actual_danger = self._compute_danger(actual_outcome)
        alt_danger = self._compute_danger(predicted_outcome)
        risk_delta = alt_danger - actual_danger

        # Extract lesson
        lesson_text, lesson_confidence = self._compare_and_lesson(
            actual_action, alternative_action,
            actual_outcome, predicted_outcome,
            actual_utility, alt_utility)

        result = CounterfactualResult(
            decision_id=decision_id if decision_id is not None else -1,
            actual_action=actual_action,
            alternative_action=alternative_action,
            actual_outcome=actual_outcome,
            predicted_outcome=predicted_outcome,
            outcome_distance=outcome_distance,
            utility_difference=utility_diff,
            regret=regret,
            relief=relief,
            risk_delta=risk_delta,
            lesson=lesson_text,
            confidence=lesson_confidence,
        )

        self._counterfactuals.append(result)

        return result

    def simulate_batch(self, state: np.ndarray,
                       alternative_actions: list[tuple[str, int]],
                       tick: int) -> list[CounterfactualResult]:
        """Simulate multiple alternatives at the same decision point."""
        results = []
        for action_name, action_id in alternative_actions:
            result = self.simulate_alternative(
                state, action_name, action_id, tick)
            results.append(result)
        return results

    def assess_risk(self, state: np.ndarray, candidate_action: str,
                    action_id: int = 0,
                    n_simulations: int = 3) -> RiskAssessment:
        """Assess the risk of taking an action in a given state.

        Runs multiple simulated trajectories with slight perturbations
        to estimate worst/best case outcomes.
        """
        state = np.asarray(state, dtype=np.float64)
        utilities = []

        for i in range(n_simulations):
            # Add small perturbation to state
            perturbed = state + np.random.randn(*state.shape) * 0.05
            predicted = self._simulate_trajectory(
                perturbed, action_id, 0.1, 0)
            util = self._evaluate_state(predicted)
            utilities.append(util)

        if not utilities:
            return RiskAssessment(
                action=candidate_action, state=state,
                recommendation="insufficient_data")

        utilities = np.array(utilities)
        predicted_util = float(np.mean(utilities))
        worst = float(np.min(utilities))
        best = float(np.max(utilities))

        # Risk is probability of utility below threshold
        danger_count = np.sum(utilities < -0.5)
        expected_risk = float(danger_count / len(utilities))

        # Max danger: how far below zero
        max_danger = max(0.0, -worst)

        # Recommendation
        if expected_risk > self.config.risk_danger_threshold:
            recommendation = "avoid"
        elif expected_risk > 0.3:
            recommendation = "caution"
        else:
            recommendation = "proceed"

        return RiskAssessment(
            action=candidate_action,
            state=state,
            predicted_utility=predicted_util,
            worst_case_utility=worst,
            best_case_utility=best,
            expected_risk=expected_risk,
            max_danger=max_danger,
            recommendation=recommendation,
        )

    def extract_lessons(self, min_strength: Optional[float] = None) -> list[Lesson]:
        """Extract actionable lessons from counterfactual comparisons.

        Aggregates patterns across multiple counterfactual results
        to form stable lessons about what works and what doesn't.
        """
        if min_strength is None:
            min_strength = self.config.lesson_min_strength

        # Aggregate by (action, context_type)
        action_outcomes: dict[str, list[float]] = defaultdict(list)
        action_pairs: dict[tuple[str, str], list[float]] = defaultdict(list)

        for cf in self._counterfactuals:
            action_outcomes[cf.actual_action].append(cf.utility_difference)
            action_pairs[(cf.actual_action, cf.alternative_action)].append(
                cf.utility_difference)

        # Create lessons from consistent patterns
        for action, diffs in action_outcomes.items():
            if len(diffs) < 2:
                continue
            avg_diff = np.mean(diffs)
            strength = min(1.0, len(diffs) / 10.0) * min(1.0, abs(avg_diff) / 0.2)

            if strength < min_strength:
                continue

            direction = "prefer" if avg_diff > 0 else "avoid"

            # Check if lesson already exists
            existing = self._find_lesson(action, direction)
            if existing:
                existing.strength = min(1.0, existing.strength + 0.1)
                existing.uses += 1
            else:
                self._create_lesson(action, direction, strength, diffs)

        # Pair-based lessons
        for (actual, alternative), diffs in action_pairs.items():
            if len(diffs) < 2:
                continue
            avg_diff = np.mean(diffs)
            if avg_diff < self.config.regret_threshold:
                continue

            # "alternative is better than actual in this context"
            lesson_text = f"prefer_{alternative}_over_{actual}"
            strength = min(1.0, len(diffs) / 10.0)

            existing = self._find_lesson(lesson_text, "prefer")
            if existing:
                existing.strength = min(1.0, existing.strength + 0.1)
            else:
                self._create_lesson(lesson_text, "prefer", strength, diffs)

        return [l for l in self._lessons.values() if l.strength >= min_strength]

    def get_regret_stats(self) -> dict:
        """Summary of regret patterns."""
        if not self._counterfactuals:
            return {"n_counterfactuals": 0, "avg_regret": 0, "avg_relief": 0}

        regrets = [cf.regret for cf in self._counterfactuals]
        reliefs = [cf.relief for cf in self._counterfactuals]
        return {
            "n_counterfactuals": len(self._counterfactuals),
            "avg_regret": float(np.mean(regrets)),
            "max_regret": float(np.max(regrets)),
            "avg_relief": float(np.mean(reliefs)),
            "max_relief": float(np.max(reliefs)),
            "regret_rate": float(np.mean([r > 0 for r in regrets])),
        }

    def summary(self) -> dict:
        """Full summary of counterfactual reasoning state."""
        return {
            "n_decisions": len(self._decisions),
            "n_counterfactuals": len(self._counterfactuals),
            "n_lessons": len(self._lessons),
            "regret_stats": self.get_regret_stats(),
        }

    # ── Private methods ──────────────────────────────────────

    def _simulate_trajectory(self, state: np.ndarray,
                             action_id: int,
                             magnitude: float,
                             tick: int) -> np.ndarray:
        """Simulate trajectory from state using DynamicsMemory."""
        if self._dm is not None:
            try:
                # Apply action as perturbation to state
                perturbed = state.copy()
                dim = min(action_id + 1, len(perturbed))
                perturbed[action_id % len(perturbed)] += magnitude

                # Use dynamics memory to predict
                prediction = self._dm.predict(perturbed)
                if prediction is not None:
                    return np.asarray(prediction, dtype=np.float64)
            except Exception:
                pass

        # Fallback: simple linear projection
        return state + np.random.randn(*state.shape) * 0.01

    def _evaluate_state(self, state: np.ndarray) -> float:
        """Evaluate utility of a state using Evaluator."""
        if self._evaluator is not None:
            try:
                result = self._evaluator.evaluate(state, lightweight=True)
                return result.total_utility
            except Exception:
                pass
        # Fallback: simple norm-based utility
        return float(-np.linalg.norm(state))

    def _compute_danger(self, state: np.ndarray) -> float:
        """Compute danger score for a state (0-1)."""
        # Simple heuristic: states far from origin or with extreme values
        extreme_count = np.sum(np.abs(state) > 0.8)
        danger = min(1.0, extreme_count / len(state))
        return float(danger)

    def _compare_and_lesson(self, actual_action: str, alt_action: str,
                            actual_outcome: np.ndarray,
                            alt_outcome: np.ndarray,
                            actual_util: float,
                            alt_util: float) -> tuple[str, float]:
        """Compare outcomes and generate a lesson string."""
        utility_diff = alt_util - actual_util

        if abs(utility_diff) < 0.05:
            return "", 0.0

        if utility_diff > 0:
            return (f"action_{alt_action}_would_have_been_better_than_{actual_action}",
                    min(1.0, abs(utility_diff)))
        else:
            return (f"action_{actual_action}_was_better_than_{alt_action}",
                    min(1.0, abs(utility_diff)))

    def _find_closest_decision(self, state: np.ndarray,
                               action: str) -> Optional[int]:
        """Find the closest matching decision point."""
        best_id = None
        best_dist = float('inf')

        for dp in self._decisions.values():
            dist = np.linalg.norm(state - dp.state)
            if dist < best_dist:
                best_dist = dist
                best_id = dp.decision_id

        return best_id

    def _quantize_state(self, state: np.ndarray) -> str:
        """Quantize state to a discrete key for indexing."""
        return tuple((state * 4).astype(int).tolist()).__str__()

    def _find_lesson(self, action: str, direction: str) -> Optional[Lesson]:
        """Find existing lesson matching action and direction."""
        for lesson in self._lessons.values():
            if lesson.action == action and lesson.direction == direction:
                return lesson
        return None

    def _create_lesson(self, action: str, direction: str,
                       strength: float, source_diffs: list[float]) -> Lesson:
        """Create a new lesson."""
        l_id = self._next_lesson_id
        self._next_lesson_id += 1

        lesson = Lesson(
            lesson_id=l_id,
            context="general",
            action=action,
            direction=direction,
            strength=strength,
            source_decisions=[],
        )

        self._lessons[l_id] = lesson
        return lesson

    def _evict_old(self) -> None:
        """Remove oldest decisions when at capacity."""
        if len(self._decisions) <= self.config.max_decision_points:
            return
        sorted_ids = sorted(self._decisions.keys())
        n_remove = len(self._decisions) - self.config.max_decision_points
        for i in range(n_remove):
            del self._decisions[sorted_ids[i]]
