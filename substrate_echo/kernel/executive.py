"""Executive Function — Governs what matters right now.

Goals are primitives. Executive Function determines:
  - Which goals exist
  - Which goals are active
  - Which goals get attention
  - How conflicts between goals are resolved
  - When goals are created from observations
  - When goals are archived or abandoned
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import time


class GoalStatus(Enum):
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"
    ARCHIVED = "archived"


class GoalTier(Enum):
    SAFETY = 0
    MAINTENANCE = 1
    ACTIVE = 2
    LEARNING = 3
    EXPLORATION = 4
    IDLE = 5


@dataclass
class GoalState:
    id: int
    target: List[float]
    description: str = ""
    embodiment_id: str = "default"
    status: GoalStatus = GoalStatus.CREATED
    tier: GoalTier = GoalTier.ACTIVE
    urgency: float = 0.5
    importance: float = 0.5
    confidence: float = 0.5
    expected_value: float = 0.5
    resource_cost: float = 0.3
    created_at: float = 0.0
    activated_at: float = 0.0
    completed_at: float = 0.0
    last_scored_at: float = 0.0
    priority_score: float = 0.0
    attention_weight: float = 0.5

    def effective_priority(self) -> float:
        if self.status in (GoalStatus.COMPLETED, GoalStatus.FAILED,
                           GoalStatus.ABANDONED, GoalStatus.ARCHIVED):
            return 0.0
        if self.status == GoalStatus.PAUSED:
            return self.priority_score * 0.1
        return self.priority_score


@dataclass
class GoalConflict:
    goal_a_id: int
    goal_b_id: int
    conflict_type: str
    resolution: str = ""


@dataclass
class ExecutiveState:
    active_goals: List[Dict[str, Any]]
    priority_weights: Dict[str, float]
    attention_focus: Dict[int, float]
    conflicts: List[Dict[str, Any]]
    uncertainty: float
    n_goals: int
    n_active: int
    n_completed: int
    n_failed: int


class PriorityScorer:
    """Scores goals based on urgency, importance, confidence,
    expected value, and resource cost."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "urgency": 0.3, "importance": 0.3,
            "confidence": 0.15, "expected_value": 0.2,
            "resource_cost_penalty": 0.05,
        }

    def score(self, goal: GoalState) -> float:
        raw = (
            self.weights["urgency"] * goal.urgency
            + self.weights["importance"] * goal.importance
            + self.weights["confidence"] * goal.confidence
            + self.weights["expected_value"] * goal.expected_value
            - self.weights["resource_cost_penalty"] * goal.resource_cost
        )
        tier_boost = max(0, (5 - goal.tier.value) * 0.05)
        return max(0.0, min(1.0, raw + tier_boost))

    def rescore_all(self, goals: List[GoalState]) -> List[GoalState]:
        for g in goals:
            g.priority_score = self.score(g)
            g.last_scored_at = time.time()
        return sorted(goals, key=lambda g: g.effective_priority(), reverse=True)


class AttentionAllocator:
    """Determines where the kernel's processing attention goes.

    Attention is a finite resource. Not every event gets full processing.
    """

    def __init__(self, max_attention: float = 10.0):
        self.max_attention = max_attention
        self._attention_map: Dict[int, float] = {}

    def allocate(self, goals: List[GoalState],
                 attractor_novelty: Optional[Dict[int, float]] = None,
                 prediction_errors: Optional[Dict[int, float]] = None
                 ) -> Dict[int, float]:
        self._attention_map.clear()

        for goal in goals:
            if goal.status != GoalStatus.ACTIVE:
                continue
            weight = goal.effective_priority() * goal.attention_weight
            for aid in range(200):
                self._attention_map[aid] = self._attention_map.get(aid, 0.0) + weight * 0.05

        if attractor_novelty:
            for aid, novelty in attractor_novelty.items():
                self._attention_map[aid] = self._attention_map.get(aid, 0.0) + novelty * 0.3

        if prediction_errors:
            for aid, error in prediction_errors.items():
                self._attention_map[aid] = self._attention_map.get(aid, 0.0) + min(1.0, error) * 0.4

        total = sum(self._attention_map.values())
        if total > self.max_attention and total > 0:
            scale = self.max_attention / total
            self._attention_map = {k: v * scale for k, v in self._attention_map.items()}

        return self._attention_map

    def get_attention(self, attractor_id: int) -> float:
        return self._attention_map.get(attractor_id, 0.0)

    def global_focus(self) -> float:
        return min(1.0, sum(self._attention_map.values()) / self.max_attention)


class GoalGenerator:
    """Generates goals from observations via rules."""

    def __init__(self):
        self._next_id = 1000
        self.rules: List[Callable] = [self._safety_rule]
        self._safety_cooldown: Dict[str, float] = {}  # embodiment_id -> last_safety_tick
        self._cooldown_ticks = 20  # Don't re-create safety goals within N ticks

    def check(self, observation: Any, current_tick: int = 0) -> List[GoalState]:
        new_goals = []
        for rule in self.rules:
            new_goals.extend(rule(observation, current_tick))
        return new_goals

    def _safety_rule(self, obs: Any, current_tick: int = 0) -> List[GoalState]:
        goals = []
        vec = obs.to_array()
        emb_id = obs.embodiment_id

        # Check cooldown
        last_tick = self._safety_cooldown.get(emb_id, -999)
        if current_tick - last_tick < self._cooldown_ticks:
            return goals

        # Only create if ANY dimension is extreme (not per-dimension)
        has_extreme = any(v < 0.05 or v > 0.95 for v in vec)
        if has_extreme:
            extreme_dims = [i for i, v in enumerate(vec) if v < 0.05 or v > 0.95]
            goals.append(GoalState(
                id=self._next_id,
                target=[0.5] * len(vec),
                description=f"Safety: dims {extreme_dims[:3]} extreme ({len(extreme_dims)} total)",
                embodiment_id=emb_id,
                status=GoalStatus.ACTIVE,
                tier=GoalTier.SAFETY,
                urgency=0.9, importance=0.9, confidence=1.0,
                expected_value=0.8, resource_cost=0.1,
                created_at=time.time(), activated_at=time.time(),
            ))
            self._next_id += 1
            self._safety_cooldown[emb_id] = current_tick
        return goals


class ExecutiveFunction:
    """Manages the goal landscape.

    Responsibilities:
      1. Goal lifecycle management
      2. Priority scoring
      3. Attention allocation
      4. Conflict detection and resolution
      5. Goal creation from observations
    """

    def __init__(self):
        self.scorer = PriorityScorer()
        self.attention = AttentionAllocator()
        self.generator = GoalGenerator()
        self._goals: Dict[int, GoalState] = {}
        self._next_id = 0
        self._conflicts: List[GoalConflict] = []

    def add_goal(self, goal: GoalState) -> int:
        if goal.id == 0:
            goal.id = self._next_id
        self._next_id = max(self._next_id, goal.id + 1)
        self._goals[goal.id] = goal
        return goal.id

    def activate_goal(self, goal_id: int):
        if goal_id in self._goals:
            g = self._goals[goal_id]
            g.status = GoalStatus.ACTIVE
            g.activated_at = time.time()

    def complete_goal(self, goal_id: int):
        if goal_id in self._goals:
            self._goals[goal_id].status = GoalStatus.COMPLETED
            self._goals[goal_id].completed_at = time.time()

    def fail_goal(self, goal_id: int):
        if goal_id in self._goals:
            self._goals[goal_id].status = GoalStatus.FAILED
            self._goals[goal_id].completed_at = time.time()

    def abandon_goal(self, goal_id: int):
        if goal_id in self._goals:
            self._goals[goal_id].status = GoalStatus.ABANDONED

    def pause_goal(self, goal_id: int):
        if goal_id in self._goals:
            self._goals[goal_id].status = GoalStatus.PAUSED

    def resume_goal(self, goal_id: int):
        if goal_id in self._goals:
            self._goals[goal_id].status = GoalStatus.ACTIVE

    def tick(self, observation: Any = None,
             attractor_novelty: Optional[Dict[int, float]] = None,
             prediction_errors: Optional[Dict[int, float]] = None,
             current_tick: int = 0
             ) -> ExecutiveState:
        if observation:
            for g in self.generator.check(observation, current_tick):
                self.add_goal(g)

        goals_list = list(self._goals.values())
        self.scorer.rescore_all(goals_list)
        self.attention.allocate(goals_list, attractor_novelty, prediction_errors)

        active = [g for g in goals_list if g.status == GoalStatus.ACTIVE]
        completed = [g for g in goals_list if g.status == GoalStatus.COMPLETED]
        failed = [g for g in goals_list if g.status == GoalStatus.FAILED]

        return ExecutiveState(
            active_goals=[{"id": g.id, "desc": g.description,
                           "score": g.priority_score, "tier": g.tier.name}
                          for g in active[:10]],
            priority_weights=self.scorer.weights,
            attention_focus={k: v for k, v in self.attention._attention_map.items()
                             if v > 0.01},
            conflicts=[{"a": c.goal_a_id, "b": c.goal_b_id,
                        "type": c.conflict_type}
                       for c in self._conflicts],
            uncertainty=1.0 - self.attention.global_focus(),
            n_goals=len(goals_list),
            n_active=len(active),
            n_completed=len(completed),
            n_failed=len(failed),
        )

    def get_goals(self) -> List[GoalState]:
        return sorted(self._goals.values(),
                      key=lambda g: g.effective_priority(), reverse=True)
