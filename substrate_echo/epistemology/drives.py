"""Homeostatic Drives — Need-based motivation system.

Pure cognitive module. No domain knowledge.

The DriveManager maintains a set of Needs. Each Need tracks:
  - current value [0, 1]
  - target value [0, 1] (shifts with game phase)
  - deficit (target - current)
  - velocity (rate of change)
  - urgency (deficit weighted by velocity)

The motivational state is a deficit vector that the affordance
layer uses to score actions via dot product with affinity vectors.

Architecture:
    DriveManager (this module)
        ↓
    Motivational State (deficit vector)
        ↓
    Affordance Tracer (assigns affinity vectors per action)
        ↓
    Action Bridge (dot product + other signals)
        ↓
    Governance Gate (policy checks)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time


# ── Need Types ───────────────────────────────────────────────────

class NeedType(Enum):
    """Abstract need categories. No domain-specific types here."""
    MINERALS = "minerals"
    GAS = "gas"
    SUPPLY = "supply"
    MILITARY = "military"
    INTEL = "intel"
    DEFENSE = "defense"
    EXPANSION = "expansion"
    TECHNOLOGY = "technology"


# ── Need ─────────────────────────────────────────────────────────

@dataclass
class Need:
    """A single homeostatic need.

    Tracks current state, target, and the rate of change
    (velocity) to capture momentum — whether the deficit
    is growing or shrinking.
    """
    need_type: NeedType
    current: float = 0.0       # [0, 1] current satisfaction level
    target: float = 0.5        # [0, 1] desired satisfaction level
    weight: float = 1.0        # priority multiplier (higher = more important)

    # Internal tracking
    _velocity: float = 0.0     # rate of change (negative = getting worse)
    _prev_current: float = 0.0
    _last_update_tick: int = 0
    _adaptation_rate: float = 0.1  # how fast velocity smooths

    @property
    def deficit(self) -> float:
        """Unsatisfied portion of this need. [0, 1]"""
        return max(0.0, min(1.0, self.target - self.current))

    @property
    def urgency(self) -> float:
        """Deficit weighted by momentum.

        If deficit is growing (velocity < 0), urgency rises.
        If deficit is shrinking (velocity > 0), urgency falls.
        """
        # velocity is change in current. Negative = current dropping = worse.
        momentum_factor = 1.0 - self._velocity  # boost when velocity negative
        momentum_factor = max(0.1, min(3.0, momentum_factor))
        return self.deficit * self.weight * momentum_factor

    @property
    def is_satisfied(self) -> bool:
        return self.deficit <= 0.05

    @property
    def velocity(self) -> float:
        """Rate of change of current value. Positive = improving."""
        return self._velocity

    def update(self, new_current: float, tick: int):
        """Update current value and recompute velocity."""
        dt = max(1, tick - self._last_update_tick) if self._last_update_tick > 0 else 1

        raw_velocity = (new_current - self._prev_current) / dt

        # Exponential smoothing
        self._velocity = (
            self._adaptation_rate * raw_velocity
            + (1.0 - self._adaptation_rate) * self._velocity
        )

        self._prev_current = self.current
        self.current = max(0.0, min(1.0, new_current))
        self._last_update_tick = tick

    def set_target(self, new_target: float):
        """Update the target (e.g., when game phase changes)."""
        self.target = max(0.0, min(1.0, new_target))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.need_type.value,
            "current": round(self.current, 3),
            "target": round(self.target, 3),
            "deficit": round(self.deficit, 3),
            "velocity": round(self._velocity, 3),
            "urgency": round(self.urgency, 3),
            "weight": round(self.weight, 3),
        }


# ── Need Model ───────────────────────────────────────────────────

@dataclass
class NeedModel:
    """Tracks adaptation of target values over time.

    If repeated observations show that a different target
    produces better outcomes, the target gradually shifts.
    """
    need_type: NeedType
    base_target: float = 0.5
    current_target: float = 0.5
    confidence: float = 0.5    # how confident in this target
    adaptation_rate: float = 0.01  # how fast target shifts
    _outcome_history: List[float] = field(default_factory=list)

    def observe_outcome(self, outcome_quality: float):
        """Record how well the current target performed.

        outcome_quality: [0, 1] — higher is better.
        """
        self._outcome_history.append(outcome_quality)
        if len(self._outcome_history) > 100:
            self._outcome_history.pop(0)

        # Adapt target based on outcomes
        if len(self._outcome_history) >= 10:
            recent_avg = sum(self._outcome_history[-10:]) / 10
            if recent_avg > 0.7:
                # Good outcomes — target might be too low, increase slightly
                self.current_target = min(
                    1.0,
                    self.current_target + self.adaptation_rate * recent_avg
                )
            elif recent_avg < 0.4:
                # Bad outcomes — target might be too high, decrease
                self.current_target = max(
                    0.0,
                    self.current_target - self.adaptation_rate * (1.0 - recent_avg)
                )

            # Update confidence based on consistency
            if len(self._outcome_history) >= 20:
                variance = sum(
                    (x - sum(self._outcome_history[-20:]) / 20) ** 2
                    for x in self._outcome_history[-20:]
                ) / 20
                self.confidence = max(0.1, min(1.0, 1.0 - variance))


# ── Drive Manager ────────────────────────────────────────────────

class DriveManager:
    """Manages all homeostatic needs.

    Pure cognitive module. No domain knowledge.
    Provides:
      - deficits(): current deficit vector
      - urgency_vector(): deficit × momentum × weight
      - update(): feed new observations
      - set_phase_targets(): shift targets for game phase
    """

    def __init__(self):
        self._needs: Dict[NeedType, Need] = {}
        self._models: Dict[NeedType, NeedModel] = {}
        self._tick = 0
        self._history: List[Dict[str, Any]] = []

    def add_need(self, need_type: NeedType,
                 initial: float = 0.0,
                 target: float = 0.5,
                 weight: float = 1.0):
        """Register a new need."""
        self._needs[need_type] = Need(
            need_type=need_type,
            current=initial,
            target=target,
            weight=weight,
        )
        self._models[need_type] = NeedModel(
            need_type=need_type,
            base_target=target,
            current_target=target,
        )

    def get_need(self, need_type: NeedType) -> Optional[Need]:
        return self._needs.get(need_type)

    @property
    def need_types(self) -> List[NeedType]:
        return list(self._needs.keys())

    def deficits(self) -> Dict[NeedType, float]:
        """Current deficit for each need."""
        return {nt: n.deficit for nt, n in self._needs.items()}

    def urgency_vector(self) -> Dict[NeedType, float]:
        """Urgency for each need (deficit × momentum × weight)."""
        return {nt: n.urgency for nt, n in self._needs.items()}

    def current_values(self) -> Dict[NeedType, float]:
        """Current satisfaction for each need."""
        return {nt: n.current for nt, n in self._needs.items()}

    def velocity_vector(self) -> Dict[NeedType, float]:
        """Rate of change for each need."""
        return {nt: n.velocity for nt, n in self._needs.items()}

    def update(self, observations: Dict[NeedType, float], tick: int):
        """Update all needs from observations.

        observations: mapping from NeedType to current observed value [0, 1]
        """
        self._tick = tick
        for nt, value in observations.items():
            if nt in self._needs:
                self._needs[nt].update(value, tick)

        # Record snapshot
        self._history.append({
            "tick": tick,
            "deficits": self.deficits(),
            "urgency": self.urgency_vector(),
        })
        if len(self._history) > 500:
            self._history.pop(0)

    def set_phase_targets(self, targets: Dict[NeedType, float]):
        """Shift targets for a new game phase.

        targets: mapping from NeedType to desired target value [0, 1]
        """
        for nt, target in targets.items():
            if nt in self._needs:
                self._needs[nt].set_target(target)
                self._models[nt].current_target = target

    def record_outcome(self, need_type: NeedType, quality: float):
        """Record outcome quality for adaptive targeting."""
        if need_type in self._models:
            self._models[need_type].observe_outcome(quality)

    def get_adapted_targets(self) -> Dict[NeedType, float]:
        """Get targets after adaptation."""
        return {nt: m.current_target for nt, m in self._models.items()}

    def render(self) -> str:
        """Human-readable motivational state."""
        lines = ["  Drive Manager State:"]
        lines.append(f"    {'Need':12s} {'Current':>7s} {'Target':>7s} "
                     f"{'Deficit':>7s} {'Velocity':>8s} {'Urgency':>7s}")
        lines.append(f"    {'-'*58}")

        sorted_needs = sorted(
            self._needs.values(),
            key=lambda n: n.urgency,
            reverse=True,
        )

        for n in sorted_needs:
            vel_arrow = "+" if n.velocity > 0.05 else ("-" if n.velocity < -0.05 else "=")
            lines.append(
                f"    {n.need_type.value:12s} "
                f"{n.current:7.3f} {n.target:7.3f} "
                f"{n.deficit:7.3f} {vel_arrow}{abs(n.velocity):.3f}  "
                f"{n.urgency:7.3f}"
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self._tick,
            "needs": {nt.value: n.to_dict() for nt, n in self._needs.items()},
            "total_deficit": sum(n.deficit for n in self._needs.values()),
            "max_urgency": max((n.urgency for n in self._needs.values()), default=0.0),
        }
