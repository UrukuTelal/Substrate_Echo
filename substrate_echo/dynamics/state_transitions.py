"""State Transition Manager — unified event language for all subsystems.

Every change in Substrate_Echo — field evolution, agent actions, memory
formation, topology events — produces a StateTransition. This creates
a common currency between physics, cognition, and simulation.

The manager validates transitions against conservation laws, maintains
a history for analysis, and distributes events to interested subsystems.

Usage:
    manager = StateTransitionManager()
    
    # Record a transition
    t = StateTransition(
        source_state=current_psv,
        target_state=new_psv,
        cause=TransitionCause.FIELD_CHANGE,
        energy_cost=0.02,
    )
    result = manager.record(t)
    
    # Query history
    recent = manager.recent(causes=[TransitionCause.AGENT_ACTION])
    total_energy = manager.total_energy_cost(window=60.0)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional
import numpy as np
import time


class TransitionCause(Enum):
    """What triggered the state change."""
    FIELD_CHANGE = auto()       # physics: ontological field evolved
    AGENT_ACTION = auto()       # cognition: agent decided and acted
    TOPOLOGY_EVENT = auto()     # physics: vacuum tunneling, foam node
    MEMORY_UPDATE = auto()      # cognition: attractor formed/merged/decayed
    SENSOR_INPUT = auto()       # perception: new sensory data arrived
    EXTERNAL_FORCING = auto()   # environment: outside influence applied
    CONSERVATION_CORRECTION = auto()  # physics: invariant enforcement


class TransitionStatus(Enum):
    """Was the transition accepted or rejected."""
    ACCEPTED = auto()
    REJECTED_ENERGY = auto()    # energy increased beyond tolerance
    REJECTED_CONSERVATION = auto()  # violated conservation law
    REJECTED_BOUNDS = auto()    # state out of valid range
    CORRECTED = auto()          # accepted but corrected to preserve invariants


@dataclass
class StateTransition:
    """A single state change event.
    
    This is the atomic unit of change in Substrate_Echo. Every subsystem
    that modifies state produces one of these.
    """
    source_state: np.ndarray          # 16D state before change
    target_state: np.ndarray          # 16D state after change
    cause: TransitionCause            # what triggered this
    
    # Optional metadata
    energy_cost: float = 0.0         # energy consumed (positive = cost)
    information_delta: float = 0.0   # change in information content
    agent_id: Optional[str] = None   # which agent caused it
    description: str = ""            # human-readable summary
    metadata: dict = field(default_factory=dict)
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    # Computed fields (set by manager)
    status: TransitionStatus = TransitionStatus.ACCEPTED
    corrected_state: Optional[np.ndarray] = None  # if conservation corrected
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class TransitionConstraint:
    """A conservation law or bound that transitions must satisfy."""
    name: str
    description: str
    check_fn: Callable[[StateTransition], tuple[bool, str]]
    tolerance: float = 1e-6
    correctable: bool = False  # can the manager auto-fix violations?
    correct_fn: Optional[Callable[[StateTransition], np.ndarray]] = None


@dataclass
class TransitionCallback:
    """Registered listener for state transitions."""
    name: str
    filter_causes: Optional[list[TransitionCause]] = None  # None = all
    callback: Callable[[StateTransition], None] = None


class StateTransitionManager:
    """Central hub for state change events.
    
    Responsibilities:
    1. Validate transitions against constraints
    2. Auto-correct if possible (conservation enforcement)
    3. Maintain history for analysis
    4. Distribute events to registered listeners
    5. Compute aggregate statistics
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.history: list[StateTransition] = []
        self.constraints: list[TransitionConstraint] = []
        self.callbacks: list[TransitionCallback] = []
        
        # Statistics
        self._total_transitions = 0
        self._total_accepted = 0
        self._total_rejected = 0
        self._total_corrected = 0
        self._total_energy_cost = 0.0
        self._total_information_delta = 0.0
        
        # Per-cause counters
        self._cause_counts: dict[TransitionCause, int] = {
            cause: 0 for cause in TransitionCause
        }
    
    def record(self, transition: StateTransition) -> StateTransition:
        """Record a state transition, validate, and distribute.
        
        Returns the transition with status set.
        """
        self._total_transitions += 1
        self._cause_counts[transition.cause] += 1
        
        # Validate against constraints
        for constraint in self.constraints:
            passed, error_msg = constraint.check_fn(transition)
            if not passed:
                transition.validation_errors.append(
                    f"{constraint.name}: {error_msg}"
                )
                
                # Try to correct
                if constraint.correctable and constraint.correct_fn:
                    corrected = constraint.correct_fn(transition)
                    transition.corrected_state = corrected
                    transition.status = TransitionStatus.CORRECTED
                    transition.target_state = corrected
                else:
                    transition.status = TransitionStatus.REJECTED_ENERGY
                    self._total_rejected += 1
                    self._notify_callbacks(transition)
                    return transition
        
        # Check bounds
        if np.any(transition.target_state < 0) or np.any(transition.target_state > 1):
            transition.status = TransitionStatus.REJECTED_BOUNDS
            transition.validation_errors.append("State out of [0, 1] bounds")
            self._total_rejected += 1
            self._notify_callbacks(transition)
            return transition
        
        # Accept
        if transition.status != TransitionStatus.CORRECTED:
            transition.status = TransitionStatus.ACCEPTED
        
        self._total_accepted += 1
        self._total_energy_cost += transition.energy_cost
        self._total_information_delta += transition.information_delta
        
        # Store in history
        self.history.append(transition)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Notify listeners
        self._notify_callbacks(transition)
        
        return transition
    
    def add_constraint(self, constraint: TransitionConstraint) -> None:
        """Add a conservation law or bound."""
        self.constraints.append(constraint)
    
    def add_callback(self, callback: TransitionCallback) -> None:
        """Register a listener for transitions."""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, transition: StateTransition) -> None:
        """Distribute transition to registered listeners."""
        for cb in self.callbacks:
            if cb.filter_causes is None or transition.cause in cb.filter_causes:
                if cb.callback is not None:
                    cb.callback(transition)
    
    # ── Query Methods ──────────────────────────────────────────────
    
    def recent(self, n: int = 10, 
               causes: Optional[list[TransitionCause]] = None) -> list[StateTransition]:
        """Get recent transitions, optionally filtered by cause."""
        if causes is None:
            return list(self.history[-n:])
        
        filtered = [t for t in self.history if t.cause in causes]
        return filtered[-n:]
    
    def total_energy_cost(self, window_seconds: Optional[float] = None) -> float:
        """Total energy cost, optionally within a time window."""
        if window_seconds is None:
            return self._total_energy_cost
        
        cutoff = time.time() - window_seconds
        return sum(
            t.energy_cost for t in self.history
            if t.timestamp > cutoff
        )
    
    def transitions_by_cause(self, cause: TransitionCause,
                             window_seconds: Optional[float] = None) -> list[StateTransition]:
        """Get all transitions of a given cause, optionally within window."""
        if window_seconds is None:
            return [t for t in self.history if t.cause == cause]
        
        cutoff = time.time() - window_seconds
        return [
            t for t in self.history
            if t.cause == cause and t.timestamp > cutoff
        ]
    
    def state_trajectory(self, n: int = 100) -> list[np.ndarray]:
        """Get the sequence of states from recent transitions."""
        return [t.target_state for t in self.history[-n:]]
    
    def information_rate(self, window_seconds: float = 60.0) -> float:
        """Information flow rate (delta per second) in recent window."""
        cutoff = time.time() - window_seconds
        recent = [t for t in self.history if t.timestamp > cutoff]
        
        if not recent:
            return 0.0
        
        total_delta = sum(abs(t.information_delta) for t in recent)
        elapsed = window_seconds if recent else 1.0
        
        return total_delta / elapsed
    
    # ── Statistics ─────────────────────────────────────────────────
    
    def stats(self) -> dict:
        """Get aggregate statistics."""
        return {
            "total_transitions": self._total_transitions,
            "accepted": self._total_accepted,
            "rejected": self._total_rejected,
            "corrected": self._total_corrected,
            "acceptance_rate": (
                self._total_accepted / max(1, self._total_transitions)
            ),
            "total_energy_cost": self._total_energy_cost,
            "total_information_delta": self._total_information_delta,
            "by_cause": {
                cause.name: count
                for cause, count in self._cause_counts.items()
                if count > 0
            },
            "history_size": len(self.history),
            "constraints_active": len(self.constraints),
            "callbacks_registered": len(self.callbacks),
        }
    
    def reset(self) -> None:
        """Clear all history and statistics."""
        self.history.clear()
        self._total_transitions = 0
        self._total_accepted = 0
        self._total_rejected = 0
        self._total_corrected = 0
        self._total_energy_cost = 0.0
        self._total_information_delta = 0.0
        for cause in self._cause_counts:
            self._cause_counts[cause] = 0
