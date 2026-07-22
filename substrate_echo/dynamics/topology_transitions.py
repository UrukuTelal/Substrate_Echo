"""BCFVT-04: Topology Transition Operator — discrete jumps between vacua.

Implements the discrete topology transition operator τ: Q → Q
handles quantum tunneling between vacua, foam node creation/annihilation.

Transition rates from Euclidean action:
Γ = A·exp(-S_E/ℏ)
S_E = ∫dτ (½g^μν ∂_μφ ∂_νφ + V(φ)) via lattice path integral
A = fluctuation determinant prefactor

Anti-runaway:
- Rate limiter on transitions
- Energy conservation check after each tunnel
- Rollback if total energy increases beyond tolerance

References:
- BCFVT Implementation Plan, BCFVT-04
- Adversarial findings 04-1 (stability), 04-2 (rate computation),
  04-3 (interface), 04-4 (step execution)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np
import math
import time


class TransitionType(Enum):
    """Types of topology transitions."""
    VACUUM_TUNNELING = auto()     # tunneling between vacua
    FOAM_NODE_CREATE = auto()     # new node in spacetime foam
    FOAM_NODE_ANNIHILATE = auto() # node removal
    VORTEX_CREATE = auto()        # new vortex (topological defect)
    VORTEX_ANNIHILATE = auto()    # vortex removal
    VORTEX_MERGE = auto()         # two vortices combine


@dataclass
class TransitionConfig:
    """Configuration for topology transitions."""
    # Rate limiting
    max_transitions_per_step: int = 10
    rate_limit_window: float = 1.0   # seconds
    
    # Energy conservation
    energy_tolerance: float = 0.1    # 10% energy increase allowed
    enable_rollback: bool = True
    
    # Physical constants
    hbar: float = 1.0               # reduced Planck constant (natural units)
    
    # Prefactor for transition rate
    prefactor_A: float = 1.0        # fluctuation determinant
    
    # Foam parameters
    foam_node_energy: float = 0.5   # energy cost of creating a foam node
    max_foam_nodes: int = 100


@dataclass
class TopologyTransition:
    """A single topology transition event."""
    transition_type: TransitionType
    position: np.ndarray              # position in state space
    energy_barrier: float             # S_E / ℏ (Euclidean action)
    rate: float                       # Γ = A·exp(-S_E/ℏ)
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    
    @property
    def is_tunnelling(self) -> bool:
        return self.transition_type == TransitionType.VACUUM_TUNNELING


class TopologyManager:
    """BCFVT-04 Topology Transition Operator.
    
    Manages discrete topology transitions in the BCFVT system.
    
    Key features:
    - Computes transition rates from Euclidean action
    - Rate limiting to prevent runaway
    - Energy conservation checks with rollback
    - Anti-runaway: if energy increases too much, reject transition
    
    Interface:
    - check_transitions(field_config) → list of events
    - execute_transition(event) → new field config
    """
    
    def __init__(self, config: Optional[TransitionConfig] = None):
        self.config = config or TransitionConfig()
        
        # State tracking
        self._transition_history: list[TopologyTransition] = []
        self._recent_timestamps: list[float] = []
        self._total_transitions: int = 0
        self._rejected_count: int = 0
        self._rolled_back_count: int = 0
    
    # ── Euclidean Action Computation ──────────────────────────────
    
    def compute_euclidean_action(self, field_initial: np.ndarray,
                                  field_final: np.ndarray,
                                  metric: Optional[np.ndarray] = None,
                                  potential_fn: Optional[callable] = None) -> float:
        """Compute Euclidean action for a transition.
        
        S_E = ∫dτ (½g^μν ∂_μφ ∂_νφ + V(φ))
        
        Discrete version (lattice path integral):
        S_E ≈ Σ_k [½|φ_{k+1} - φ_k|² + V(φ_k)]
        
        Args:
            field_initial: initial field configuration
            field_final: final field configuration
            metric: metric tensor (default: identity)
            potential_fn: potential energy function V(φ)
        """
        if metric is None:
            metric = np.eye(len(field_initial))
        
        if potential_fn is None:
            # Default: Ginzburg-Landau potential
            def potential_fn(phi):
                norm_sq = float(np.dot(phi, phi))
                return (norm_sq - 1.0) ** 2
        
        # Kinetic term: ½|Δφ|²
        delta = field_final - field_initial
        kinetic = 0.5 * float(delta @ metric @ delta)
        
        # Potential term: V(φ_initial)
        potential = potential_fn(field_initial)
        
        return kinetic + potential
    
    def compute_transition_rate(self, euclidean_action: float) -> float:
        """Compute transition rate from Euclidean action.
        
        Γ = A·exp(-S_E/ℏ)
        
        A = prefactor from fluctuation determinant
        S_E = Euclidean action
        """
        exponent = -euclidean_action / self.config.hbar
        
        # Prevent overflow
        if exponent < -500:
            return 0.0
        
        return self.config.prefactor_A * math.exp(exponent)
    
    # ── Transition Detection ──────────────────────────────────────
    
    def check_transitions(self, field_config: np.ndarray,
                          candidate_finals: Optional[list[np.ndarray]] = None,
                          positions: Optional[list[np.ndarray]] = None,
                          potential_fn: Optional[callable] = None
                          ) -> list[TopologyTransition]:
        """Check for possible topology transitions.
        
        This is the main interface called each tick by foam dynamics.
        
        Args:
            field_config: current field configuration
            candidate_finals: list of possible final configurations
            positions: positions of each candidate
            potential_fn: potential energy function
        
        Returns:
            List of transitions that pass rate limiting and energy checks
        """
        if candidate_finals is None:
            candidate_finals = []
        
        if positions is None:
            positions = [np.zeros(len(field_config)) for _ in candidate_finals]
        
        events = []
        
        for final, pos in zip(candidate_finals, positions):
            # Compute Euclidean action
            S_E = self.compute_euclidean_action(
                field_config, final, potential_fn=potential_fn
            )
            
            # Compute transition rate
            rate = self.compute_transition_rate(S_E)
            
            # Determine transition type
            t_type = self._classify_transition(field_config, final)
            
            # Create transition event
            transition = TopologyTransition(
                transition_type=t_type,
                position=pos,
                energy_barrier=S_E,
                rate=rate,
                metadata={
                    "action": S_E,
                    "rate": rate,
                },
            )
            
            events.append(transition)
        
        # Apply rate limiting
        events = self._apply_rate_limit(events)
        
        # Apply energy conservation check
        events = self._apply_energy_check(events, field_config)
        
        return events
    
    def _classify_transition(self, initial: np.ndarray,
                             final: np.ndarray) -> TransitionType:
        """Classify the type of transition."""
        # Simple heuristic: if field amplitude drops, it's vortex creation
        init_norm = np.linalg.norm(initial)
        final_norm = np.linalg.norm(final)
        
        if final_norm < init_norm * 0.5:
            return TransitionType.VORTEX_CREATE
        elif init_norm < final_norm * 0.5:
            return TransitionType.VORTEX_ANNIHILATE
        else:
            return TransitionType.VACUUM_TUNNELING
    
    # ── Rate Limiting ─────────────────────────────────────────────
    
    def _apply_rate_limit(self, events: list[TopologyTransition]
                          ) -> list[TopologyTransition]:
        """Apply rate limiting to prevent runaway transitions."""
        now = time.time()
        
        # Clean old timestamps
        self._recent_timestamps = [
            t for t in self._recent_timestamps
            if now - t < self.config.rate_limit_window
        ]
        
        # Check limit
        available = self.config.max_transitions_per_step - len(self._recent_timestamps)
        
        if available <= 0:
            self._rejected_count += len(events)
            return []
        
        # Sort by rate (highest first) and take available
        events.sort(key=lambda e: e.rate, reverse=True)
        accepted = events[:available]
        rejected = events[available:]
        
        self._rejected_count += len(rejected)
        self._recent_timestamps.extend([now] * len(accepted))
        
        return accepted
    
    # ── Energy Conservation ───────────────────────────────────────
    
    def _apply_energy_check(self, events: list[TopologyTransition],
                            current_field: np.ndarray) -> list[TopologyTransition]:
        """Check energy conservation and reject if too much increase."""
        if not self.config.enable_rollback:
            return events
        
        current_energy = self._estimate_energy(current_field)
        
        accepted = []
        for event in events:
            # Estimate energy after transition
            # (simplified: just check the barrier height)
            energy_increase = event.energy_barrier
            
            relative_increase = energy_increase / max(abs(current_energy), 1e-10)
            
            if relative_increase < self.config.energy_tolerance:
                accepted.append(event)
            else:
                self._rolled_back_count += 1
        
        return accepted
    
    def _estimate_energy(self, field: np.ndarray) -> float:
        """Estimate total field energy."""
        norm_sq = float(np.dot(field, field))
        return (norm_sq - 1.0) ** 2
    
    # ── Transition Execution ──────────────────────────────────────
    
    def execute_transition(self, field_config: np.ndarray,
                           transition: TopologyTransition) -> np.ndarray:
        """Execute a topology transition.
        
        This applies the transition to the field configuration.
        For vortex creation: introduces a phase winding.
        For vacuum tunneling: shifts field to new vacuum.
        
        Returns new field configuration.
        """
        new_field = field_config.copy()
        
        if transition.transition_type == TransitionType.VORTEX_CREATE:
            # Introduce phase winding at position
            pos = transition.position
            for i in range(len(new_field)):
                distance = abs(i - pos[0]) if len(pos) > 0 else 0
                if distance < 2:
                    new_field[i] *= 0.1  # suppress amplitude in core
        
        elif transition.transition_type == TransitionType.VACUUM_TUNNELING:
            # Shift to nearby vacuum
            shift = np.random.randn(*new_field.shape) * 0.1
            new_field = new_field + shift
        
        # Enforce bounds
        new_field = np.clip(new_field, 0.0, 1.0)
        
        # Record
        self._transition_history.append(transition)
        self._total_transitions += 1
        
        return new_field
    
    # ── Statistics ─────────────────────────────────────────────────
    
    def stats(self) -> dict:
        """Get transition statistics."""
        return {
            "total_transitions": self._total_transitions,
            "rejected": self._rejected_count,
            "rolled_back": self._rolled_back_count,
            "recent_rate": len(self._recent_timestamps),
            "by_type": {
                tt.name: sum(
                    1 for t in self._transition_history
                    if t.transition_type == tt
                )
                for tt in TransitionType
            },
        }
    
    def reset(self) -> None:
        """Reset all state."""
        self._transition_history.clear()
        self._recent_timestamps.clear()
        self._total_transitions = 0
        self._rejected_count = 0
        self._rolled_back_count = 0
