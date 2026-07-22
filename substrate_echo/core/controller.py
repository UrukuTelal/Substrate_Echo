"""Controller — computes feasible ΔPSV from desired targets.

The controller translates planner intent into physically realizable
state changes, subject to conservation, topology, and actuator constraints.
This is the bridge between "where I want to go" and "how I can get there."
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ControlConfig:
    """Configuration for the controller."""
    gain: float = 0.3              # proportional gain
    max_delta: float = 0.2         # max |ΔPSV| per step
    min_delta: float = 1e-4        # ignore tiny corrections
    energy_budget: float = 0.1     # max energy per action
    conservation_enforce: bool = True
    norm_target: float = 1.0       # target norm for conservation
    damping: float = 0.0           # velocity damping (0 = no damping)


class Controller:
    """Computes feasible ΔPSV from desired targets.
    
    Uses proportional control with constraint enforcement:
    
        Δ = K * (target - current)
        Δ = clamp(Δ, max_magnitude)
        Δ = project(Δ, conservation_manifold)
    
    The controller doesn't know about dynamics — it only knows
    "how far do I need to go" and "what are the constraints."
    """
    
    def __init__(self, config: Optional[ControlConfig] = None):
        self.config = config or ControlConfig()
    
    def compute_control(self, current: np.ndarray,
                        target: np.ndarray) -> 'ControlOutput':
        """Compute the control action to move from current toward target.
        
        Returns a ControlOutput with the feasible delta and diagnostics.
        """
        current = np.asarray(current, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        
        error = target - current
        error_mag = np.linalg.norm(error)
        
        if error_mag < self.config.min_delta:
            return ControlOutput(
                delta=np.zeros_like(current),
                error_magnitude=error_mag,
                constrained=False,
                reason="noop",
            )
        
        # Proportional control
        delta = self.config.gain * error
        
        # Clamp magnitude
        delta_mag = np.linalg.norm(delta)
        constrained = False
        reason = "none"
        
        if delta_mag > self.config.max_delta:
            delta = delta * (self.config.max_delta / delta_mag)
            constrained = True
            reason = "magnitude_clamp"
        
        # Energy budget
        energy = float(np.sum(delta ** 2))
        if energy > self.config.energy_budget:
            scale = np.sqrt(self.config.energy_budget / energy)
            delta = delta * scale
            constrained = True
            reason = "energy_budget"
        
        # Conservation enforcement
        if self.config.conservation_enforce:
            delta = self._enforce_conservation(delta, current)
        
        # Damping
        if self.config.damping > 0:
            delta = delta * (1.0 - self.config.damping)
        
        return ControlOutput(
            delta=delta,
            error_magnitude=error_mag,
            final_magnitude=float(np.linalg.norm(delta)),
            constrained=constrained,
            reason=reason,
        )
    
    def compute_trajectory_control(self, current: np.ndarray,
                                    target_trajectory: list[np.ndarray]
                                    ) -> list['ControlOutput']:
        """Compute a sequence of controls to follow a target trajectory."""
        controls = []
        state = current.copy()
        
        for target in target_trajectory:
            output = self.compute_control(state, target)
            controls.append(output)
            state = state + output.delta
            state = np.clip(state, 0.0, 1.0)
        
        return controls
    
    def _enforce_conservation(self, delta: np.ndarray,
                              current: np.ndarray) -> np.ndarray:
        """Project delta onto the norm-conservation manifold.
        
        Ensures that applying delta doesn't violate ||psi|| ≈ norm_target.
        Uses a simple projection: subtract the component along the current state.
        """
        current_norm = np.linalg.norm(current)
        if current_norm < 1e-10:
            return delta
        
        # Component of delta along current state direction
        direction = current / current_norm
        radial_component = np.dot(delta, direction)
        
        # Only enforce if moving outward would increase norm too much
        projected_norm = np.linalg.norm(current + delta)
        if projected_norm > self.config.norm_target * 1.1:
            # Remove radial component that increases norm
            delta = delta - radial_component * direction
        elif projected_norm < self.config.norm_target * 0.9:
            # Add radial component to increase norm
            needed = self.config.norm_target - projected_norm
            delta = delta + needed * direction * 0.5
        
        return delta


@dataclass
class ControlOutput:
    """Result of a control computation."""
    delta: np.ndarray
    error_magnitude: float = 0.0
    final_magnitude: float = 0.0
    constrained: bool = False
    reason: str = "none"
