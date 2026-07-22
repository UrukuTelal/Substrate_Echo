"""Simulator — models interventions: (state + action) → predicted future.

Unlike WorldModel (passive evolution), Simulator applies control inputs
and predicts the resulting trajectory. This is the core of model-based planning.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto
import numpy as np

from .world_model import WorldModel


@dataclass
class SimConfig:
    """Configuration for simulation."""
    prediction_horizon: int = 50
    dt: float = 1.0          # per-step (matches DynamicsMemory velocity scale)
    action_application: str = "impulse"  # "impulse" (one-shot) or "sustained" (continuous)


@dataclass
class ActionDelta:
    """A PSV-level action: additive perturbation to pillar state.
    
    Can originate from a high-level ActionType or be generated directly.
    """
    delta: np.ndarray                # 16D perturbation vector
    source_action: Optional[str] = None  # high-level action name (e.g., "DEFEND")
    description: str = ""
    magnitude: float = 0.0
    
    def __post_init__(self):
        self.delta = np.asarray(self.delta, dtype=np.float64)
        if self.magnitude == 0.0:
            self.magnitude = float(np.linalg.norm(self.delta))
    
    @staticmethod
    def pillar_boost(pillar_idx: int, amount: float = 0.1,
                     dim: int = 16) -> 'ActionDelta':
        """Create an action that boosts a specific pillar."""
        delta = np.zeros(dim)
        delta[pillar_idx] = amount
        return ActionDelta(delta=delta, description=f"boost_pillar_{pillar_idx}")
    
    @staticmethod
    def pillar_suppress(pillar_idx: int, amount: float = 0.1,
                        dim: int = 16) -> 'ActionDelta':
        """Create an action that suppresses a specific pillar."""
        delta = np.zeros(dim)
        delta[pillar_idx] = -amount
        return ActionDelta(delta=delta, description=f"suppress_pillar_{pillar_idx}")
    
    @staticmethod
    def random(dim: int = 16, magnitude: float = 0.1,
               rng: Optional[np.random.RandomState] = None) -> 'ActionDelta':
        """Create a random action with given magnitude."""
        if rng is None:
            rng = np.random.RandomState()
        direction = rng.randn(dim)
        direction /= np.linalg.norm(direction) + 1e-10
        delta = direction * magnitude
        return ActionDelta(delta=delta, description="random")
    
    @staticmethod
    def toward_target(target: np.ndarray, current: np.ndarray,
                      max_magnitude: float = 0.2) -> 'ActionDelta':
        """Create an action that moves toward a target state."""
        error = target - current
        mag = np.linalg.norm(error)
        if mag < 1e-10:
            return ActionDelta(delta=np.zeros_like(target), description="noop")
        if mag > max_magnitude:
            error = error * (max_magnitude / mag)
        return ActionDelta(delta=error, description="toward_target")


@dataclass
class SimResult:
    """Result of a simulation."""
    initial_state: np.ndarray
    final_state: np.ndarray
    trajectory: list[np.ndarray]
    basin_transitions: list[int]
    stability: str
    confidence: float
    
    @property
    def total_basin_changes(self) -> int:
        """Number of basin transitions during simulation."""
        if len(self.basin_transitions) < 2:
            return 0
        return sum(1 for a, b in zip(self.basin_transitions, self.basin_transitions[1:]) if a != b)


class Simulator:
    """Simulates (state + action) → predicted future.
    
    The simulator applies an ActionDelta to the current state,
    then integrates the learned dynamics forward. This produces
    a predicted trajectory that the Planner can evaluate.
    """
    
    def __init__(self, world_model: WorldModel,
                 config: Optional[SimConfig] = None):
        self.world_model = world_model
        self.config = config or SimConfig()
    
    def simulate(self, state: np.ndarray, action: ActionDelta,
                 steps: int = None, diagnostics: bool = True) -> SimResult:
        """Simulate one action applied to the current state.
        
        1. Apply action: state' = state + action.delta
        2. Integrate V(x) forward for `steps` steps
        3. Track basin transitions and final stability
        
        diagnostics=False skips stability/confidence analysis (expensive for local model).
        """
        if steps is None:
            steps = self.config.prediction_horizon
        
        state = np.asarray(state, dtype=np.float64)
        
        # Apply action
        x = state + action.delta
        x = np.clip(x, 0.0, 1.0)
        
        # Integrate forward
        trajectory = [x.copy()]
        basin_transitions = []
        
        # Initial basin
        initial_basin = self.world_model.get_basin(x)
        basin_transitions.append(initial_basin)
        
        for _ in range(steps):
            v = self.world_model.predict_velocity(x)
            x = x + v * self.config.dt
            x = np.clip(x, 0.0, 1.0)
            trajectory.append(x.copy())
            
            # Track basin (check every 5 steps for efficiency)
            if len(trajectory) % 5 == 0:
                basin = self.world_model.get_basin(x)
                basin_transitions.append(basin)
        
        # Final state analysis
        final_basin = self.world_model.get_basin(x)
        if basin_transitions[-1] != final_basin:
            basin_transitions.append(final_basin)
        
        if diagnostics:
            stability = self.world_model.get_stability(x)
            confidence = self.world_model.prediction_confidence(x)
        else:
            stability = {'classification': 'unknown'}
            confidence = 0.5
        
        return SimResult(
            initial_state=state,
            final_state=x,
            trajectory=trajectory,
            basin_transitions=basin_transitions,
            stability=stability['classification'],
            confidence=confidence,
        )
    
    def simulate_batch(self, state: np.ndarray,
                       actions: list[ActionDelta]) -> list[SimResult]:
        """Simulate multiple candidate actions from the same state."""
        return [self.simulate(state, action) for action in actions]
    
    def simulate_controlled(self, state: np.ndarray,
                            target_psv: np.ndarray,
                            gain: float = 0.3,
                            steps: int = None) -> SimResult:
        """Simulate with proportional control toward a target.
        
        Each step: delta = gain * (target - current)
        """
        if steps is None:
            steps = self.config.prediction_horizon
        
        state = np.asarray(state, dtype=np.float64)
        target_psv = np.asarray(target_psv, dtype=np.float64)
        
        x = state.copy()
        trajectory = [x.copy()]
        basin_transitions = [self.world_model.get_basin(x)]
        
        for _ in range(steps):
            # Control: move toward target
            error = target_psv - x
            control = gain * error
            control_mag = np.linalg.norm(control)
            if control_mag > 0.2:
                control = control * (0.2 / control_mag)
            
            # Natural dynamics + control
            v = self.world_model.predict_velocity(x)
            x = x + v + control
            x = np.clip(x, 0.0, 1.0)
            trajectory.append(x.copy())
            
            if len(trajectory) % 5 == 0:
                basin_transitions.append(self.world_model.get_basin(x))
        
        final_basin = self.world_model.get_basin(x)
        if basin_transitions[-1] != final_basin:
            basin_transitions.append(final_basin)
        
        stability = self.world_model.get_stability(x)
        confidence = self.world_model.prediction_confidence(x)
        
        return SimResult(
            initial_state=state,
            final_state=x,
            trajectory=trajectory,
            basin_transitions=basin_transitions,
            stability=stability['classification'],
            confidence=confidence,
        )
