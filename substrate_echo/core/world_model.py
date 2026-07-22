"""World Model — wraps DynamicsMemory to answer: "What happens if nothing intervenes?"

Provides passive prediction, structure discovery, and uncertainty estimation.
The world model is descriptive — it knows how the world evolves, not how to change it.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .dynamics_memory import DynamicsMemory


@dataclass
class WorldModelConfig:
    prediction_horizon: int = 50
    confidence_decay: float = 0.02  # per-step confidence decay for extrapolation


class WorldModel:
    """Passive world model built on learned dynamics.
    
    Answers questions like:
    - "Where will this state go?" (predict)
    - "What are the stable states?" (attractors)
    - "Which basin is this in?" (basin classification)
    - "How well do I know this region?" (coverage/confidence)
    """
    
    def __init__(self, dynamics_memory: DynamicsMemory,
                 config: Optional[WorldModelConfig] = None):
        self.memory = dynamics_memory
        self.config = config or WorldModelConfig()
    
    # ── Passive Prediction ──────────────────────────────────────
    
    def predict(self, state: np.ndarray, steps: int = None) -> np.ndarray:
        """Predict future state by integrating V(x) forward."""
        if steps is None:
            steps = self.config.prediction_horizon
        return self.memory.predict(state, steps=steps, dt=1.0)
    
    def predict_trajectory(self, state: np.ndarray,
                           steps: int = None) -> list[np.ndarray]:
        """Return full predicted trajectory (for visualization/evaluation)."""
        if steps is None:
            steps = self.config.prediction_horizon
        
        trajectory = [state.copy()]
        x = state.copy()
        for _ in range(steps):
            v = self.memory.predict_velocity(x)
            x = x + v  # dt=1.0 (velocities are per-encode-step)
            x = np.clip(x, 0.0, 1.0)
            trajectory.append(x.copy())
        return trajectory
    
    def predict_velocity(self, state: np.ndarray) -> np.ndarray:
        """Instantaneous velocity at a state."""
        return self.memory.predict_velocity(state)
    
    # ── Structure Discovery ─────────────────────────────────────
    
    def get_attractors(self) -> list[np.ndarray]:
        """Discovered stable states."""
        if not self.memory._attractors:
            self.memory.discover_attractors()
        return self.memory._attractors
    
    def get_basin(self, state: np.ndarray) -> int:
        """Which basin does this state belong to?"""
        return self.memory.basin_of(state)
    
    def get_stability(self, state: np.ndarray) -> dict:
        """Classify a state: attractor, repellor, saddle, marginal."""
        return self.memory.stability_at(state)
    
    def get_n_attractors(self) -> int:
        """Number of discovered attractors."""
        return len(self.get_attractors())
    
    # ── Transition Reasoning ────────────────────────────────────
    
    def transition_probability(self, from_basin: int,
                               steps: int = 100) -> dict[int, float]:
        """Probability of transitioning from one basin to another."""
        return self.memory.transition_probability(from_basin, steps)
    
    def would_transition(self, state: np.ndarray,
                         target_basin: int) -> float:
        """Probability that this state ends up in target_basin.
        
        Integrates forward and checks convergence.
        """
        x = state.copy()
        for _ in range(200):
            v = self.memory.predict_velocity(x)
            x = x + v
            x = np.clip(x, 0.0, 1.0)
        
        # Check which basin we ended up in
        attractors = self.get_attractors()
        if not attractors:
            return 0.0
        
        dists = [np.linalg.norm(x - a) for a in attractors]
        best = np.argmin(dists)
        
        if best == target_basin and dists[best] < 0.5:
            return 1.0
        return 0.0
    
    # ── Uncertainty ─────────────────────────────────────────────
    
    def prediction_confidence(self, state: np.ndarray) -> float:
        """How confident are we in predictions at this state?
        
        Based on:
        1. Distance to nearest training point (coverage)
        2. Stability of the local dynamics
        """
        if not self.memory._fitted:
            return 0.0
        
        # Check if state is within training data range
        if self.memory._states:
            states_arr = np.array(self.memory._states)
            dists = np.linalg.norm(states_arr - state, axis=1)
            min_dist = float(np.min(dists))
            # Confidence decays with distance from training data
            coverage = np.exp(-min_dist / 0.3)
        else:
            coverage = 0.0
        
        # Stability bonus (stable states are more predictable)
        stability = self.memory.stability_at(state)
        if stability['classification'] == 'attractor':
            stability_bonus = 0.3
        elif stability['classification'] == 'marginal':
            stability_bonus = 0.1
        else:
            stability_bonus = 0.0
        
        return min(1.0, coverage + stability_bonus)
    
    def coverage(self, state: np.ndarray) -> float:
        """How well-known is this region? [0, 1]"""
        if not self.memory._states:
            return 0.0
        
        states_arr = np.array(self.memory._states)
        dists = np.linalg.norm(states_arr - state, axis=1)
        min_dist = float(np.min(dists))
        return float(np.exp(-min_dist / 0.3))
    
    # ── Interface ───────────────────────────────────────────────
    
    def share_observations(self, other_world_model: 'WorldModel',
                           max_samples: int = 200) -> int:
        """Import observations from another agent's world model.

        This enables social learning: agents share what they've learned
        about the world. Only shares samples that are novel (far from
        existing training data) to avoid redundancy.

        Returns number of samples imported.
        """
        if not other_world_model.memory._states:
            return 0
        
        other_states = other_world_model.memory._states
        other_vels = other_world_model.memory._velocities
        
        imported = 0
        for i in range(min(len(other_states), max_samples)):
            state = other_states[i]
            vel = other_vels[i] if i < len(other_vels) else np.zeros_like(state)
            
            # Check if this sample is novel enough to import
            if self.memory._states:
                states_arr = np.array(self.memory._states)
                dists = np.linalg.norm(states_arr - state, axis=1)
                min_dist = float(np.min(dists))
                # Only import if far from existing data
                if min_dist < 0.1:
                    continue
            
            self.memory._states.append(state.copy())
            self.memory._velocities.append(vel.copy())
            imported += 1
        
        # Retrain if we imported enough new data
        if imported > 10:
            self.memory._fit_dynamics()
        
        return imported
    
    def stats(self) -> dict:
        return {
            "fitted": self.memory._fitted,
            "n_attractors": self.get_n_attractors(),
            "n_training_samples": len(self.memory._states),
            "attractors": [a.tolist() for a in self.get_attractors()],
        }
