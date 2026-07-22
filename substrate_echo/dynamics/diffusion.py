"""Diffusion Tensor — inter-pillar coupling matrix.

The diffusion tensor D is a 16×16 matrix that controls how pillar states
influence each other over time. Off-diagonal entries represent coupling
strengths between pillars.

Default: identity (pillars independent)
Learned: correlations between pillars strengthen coupling
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DiffusionConfig:
    """Configuration for diffusion tensor."""
    base_diffusion: float = 0.01  # default diffusion rate
    learning_rate: float = 0.001  # how fast correlations are learned
    decay_rate: float = 0.999  # how fast unused couplings decay
    min_value: float = 0.0  # minimum coupling
    max_value: float = 1.0  # maximum coupling
    diagonal_boost: float = 2.0  # self-coupling multiplier


class DiffusionTensor:
    """16×16 inter-pillar coupling matrix.
    
    Controls how pillar states influence each other:
    - Diagonal: self-coupling (pillar's own dynamics)
    - Off-diagonal: cross-coupling (how pillars affect each other)
    
    Usage:
        diff = DiffusionTensor()
        diff.update_from_correlation(observed_correlations)
        new_state = diff.apply(state)
    """
    
    PILLAR_NAMES = [
        "Awareness", "Willpower", "Force", "Influence",
        "Resistance", "Integrity", "Cohesion", "Relation",
        "Presence", "Warmth", "Memory", "Attraction",
        "Harm", "Distortion", "Flux", "Depth",
    ]
    
    def __init__(self, config: Optional[DiffusionConfig] = None):
        self.config = config or DiffusionConfig()
        self.tensor = np.eye(16, dtype=np.float64) * self.config.base_diffusion
        self._correlation_buffer: list[np.ndarray] = []
        self._update_count = 0
    
    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply diffusion to state vector.
        
        Returns D·(0 - state) as the diffusion force.
        """
        return self.tensor @ (np.zeros_like(state) - state) * 0.1
    
    def update_from_observation(self, state_delta: np.ndarray) -> None:
        """Update tensor based on observed state changes.
        
        If pillar i and pillar j change together, strengthen their coupling.
        """
        self._correlation_buffer.append(state_delta.copy())
        
        # Keep buffer bounded
        if len(self._correlation_buffer) > 100:
            self._correlation_buffer.pop(0)
        
        # Update every 10 observations
        if len(self._correlation_buffer) >= 10:
            self._update_from_buffer()
    
    def _update_from_buffer(self) -> None:
        """Compute correlations and update tensor."""
        if not self._correlation_buffer:
            return
        
        data = np.array(self._correlation_buffer)
        n_obs = len(data)
        
        if n_obs < 2:
            return
        
        # Compute correlation matrix
        mean = np.mean(data, axis=0)
        centered = data - mean
        cov = (centered.T @ centered) / (n_obs - 1)
        
        # Normalize to [0, 1]
        std = np.sqrt(np.diag(cov))
        std[std < 1e-10] = 1.0  # avoid division by zero
        corr = cov / np.outer(std, std)
        corr = (corr + 1) / 2  # map [-1, 1] → [0, 1]
        
        # Update tensor with exponential moving average
        lr = self.config.learning_rate
        self.tensor = (1 - lr) * self.tensor + lr * corr * self.config.base_diffusion
        
        # Apply constraints
        np.fill_diagonal(self.tensor, np.diag(self.tensor) * self.config.diagonal_boost)
        self.tensor = np.clip(self.tensor, self.config.min_value, self.config.max_value)
        
        # Decay unused couplings
        self.tensor *= self.config.decay_rate
        np.fill_diagonal(self.tensor, np.diag(self.tensor) / self.config.decay_rate)
        
        self._update_count += 1
        self._correlation_buffer.clear()
    
    def get_coupling_strength(self, pillar_i: int, pillar_j: int) -> float:
        """Get coupling strength between two pillars."""
        return float(self.tensor[pillar_i, pillar_j])
    
    def get_strongest_couplings(self, top_k: int = 5) -> list[tuple[str, str, float]]:
        """Get the strongest cross-couplings."""
        # Mask diagonal
        masked = self.tensor.copy()
        np.fill_diagonal(masked, 0)
        
        # Find top-k
        indices = np.unravel_index(
            np.argsort(masked.ravel())[-top_k:],
            masked.shape
        )
        
        result = []
        for i, j in zip(indices[0], indices[1]):
            result.append((
                self.PILLAR_NAMES[i],
                self.PILLAR_NAMES[j],
                float(masked[i, j])
            ))
        
        return list(reversed(result))
    
    def get_pillar_influence(self, pillar: int) -> dict[str, float]:
        """Get how much each other pillar influences this one."""
        return {
            self.PILLAR_NAMES[j]: float(self.tensor[j, pillar])
            for j in range(16)
        }
    
    def set_manual_coupling(self, pillar_i: int, pillar_j: int, strength: float) -> None:
        """Manually set coupling between two pillars."""
        self.tensor[pillar_i, pillar_j] = strength
        self.tensor[pillar_j, pillar_i] = strength  # symmetric
    
    def reset(self) -> None:
        """Reset to identity (pillars independent)."""
        self.tensor = np.eye(16, dtype=np.float64) * self.config.base_diffusion
        self._correlation_buffer.clear()
        self._update_count = 0
