"""Probabilistic Pillar State Vector — P6.10

Upgrade PSV from deterministic (16 floats) to probabilistic
(16 GaussianDims: mean + variance per pillar).

The current ecosystem uses deterministic PSV everywhere:
- Engine: float pillars[16]
- Substrate_Echo: np.ndarray of shape (16,)
- VNES-Lab: theta/phi angles

This module adds uncertainty. Each pillar is a GaussianDim
with mean ∈ [0,1] and variance ∈ (0,1]. Observations update
via Bayesian inference (same math as HSVState).

Why this matters:
1. Conflicting observations: agent shows high Willpower then
   low Willpower — variance increases, reflecting uncertainty
2. Consistent observations: variance decreases, confidence grows
3. Prediction: can predict range of future states, not just point
4. Risk assessment: high-variance pillars are unreliable

Usage:
    psv = ProbabilisticPSV()
    psv.update_from_observation({"Awareness": 0.8, "Force": 0.3})
    print(psv.confidence)  # 1 - mean variance
    
    # Convert to deterministic (for Engine)
    det = psv.to_deterministic()  # np.ndarray shape (16,)
    
    # Convert from deterministic (from Engine)
    psv2 = ProbabilisticPSV.from_deterministic(det)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from substrate_echo.core.human_state import GaussianDim


PILLAR_NAMES = [
    "Awareness", "Willpower", "Force", "Influence",
    "Resistance", "Integrity", "Cohesion", "Relation",
    "Presence", "Warmth", "Memory", "Attraction",
    "Harm", "Distortion", "Flux", "Depth",
]

PILLAR_NAME_TO_INDEX = {name: i for i, name in enumerate(PILLAR_NAMES)}

NUM_PILLARS = 16


@dataclass
class ProbabilisticPSV:
    """Probabilistic 16D Pillar State Vector.
    
    Each pillar is a GaussianDim with mean ∈ [0,1] and variance ∈ (0,1].
    Supports Bayesian updates, deterministic conversion, and
    uncertainty-aware operations.
    """
    
    pillars: list[GaussianDim] = None
    
    def __post_init__(self):
        if self.pillars is None:
            self.pillars = [GaussianDim(0.5, 0.25) for _ in range(NUM_PILLARS)]
    
    @property
    def means(self) -> np.ndarray:
        return np.array([p.mean for p in self.pillars])
    
    @property
    def variances(self) -> np.ndarray:
        return np.array([p.variance for p in self.pillars])
    
    @property
    def uncertainty(self) -> float:
        """Mean variance across all pillars."""
        return float(np.mean(self.variances))
    
    @property
    def confidence(self) -> float:
        """1 - uncertainty."""
        return 1.0 - self.uncertainty
    
    @property
    def dominant_pillar(self) -> int:
        return int(np.argmax(self.means))
    
    @property
    def weakest_pillar(self) -> int:
        return int(np.argmin(self.means))
    
    def to_deterministic(self) -> np.ndarray:
        """Convert to flat 16D array of means (for Engine compatibility)."""
        return self.means.copy()
    
    @classmethod
    def from_deterministic(cls, arr: np.ndarray,
                           initial_variance: float = 0.25) -> ProbabilisticPSV:
        """Create from deterministic 16D array."""
        arr = np.asarray(arr, dtype=np.float64)
        pillars = [GaussianDim(mean=float(arr[i]), variance=initial_variance)
                   for i in range(NUM_PILLARS)]
        return cls(pillars=pillars)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> ProbabilisticPSV:
        """Create from 32D array: [16 means, 16 variances]."""
        arr = np.asarray(arr, dtype=np.float64)
        pillars = [GaussianDim(mean=float(arr[i]),
                               variance=float(arr[i + NUM_PILLARS]))
                   for i in range(NUM_PILLARS)]
        return cls(pillars=pillars)
    
    def to_array(self) -> np.ndarray:
        """Serialize to 32D array: [16 means, 16 variances]."""
        return np.concatenate([self.means, self.variances])
    
    def update_from_observation(self, observations: dict[str, float],
                                 confidence: float = 0.5) -> None:
        """Update pillars from observed values.
        
        Args:
            observations: dict mapping pillar names to observed values (0-1)
            confidence: how much to trust the observation (0-1)
        """
        for name, value in observations.items():
            if name in PILLAR_NAME_TO_INDEX:
                idx = PILLAR_NAME_TO_INDEX[name]
                noise = 0.1 + (1.0 - confidence) * 0.4
                self.pillars[idx].update(value, noise)
    
    def update_from_array(self, observed: np.ndarray,
                          noise: float = 0.2) -> None:
        """Update all pillars from a deterministic observation vector."""
        observed = np.asarray(observed, dtype=np.float64)
        for i in range(min(NUM_PILLARS, len(observed))):
            self.pillars[i].update(float(observed[i]), noise)
    
    def similarity(self, other: ProbabilisticPSV) -> float:
        """Cosine similarity of means."""
        a, b = self.means, other.means
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    
    def distance(self, other: ProbabilisticPSV) -> float:
        """Euclidean distance between means."""
        return float(np.linalg.norm(self.means - other.means))
    
    def coherence(self) -> float:
        """How internally aligned the pillars are (0-1)."""
        return float(1.0 - np.std(self.means))
    
    def pillar_summary(self) -> dict[str, float]:
        """Named dictionary of pillar means."""
        return {PILLAR_NAMES[i]: round(self.pillars[i].mean, 4)
                for i in range(NUM_PILLARS)}
    
    def uncertain_pillars(self, threshold: float = 0.3) -> list[str]:
        """Pillars with variance above threshold (unreliable)."""
        return [PILLAR_NAMES[i] for i in range(NUM_PILLARS)
                if self.pillars[i].variance > threshold]
    
    def copy(self) -> ProbabilisticPSV:
        new_pillars = [GaussianDim(p.mean, p.variance) for p in self.pillars]
        return ProbabilisticPSV(pillars=new_pillars)
