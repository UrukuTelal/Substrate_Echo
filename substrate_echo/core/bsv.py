"""Biological State Vector (BSV) — P6.6

Standardized biological state representation bridging Engine's
BioState/CellState and Substrate_Echo's HSVState.

The BSV is a probabilistic vector capturing the biological
dimensions shared across all living systems: metabolism,
integrity, stress, energy, health, and reproduction readiness.

Each dimension is a GaussianDim (mean + variance), enabling
Kalman-style Bayesian updates from observations.

This standardizes the scattered biological state across:
- Engine BioState: food_level, offspring_count, has_shelter
- Engine BiologicalProjection: metabolic_rate, membrane_integrity, stress_level
- Substrate_Echo HSVState: arousal, fatigue (human-specific)

The BSV is organism-agnostic — it captures what ALL living
systems share, not what makes species specific.

Usage:
    bsv = BiologicalStateVector()
    bsv.update_from_observation({
        "metabolic_rate": 0.8,
        "stress_level": 0.3,
        "energy_level": 0.6,
    })
    
    print(bsv.health)  # composite health score
    print(bsv.to_array())  # 14D [means; variances]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from substrate_echo.core.human_state import GaussianDim


# Biological dimensions shared across all living systems
BSV_DIMENSIONS = [
    "metabolic_rate",       # energy processing speed (0=starving, 1=peak)
    "membrane_integrity",   # structural wholeness (0=damaged, 1=intact)
    "stress_level",         # current stress (0=calm, 1=crisis)
    "energy_level",         # stored energy (0=depleted, 1=full)
    "health",               # overall viability (0=dying, 1=thriving)
    "reproductive_readiness",  # capacity to reproduce (0=none, 1=ready)
    "sensory_noise",        # perception reliability (0=blind, 1=clear)
    "adaptation_rate",      # speed of response to change (0=rigid, 1=plastic)
]

NUM_BSV_DIMS = len(BSV_DIMENSIONS)


@dataclass
class BiologicalStateVector:
    """Probabilistic biological state for any living system.
    
    8 dimensions capturing what all organisms share:
    metabolic_rate, membrane_integrity, stress_level, energy_level,
    health, reproductive_readiness, sensory_noise, adaptation_rate.
    
    Each dimension is a GaussianDim with mean ∈ [0,1] and variance ∈ (0,1].
    """
    
    metabolic_rate: GaussianDim = None
    membrane_integrity: GaussianDim = None
    stress_level: GaussianDim = None
    energy_level: GaussianDim = None
    health: GaussianDim = None
    reproductive_readiness: GaussianDim = None
    sensory_noise: GaussianDim = None
    adaptation_rate: GaussianDim = None
    
    def __post_init__(self):
        defaults = {
            "metabolic_rate": GaussianDim(0.5, 0.25),
            "membrane_integrity": GaussianDim(0.8, 0.1),
            "stress_level": GaussianDim(0.2, 0.25),
            "energy_level": GaussianDim(0.6, 0.2),
            "health": GaussianDim(0.7, 0.15),
            "reproductive_readiness": GaussianDim(0.3, 0.3),
            "sensory_noise": GaussianDim(0.5, 0.2),
            "adaptation_rate": GaussianDim(0.5, 0.25),
        }
        for dim_name, default in defaults.items():
            if getattr(self, dim_name) is None:
                setattr(self, dim_name, default)
    
    @property
    def dim_names(self) -> list[str]:
        return list(BSV_DIMENSIONS)
    
    @property
    def means(self) -> np.ndarray:
        return np.array([getattr(self, d).mean for d in BSV_DIMENSIONS])
    
    @property
    def variances(self) -> np.ndarray:
        return np.array([getattr(self, d).variance for d in BSV_DIMENSIONS])
    
    @property
    def uncertainty(self) -> float:
        return float(np.mean(self.variances))
    
    @property
    def confidence(self) -> float:
        return 1.0 - self.uncertainty
    
    def to_array(self) -> np.ndarray:
        """14D array: [8 means, 6... wait 8 variances]"""
        return np.concatenate([self.means, self.variances])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> BiologicalStateVector:
        arr = np.asarray(arr, dtype=np.float64)
        kwargs = {}
        for i, dim_name in enumerate(BSV_DIMENSIONS):
            kwargs[dim_name] = GaussianDim(mean=float(arr[i]),
                                           variance=float(arr[i + NUM_BSV_DIMS]))
        return cls(**kwargs)
    
    def update_from_observation(self, observations: dict[str, float],
                                 confidence: float = 0.5) -> None:
        """Update dimensions from observed values.
        
        Args:
            observations: dict mapping dimension names to observed values (0-1)
            confidence: how much to trust the observation (0-1)
        """
        for dim_name, value in observations.items():
            if dim_name in BSV_DIMENSIONS:
                dim = getattr(self, dim_name)
                # Use inverse variance as observation noise
                noise = 0.1 + (1.0 - confidence) * 0.4
                dim.update(value, noise)
    
    def update_from_engine_bio_state(self, food_level: float,
                                      has_shelter: bool,
                                      offspring_count: int) -> None:
        """Bridge from Engine's BioState fields."""
        self.energy_level.update(food_level / 100.0, 0.15)
        if has_shelter:
            self.membrane_integrity.update(0.8, 0.2)
        self.reproductive_readiness.update(
            min(1.0, offspring_count * 0.2), 0.3)
    
    def update_from_engine_projection(self, metabolic_rate: float,
                                       membrane_integrity: float,
                                       stress_level: float) -> None:
        """Bridge from Engine's BiologicalProjection."""
        self.metabolic_rate.update(metabolic_rate, 0.15)
        self.membrane_integrity.update(membrane_integrity, 0.15)
        self.stress_level.update(stress_level, 0.15)
    
    def composite_health(self) -> float:
        """Weighted health score across all dimensions."""
        w = {
            "metabolic_rate": 0.15,
            "membrane_integrity": 0.2,
            "stress_level": -0.15,  # inverted: high stress = low health
            "energy_level": 0.2,
            "health": 0.2,
            "reproductive_readiness": 0.05,
            "sensory_noise": 0.0,
            "adaptation_rate": 0.05,
        }
        score = 0.0
        for dim_name, weight in w.items():
            score += weight * getattr(self, dim_name).mean
        return max(0.0, min(1.0, score))
    
    def is_viable(self, threshold: float = 0.3) -> bool:
        """Is this organism alive?"""
        return self.composite_health() > threshold
    
    def to_dict(self) -> dict:
        return {d: {"mean": getattr(self, d).mean,
                     "variance": getattr(self, d).variance}
                for d in BSV_DIMENSIONS}
