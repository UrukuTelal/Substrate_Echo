"""Metric Interface — API contract for dynamic metric evolution (BCFVT-01).

Defines the interface that BCFVT-01 Ricci flow will implement.
Currently provides stub behavior so the system can run without the math.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MetricTensor:
    """16×16 metric tensor G_ij that defines distances in pillar space."""
    matrix: np.ndarray
    
    @classmethod
    def identity(cls) -> MetricTensor:
        """Default Euclidean metric."""
        return cls(matrix=np.eye(16, dtype=np.float64))
    
    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Geodesic distance between two points under this metric."""
        diff = a - b
        return float(np.sqrt(diff @ self.matrix @ diff))
    
    def inner_product(self, a: np.ndarray, b: np.ndarray) -> float:
        """Inner product under this metric."""
        return float(a @ self.matrix @ b)


class MetricInterface:
    """Interface for dynamic metric evolution.
    
    Stub: returns identity metric (Euclidean distances).
    After BCFVT-01: Ricci flow ∂G/∂t = -2R_ij + κT_ij
    """
    
    def __init__(self):
        self.metric = MetricTensor.identity()
        self._evolution_count = 0
    
    def get_metric(self) -> MetricTensor:
        """Get current metric tensor."""
        return self.metric
    
    def evolve(self, field_state: np.ndarray, dt: float = 0.01) -> MetricTensor:
        """Evolve metric based on field state.
        
        Stub: metric stays identity.
        BCFVT-01: R_ij from Christoffel symbols, T_ij from field stress-energy.
        """
        self._evolution_count += 1
        return self.metric
    
    def compute_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two points in state space."""
        return self.metric.distance(a, b)
    
    def stats(self) -> dict:
        return {
            "evolution_count": self._evolution_count,
            "metric_type": "identity" if self._evolution_count == 0 else "evolved",
        }
