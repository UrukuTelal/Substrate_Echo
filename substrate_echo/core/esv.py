"""Environmental State Vector (ESV) — P6.7

Standardized environmental state capturing what any location
or cell in the world can express.

Bridges Engine's CellState (temp, depth, flux, population)
with Substrate_Echo's spatial memory and affordance systems.

Each dimension is a GaussianDim — the environment is observed
through noisy sensors, so probabilistic representation is natural.

Dimensions:
- temperature: thermal state (0=cold, 1=hot)
- humidity: moisture level (0=dry, 1=wet)
- light: illumination (0=dark, 1=bright)
- resource_density: available resources (0=barren, 1=rich)
- hazard_level: danger (0=safe, 1=lethal)
- population_density: organism count (0=empty, 1=crowded)
- terrain_complexity: navigation difficulty (0=open, 1=impassable)
- pollution: contamination (0=pristine, 1=toxic)

Usage:
    esv = EnvironmentalStateVector()
    esv.update_from_observation({
        "temperature": 0.6,
        "hazard_level": 0.1,
        "resource_density": 0.8,
    })
    
    # Bridge from Engine CellState
    esv.update_from_engine_cellstate(temp=22.0, depth=0.5, flux=0.3)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from substrate_echo.core.human_state import GaussianDim


ESV_DIMENSIONS = [
    "temperature",         # thermal state (0=cold, 1=hot)
    "humidity",            # moisture (0=dry, 1=wet)
    "light",               # illumination (0=dark, 1=bright)
    "resource_density",    # available resources (0=barren, 1=rich)
    "hazard_level",        # danger (0=safe, 1=lethal)
    "population_density",  # organism count (0=empty, 1=crowded)
    "terrain_complexity",  # navigation difficulty (0=open, 1=impassable)
    "pollution",           # contamination (0=pristine, 1=toxic)
]

NUM_ESV_DIMS = len(ESV_DIMENSIONS)


@dataclass
class EnvironmentalStateVector:
    """Probabilistic environmental state for any location.
    
    8 dimensions capturing what any environment can express.
    Each dimension is a GaussianDim with mean ∈ [0,1] and variance ∈ (0,1].
    """
    
    temperature: GaussianDim = None
    humidity: GaussianDim = None
    light: GaussianDim = None
    resource_density: GaussianDim = None
    hazard_level: GaussianDim = None
    population_density: GaussianDim = None
    terrain_complexity: GaussianDim = None
    pollution: GaussianDim = None
    
    def __post_init__(self):
        defaults = {
            "temperature": GaussianDim(0.5, 0.2),
            "humidity": GaussianDim(0.5, 0.2),
            "light": GaussianDim(0.5, 0.2),
            "resource_density": GaussianDim(0.5, 0.25),
            "hazard_level": GaussianDim(0.1, 0.2),
            "population_density": GaussianDim(0.3, 0.25),
            "terrain_complexity": GaussianDim(0.3, 0.2),
            "pollution": GaussianDim(0.1, 0.15),
        }
        for dim_name, default in defaults.items():
            if getattr(self, dim_name) is None:
                setattr(self, dim_name, default)
    
    @property
    def dim_names(self) -> list[str]:
        return list(ESV_DIMENSIONS)
    
    @property
    def means(self) -> np.ndarray:
        return np.array([getattr(self, d).mean for d in ESV_DIMENSIONS])
    
    @property
    def variances(self) -> np.ndarray:
        return np.array([getattr(self, d).variance for d in ESV_DIMENSIONS])
    
    @property
    def uncertainty(self) -> float:
        return float(np.mean(self.variances))
    
    def to_array(self) -> np.ndarray:
        return np.concatenate([self.means, self.variances])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> EnvironmentalStateVector:
        arr = np.asarray(arr, dtype=np.float64)
        kwargs = {}
        for i, dim_name in enumerate(ESV_DIMENSIONS):
            kwargs[dim_name] = GaussianDim(mean=float(arr[i]),
                                           variance=float(arr[i + NUM_ESV_DIMS]))
        return cls(**kwargs)
    
    def update_from_observation(self, observations: dict[str, float],
                                 confidence: float = 0.5) -> None:
        """Update dimensions from sensor readings."""
        for dim_name, value in observations.items():
            if dim_name in ESV_DIMENSIONS:
                dim = getattr(self, dim_name)
                noise = 0.1 + (1.0 - confidence) * 0.4
                dim.update(value, noise)
    
    def update_from_engine_cellstate(self, temp: float, depth: float,
                                      flux: float) -> None:
        """Bridge from Engine's CellState.
        
        Args:
            temp: temperature in Celsius (mapped to 0-1)
            depth: depth in km (mapped to light/inverse)
            flux: flux 0-1
        """
        # Map temperature: 0°C→0.2, 20°C→0.5, 40°C→0.8
        temp_norm = max(0.0, min(1.0, (temp + 20.0) / 60.0))
        self.temperature.update(temp_norm, 0.15)
        
        # Depth reduces light
        light_val = max(0.0, 1.0 - depth * 0.5)
        self.light.update(light_val, 0.2)
        
        # Flux maps to resource density
        self.resource_density.update(flux, 0.2)
    
    def habitability(self) -> float:
        """How habitable is this environment for biological life?
        
        High habitability = moderate temp, good light, resources,
        low hazard, low pollution.
        """
        temp_ideal = 1.0 - abs(self.temperature.mean - 0.5) * 2.0
        return (
            0.25 * temp_ideal +
            0.20 * self.light.mean +
            0.25 * self.resource_density.mean +
            0.15 * (1.0 - self.hazard_level.mean) +
            0.10 * (1.0 - self.pollution.mean) +
            0.05 * (1.0 - self.terrain_complexity.mean)
        )
    
    def danger_score(self) -> float:
        """Overall danger level."""
        return (
            0.4 * self.hazard_level.mean +
            0.3 * self.pollution.mean +
            0.2 * self.terrain_complexity.mean +
            0.1 * min(1.0, abs(self.temperature.mean - 0.5) * 2.0)
        )
    
    def to_dict(self) -> dict:
        return {d: {"mean": getattr(self, d).mean,
                     "variance": getattr(self, d).variance}
                for d in ESV_DIMENSIONS}
