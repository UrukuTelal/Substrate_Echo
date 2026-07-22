"""Ontological Field — dynamic 16D state space with field evolution.

The 16D PSV is not just a static vector — it's a field that evolves over time.
Experiences inject energy into the field, forming attractors (memories) and
repulsors (avoidances). The field's gradient drives motivation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math
import time

import numpy as np


@dataclass
class Attractor:
    """A stable pattern in the ontological field — the basis of memory.
    
    Attractors form when the field is repeatedly stimulated in a similar pattern.
    They represent stable concepts, memories, or identity features.
    """
    center: np.ndarray       # 16D center point
    basin_width: float = 0.3  # how wide the attraction basin is
    strength: float = 1.0     # persistence (0-1)
    access_count: int = 0
    formed_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    decay_rate: float = 0.0001  # per second
    label: str = ""           # optional human-readable label
    metadata: dict = field(default_factory=dict)

    def distance_to(self, point: np.ndarray) -> float:
        """Euclidean distance from this attractor to a point."""
        return float(np.linalg.norm(self.center - point))
    
    def is_in_basin(self, point: np.ndarray) -> bool:
        """Is a point within this attractor's basin of attraction?"""
        return self.distance_to(point) <= self.basin_width
    
    def pull(self, point: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Compute the pull vector from this attractor on a point."""
        diff = self.center - point
        dist = np.linalg.norm(diff)
        if dist < 1e-8:
            return np.zeros(16)
        # Stronger pull when closer and stronger attractor
        pull_strength = strength * self.strength / (1.0 + dist)
        return diff * pull_strength
    
    def access(self) -> None:
        """Record an access (strengthens the attractor)."""
        self.access_count += 1
        self.last_accessed = time.time()
        self.strength = min(1.0, self.strength + 0.02)
    
    def decay(self, dt: float) -> None:
        """Attractor weakens without access."""
        time_since_access = time.time() - self.last_accessed
        hours = time_since_access / 3600.0
        self.strength *= math.exp(-self.decay_rate * hours * dt)
        self.strength = max(0.0, self.strength)
    
    @property
    def is_viable(self) -> bool:
        return self.strength > 0.01


@dataclass
class Repulsor:
    """An unstable point that pushes state away — represents avoidances."""
    center: np.ndarray
    strength: float = 0.5
    radius: float = 0.2

    def push(self, point: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Compute the push vector away from this repulsor."""
        diff = point - self.center
        dist = np.linalg.norm(diff)
        if dist < 1e-8 or dist > self.radius:
            return np.zeros(16)
        push_strength = strength * self.strength * (1.0 - dist / self.radius)
        return diff * push_strength


class OntologicalField:
    """Dynamic field over the 16D ontological state space.
    
    Evolution equation:
        ∂ψ/∂t = D·∇²ψ + F(ψ) + η(t)
    
    Where:
        ψ = 16D PSV field
        D = diffusion tensor (inter-pillar coupling)
        F(ψ) = local dynamics (attractors, repulsors)
        η(t) = sensory noise/input
    """
    
    NUM_PILLARS = 16
    
    def __init__(self):
        # Current state
        self.state = np.zeros(self.NUM_PILLARS, dtype=np.float64)
        self.state_velocity = np.zeros(self.NUM_PILLARS, dtype=np.float64)
        
        # Field structures
        self.attractors: list[Attractor] = []
        self.repulsors: list[Repulsor] = []
        
        # Diffusion tensor (16x16 inter-pillar coupling)
        self.diffusion = np.eye(self.NUM_PILLARS, dtype=np.float64) * 0.01
        
        # Sensory input buffer
        self.sensory_buffer = np.zeros(self.NUM_PILLARS, dtype=np.float64)
        self.sensory_weight = 0.1  # how much sensory input affects field
        
        # Timing
        self.last_evolution = time.time()
    
    def set_state(self, values: np.ndarray) -> None:
        """Set the field state directly."""
        self.state = np.clip(values.astype(np.float64), 0.0, 1.0)
    
    def inject_sensory(self, sensory: np.ndarray, weight: float = None) -> None:
        """Inject sensory input into the field."""
        w = weight if weight is not None else self.sensory_weight
        self.sensory_buffer += sensory * w
    
    def evolve(self, dt: float) -> None:
        """Step the field forward by dt seconds.
        
        Combines:
        1. Diffusion (inter-pillar spreading)
        2. Attractor pulls
        3. Repulsor pushes
        4. Sensory input
        5. Damping (field doesn't grow unbounded)
        """
        # Diffusion: D·∇²ψ (simplified as coupling to neighbors)
        diffusion_force = self.diffusion @ (np.zeros(16) - self.state) * 0.01
        
        # Attractor pulls
        attractor_force = np.zeros(16)
        for att in self.attractors:
            attractor_force += att.pull(self.state, strength=0.05)
        
        # Repulsor pushes
        repulsor_force = np.zeros(16)
        for rep in self.repulsors:
            repulsor_force += rep.push(self.state, strength=0.05)
        
        # Sensory input
        sensory_force = self.sensory_buffer.copy()
        self.sensory_buffer *= 0.5  # buffer decays
        
        # Total force
        total_force = diffusion_force + attractor_force + repulsor_force + sensory_force
        
        # Update state
        self.state_velocity = self.state_velocity * 0.9 + total_force * 0.1
        self.state += self.state_velocity * dt
        
        # Clamp to [0, 1]
        self.state = np.clip(self.state, 0.0, 1.0)
        
        self.last_evolution = time.time()
    
    def form_attractor(self, pattern: np.ndarray, strength: float = 0.8,
                       label: str = "") -> Attractor:
        """Crystallize a new attractor from a pattern."""
        # Check if similar attractor already exists
        for att in self.attractors:
            if att.is_in_basin(pattern):
                att.access()
                return att
        
        # Create new attractor
        new_att = Attractor(
            center=pattern.copy(),
            basin_width=0.3,
            strength=strength,
            label=label,
        )
        self.attractors.append(new_att)
        return new_att
    
    def form_repulsor(self, point: np.ndarray, strength: float = 0.5) -> Repulsor:
        """Create a new repulsor (avoidance pattern)."""
        rep = Repulsor(center=point.copy(), strength=strength)
        self.repulsors.append(rep)
        return rep
    
    def find_nearest_attractors(self, point: np.ndarray, k: int = 5) -> list[Attractor]:
        """Find k nearest attractors to a point."""
        ranked = sorted(self.attractors, key=lambda a: a.distance_to(point))
        return [a for a in ranked[:k] if a.is_viable]
    
    def motivation_gradient(self, target: np.ndarray) -> np.ndarray:
        """Compute the gradient (direction) from current state toward target.
        
        This is the motivation drive — the field "wants" to move toward targets.
        """
        return target - self.state
    
    def consolidate(self, dt: float) -> int:
        """Consolidate the field: decay weak attractors, remove dead ones."""
        for att in self.attractors:
            att.decay(dt)
        before = len(self.attractors)
        self.attractors = [a for a in self.attractors if a.is_viable]
        return before - len(self.attractors)
    
    def coherence(self) -> float:
        """How internally aligned the field is (0-1)."""
        if not self.attractors:
            return 0.5
        centers = np.array([a.center for a in self.attractors])
        mean_center = np.mean(centers, axis=0)
        spread = np.mean(np.linalg.norm(centers - mean_center, axis=1))
        return max(0.0, 1.0 - spread)
    
    def identity_pattern(self) -> Optional[np.ndarray]:
        """The persistent attractor cluster = the agent's identity.
        
        Identity is the weighted mean of all strong attractors.
        """
        strong = [a for a in self.attractors if a.strength > 0.5]
        if not strong:
            return None
        weights = np.array([a.strength for a in strong])
        centers = np.array([a.center for a in strong])
        return np.average(centers, axis=0, weights=weights)
    
    def to_dict(self) -> dict:
        return {
            "state": self.state.tolist(),
            "attractor_count": len(self.attractors),
            "repulsor_count": len(self.repulsors),
            "coherence": round(self.coherence(), 3),
            "has_identity": self.identity_pattern() is not None,
        }
