"""Attractor Dynamics — formation, decay, strengthening, merging.

Manages the lifecycle of attractors in the ontological field:
- Formation: repeated stimulation → crystallization
- Decay: exponential forgetting for unused attractors
- Strengthening: frequent access increases strength
- Merging: similar attractors combine when overlap > threshold
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time


@dataclass
class DynamicsConfig:
    """Configuration for attractor dynamics."""
    formation_threshold: float = 0.3  # similarity to form new attractor
    merge_threshold: float = 0.9  # similarity to merge attractors
    decay_rate: float = 0.001  # exponential decay per second
    strengthen_rate: float = 0.01  # strength increase per access
    min_strength: float = 0.05  # minimum before pruning
    max_attractors: int = 1000  # maximum number of attractors
    access_history_size: int = 100  # keep last N access times


class AttractorDynamics:
    """Manages attractor lifecycle in the ontological field.
    
    Usage:
        dynamics = AttractorDynamics()
        dynamics.record_access(attractor, time)
        dynamics.decay_all(current_time)
        dynamics.merge_similar(attractors)
    """
    
    def __init__(self, config: Optional[DynamicsConfig] = None):
        self.config = config or DynamicsConfig()
        self._access_history: dict[str, list[float]] = {}
        self._formation_times: dict[str, float] = {}
    
    def record_access(self, attractor_label: str, strength: float,
                      current_time: Optional[float] = None) -> float:
        """Record access to an attractor and return new strength.
        
        Increases strength based on access frequency.
        """
        t = current_time or time.time()
        
        if attractor_label not in self._access_history:
            self._access_history[attractor_label] = []
        
        history = self._access_history[attractor_label]
        history.append(t)
        
        # Keep bounded
        if len(history) > self.config.access_history_size:
            history.pop(0)
        
        # Strengthen based on recent access frequency
        recent_cutoff = t - 60.0  # last minute
        recent_accesses = sum(1 for t_acc in history if t_acc > recent_cutoff)
        
        # More recent accesses → more strengthening
        frequency_bonus = min(recent_accesses / 10.0, 1.0) * self.config.strengthen_rate
        new_strength = min(1.0, strength + frequency_bonus)
        
        return new_strength
    
    def compute_decay(self, attractor_label: str, current_strength: float,
                      current_time: Optional[float] = None) -> float:
        """Compute decayed strength for an attractor.
        
        Exponential decay based on time since last access.
        """
        t = current_time or time.time()
        
        if attractor_label not in self._access_history:
            # Never accessed → decay from formation
            if attractor_label in self._formation_times:
                last_access = self._formation_times[attractor_label]
            else:
                return current_strength  # no history, no decay
        else:
            history = self._access_history[attractor_label]
            if not history:
                return current_strength
            last_access = max(history)
        
        time_since_access = t - last_access
        decay = np.exp(-self.config.decay_rate * time_since_access)
        
        return current_strength * decay
    
    def should_merge(self, attractor_a: np.ndarray, attractor_b: np.ndarray) -> bool:
        """Check if two attractors should merge based on similarity."""
        norm_a = np.linalg.norm(attractor_a)
        norm_b = np.linalg.norm(attractor_b)
        
        if norm_a < 1e-12 or norm_b < 1e-12:
            return False
        
        similarity = np.dot(attractor_a, attractor_b) / (norm_a * norm_b)
        return similarity >= self.config.merge_threshold
    
    def merge_attractors(self, center_a: np.ndarray, strength_a: float,
                         center_b: np.ndarray, strength_b: float) -> tuple[np.ndarray, float]:
        """Merge two attractors into one.
        
        New center is weighted average, new strength is sum (capped).
        """
        total = strength_a + strength_b
        if total < 1e-12:
            return center_a, strength_a
        
        weight_a = strength_a / total
        weight_b = strength_b / total
        
        new_center = weight_a * center_a + weight_b * center_b
        new_strength = min(1.0, total * 0.9)  # slight reduction to prevent runaway
        
        return new_center, new_strength
    
    def should_prune(self, strength: float) -> bool:
        """Check if an attractor should be pruned."""
        return strength < self.config.min_strength
    
    def register_formation(self, label: str, current_time: Optional[float] = None) -> None:
        """Register when an attractor was formed."""
        self._formation_times[label] = current_time or time.time()
    
    def get_age(self, label: str, current_time: Optional[float] = None) -> float:
        """Get age of an attractor in seconds."""
        t = current_time or time.time()
        if label in self._formation_times:
            return t - self._formation_times[label]
        return 0.0
    
    def get_access_frequency(self, label: str, window_seconds: float = 60.0,
                             current_time: Optional[float] = None) -> float:
        """Get access frequency (accesses per second) in recent window."""
        t = current_time or time.time()
        
        if label not in self._access_history:
            return 0.0
        
        cutoff = t - window_seconds
        recent = sum(1 for t_acc in self._access_history[label] if t_acc > cutoff)
        
        return recent / window_seconds
    
    def get_stats(self) -> dict:
        """Get dynamics statistics."""
        return {
            "tracked_attractors": len(self._access_history),
            "formation_times": len(self._formation_times),
            "total_accesses": sum(len(h) for h in self._access_history.values()),
        }
