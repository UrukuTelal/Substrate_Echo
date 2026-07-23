"""Basin Topology — Metrics for the geometry of cognition.

Measures the structure of the energy landscape over time:
- Basin depth: how deep each basin is (energy at center vs boundary)
- Basin volume: estimated volume of each basin's influence
- Basin lifetime: how long each attractor persists
- Merge/split frequency: how often attractors combine or divide
- Basin entropy: diversity of basin sizes
- Persistence: resistance to perturbation

These metrics describe the geometry of cognition, not just the quantity.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math
import numpy as np


@dataclass
class BasinMetrics:
    """Snapshot of basin topology at a point in time."""
    n_attractors: int
    depths: List[float]        # energy depth of each basin
    volumes: List[float]       # estimated volume of each basin
    mean_depth: float
    mean_volume: float
    depth_entropy: float       # entropy of depth distribution
    volume_entropy: float      # entropy of volume distribution
    basin_balance: float       # 1.0 = all basins equal, 0.0 = one dominates
    total_energy: float        # sum of all basin energies


@dataclass
class BasinEvent:
    """Record of a structural change in the landscape."""
    event_type: str  # "birth", "death", "merge", "split", "strengthen", "weaken"
    tick: int
    attractor_ids: List[int]
    description: str = ""


@dataclass
class AttractorState:
    """State of an individual attractor with plasticity properties."""
    id: int
    center: np.ndarray
    strength: float = 0.5
    created_tick: int = 0
    last_accessed_tick: int = 0
    access_count: int = 0
    stability: float = 0.5     # resists change (increases with age + access)
    plasticity: float = 0.5    # ability to shift center (decreases with stability)
    novelty: float = 1.0       # decreases with time and repetition
    confidence: float = 0.5    # increases with consistent evidence
    center_velocity: np.ndarray = field(default_factory=lambda: np.zeros(16))

    def access(self, tick: int):
        self.access_count += 1
        self.last_accessed_tick = tick
        self.stability = min(1.0, self.stability + 0.02)
        self.plasticity = max(0.1, 1.0 - self.stability)
        self.novelty = max(0.0, self.novelty - 0.05)
        self.confidence = min(1.0, self.confidence + 0.03)

    def decay(self, tick: int):
        ticks_idle = tick - self.last_accessed_tick
        if ticks_idle > 100:
            decay_rate = 0.001 * (1.0 - self.stability)
            self.strength = max(0.05, self.strength - decay_rate * ticks_idle)
            self.novelty = min(1.0, self.novelty + 0.001 * ticks_idle)

    def shift(self, new_center: np.ndarray, learning_rate: float = 0.1):
        """Shift center toward new evidence, weighted by plasticity."""
        delta = new_center - self.center
        effective_rate = learning_rate * self.plasticity
        self.center_velocity = delta * effective_rate
        self.center = self.center + self.center_velocity
        # Stability resists movement — more stable attractors shift less
        self.stability = max(0.1, self.stability - 0.01 * np.linalg.norm(delta))


class BasinTopology:
    """Tracks basin topology over time and detects structural changes."""

    def __init__(self, sigma: float = 0.3):
        self.sigma = sigma
        self.attractors: Dict[int, AttractorState] = {}
        self._next_id = 0
        self.history: List[Tuple[int, BasinMetrics]] = []
        self.events: List[BasinEvent] = []
        self._prev_centers: Optional[np.ndarray] = None

    def add_attractor(self, center: np.ndarray, tick: int,
                      strength: float = 0.5) -> int:
        aid = self._next_id
        self._next_id += 1
        self.attractors[aid] = AttractorState(
            id=aid, center=center.copy(), strength=strength,
            created_tick=tick, last_accessed_tick=tick)
        self.events.append(BasinEvent("birth", tick, [aid],
                                      f"New attractor at norm={np.linalg.norm(center):.3f}"))
        return aid

    def remove_attractor(self, aid: int, tick: int, reason: str = "decay"):
        if aid in self.attractors:
            del self.attractors[aid]
            self.events.append(BasinEvent("death", tick, [aid], reason))

    def detect_events(self, current_centers: np.ndarray, tick: int):
        """Compare current attractor layout to previous, detect merges/splits."""
        if self._prev_centers is None or len(current_centers) == 0:
            self._prev_centers = current_centers.copy() if len(current_centers) > 0 else np.array([])
            return

        prev = self._prev_centers
        curr = current_centers

        # Detect merges: two prev centers now map to same curr center
        if len(prev) >= 2 and len(curr) < len(prev):
            # Find which prev centers disappeared
            for p in prev:
                dists_to_curr = [np.linalg.norm(p - c) for c in curr]
                if min(dists_to_curr) > self.sigma * 2:
                    # This center disappeared — find what it merged into
                    closest_curr = curr[np.argmin([np.linalg.norm(p - c) for c in curr])]
                    self.events.append(BasinEvent(
                        "merge", tick, [],
                        f"Attractor merged toward norm={np.linalg.norm(closest_curr):.3f}"))

        # Detect splits: one prev center now maps to multiple curr centers
        if len(curr) > len(prev) and len(prev) > 0:
            new_count = len(curr) - len(prev)
            self.events.append(BasinEvent(
                "split", tick, [],
                f"{new_count} new attractor(s) appeared"))

        self._prev_centers = curr.copy()

    def compute_metrics(self, states: Optional[np.ndarray] = None) -> BasinMetrics:
        """Compute current basin topology metrics.

        Depth: energy contrast between basin center and basin boundary.
        Volume: reciprocal of nearest-neighbor distance (higher = more isolated).
        """
        if not self.attractors:
            return BasinMetrics(0, [], [], 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        centers = np.array([a.center for a in self.attractors.values()])
        strengths = np.array([a.strength for a in self.attractors.values()])
        dim = centers.shape[1]

        # Basin depth: energy at center vs average energy at boundary
        # Sample multiple boundary points per basin for robustness
        depths = []
        for c in centers:
            e_center = self._energy_at(c, centers, strengths)
            boundary_energies = []
            for _ in range(5):
                boundary = c + np.random.randn(dim) * self.sigma
                boundary = np.clip(boundary, 0.0, 1.0)
                boundary_energies.append(self._energy_at(boundary, centers, strengths))
            e_boundary = np.mean(boundary_energies)
            depths.append(abs(e_center - e_boundary))

        # Basin volume: use nearest-neighbor distance as isolation measure
        # Higher value = more isolated = larger effective basin
        volumes = []
        for i, c in enumerate(centers):
            dists = [np.linalg.norm(c - other)
                     for j, other in enumerate(centers) if j != i]
            if dists:
                nn_dist = min(dists)
                # Convert to a volume-like measure: further = bigger basin
                vol = nn_dist ** 2  # squared distance, scale-invariant
            else:
                vol = self.sigma ** 2
            volumes.append(vol)

        # Normalize for entropy
        total_depth = sum(depths) + 1e-10
        total_volume = sum(volumes) + 1e-10
        depth_probs = np.array(depths) / total_depth
        volume_probs = np.array(volumes) / total_volume

        depth_entropy = -np.sum(depth_probs * np.log(depth_probs + 1e-10))
        volume_entropy = -np.sum(volume_probs * np.log(volume_probs + 1e-10))

        # Basin balance: 1.0 = all equal, 0.0 = one dominates
        n = len(strengths)
        if n > 1:
            balance = 1.0 - np.std(strengths) / (np.mean(strengths) + 1e-10)
        else:
            balance = 1.0

        return BasinMetrics(
            n_attractors=n,
            depths=depths,
            volumes=volumes,
            mean_depth=np.mean(depths),
            mean_volume=np.mean(volumes),
            depth_entropy=float(depth_entropy),
            volume_entropy=float(volume_entropy),
            basin_balance=float(balance),
            total_energy=float(np.sum(strengths)),
        )

    def _energy_at(self, x, centers, strengths):
        e = 0.0
        for c, s in zip(centers, strengths):
            dist2 = np.sum((x - c) ** 2)
            e -= s * np.exp(-dist2 / (2 * self.sigma ** 2))
        return e

    def record_snapshot(self, tick: int, states: Optional[np.ndarray] = None):
        metrics = self.compute_metrics(states)
        self.history.append((tick, metrics))

    def summary(self) -> dict:
        if not self.history:
            return {}
        ticks = [t for t, _ in self.history]
        counts = [m.n_attractors for _, m in self.history]
        depths = [m.mean_depth for _, m in self.history]
        entropies = [m.volume_entropy for _, m in self.history]
        return {
            "ticks": ticks,
            "n_attractors": counts,
            "mean_depth": depths,
            "volume_entropy": entropies,
            "n_events": len(self.events),
            "births": sum(1 for e in self.events if e.event_type == "birth"),
            "deaths": sum(1 for e in self.events if e.event_type == "death"),
            "merges": sum(1 for e in self.events if e.event_type == "merge"),
            "splits": sum(1 for e in self.events if e.event_type == "split"),
        }
