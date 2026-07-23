"""Abstraction — Meta-attractor creation from correlated attractors.

When multiple base attractors co-activate repeatedly, the system creates
a meta-attractor representing their shared structure. This is not merging
(destroying the originals) but abstraction (building hierarchy).

The hierarchy emerges naturally:
  base attractors → correlated clusters → meta-attractors → ...

Additionally, a finite cognitive budget forces competition:
  total energy is conserved, so strengthening one attractor weakens others.
  This produces forgetting without inventing a forgetting algorithm.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import numpy as np


@dataclass
class MetaAttractor:
    """A higher-order attractor representing correlated base attractors."""
    id: int
    center: np.ndarray
    children: List[int]        # IDs of base attractors this represents
    strength: float = 0.0
    created_tick: int = 0
    last_accessed_tick: int = 0
    access_count: int = 0
    sigma: float = 0.5         # coarser resolution than base attractors
    stability: float = 0.5
    confidence: float = 0.5
    level: int = 1             # 1 = meta, 2 = meta-meta, etc.


@dataclass
class CorrelationSnapshot:
    """Pairwise correlation between two attractors at a point in time."""
    id_a: int
    id_b: int
    correlation: float
    co_activations: int
    total_observations: int


class AttractorCorrelation:
    """Tracks which attractors co-activate over time.

    Co-activation means both attractors are "near" a state within
    the same time window.
    """

    def __init__(self, window_size: int = 50, proximity_threshold: float = 0.25):
        self.window_size = window_size
        self.proximity_threshold = proximity_threshold
        # Sliding window of (tick, nearest_attractor_id) pairs
        self.activation_history: List[Tuple[int, int]] = []
        # Pairwise co-activation counts
        self.co_activation: Dict[Tuple[int, int], int] = {}
        self.total_observations: Dict[int, int] = {}

    def record_activation(self, tick: int, attractor_id: int):
        self.activation_history.append((tick, attractor_id))
        if len(self.activation_history) > self.window_size:
            self.activation_history.pop(0)
        self.total_observations[attractor_id] = (
            self.total_observations.get(attractor_id, 0) + 1)

    def get_correlation(self, id_a: int, id_b: int) -> float:
        """Co-activation correlation.

        What fraction of each attractor's activations occur within
        a short time window of the other's activations?
        """
        if id_a == id_b:
            return 1.0
        ticks_a = [t for t, aid in self.activation_history if aid == id_a]
        ticks_b = [t for t, aid in self.activation_history if aid == id_b]
        if not ticks_a or not ticks_b:
            return 0.0

        time_window = 30  # ticks

        # Count how many of A's activations are near any of B's
        shared_a = sum(
            1 for ta in ticks_a
            if any(abs(ta - tb) <= time_window for tb in ticks_b))
        # Count how many of B's activations are near any of A's
        shared_b = sum(
            1 for tb in ticks_b
            if any(abs(tb - ta) <= time_window for ta in ticks_a))

        frac_a = shared_a / len(ticks_a)
        frac_b = shared_b / len(ticks_b)
        return (frac_a + frac_b) / 2

    def get_correlation_matrix(self, attractor_ids: List[int]) -> np.ndarray:
        n = len(attractor_ids)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self.get_correlation(
                        attractor_ids[i], attractor_ids[j])
        return matrix

    def find_correlated_clusters(self, attractor_ids: List[int],
                                  threshold: float = 0.3,
                                  min_cluster_size: int = 2) -> List[List[int]]:
        """Find clusters of attractors with mutual correlation above threshold.

        Uses a simple connected-component approach.
        """
        if len(attractor_ids) < 2:
            return []

        # Build adjacency graph
        adj: Dict[int, Set[int]] = {aid: set() for aid in attractor_ids}
        for i, a in enumerate(attractor_ids):
            for b in attractor_ids[i+1:]:
                corr = self.get_correlation(a, b)
                if corr >= threshold:
                    adj[a].add(b)
                    adj[b].add(a)

        # Connected components via BFS
        visited = set()
        clusters = []
        for aid in attractor_ids:
            if aid in visited:
                continue
            cluster = []
            queue = [aid]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            if len(cluster) >= min_cluster_size:
                clusters.append(sorted(cluster))

        return clusters


class CognitiveBudget:
    """Finite cognitive energy that attractors compete for.

    Total energy is conserved. Each tick, energy is redistributed:
    - Used attractors gain energy (reinforcement)
    - Unused attractors lose energy (passive decay via competition)
    - New attractors start with minimum energy
    """

    def __init__(self, total_energy: float = 10.0, maintenance_rate: float = 0.01,
                 reinforcement_rate: float = 0.02, min_strength: float = 0.02):
        self.total_energy = total_energy
        self.maintenance_rate = maintenance_rate
        self.reinforcement_rate = reinforcement_rate
        self.min_strength = min_strength

    def redistribute(self, attractor_strengths: Dict[int, float],
                     accessed_ids: Set[int],
                     all_ids: Set[int]) -> Dict[int, float]:
        """Redistribute energy based on usage.

        accessed_ids: attractors that were activated this tick
        all_ids: all attractor IDs
        """
        new_strengths = dict(attractor_strengths)

        # Every attractor pays maintenance cost
        for aid in all_ids:
            current = new_strengths.get(aid, self.min_strength)
            cost = self.maintenance_rate * current
            new_strengths[aid] = max(self.min_strength, current - cost)

        # Accessed attractors gain reinforcement
        for aid in accessed_ids:
            current = new_strengths.get(aid, self.min_strength)
            new_strengths[aid] = min(1.0, current + self.reinforcement_rate)

        # Scale to conserve total energy
        total = sum(new_strengths.get(aid, self.min_strength) for aid in all_ids)
        if total > 0 and abs(total - self.total_energy) > 0.1:
            scale = self.total_energy / total
            for aid in all_ids:
                new_strengths[aid] = max(
                    self.min_strength,
                    new_strengths.get(aid, self.min_strength) * scale)

        return new_strengths


class AbstractionEngine:
    """Detects correlated attractor clusters and creates meta-attractors.

    The abstraction hierarchy:
      Level 0: base attractors (directly from convergence detection)
      Level 1: meta-attractors (from correlated Level 0 clusters)
      Level 2: meta-meta-attractors (from correlated Level 1 clusters)
      ...
    """

    def __init__(self, correlation_threshold: float = 0.3,
                 min_cluster_size: int = 2,
                 meta_sigma: float = 0.5):
        self.correlation = AttractorCorrelation(
            window_size=50, proximity_threshold=0.25)
        self.budget = CognitiveBudget(
            total_energy=10.0, maintenance_rate=0.01,
            reinforcement_rate=0.02, min_strength=0.02)
        self.correlation_threshold = correlation_threshold
        self.min_cluster_size = min_cluster_size
        self.meta_sigma = meta_sigma
        self.meta_attractors: Dict[int, MetaAttractor] = {}
        self._next_meta_id = 1000
        self.abstraction_events: List[Dict] = []

    def update(self, tick: int, state: np.ndarray,
               base_attractors: Dict[int, Tuple[np.ndarray, float]]):
        """One tick of the abstraction engine.

        base_attractors: {id: (center, strength)}
        """
        if not base_attractors:
            return

        # Find nearest base attractor
        min_dist = float('inf')
        nearest_id = -1
        for aid, (center, _) in base_attractors.items():
            d = np.linalg.norm(state - center)
            if d < min_dist:
                min_dist = d
                nearest_id = aid

        # Record activation if within proximity
        if nearest_id >= 0 and min_dist < self.correlation.proximity_threshold:
            self.correlation.record_activation(tick, nearest_id)

    def check_abstraction(self, tick: int,
                          base_attractors: Dict[int, Tuple[np.ndarray, float]]
                          ) -> List[MetaAttractor]:
        """Check if any correlated clusters are ready for abstraction.

        Returns newly created meta-attractors.
        """
        if len(base_attractors) < self.min_cluster_size:
            return []

        ids = list(base_attractors.keys())
        clusters = self.correlation.find_correlated_clusters(
            ids, self.correlation_threshold, self.min_cluster_size)

        new_metas = []
        for cluster in clusters:
            # Check if this cluster is already represented by a meta-attractor
            if self._cluster_already_abstracted(cluster):
                continue

            # Create meta-attractor at the centroid of the cluster
            centers = np.array([base_attractors[cid][0] for cid in cluster])
            strengths = [base_attractors[cid][1] for cid in cluster]
            centroid = np.mean(centers, axis=0)
            total_strength = sum(strengths)

            meta = MetaAttractor(
                id=self._next_meta_id,
                center=centroid,
                children=cluster,
                strength=total_strength * 0.5,  # meta gets half the combined
                created_tick=tick,
                last_accessed_tick=tick,
                sigma=self.meta_sigma,
                level=1,
            )
            self._next_meta_id += 1
            self.meta_attractors[meta.id] = meta
            new_metas.append(meta)

            event = {
                "type": "abstraction",
                "tick": tick,
                "meta_id": meta.id,
                "children": cluster,
                "correlations": [
                    self.correlation.get_correlation(cluster[i], cluster[j])
                    for i in range(len(cluster))
                    for j in range(i+1, len(cluster))
                ],
                "centroid_norm": float(np.linalg.norm(centroid)),
                "total_strength": total_strength,
            }
            self.abstraction_events.append(event)

        return new_metas

    def _cluster_already_abstracted(self, cluster: List[int]) -> bool:
        for meta in self.meta_attractors.values():
            if sorted(meta.children) == sorted(cluster):
                return True
        return False

    def get_all_centers(self, include_meta: bool = True) -> List[np.ndarray]:
        """Get all attractor centers (base + meta) for energy landscape."""
        centers = []
        for meta in self.meta_attractors.values():
            if include_meta:
                centers.append(meta.center)
        return centers

    def energy_contribution(self, x: np.ndarray) -> float:
        """Compute meta-attractor energy at point x."""
        e = 0.0
        for meta in self.meta_attractors.values():
            dist2 = np.sum((x - meta.center) ** 2)
            e -= meta.strength * np.exp(-dist2 / (2 * meta.sigma ** 2))
        return e

    def gradient_contribution(self, x: np.ndarray) -> np.ndarray:
        """Compute meta-attractor gradient at point x."""
        grad = np.zeros_like(x)
        for meta in self.meta_attractors.values():
            diff = x - meta.center
            dist2 = np.sum(diff ** 2)
            grad += meta.strength * diff / (meta.sigma ** 2) * np.exp(
                -dist2 / (2 * meta.sigma ** 2))
        return grad

    def summary(self) -> Dict:
        return {
            "n_meta": len(self.meta_attractors),
            "n_abstraction_events": len(self.abstraction_events),
            "meta_details": [
                {
                    "id": m.id,
                    "children": m.children,
                    "strength": m.strength,
                    "n_children": len(m.children),
                }
                for m in self.meta_attractors.values()
            ],
        }
