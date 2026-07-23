"""EXP-SUB-004: Abstraction Hierarchy — From Correlation to Meta-Attractors

Tests whether the system creates meta-attractors from correlated
base attractors, building a hierarchy of abstraction.

Design:
  - 5 behavioral modes with overlapping structure
  - Modes 0,1,2 share dimensions (correlated attractors expected)
  - Mode 3 is the "abstract" version (all shared dimensions)
  - Mode 4 is unrelated (control)
  - Abstraction engine tracks correlations and creates meta-attractors
  - Cognitive budget forces competition (finite energy)
"""
import sys
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from substrate_echo.core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig
from substrate_echo.dynamics.basin_topology import BasinTopology
from substrate_echo.dynamics.abstraction import AbstractionEngine


class EnergyLandscape:
    def __init__(self, sigma=0.3):
        self.sigma = sigma
        self.attractors = []  # list of (center, strength, tick)

    def add_attractor(self, center, strength=0.5, tick=0):
        self.attractors.append((center.copy(), strength, tick))

    def energy(self, x):
        if not self.attractors:
            return 0.0
        e = 0.0
        for center, strength, _ in self.attractors:
            dist2 = np.sum((x - center) ** 2)
            e -= strength * np.exp(-dist2 / (2 * self.sigma ** 2))
        return e

    def gradient(self, x):
        if not self.attractors:
            return np.zeros_like(x)
        grad = np.zeros_like(x)
        for center, strength, _ in self.attractors:
            diff = x - center
            dist2 = np.sum(diff ** 2)
            grad += strength * diff / (self.sigma ** 2) * np.exp(
                -dist2 / (2 * self.sigma ** 2))
        return grad


class ConvergenceDetector:
    def __init__(self, window_size=40, radius=0.12, min_cluster=6):
        self.window_size = window_size
        self.radius = radius
        self.min_cluster = min_cluster
        self.endpoints = []

    def add(self, state):
        self.endpoints.append(state.copy())
        if len(self.endpoints) > self.window_size:
            self.endpoints.pop(0)

    def detect(self):
        if len(self.endpoints) < self.min_cluster:
            return []
        arr = np.array(self.endpoints)
        assigned = np.zeros(len(arr), dtype=int) - 1
        clusters = []
        for i, pt in enumerate(arr):
            if assigned[i] >= 0:
                continue
            dists = np.linalg.norm(arr - pt, axis=1)
            neighbors = np.where(dists < self.radius)[0]
            if len(neighbors) >= self.min_cluster:
                center = arr[neighbors].mean(axis=0)
                clusters.append((center, len(neighbors)))
                assigned[neighbors] = len(clusters) - 1
        return clusters


class CoupledDynamics:
    def __init__(self, dm, landscape, abstraction_engine=None, coupling_strength=0.3):
        self.dm = dm
        self.landscape = landscape
        self.abstraction = abstraction_engine
        self.lambda_ = coupling_strength

    def predict_velocity(self, x):
        v_learned = self.dm.predict_velocity(x)
        v_attractor = -self.landscape.gradient(x)
        v_meta = np.zeros_like(x)
        if self.abstraction:
            v_meta = -self.abstraction.gradient_contribution(x)
        return v_learned + self.lambda_ * (v_attractor + 0.5 * v_meta)


def generate_correlated_trajectory(n_ticks=4000, seed=42):
    """Generate trajectory with overlapping behavioral modes.

    Modes 0,1,2 share dimensions 0,1 (the "concept" core).
    Mode 3 activates all shared dimensions (the "abstract" form).
    Mode 4 is unrelated (control).

    This structure should cause modes 0,1,2 to correlate,
    leading to a meta-attractor representing the shared pattern.
    """
    rng = np.random.RandomState(seed)
    state = np.full(16, 0.5)
    states, modes = [], []

    # Core concept: dimensions 0,1 are always active
    # Variations add different peripheral dimensions
    targets = {
        0: lambda: _make_target(0, [0, 1, 2, 3],   [0.8, 0.7, 0.6, 0.3]),   # concept variant A
        1: lambda: _make_target(0, [0, 1, 4, 5],   [0.75, 0.8, 0.5, 0.4]),  # concept variant B
        2: lambda: _make_target(0, [0, 1, 6, 7],   [0.7, 0.75, 0.55, 0.35]),# concept variant C
        3: lambda: _make_target(0, [0, 1, 2, 3, 4, 5, 6, 7], [0.75]*8),     # abstract (all shared)
        4: lambda: _make_target(0, [10, 11, 12, 13], [0.8, 0.7, 0.6, 0.5]), # unrelated
    }

    for tick in range(n_ticks):
        # Cycle through modes, but concept variants appear more often
        r = rng.random()
        if r < 0.30:
            mode = 0
        elif r < 0.55:
            mode = 1
        elif r < 0.75:
            mode = 2
        elif r < 0.85:
            mode = 3
        else:
            mode = 4

        target = targets[mode]()
        state = state + (target - state) * 0.12 + rng.randn(16) * 0.015
        state = np.clip(state, 0.0, 1.0)
        states.append(state.copy())
        modes.append(mode)

    return np.array(states), modes


def _make_target(_, indices, values):
    t = np.zeros(16)
    for i, v in zip(indices, values):
        t[i] = v
    return t


def compute_coherence(assignments, labels):
    clusters = {}
    for a, l in zip(assignments, labels):
        clusters.setdefault(a, Counter())[l] += 1
    total = correct = 0
    for counts in clusters.values():
        correct += counts.most_common(1)[0][1]
        total += sum(counts.values())
    return correct / total if total > 0 else 0.0


def assign_to_attractors(states, attractor_centers):
    if not attractor_centers:
        return [-1] * len(states)
    assignments = []
    for s in states:
        dists = [np.linalg.norm(s - c) for c in attractor_centers]
        assignments.append(int(np.argmin(dists)))
    return assignments


def run_experiment(n_ticks=4000, seed=42):
    print("=" * 70)
    print("EXP-SUB-004: Abstraction Hierarchy")
    print("=" * 70)
    print(f"Ticks: {n_ticks}, Seed: {seed}")
    print()

    states, modes = generate_correlated_trajectory(n_ticks, seed)
    print(f"Generated {len(states)} states, {len(set(modes))} modes")
    mode_counts = Counter(modes)
    for m in sorted(mode_counts):
        print(f"  Mode {m}: {mode_counts[m]} ticks")

    # ── Setup ────────────────────────────────────────────────────
    dm = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(
        model_type='global', min_samples_for_fit=50, max_samples=4000,
        attractor_samples=300, attractor_integration_steps=200))
    landscape = EnergyLandscape(sigma=0.3)
    detector = ConvergenceDetector(window_size=40, radius=0.12, min_cluster=6)
    abstraction = AbstractionEngine(
        correlation_threshold=0.25, min_cluster_size=2, meta_sigma=0.5)
    coupled = CoupledDynamics(dm, landscape, abstraction, coupling_strength=0.3)
    topology = BasinTopology(sigma=0.3)

    prev_state = None
    convergence_events = 0

    # Track base attractors as {id: (center, strength)}
    base_attractor_map = {}

    print("\n--- Running coupled dynamics with abstraction engine ---")

    for tick in range(len(states)):
        state = states[tick]

        if prev_state is not None:
            velocity = state - prev_state
            dm._states.append(prev_state.copy())
            dm._velocities.append(velocity.copy())
            if len(dm._states) >= dm.config.min_samples_for_fit:
                if len(dm._states) % 50 == 0:
                    dm._fit_dynamics()

        # Convergence detection → new base attractors
        detector.add(state)
        if tick % 100 == 0 and tick > 200:
            clusters = detector.detect()
            new_attractors = 0
            for center, size in clusters:
                is_new = True
                for aid, (existing, _) in base_attractor_map.items():
                    if np.linalg.norm(center - existing) < 0.2:
                        is_new = False
                        break
                if is_new:
                    strength = min(1.0, size / 20.0)
                    aid = len(base_attractor_map)
                    base_attractor_map[aid] = (center.copy(), strength)
                    landscape.add_attractor(center, strength, tick)
                    topology.add_attractor(center, tick, strength)
                    new_attractors += 1
                    convergence_events += 1

            if new_attractors > 0:
                print(f"  tick {tick:4d}: +{new_attractors} base attractor(s), "
                      f"total={len(base_attractor_map)}")

        # Abstraction engine: track correlations and create meta-attractors
        if tick % 10 == 0 and base_attractor_map:
            abstraction.update(tick, state, base_attractor_map)

        if tick % 500 == 0 and tick > 500 and base_attractor_map:
            new_metas = abstraction.check_abstraction(tick, base_attractor_map)
            for meta in new_metas:
                print(f"  tick {tick:4d}: META-ABSTRACTION: "
                      f"M{meta.id} created from {meta.children} "
                      f"(strength={meta.strength:.3f})")
                # Add meta-attractor to landscape with coarser sigma
                landscape.attractors.append(
                    (meta.center.copy(), meta.strength * 0.5, tick))

        # Record topology snapshot
        if tick % 500 == 0 and tick > 0:
            topology.record_snapshot(tick)

        prev_state = state.copy()

    # ── Final measurements ───────────────────────────────────────
    print("\n--- Abstraction Summary ---")
    abs_summary = abstraction.summary()
    print(f"  Meta-attractors created: {abs_summary['n_meta']}")
    print(f"  Abstraction events: {abs_summary['n_abstraction_events']}")
    for m in abs_summary['meta_details']:
        print(f"    M{m['id']}: children={m['children']}, "
              f"strength={m['strength']:.3f}")

    # Correlation matrix
    if len(base_attractor_map) > 1:
        print("\n--- Attractor Correlation Matrix ---")
        ids = sorted(base_attractor_map.keys())
        corr_matrix = abstraction.correlation.get_correlation_matrix(ids)
        header = "     " + "".join(f"  A{a:02d} " for a in ids)
        print(header)
        for i, a in enumerate(ids):
            row = f"  A{a:02d} "
            for j, b in enumerate(ids):
                row += f" {corr_matrix[i][j]:.2f}  "
            print(row)

    # Coherence
    all_centers = [c for c, _ in base_attractor_map.values()]
    for meta in abstraction.meta_attractors.values():
        all_centers.append(meta.center)
    if all_centers:
        assign = assign_to_attractors(states, all_centers)
        coherence = compute_coherence(assign, modes)
    else:
        coherence = 0.0

    # Base-only coherence for comparison
    base_centers = [c for c, _ in base_attractor_map.values()]
    if base_centers:
        base_assign = assign_to_attractors(states, base_centers)
        base_coherence = compute_coherence(base_assign, modes)
    else:
        base_coherence = 0.0

    print(f"\n--- Coherence ---")
    print(f"  Base attractors only: {base_coherence:.3f}")
    print(f"  Base + meta-attractors: {coherence:.3f}")

    # Topology summary
    print("\n--- Topology ---")
    topo_summary = topology.summary()
    print(f"  Births: {topo_summary.get('births', 0)}")
    print(f"  Deaths: {topo_summary.get('deaths', 0)}")
    print(f"  Merges: {topo_summary.get('merges', 0)}")
    print(f"  Splits: {topo_summary.get('splits', 0)}")

    # Meta-attractor influence test
    print("\n--- Meta-Attractor Influence ---")
    if abstraction.meta_attractors:
        # Does the meta-attractor sit between its children?
        for meta in abstraction.meta_attractors.values():
            child_centers = [base_attractor_map[c][0] for c in meta.children
                            if c in base_attractor_map]
            if child_centers:
                mean_child = np.mean(child_centers, axis=0)
                dist_to_mean = np.linalg.norm(meta.center - mean_child)
                print(f"  M{meta.id}: distance to mean of children = {dist_to_mean:.4f}")
                # Is the meta-attractor's region between the children?
                for c in meta.children:
                    if c in base_attractor_map:
                        d = np.linalg.norm(meta.center - base_attractor_map[c][0])
                        print(f"    -> A{c}: distance = {d:.4f}")

    print("\n" + "=" * 70)
    if abs_summary['n_meta'] > 0:
        print("RESULT: Meta-attractors created from correlated base attractors.")
        print("The system builds hierarchy through correlation -> abstraction.")
    else:
        print("RESULT: No meta-attractors formed. Correlation may need")
        print("longer runs or lower threshold.")
    print("=" * 70)

    return {
        "n_base": len(base_attractor_map),
        "n_meta": abs_summary['n_meta'],
        "base_coherence": base_coherence,
        "total_coherence": coherence,
        "correlation_events": abs_summary['n_abstraction_events'],
    }


if __name__ == "__main__":
    result = run_experiment(n_ticks=4000, seed=42)
