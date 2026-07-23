"""EXP-SUB-003: Basin Topology & Plasticity — Geometry of Cognition

Measures how the landscape's structure evolves over time:
  - Basin depth: energy depth of each basin
  - Basin volume: estimated influence radius
  - Basin lifetime: how long each attractor persists
  - Merge/split frequency: structural reorganization events
  - Basin entropy: diversity of basin sizes
  - Persistence: resistance to perturbation

Plus plasticity properties on each attractor:
  - strength: current energy contribution
  - stability: resists change (increases with age + access)
  - plasticity: ability to shift center (decreases with stability)
  - novelty: decreases with time and repetition
  - confidence: increases with consistent evidence

These metrics describe the geometry of cognition, not just the quantity.
"""
import sys
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from substrate_echo.core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig
from substrate_echo.dynamics.basin_topology import BasinTopology, AttractorState


# ── Energy Landscape (same as EXP-SUB-002) ────────────────────────

class EnergyLandscape:
    def __init__(self, sigma=0.3):
        self.sigma = sigma
        self.attractors = []

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


# ── Convergence Detector ──────────────────────────────────────────

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


# ── Coupled Dynamics ──────────────────────────────────────────────

class CoupledDynamics:
    def __init__(self, dm, landscape, coupling_strength=0.3):
        self.dm = dm
        self.landscape = landscape
        self.lambda_ = coupling_strength

    def predict_velocity(self, x):
        v_learned = self.dm.predict_velocity(x)
        v_attractor = -self.landscape.gradient(x)
        return v_learned + self.lambda_ * v_attractor


# ── Trajectory Generation ─────────────────────────────────────────

def generate_diverse_trajectory(n_ticks=3000, seed=42):
    rng = np.random.RandomState(seed)
    state = np.full(16, 0.5)
    states, modes = [], []
    targets = {
        0: lambda: _make_target(0, [0, 14, 15], [0.8, 0.7, 0.6]),
        1: lambda: np.full(16, 0.3),
        2: lambda: _make_target(0, [7, 8, 9], [0.9, 0.7, 0.8]),
        3: lambda: _make_target(0, [1, 5, 10], [0.85, 0.8, 0.75]),
        4: lambda: _make_target(0, [3, 11, 12], [0.7, 0.8, 0.65]),
    }
    for tick in range(n_ticks):
        mode = (tick // 80) % 5
        target = targets[mode]()
        state = state + (target - state) * 0.1 + rng.randn(16) * 0.02
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


# ── Experiment ────────────────────────────────────────────────────

def run_experiment(n_ticks=3000, seed=42):
    print("=" * 70)
    print("EXP-SUB-003: Basin Topology & Plasticity")
    print("=" * 70)
    print(f"Ticks: {n_ticks}, Seed: {seed}")
    print()

    states, modes = generate_diverse_trajectory(n_ticks, seed)
    print(f"Generated {len(states)} states, {len(set(modes))} modes")

    # ── Setup coupled system ─────────────────────────────────────
    dm = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(
        model_type='global', min_samples_for_fit=50, max_samples=3000,
        attractor_samples=300, attractor_integration_steps=200))
    landscape = EnergyLandscape(sigma=0.3)
    detector = ConvergenceDetector(window_size=40, radius=0.12, min_cluster=6)
    coupled = CoupledDynamics(dm, landscape, coupling_strength=0.3)
    topology = BasinTopology(sigma=0.3)

    prev_state = None
    convergence_events = 0

    print("\n--- Running coupled dynamics with topology tracking ---")

    for tick in range(len(states)):
        state = states[tick]

        if prev_state is not None:
            velocity = state - prev_state
            dm._states.append(prev_state.copy())
            dm._velocities.append(velocity.copy())
            if len(dm._states) >= dm.config.min_samples_for_fit:
                if len(dm._states) % 50 == 0:
                    dm._fit_dynamics()

        detector.add(state)
        if tick % 100 == 0 and tick > 200:
            clusters = detector.detect()
            new_attractors = 0
            for center, size in clusters:
                is_new = True
                for existing in landscape.attractors:
                    if np.linalg.norm(center - existing[0]) < 0.2:
                        is_new = False
                        break
                if is_new:
                    strength = min(1.0, size / 20.0)
                    landscape.add_attractor(center, strength, tick)
                    topology.add_attractor(center, tick, strength)
                    new_attractors += 1
                    convergence_events += 1

            if new_attractors > 0:
                print(f"  tick {tick:4d}: +{new_attractors} attractor(s), "
                      f"total={len(landscape.attractors)}, "
                      f"convergences={convergence_events}")

        # Record topology snapshot every 200 ticks
        if tick % 200 == 0 and tick > 0:
            # Update plasticity: access attractors near current state
            for aid, astate in topology.attractors.items():
                dist = np.linalg.norm(state - astate.center)
                if dist < 0.3:
                    astate.access(tick)
                elif tick - astate.last_accessed_tick > 200:
                    astate.decay(tick)

            # Detect structural events
            if landscape.attractors:
                curr_centers = np.array([c for c, _, _ in landscape.attractors])
                topology.detect_events(curr_centers, tick)

            # Record snapshot
            topology.record_snapshot(tick)

        prev_state = state.copy()

    # ── Final measurements ───────────────────────────────────────
    print("\n--- Basin Topology Summary ---")
    summary = topology.summary()
    print(f"  Structural events: {summary.get('n_events', 0)}")
    print(f"    Births: {summary.get('births', 0)}")
    print(f"    Deaths: {summary.get('deaths', 0)}")
    print(f"    Merges: {summary.get('merges', 0)}")
    print(f"    Splits: {summary.get('splits', 0)}")

    # Final metrics
    final_metrics = topology.compute_metrics()
    print(f"\n  Final basin topology:")
    print(f"    Attractors: {final_metrics.n_attractors}")
    print(f"    Mean depth: {final_metrics.mean_depth:.4f}")
    print(f"    Mean volume: {final_metrics.mean_volume:.6f}")
    print(f"    Depth entropy: {final_metrics.depth_entropy:.3f}")
    print(f"    Volume entropy: {final_metrics.volume_entropy:.3f}")
    print(f"    Basin balance: {final_metrics.basin_balance:.3f}")
    print(f"    Total energy: {final_metrics.total_energy:.3f}")

    # ── Plasticity state of each attractor ───────────────────────
    print("\n--- Attractor Plasticity ---")
    for aid, astate in sorted(topology.attractors.items()):
        print(f"  A{aid:02d}: "
              f"strength={astate.strength:.3f} "
              f"stability={astate.stability:.3f} "
              f"plasticity={astate.plasticity:.3f} "
              f"novelty={astate.novelty:.3f} "
              f"confidence={astate.confidence:.3f} "
              f"accesses={astate.access_count}")

    # ── Coherence comparison ─────────────────────────────────────
    final_centers = [c for c, _, _ in landscape.attractors]
    if final_centers:
        assign = assign_to_attractors(states, final_centers)
        coherence = compute_coherence(assign, modes)
    else:
        coherence = 0.0

    print(f"\n--- Final Coherence: {coherence:.3f} ---")

    # ── Topology evolution over time ─────────────────────────────
    print("\n--- Topology Evolution ---")
    for tick, metrics in topology.history:
        print(f"  tick {tick:4d}: "
              f"attractors={metrics.n_attractors:3d} "
              f"depth={metrics.mean_depth:.4f} "
              f"entropy={metrics.volume_entropy:.3f} "
              f"balance={metrics.basin_balance:.3f}")

    print("\n" + "=" * 70)
    print("RESULT: Basin topology tracked. Plasticity properties active.")
    print("=" * 70)

    return {
        "final_attractors": final_metrics.n_attractors,
        "final_coherence": coherence,
        "mean_depth": final_metrics.mean_depth,
        "volume_entropy": final_metrics.volume_entropy,
        "basin_balance": final_metrics.basin_balance,
        "n_events": summary.get('n_events', 0),
        "n_births": summary.get('births', 0),
        "n_deaths": summary.get('deaths', 0),
        "n_merges": summary.get('merges', 0),
        "n_splits": summary.get('splits', 0),
        "attractor_plasticity": {
            aid: {
                "stability": a.stability,
                "plasticity": a.plasticity,
                "novelty": a.novelty,
                "confidence": a.confidence,
            }
            for aid, a in topology.attractors.items()
        },
    }


if __name__ == "__main__":
    result = run_experiment(n_ticks=3000, seed=42)
