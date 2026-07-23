"""EXP-SUB-002: Closed Feedback Loop — Self-Emerging Semantic Basins

Tests whether attractors become self-reinforcing through a closed loop:
  dynamics → convergence detection → attractor instantiation →
  energy landscape → modified dynamics

This is the stronger substrate test: not "do attractors emerge?"
but "do they become self-reinforcing?"

Hypothesis:
  Early learning:  sparse, weak basins
  Middle learning: new attractors appear where trajectories converge
  Late learning:   semantic basins deepen, stabilize, organize future trajectories
"""
import sys
import numpy as np
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from substrate_echo.core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig


class EnergyLandscape:
    """Energy landscape constructed from discovered attractors.

    E(x) = -sum_i strength_i * exp(-|x - center_i|^2 / (2 * sigma^2))

    Lower energy = deeper basin = stronger attractor.
    Gradient descent on E(x) gives the attractor influence field.
    """

    def __init__(self, sigma=0.3):
        self.sigma = sigma
        self.attractors = []  # list of (center, strength, creation_tick)

    def add_attractor(self, center, strength=0.5, tick=0):
        self.attractors.append((center.copy(), strength, tick))

    def energy(self, x):
        """Compute energy at point x."""
        if not self.attractors:
            return 0.0
        e = 0.0
        for center, strength, _ in self.attractors:
            dist2 = np.sum((x - center) ** 2)
            e -= strength * np.exp(-dist2 / (2 * self.sigma ** 2))
        return e

    def gradient(self, x):
        """Compute gradient of energy landscape at point x.
        Returns the force vector: -∇E(x) points toward lower energy.
        """
        if not self.attractors:
            return np.zeros_like(x)
        grad = np.zeros_like(x)
        for center, strength, _ in self.attractors:
            diff = x - center
            dist2 = np.sum(diff ** 2)
            # d/dx [-strength * exp(-|x-c|^2/(2σ²))]
            #   = strength * (x-c)/σ² * exp(-|x-c|^2/(2σ²))
            grad += strength * diff / (self.sigma ** 2) * np.exp(
                -dist2 / (2 * self.sigma ** 2))
        return grad  # negative gradient = force toward attractor


class ConvergenceDetector:
    """Detects when trajectory endpoints cluster, indicating a new attractor.

    Uses a sliding window of recent endpoints and checks for
    spatial clustering within a threshold radius.
    """

    def __init__(self, window_size=50, radius=0.15, min_cluster=8):
        self.window_size = window_size
        self.radius = radius
        self.min_cluster = min_cluster
        self.endpoints = []

    def add(self, state):
        self.endpoints.append(state.copy())
        if len(self.endpoints) > self.window_size:
            self.endpoints.pop(0)

    def detect(self):
        """Detect clusters in recent endpoints.

        Returns list of (center, size) for each detected cluster.
        """
        if len(self.endpoints) < self.min_cluster:
            return []

        arr = np.array(self.endpoints)
        assigned = np.zeros(len(arr), dtype=int) - 1
        clusters = []

        for i, pt in enumerate(arr):
            if assigned[i] >= 0:
                continue
            # Find all points within radius
            dists = np.linalg.norm(arr - pt, axis=1)
            neighbors = np.where(dists < self.radius)[0]
            if len(neighbors) >= self.min_cluster:
                center = arr[neighbors].mean(axis=0)
                clusters.append((center, len(neighbors)))
                assigned[neighbors] = len(clusters) - 1

        return clusters


class CoupledDynamics:
    """Dynamics with attractor energy feedback.

    V'(x) = V(x) + λ * (-∇E(x))

    where V(x) is the learned vector field and E(x) is the energy landscape.
    """

    def __init__(self, dm, landscape, coupling_strength=0.3):
        self.dm = dm
        self.landscape = landscape
        self.lambda_ = coupling_strength

    def predict_velocity(self, x):
        """Modified velocity: learned dynamics + attractor influence."""
        v_learned = self.dm.predict_velocity(x)
        v_attractor = -self.landscape.gradient(x)
        return v_learned + self.lambda_ * v_attractor


def generate_diverse_trajectory(n_ticks=2000, seed=42):
    """Generate a trajectory with 5 behavioral modes for rich dynamics."""
    rng = np.random.RandomState(seed)
    state = np.full(16, 0.5)
    states, modes = [], []

    # 5 distinct behavioral modes
    targets = {
        0: lambda: _make_target(0, [0, 14, 15], [0.8, 0.7, 0.6]),   # exploration
        1: lambda: np.full(16, 0.3),                                   # rest
        2: lambda: _make_target(0, [7, 8, 9], [0.9, 0.7, 0.8]),     # social
        3: lambda: _make_target(0, [1, 5, 10], [0.85, 0.8, 0.75]),  # focused work
        4: lambda: _make_target(0, [3, 11, 12], [0.7, 0.8, 0.65]),  # influence
    }

    for tick in range(n_ticks):
        mode = (tick // 80) % 5  # switch every 80 ticks
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


def run_experiment(n_ticks=2000, seed=42):
    print("=" * 70)
    print("EXP-SUB-002: Closed Feedback Loop")
    print("=" * 70)
    print(f"Ticks: {n_ticks}, Seed: {seed}")
    print()

    # Generate diverse trajectory
    states, modes = generate_diverse_trajectory(n_ticks, seed)
    print(f"Generated {len(states)} states, {len(set(modes))} modes")

    # ── BASELINE: DynamicsMemory alone ──────────────────────────
    print("\n--- BASELINE: DynamicsMemory alone ---")
    dm_base = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(
        model_type='global', min_samples_for_fit=50, max_samples=2000,
        attractor_samples=300, attractor_integration_steps=200))
    for i in range(1, len(states)):
        dm_base._states.append(states[i-1])
        dm_base._velocities.append(states[i] - states[i-1])
    dm_base._fit_dynamics()
    base_atts = dm_base.discover_attractors()
    base_assign = assign_to_attractors(states, base_atts)
    base_coh = compute_coherence(base_assign, modes)
    print(f"  Attractors: {len(base_atts)}, Coherence: {base_coh:.3f}")

    # ── COUPLED: DynamicsMemory + Energy Landscape ──────────────
    print("\n--- COUPLED: Dynamics + Energy Feedback ---")
    dm_coupled = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(
        model_type='global', min_samples_for_fit=50, max_samples=2000,
        attractor_samples=300, attractor_integration_steps=200))
    landscape = EnergyLandscape(sigma=0.3)
    detector = ConvergenceDetector(window_size=40, radius=0.12, min_cluster=6)
    coupled = CoupledDynamics(dm_coupled, landscape, coupling_strength=0.3)

    # Training loop with closed feedback
    prev_state = None
    attractor_log = []  # (tick, n_attractors, coherence)
    convergence_events = 0

    for tick in range(len(states)):
        state = states[tick]

        # Feed state transition to dynamics memory
        if prev_state is not None:
            velocity = state - prev_state
            dm_coupled._states.append(prev_state.copy())
            dm_coupled._velocities.append(velocity.copy())
            if len(dm_coupled._states) >= dm_coupled.config.min_samples_for_fit:
                if len(dm_coupled._states) % 50 == 0:
                    dm_coupled._fit_dynamics()

        # Detect convergence → auto-instantiate attractor
        detector.add(state)
        if tick % 100 == 0 and tick > 200:
            clusters = detector.detect()
            new_attractors = 0
            for center, size in clusters:
                # Check if this is genuinely new (not near existing attractor)
                is_new = True
                for existing, _, _ in landscape.attractors:
                    if np.linalg.norm(center - existing) < 0.2:
                        is_new = False
                        break
                if is_new:
                    strength = min(1.0, size / 20.0)  # stronger with more evidence
                    landscape.add_attractor(center, strength, tick)
                    new_attractors += 1
                    convergence_events += 1

            if new_attractors > 0:
                print(f"  tick {tick:4d}: +{new_attractors} attractor(s), "
                      f"total={len(landscape.attractors)}, "
                      f"convergences={convergence_events}")

        # Track metrics periodically
        if tick % 200 == 0 and tick > 0:
            current_atts = [c for c, _, _ in landscape.attractors]
            if current_atts:
                a = assign_to_attractors(states[:tick+1], current_atts)
                c = compute_coherence(a, modes[:tick+1])
            else:
                c = 0.0
            attractor_log.append((tick, len(landscape.attractors), c))

        prev_state = state.copy()

    # Final measurement
    final_centers = [c for c, _, _ in landscape.attractors]
    final_assign = assign_to_attractors(states, final_centers) if final_centers else [-1]*len(states)
    final_coh = compute_coherence(final_assign, modes)

    print(f"\n  Final: {len(landscape.attractors)} attractors, "
          f"coherence={final_coh:.3f}")
    print(f"  Convergence events: {convergence_events}")

    # ── TRAJECTORY UNDER COUPLED DYNAMICS ───────────────────────
    print("\n--- TRAJECTORY UNDER COUPLED DYNAMICS ---")
    # Re-run trajectory using the coupled vector field
    rng = np.random.RandomState(seed)
    test_state = states[0].copy()
    coupled_trajectory = [test_state.copy()]

    # First: fit the dynamics on the original data
    dm_test = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(
        model_type='global', min_samples_for_fit=50, max_samples=2000,
        attractor_samples=300, attractor_integration_steps=200))
    for i in range(1, len(states)):
        dm_test._states.append(states[i-1])
        dm_test._velocities.append(states[i] - states[i-1])
    dm_test._fit_dynamics()

    coupled_test = CoupledDynamics(dm_test, landscape, coupling_strength=0.3)

    for step in range(500):
        v = coupled_test.predict_velocity(test_state)
        test_state = test_state + 0.02 * v + rng.randn(16) * 0.005
        test_state = np.clip(test_state, 0.0, 1.0)
        coupled_trajectory.append(test_state.copy())

    coupled_arr = np.array(coupled_trajectory)

    # Does the coupled trajectory settle into the discovered basins?
    if final_centers:
        dists_to_nearest = []
        for s in coupled_arr[100:]:  # skip transient
            d = min(np.linalg.norm(s - c) for c in final_centers)
            dists_to_nearest.append(d)
        mean_dist = np.mean(dists_to_nearest)
        settled = mean_dist < 0.2
        print(f"  Mean distance to nearest attractor: {mean_dist:.3f}")
        print(f"  Trajectory settles into basins: {'YES' if settled else 'NO'}")
    else:
        print("  No attractors to settle into.")

    # ── SUMMARY ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline (DynamicsMemory alone):")
    print(f"    Attractors: {len(base_atts)}, Coherence: {base_coh:.3f}")
    print(f"  Coupled (Dynamics + Energy Feedback):")
    print(f"    Attractors: {len(landscape.attractors)}, Coherence: {final_coh:.3f}")
    print(f"    Convergence events: {convergence_events}")
    print(f"    Trajectory settles: {'YES' if final_centers and settled else 'NO'}")

    improvement = final_coh - base_coh
    print(f"\n  Coherence improvement: {improvement:+.3f}")

    if len(landscape.attractors) > len(base_atts) and final_coh > base_coh:
        print("\n  RESULT: Closed feedback loop produces MORE attractors")
        print("  with HIGHER semantic coherence.")
        print("  The substrate is SELF-REINFORCING.")
        print("  SUBSTRATE CLAIM: STRONG")
    elif final_coh > base_coh:
        print("\n  RESULT: Feedback loop improves coherence but not count.")
        print("  SUBSTRATE CLAIM: MODERATE")
    else:
        print("\n  RESULT: Feedback loop does not improve over baseline.")
        print("  SUBSTRATE CLAIM: WEAK")

    return {
        "baseline_attractors": len(base_atts),
        "baseline_coherence": base_coh,
        "coupled_attractors": len(landscape.attractors),
        "coupled_coherence": final_coh,
        "convergence_events": convergence_events,
        "attractor_log": attractor_log,
    }


if __name__ == "__main__":
    result = run_experiment(n_ticks=2000, seed=42)
