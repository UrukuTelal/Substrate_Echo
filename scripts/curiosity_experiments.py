"""Curiosity Experiments — demonstrating self-extinguishing learning.

1. Curiosity Decay: visit unknown region, retrain, measure error decrease
2. Environmental Change: move attractor, measure spike and recovery
3. Information Gain Planning: curiosity as control signal for goal generation
"""

import sys
import time
import numpy as np

sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.dynamics_memory import DynamicsMemory


# ── Experiment 1: Curiosity Decay ────────────────────────────────

def experiment_curiosity_decay():
    """Visit unknown region repeatedly, retrain, measure prediction error.

    Expected curve:
        Visit 1:   error ~0.9
        Visit 5:   error ~0.4
        Visit 10:  error ~0.2
        Visit 20:  error ~0.07
        Visit 50:  error ~0.02

    If the curve decreases monotonically, curiosity is self-extinguishing.
    """
    print("\n=== Experiment 1: Curiosity Decay ===")
    rng = np.random.RandomState(42)
    dim = 16

    # Train on KNOWN dynamics: v = -0.5*(x - 0.5) in region [0.3, 0.7]
    dm = DynamicsMemory(dim=dim)
    for _ in range(500):
        x = rng.uniform(0.3, 0.7, dim)
        v = -0.5 * (x - 0.5)
        dm._states.append(x)
        dm._velocities.append(v)
    dm._fit_dynamics()

    # Define UNKNOWN region: [0.0, 0.2] x 16
    # Different dynamics: v = +0.3*(x - 0.1) (diverges from 0.1)
    unknown_center = np.full(dim, 0.1)
    unknown_dynamics = lambda x: 0.3 * (x - 0.1)

    # Measure baseline error in unknown region BEFORE visiting
    baseline_errors = []
    for _ in range(50):
        x = rng.uniform(0.0, 0.2, dim)
        actual_v = unknown_dynamics(x)
        error = dm.prediction_error(x, actual_v)
        baseline_errors.append(error)
    baseline_avg = np.mean(baseline_errors)

    # Visit unknown region repeatedly, collect data, retrain
    visit_counts = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
    errors_per_visit = []

    for n_visits in visit_counts:
        # Add n_visits new samples from unknown region
        for _ in range(n_visits):
            x = rng.uniform(0.0, 0.2, dim)
            v = unknown_dynamics(x) + rng.randn(dim) * 0.01  # small noise
            dm._states.append(x)
            dm._velocities.append(v)

        # Retrain
        dm._fit_dynamics()

        # Measure error in unknown region
        errors = []
        for _ in range(50):
            x = rng.uniform(0.0, 0.2, dim)
            actual_v = unknown_dynamics(x)
            error = dm.prediction_error(x, actual_v)
            errors.append(error)
        avg_error = np.mean(errors)
        errors_per_visit.append(avg_error)

    # Print curve
    print(f"  Baseline error (before any visits): {baseline_avg:.6f}")
    print(f"  {'Visits':>8} {'Error':>10} {'Reduction':>10}")
    print(f"  {'--------':>8} {'----------':>10} {'----------':>10}")
    for n, err in zip(visit_counts, errors_per_visit):
        reduction = (baseline_avg - err) / baseline_avg * 100
        print(f"  {n:>8} {err:>10.6f} {reduction:>9.1f}%")

    # Check monotonic decrease
    is_monotonic = all(errors_per_visit[i] >= errors_per_visit[i+1]
                       for i in range(len(errors_per_visit) - 1))
    total_reduction = (baseline_avg - errors_per_visit[-1]) / baseline_avg

    print(f"\n  Monotonic decrease: {'YES' if is_monotonic else 'NO'}")
    print(f"  Total reduction: {total_reduction:.1%}")
    print(f"  Self-extinguishing: {'YES' if is_monotonic and total_reduction > 0.5 else 'NO'}")

    return {
        "baseline_error": baseline_avg,
        "visit_counts": visit_counts,
        "errors": errors_per_visit,
        "monotonic": is_monotonic,
        "total_reduction": total_reduction,
    }


# ── Experiment 2: Environmental Change ───────────────────────────

def experiment_environmental_change():
    """Move attractor after world is learned, measure spike and recovery.

    Expected:
        Phase 1 (learn): error falls to ~0
        Phase 2 (change): error spikes
        Phase 3 (relearn): error falls again

    This demonstrates adaptation, not just memorization.
    """
    print("\n=== Experiment 2: Environmental Change ===")
    rng = np.random.RandomState(123)
    dim = 16

    dm = DynamicsMemory(dim=dim)

    # Phase 1: Learn original dynamics
    # v = -0.8*(x - 0.5) — converges to 0.5
    original_dynamics = lambda x: -0.8 * (x - 0.5)

    print("  Phase 1: Learning original dynamics...")
    phase1_errors = []
    for episode in range(20):
        for _ in range(50):
            x = rng.uniform(0.1, 0.9, dim)
            v = original_dynamics(x) + rng.randn(dim) * 0.005
            dm._states.append(x)
            dm._velocities.append(v)
        dm._fit_dynamics()

        # Measure error
        errors = []
        for _ in range(100):
            x = rng.uniform(0.1, 0.9, dim)
            actual_v = original_dynamics(x)
            error = dm.prediction_error(x, actual_v)
            errors.append(error)
        phase1_errors.append(np.mean(errors))

    print(f"    Episode 1 error: {phase1_errors[0]:.6f}")
    print(f"    Episode 20 error: {phase1_errors[-1]:.6f}")

    # Phase 2: ENVIRONMENT CHANGES — attractor moves from 0.5 to 0.2
    new_attractor = 0.2
    changed_dynamics = lambda x: -0.8 * (x - new_attractor)

    print(f"\n  Phase 2: Attractor moved from 0.5 to {new_attractor}...")
    phase2_errors = []
    for episode in range(20):
        for _ in range(50):
            x = rng.uniform(0.1, 0.9, dim)
            v = changed_dynamics(x) + rng.randn(dim) * 0.005
            dm._states.append(x)
            dm._velocities.append(v)
        dm._fit_dynamics()

        errors = []
        for _ in range(100):
            x = rng.uniform(0.1, 0.9, dim)
            actual_v = changed_dynamics(x)
            error = dm.prediction_error(x, actual_v)
            errors.append(error)
        phase2_errors.append(np.mean(errors))

    spike = phase2_errors[0]
    recovery = phase2_errors[-1]
    pre_change = phase1_errors[-1]

    print(f"    Pre-change error: {pre_change:.6f}")
    print(f"    Post-change spike: {spike:.6f} ({spike/pre_change:.1f}x)")
    print(f"    After relearning: {recovery:.6f}")

    # Check spike
    has_spike = spike > pre_change * 2
    has_recovery = recovery < spike * 0.3
    adapts = has_spike and has_recovery

    print(f"\n  Error spike: {'YES' if has_spike else 'NO'} ({spike/pre_change:.1f}x)")
    print(f"  Recovery: {'YES' if has_recovery else 'NO'}")
    print(f"  Adaptation: {'YES' if adapts else 'NO'}")

    return {
        "phase1_errors": phase1_errors,
        "phase2_errors": phase2_errors,
        "spike_ratio": spike / pre_change,
        "recovery_ratio": recovery / spike,
        "adapts": adapts,
    }


# ── Experiment 3: Information Gain Planning ──────────────────────

def experiment_information_gain():
    """Demonstrate curiosity as a control signal.

    Agent has two choices:
    A) Visit known region (low error, low information gain)
    B) Visit unknown region (high error, high information gain)

    An information-seeking agent should prefer B.
    Uses novelty (distance to training data) instead of prediction variance.
    """
    print("\n=== Experiment 3: Information Gain Planning ===")
    rng = np.random.RandomState(456)
    dim = 16

    dm = DynamicsMemory(dim=dim)

    # Train on region A (center)
    for _ in range(300):
        x = rng.uniform(0.4, 0.6, dim)
        v = -0.5 * (x - 0.5) + rng.randn(dim) * 0.005
        dm._states.append(x)
        dm._velocities.append(v)
    dm._fit_dynamics()

    # Region B (edge) is unknown
    region_a_center = np.full(dim, 0.5)
    region_b_center = np.full(dim, 0.1)

    # Compute information gain for each region using novelty
    def estimate_info_gain(center, n_samples=100):
        gains = []
        for _ in range(n_samples):
            x = center + rng.randn(dim) * 0.1
            x = np.clip(x, 0.0, 1.0)
            gain = dm.information_gain(x)
            gains.append(gain)
        return np.mean(gains)

    info_gain_a = estimate_info_gain(region_a_center)
    info_gain_b = estimate_info_gain(region_b_center)

    # Also measure actual prediction error
    def estimate_true_error(center, true_dynamics, n_samples=100):
        errors = []
        for _ in range(n_samples):
            x = center + rng.randn(dim) * 0.1
            x = np.clip(x, 0.0, 1.0)
            actual_v = true_dynamics(x)
            error = dm.prediction_error(x, actual_v)
            errors.append(error)
        return np.mean(errors)

    true_dynamics_b = lambda x: 0.3 * (x - 0.1)
    true_error_a = estimate_true_error(region_a_center, lambda x: -0.5 * (x - 0.5))
    true_error_b = estimate_true_error(region_b_center, true_dynamics_b)

    # Also show raw novelty scores
    def estimate_novelty(center, n_samples=100):
        novelties = []
        for _ in range(n_samples):
            x = center + rng.randn(dim) * 0.1
            x = np.clip(x, 0.0, 1.0)
            nov = dm.novelty(x)
            novelties.append(nov)
        return np.mean(novelties)

    nov_a = estimate_novelty(region_a_center)
    nov_b = estimate_novelty(region_b_center)

    print(f"  Region A (known):")
    print(f"    Novelty:          {nov_a:.6f}")
    print(f"    Info gain:        {info_gain_a:.6f}")
    print(f"    True pred error:  {true_error_a:.6f}")
    print(f"  Region B (unknown):")
    print(f"    Novelty:          {nov_b:.6f}")
    print(f"    Info gain:        {info_gain_b:.6f}")
    print(f"    True pred error:  {true_error_b:.6f}")
    print(f"  Ratio (B/A):")
    print(f"    Novelty:   {nov_b/max(nov_a,1e-10):.1f}x")
    print(f"    Info gain: {info_gain_b/max(info_gain_a,1e-10):.1f}x")
    print(f"    True error: {true_error_b/max(true_error_a,1e-10):.1f}x")

    prefers_b = info_gain_b > info_gain_a
    actually_b = true_error_b > true_error_a

    print(f"\n  Agent prefers B (info gain): {'YES' if prefers_b else 'NO'}")
    print(f"  B actually more informative: {'YES' if actually_b else 'NO'}")
    print(f"  Aligned: {'YES' if prefers_b == actually_b else 'NO'}")

    return {
        "novelty_a": nov_a,
        "novelty_b": nov_b,
        "info_gain_a": info_gain_a,
        "info_gain_b": info_gain_b,
        "true_error_a": true_error_a,
        "true_error_b": true_error_b,
        "prefers_b": prefers_b,
        "aligned": prefers_b == actually_b,
    }


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Curiosity Experiments")
    print("=" * 60)

    experiment_curiosity_decay()
    experiment_environmental_change()
    experiment_information_gain()

    print("\n" + "=" * 60)
    print("All curiosity experiments complete.")
    print("=" * 60)
