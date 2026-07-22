"""Social Learning Experiments — agents share world models.

1. Knowledge sharing: agents learn faster by sharing observations
2. Emergent specialization: agents explore different regions
3. Collective intelligence: group knowledge exceeds individual knowledge
"""

import sys
import time
import numpy as np

sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.dynamics_memory import DynamicsMemory
from substrate_echo.core.world_model import WorldModel


# ── Experiment 1: Knowledge Sharing ──────────────────────────────

def experiment_knowledge_sharing():
    """Two agents learn faster by sharing than alone.

    Agent A learns region [0.4, 0.6].
    Agent B learns region [0.0, 0.2].
    After sharing, both should predict both regions better.
    """
    print("\n=== Experiment 1: Knowledge Sharing ===")
    rng = np.random.RandomState(42)
    dim = 16

    # True dynamics: two basins
    def dynamics_a(x):
        """Basin at 0.5 (center)."""
        return -0.8 * (x - 0.5)

    def dynamics_b(x):
        """Basin at 0.1 (edge)."""
        return 0.3 * (x - 0.1)

    def true_dynamics(x):
        """Mix: center for x>0.3, edge for x<0.3."""
        return np.where(x > 0.3, dynamics_a(x), dynamics_b(x))

    # Agent A: trains on center only
    wm_a = WorldModel(DynamicsMemory(dim=dim))
    for _ in range(300):
        x = rng.uniform(0.35, 0.65, dim)
        wm_a.memory._states.append(x)
        wm_a.memory._velocities.append(dynamics_a(x) + rng.randn(dim) * 0.005)
    wm_a.memory._fit_dynamics()

    # Agent B: trains on edge only
    wm_b = WorldModel(DynamicsMemory(dim=dim))
    for _ in range(300):
        x = rng.uniform(0.0, 0.25, dim)
        wm_b.memory._states.append(x)
        wm_b.memory._velocities.append(dynamics_b(x) + rng.randn(dim) * 0.005)
    wm_b.memory._fit_dynamics()

    # Measure individual performance
    def measure_error(wm, region, dynamics, n=100):
        errors = []
        for _ in range(n):
            x = rng.uniform(*region, dim)
            error = wm.memory.prediction_error(x, dynamics(x))
            errors.append(error)
        return np.mean(errors)

    # Before sharing
    a_center_err = measure_error(wm_a, (0.35, 0.65), dynamics_a)
    a_edge_err = measure_error(wm_a, (0.0, 0.25), dynamics_b)
    b_center_err = measure_error(wm_b, (0.35, 0.65), dynamics_a)
    b_edge_err = measure_error(wm_b, (0.0, 0.25), dynamics_b)

    print(f"  Before sharing:")
    print(f"    Agent A: center={a_center_err:.6f}, edge={a_edge_err:.6f}")
    print(f"    Agent B: center={b_center_err:.6f}, edge={b_edge_err:.6f}")

    # Share
    imported_a = wm_a.share_observations(wm_b)
    imported_b = wm_b.share_observations(wm_a)

    # After sharing
    a_center_err2 = measure_error(wm_a, (0.35, 0.65), dynamics_a)
    a_edge_err2 = measure_error(wm_a, (0.0, 0.25), dynamics_b)
    b_center_err2 = measure_error(wm_b, (0.35, 0.65), dynamics_a)
    b_edge_err2 = measure_error(wm_b, (0.0, 0.25), dynamics_b)

    print(f"\n  After sharing:")
    print(f"    Imported: A<={imported_a}, B<={imported_b}")
    print(f"    Agent A: center={a_center_err2:.6f}, edge={a_edge_err2:.6f}")
    print(f"    Agent B: center={b_center_err2:.6f}, edge={b_edge_err2:.6f}")

    a_improved = a_edge_err2 < a_edge_err
    b_improved = b_center_err2 < b_center_err

    print(f"\n  Agent A edge improvement: {'YES' if a_improved else 'NO'} "
          f"({a_edge_err:.6f} -> {a_edge_err2:.6f})")
    print(f"  Agent B center improvement: {'YES' if b_improved else 'NO'} "
          f"({b_center_err:.6f} -> {b_center_err2:.6f})")

    return {
        "imported_a": imported_a, "imported_b": imported_b,
        "a_improved": a_improved, "b_improved": b_improved,
    }


# ── Experiment 2: Emergent Specialization ────────────────────────

def experiment_specialization():
    """Multiple agents explore different regions, emerge as specialists.

    3 agents start at random positions. Each explores using curiosity.
    After 1000 ticks, measure which regions each agent knows best.
    """
    print("\n=== Experiment 2: Emergent Specialization ===")
    rng = np.random.RandomState(42)
    dim = 16
    n_agents = 3
    n_ticks = 1000

    # True dynamics: 3 basins at different locations
    basin_centers = [
        np.full(dim, 0.2),  # basin A
        np.full(dim, 0.5),  # basin B
        np.full(dim, 0.8),  # basin C
    ]

    def true_dynamics(x):
        # Nearest basin determines dynamics
        dists = [np.linalg.norm(x - c) for c in basin_centers]
        nearest = np.argmin(dists)
        return -0.5 * (x - basin_centers[nearest])

    # Create agents
    agents = []
    for i in range(n_agents):
        dm = DynamicsMemory(dim=dim)
        wm = WorldModel(dm)
        agents.append({
            "wm": wm,
            "explored_regions": set(),
            "tick_count": 0,
        })

    # Simulate exploration
    for tick in range(n_ticks):
        for i, agent in enumerate(agents):
            # Each agent explores its current region + some noise
            if agent["tick_count"] == 0:
                # Start at random position
                x = rng.uniform(0.0, 1.0, dim)
            else:
                # Move toward highest novelty region
                candidates = [rng.uniform(0.0, 1.0, dim) for _ in range(10)]
                novelties = [agent["wm"].memory.novelty(c) if agent["wm"].memory._states else 1.0
                             for c in candidates]
                x = candidates[np.argmax(novelties)]

            # Observe dynamics
            v = true_dynamics(x) + rng.randn(dim) * 0.005

            # Learn
            agent["wm"].memory._states.append(x.copy())
            agent["wm"].memory._velocities.append(v.copy())
            agent["tick_count"] += 1

            # Fit every 100 ticks
            if agent["tick_count"] % 100 == 0:
                agent["wm"].memory._fit_dynamics()

            # Track which basin this agent is near
            dists = [np.linalg.norm(x - c) for c in basin_centers]
            nearest = np.argmin(dists)
            agent["explored_regions"].add(nearest)

    # Measure specialization
    print(f"  Agents explored {n_ticks} ticks each")
    for i, agent in enumerate(agents):
        agent["wm"].memory._fit_dynamics()
        regions = agent["explored_regions"]
        n_samples = len(agent["wm"].memory._states)

        # Measure error in each basin
        errors = []
        for b, center in enumerate(basin_centers):
            err = np.mean([
                agent["wm"].memory.prediction_error(
                    rng.uniform(0.0, 1.0, dim),
                    true_dynamics(rng.uniform(0.0, 1.0, dim)))
                for _ in range(30)])
            errors.append(err)

        best_basin = np.argmin(errors)
        print(f"  Agent {i}: {n_samples} samples, "
              f"best basin={best_basin} ({errors[best_basin]:.6f}), "
              f"explored regions={[r for r in regions]}")

    # Check if agents specialized (different best basins)
    best_basins = []
    for agent in agents:
        errors = []
        for center in basin_centers:
            err = np.mean([
                agent["wm"].memory.prediction_error(
                    rng.uniform(0.0, 1.0, dim),
                    true_dynamics(rng.uniform(0.0, 1.0, dim)))
                for _ in range(30)])
            errors.append(err)
        best_basins.append(np.argmin(errors))

    n_unique = len(set(best_basins))
    print(f"\n  Unique specializations: {n_unique}/{n_agents}")
    print(f"  Specialization: {'YES' if n_unique > 1 else 'NO'}")

    return {"n_unique": n_unique, "best_basins": best_basins}


# ── Experiment 3: Collective Intelligence ────────────────────────

def experiment_collective_intelligence():
    """Group of 5 agents, each learning locally, then sharing.

    Compare: individual error vs collective error after sharing.
    """
    print("\n=== Experiment 3: Collective Intelligence ===")
    rng = np.random.RandomState(42)
    dim = 16
    n_agents = 5

    # True dynamics: complex landscape with 5 basins
    basin_centers = [rng.uniform(0.2, 0.8, dim) for _ in range(n_agents)]

    def true_dynamics(x):
        dists = [np.linalg.norm(x - c) for c in basin_centers]
        nearest = np.argmin(dists)
        return -0.5 * (x - basin_centers[nearest])

    # Each agent learns from a random slice of the space
    agents = []
    for i in range(n_agents):
        dm = DynamicsMemory(dim=dim)
        wm = WorldModel(dm)
        # Learn from a random region
        center = rng.uniform(0.2, 0.8, dim)
        for _ in range(100):
            x = center + rng.randn(dim) * 0.15
            x = np.clip(x, 0.0, 1.0)
            wm.memory._states.append(x)
            wm.memory._velocities.append(true_dynamics(x) + rng.randn(dim) * 0.01)
        wm.memory._fit_dynamics()
        agents.append(wm)

    # Measure individual errors
    individual_errors = []
    for wm in agents:
        err = np.mean([
            wm.memory.prediction_error(rng.uniform(0.0, 1.0, dim),
                                        true_dynamics(rng.uniform(0.0, 1.0, dim)))
            for _ in range(100)])
        individual_errors.append(err)

    avg_individual = np.mean(individual_errors)

    # Share: each agent shares with all others
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                agents[i].share_observations(agents[j], max_samples=100)

    # Measure collective errors
    collective_errors = []
    for wm in agents:
        err = np.mean([
            wm.memory.prediction_error(rng.uniform(0.0, 1.0, dim),
                                        true_dynamics(rng.uniform(0.0, 1.0, dim)))
            for _ in range(100)])
        collective_errors.append(err)

    avg_collective = np.mean(collective_errors)
    improvement = (avg_individual - avg_collective) / avg_individual * 100

    print(f"  Individual avg error: {avg_individual:.6f}")
    print(f"  Collective avg error: {avg_collective:.6f}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Collective > Individual: {'YES' if avg_collective < avg_individual else 'NO'}")

    return {
        "individual": avg_individual,
        "collective": avg_collective,
        "improvement_pct": improvement,
    }


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Social Learning Experiments")
    print("=" * 60)

    experiment_knowledge_sharing()
    experiment_specialization()
    experiment_collective_intelligence()

    print("\n" + "=" * 60)
    print("All social learning experiments complete.")
    print("=" * 60)
