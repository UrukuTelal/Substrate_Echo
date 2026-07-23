"""EXP-SUB-001: Substrate Emergence Test

Core question: Can Substrate_Echo remember something it was never told to remember?

Method:
1. Run an IntegratedAgent with DynamicsMemory (NO AttractorMemory)
2. Feed it experiences through the normal cognitive loop
3. After training, call discover_attractors() on the learned vector field
4. Measure whether stable semantic clusters emerge WITHOUT form_attractor()

If attractors emerge from the learned dynamics alone, the substrate claim is strong.
If no attractors emerge, the AttractorMemory layer is doing the cognitive heavy lifting.
"""
import sys
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from substrate_echo.core.integrated_agent import IntegratedAgent, IntegratedAgentConfig
from substrate_echo.core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig
from substrate_echo.core.world import World, WorldConfig
from substrate_echo.core.meta_cognition import MetaCognition


def run_emergence_experiment(n_ticks=500, n_agents=3, seed=42, model_type="global"):
    """Run the substrate emergence experiment.
    
    Returns a dict with:
    - attractors_discovered: number of attractors found
    - attractor_positions: list of 16D attractor positions
    - state_clusters: which states gravitate toward which attractors
    - trajectory_diversity: how varied were the agent's experiences
    - semantic_coherence: do attractors correspond to distinct experience types
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("EXP-SUB-001: Substrate Emergence Test")
    print("=" * 70)
    print(f"Ticks: {n_ticks}, Agents: {n_agents}")
    print("CRITICAL: No AttractorMemory. Only DynamicsMemory.")
    print()
    
    # ── Setup World ──────────────────────────────────────────────
    world_config = WorldConfig(
        grid_size=10.0,
        n_resources=30,
        storm_interval=500,
    )
    world = World(world_config)
    
    # Add agents to world
    agent_positions = []
    for i in range(n_agents):
        pos = np.random.uniform(1, 9, 2)
        world.add_agent(i, position=pos)
        agent_positions.append(pos)
    
    # ── Setup Agent with DynamicsMemory ONLY ────────────────────
    agent_config = IntegratedAgentConfig(
        context_dim=16,
        curiosity_drive=0.4,
        social_drive=0.2,
        survival_drive=0.4,
    )
    
    # DynamicsMemory with local linear model (can discover multi-basin)
    dm_config = DynamicsMemoryConfig(
        model_type=model_type,
        k_neighbors=20,
        bandwidth=0.5,
        min_samples_for_fit=30,
        max_samples=1000,
        attractor_samples=200,
        attractor_integration_steps=100,
        dedup_threshold=0.3,
    )
    dm = DynamicsMemory(dim=16, config=dm_config)
    
    # Create agent
    agent = IntegratedAgent(
        agent_id=0,
        config=agent_config,
    )
    
    # Replace the agent's default DynamicsMemory with our configured one
    # This is the key: we control the dynamics memory parameters
    agent.dynamics_memory = dm
    # Wire it into modules that reference it
    agent.counterfactual._dm = dm
    agent.experience_scheduler._dm = dm
    
    # Track all states for analysis
    all_states = []
    all_actions = []
    state_types = []  # categorize what the agent was doing
    prev_state = None
    
    # ── Run Cognitive Loop ──────────────────────────────────────
    print("Running cognitive loop...")
    t0 = time.time()
    
    for tick in range(n_ticks):
        # Pick an agent to run this tick (round-robin)
        agent_id = tick % n_agents
        
        # Observe world
        obs = world.observe(agent_id)
        
        # Think (full 16-step cycle)
        action = agent.think(observation=obs)
        
        # Apply action
        world.apply_action(agent_id, action)
        
        # Record state and feed DynamicsMemory directly
        state = agent._state_16d.copy()
        all_states.append(state)
        
        # Feed state transitions into DynamicsMemory
        # This is the critical wiring that IntegratedAgent._record_experience misses
        if prev_state is not None:
            velocity = state - prev_state
            dm._states.append(prev_state.copy())
            dm._velocities.append(velocity.copy())
            if dm._local_model is not None:
                dm._local_model.add_sample(prev_state, velocity)
            # Fit when we have enough data
            if len(dm._states) >= dm.config.min_samples_for_fit:
                if dm.config.model_type == "global":
                    if len(dm._states) % 50 == 0:
                        dm._fit_dynamics()
                else:
                    dm._fitted = True
        prev_state = state.copy()
        
        # Categorize the experience
        exp_type = _categorize_state(state, obs, action)
        state_types.append(exp_type)
        
        # World tick
        world.tick()
        
        if tick % 500 == 0:
            n_traces = len(dm._states)
            fitted = dm._fitted
            print(f"  tick {tick:5d}: traces={n_traces:4d}, fitted={fitted}")
    
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({n_ticks/elapsed:.0f} t/s)")
    print(f"Total states collected: {len(all_states)}")
    print(f"Total traces in DynamicsMemory: {len(dm._states)}")
    print()
    
    # ── Discover Attractors from Learned Dynamics ───────────────
    print("=" * 70)
    print("Discovering attractors from learned vector field...")
    print("=" * 70)
    
    attractors = dm.discover_attractors()
    
    print(f"\nAttractors discovered: {len(attractors)}")
    for i, att in enumerate(attractors):
        print(f"  Attractor {i}: [{att[0]:.3f}, {att[1]:.3f}, ..., {att[-1]:.3f}]")
    
    # ── Analyze Basin Assignment ────────────────────────────────
    print("\n" + "=" * 70)
    print("Basin Analysis: Which experiences gravitate to which attractor?")
    print("=" * 70)
    
    basin_assignments = []
    basin_type_counts = []
    
    for i, att in enumerate(attractors):
        basin_type_counts.append({})
    
    for state, exp_type in zip(all_states, state_types):
        basin = dm.basin_of(state)
        basin_assignments.append(basin)
        if basin >= 0 and basin < len(attractors):
            counts = basin_type_counts[basin]
            counts[exp_type] = counts.get(exp_type, 0) + 1
    
    # Print basin compositions
    for i, counts in enumerate(basin_type_counts):
        if counts:
            total = sum(counts.values())
            print(f"\n  Basin {i} ({total} states):")
            for exp_type, count in sorted(counts.items(),
                                          key=lambda x: -x[1]):
                pct = 100 * count / total
                print(f"    {exp_type:25s}: {count:4d} ({pct:5.1f}%)")
    
    # ── Semantic Coherence Score ────────────────────────────────
    print("\n" + "=" * 70)
    print("Semantic Coherence: Are attractors semantically distinct?")
    print("=" * 70)
    
    coherence = _compute_coherence(basin_type_counts, basin_assignments)
    print(f"\n  Coherence score: {coherence:.3f}")
    print(f"  (1.0 = each basin is pure single concept)")
    print(f"  (0.0 = all basins have identical mix)")
    
    # ── Trajectory Diversity ────────────────────────────────────
    states_arr = np.array(all_states)
    diversity = _compute_trajectory_diversity(states_arr)
    print(f"\n  Trajectory diversity: {diversity:.3f}")
    print(f"  (Fraction of state space explored)")
    
    # ── Velocity Field Quality ──────────────────────────────────
    print("\n" + "=" * 70)
    print("Velocity Field Quality")
    print("=" * 70)
    
    if dm.A is not None:
        # Check eigenvalues of learned A matrix
        eigvals = np.linalg.eigvals(dm.A)
        stable = sum(1 for e in eigvals if np.real(e) < 0)
        unstable = sum(1 for e in eigvals if np.real(e) > 0)
        oscillatory = sum(1 for e in eigvals
                          if abs(np.imag(e)) > 0.01 and abs(np.real(e)) < 0.01)
        print(f"  Eigenvalues: {len(eigvals)} total")
        print(f"    Stable (Re<0): {stable}")
        print(f"    Unstable (Re>0): {unstable}")
        print(f"    Oscillatory: {oscillatory}")
        
        # Check norm of A (how strong are the dynamics)
        a_norm = np.linalg.norm(dm.A, 'fro')
        print(f"  ||A||_F = {a_norm:.4f}")
        
        # Residual: how well does V(x) = Ax+b fit the data?
        if dm._states:
            states_sample = np.array(dm._states[-200:])
            vels_actual = np.array(dm._velocities[-200:])
            vels_predicted = states_sample @ dm.A.T + dm.b
            residual = np.mean(np.linalg.norm(
                vels_actual - vels_predicted, axis=1))
            print(f"  Mean prediction residual: {residual:.6f}")
    
    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    result = {
        "attractors_discovered": len(attractors),
        "attractors": [a.tolist() for a in attractors],
        "basin_assignments": basin_assignments,
        "basin_type_counts": basin_type_counts,
        "coherence": coherence,
        "trajectory_diversity": diversity,
        "n_traces": len(dm._states),
        "fitted": dm._fitted,
        "eigenvalues": [float(abs(e)) for e in eigvals] if dm.A is not None else [],
    }
    
    if len(attractors) == 0:
        print("\n  RESULT: No attractors discovered.")
        print("  The learned dynamics have no fixed points in [0,1]^16.")
        print("  This means the vector field is divergent or cyclic.")
        print("  SUBSTRATE CLAIM: WEAK — dynamics don't converge.")
    elif coherence > 0.4:
        print(f"\n  RESULT: {len(attractors)} attractors with coherence {coherence:.3f}")
        print("  Semantic clusters EMERGED without form_attractor().")
        print("  The dynamics naturally created distinct basins of attraction.")
        print("  SUBSTRATE CLAIM: STRONG — remember what it was never told.")
    elif coherence > 0.2:
        print(f"\n  RESULT: {len(attractors)} attractors with coherence {coherence:.3f}")
        print("  Some basin differentiation, but mixed.")
        print("  SUBSTRATE CLAIM: MODERATE — partial emergence.")
    else:
        print(f"\n  RESULT: {len(attractors)} attractors with coherence {coherence:.3f}")
        print("  Attractors exist but basins are undifferentiated.")
        print("  SUBSTRATE CLAIM: WEAK — attractors are geometric, not semantic.")
    
    return result


def _categorize_state(state, obs, action):
    """Categorize what kind of experience this is based on state features."""
    # Use the dominant pillar values to categorize
    pillar_names = [
        "awareness", "willpower", "force", "influence", "resistance",
        "integrity", "cohesion", "relation", "presence", "warmth",
        "memory", "attraction", "harm", "distortion", "flux", "depth"
    ]
    
    # Find dominant pillars
    top_indices = np.argsort(state)[-3:]
    dominant = [pillar_names[i] for i in top_indices]
    
    # Categorize based on dominant features
    if state[10] > 0.6:  # memory pillar
        return "memory_encoding"
    elif state[11] > 0.6:  # attraction pillar
        return "attraction"
    elif state[8] > 0.6:  # presence pillar
        return "presence"
    elif state[14] > 0.5:  # flux pillar
        return "flux"
    elif state[1] > 0.6:  # willpower pillar
        return "willpower"
    elif state[5] > 0.6:  # integrity pillar
        return "integrity"
    elif state[15] > 0.5:  # depth pillar
        return "depth"
    elif np.std(state) < 0.1:
        return "homogeneous"
    else:
        return "mixed"


def _compute_coherence(basin_type_counts, basin_assignments):
    """Compute semantic coherence of basin assignments.
    
    High coherence = each basin contains mostly one type.
    Low coherence = all basins have similar mixes.
    """
    if not basin_type_counts:
        return 0.0
    
    # For each basin, compute the entropy of its type distribution
    entropies = []
    for counts in basin_type_counts:
        if not counts:
            continue
        total = sum(counts.values())
        if total == 0:
            continue
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        entropies.append(entropy)
    
    if not entropies:
        return 0.0
    
    # Average entropy (lower = more coherent)
    avg_entropy = np.mean(entropies)
    # Normalize by max possible entropy (log of number of types)
    n_types = len(set(c for counts in basin_type_counts
                      for c in counts.keys()))
    max_entropy = np.log(n_types + 1e-10)
    
    if max_entropy < 1e-10:
        return 1.0
    
    # Coherence = 1 - normalized entropy
    coherence = 1.0 - (avg_entropy / max_entropy)
    return float(coherence)


def _compute_trajectory_diversity(states):
    """Compute how much of the state space was explored."""
    if len(states) < 10:
        return 0.0
    
    # Use PCA to project to 2D and measure spread
    from numpy.linalg import svd
    centered = states - states.mean(axis=0)
    _, s, _ = svd(centered, full_matrices=False)
    
    # Fraction of variance captured by top 2 components
    total_var = np.sum(s ** 2)
    if total_var < 1e-10:
        return 0.0
    
    top2_var = np.sum(s[:2] ** 2)
    spread = top2_var / total_var
    
    # Also measure range coverage
    ranges = states.max(axis=0) - states.min(axis=0)
    avg_range = np.mean(ranges)
    
    # Combined metric
    diversity = 0.5 * spread + 0.5 * min(avg_range / 0.8, 1.0)
    return float(diversity)


if __name__ == "__main__":
    print("=" * 70)
    print("PART 1: Global Linear Model")
    print("=" * 70)
    result_global = run_emergence_experiment(
        n_ticks=500, n_agents=3, seed=42, model_type="global")
    
    print("\n\n")
    print("=" * 70)
    print("PART 2: Local Linear Model (k-NN)")
    print("=" * 70)
    result_local = run_emergence_experiment(
        n_ticks=500, n_agents=3, seed=42, model_type="local")
    
    print("\n\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  Global: {result_global['attractors_discovered']} attractors, "
          f"coherence={result_global['coherence']:.3f}")
    print(f"  Local:  {result_local['attractors_discovered']} attractors, "
          f"coherence={result_local['coherence']:.3f}")
    print("=" * 70)
