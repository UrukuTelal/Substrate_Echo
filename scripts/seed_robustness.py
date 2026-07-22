#!/usr/bin/env python3
"""
A. Seed Robustness — 1000 independent seeds.

Measures distributions of:
- Final norm, energy, conservation rate
- Diffusion tensor structure (Frobenius, block ratio)
- Agent coherence/diversity
- Vortex counts (on synthetic 2D fields)

Answers: Are outcomes reproducible? Does the phase boundary move?
"""
import sys, os, json, time
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.field_evolution import FieldEvolver, FieldConfig
from substrate_echo.dynamics.conservation import ConservationFramework
from substrate_echo.dynamics.pillar_coupling import PillarCoupling
from substrate_echo.dynamics.diffusion import DiffusionTensor
from substrate_echo.core.multi_agent_dynamics import SocialField, SocialConfig


PILLAR_NAMES = [
    "Awareness", "Willpower", "Force", "Influence",
    "Resistance", "Integrity", "Cohesion", "Relation",
    "Presence", "Warmth", "Memory", "Attraction",
    "Harm", "Distortion", "Flux", "Depth",
]

GROUPS = {
    'cognitive':  [0, 1, 2, 3],
    'structural': [4, 5, 6, 7],
    'emotional':  [8, 9, 10, 11],
    'metaphoric': [12, 13, 14, 15],
}


def single_seed_run(seed, n_ticks=200, n_agents=5):
    """Run one complete simulation with a given random seed."""
    rng = np.random.RandomState(seed)

    # Field dynamics
    cfg = FieldConfig(dt=0.01, gamma=0.01, lambda_gl=1.0, temperature=0.001)
    evolver = FieldEvolver(cfg)
    conservation = ConservationFramework(enabled=True)
    coupling = PillarCoupling()

    # Random initial field
    field = rng.uniform(0.1, 0.9, 16)
    conservation.set_norm_target(float(np.linalg.norm(field)))

    # Diffusion tensor
    diff = DiffusionTensor()

    # Agents
    sf = SocialField(SocialConfig(influence_strength=0.05))
    roles = ['perceiver', 'analyzer', 'synthesizer', 'guardian', 'explorer']
    for i in range(n_agents):
        state = rng.uniform(0.2, 0.8, 16)
        sf.add_agent(f"a_{i}", roles[i % len(roles)], state)

    # Tracking
    norm_history = []
    energy_history = []
    coherence_history = []

    for tick in range(n_ticks):
        # Field evolution
        rhs = evolver.rhs(field, cfg.dt)
        field = field + cfg.dt * rhs
        field = np.clip(field, 0.0, 1.0)

        # Projection
        pillar = coupling.project_to_pillars(field)

        # Diffusion learning
        state_delta = rng.randn(16) * 0.01
        diff.update_from_observation(pillar + state_delta)

        # Agent dynamics
        for aid in sf.get_all_agents():
            influence = sf.compute_social_influence(aid)
            agent = sf.get_agent(aid)
            agent.state = np.clip(agent.state + influence, 0.0, 1.0)

        norm_history.append(float(np.linalg.norm(pillar)))
        energy_history.append(evolver.compute_energy(field))
        coherence_history.append(sf.compute_collective_coherence())

    # Final metrics
    norm_arr = np.array(norm_history)
    energy_arr = np.array(energy_history)
    coherence_arr = np.array(coherence_history)

    # Diffusion structure
    off_diag = diff.tensor.copy()
    np.fill_diagonal(off_diag, 0)
    frobenius = float(np.linalg.norm(off_diag))

    within_group = 0.0
    between_group = 0.0
    wc = 0
    bc = 0
    for g_indices in GROUPS.values():
        for i in g_indices:
            for j in g_indices:
                if i != j:
                    within_group += abs(diff.tensor[i, j])
                    wc += 1
    for i in range(16):
        for j in range(16):
            if i != j:
                i_g = next((g for g, idx in GROUPS.items() if i in idx), None)
                j_g = next((g for g, idx in GROUPS.items() if j in idx), None)
                if i_g != j_g:
                    between_group += abs(diff.tensor[i, j])
                    bc += 1

    block_ratio = (within_group / max(wc, 1)) / max(between_group / max(bc, 1), 1e-10)

    return {
        'seed': seed,
        'final_norm': norm_arr[-1],
        'norm_std': float(np.std(norm_arr)),
        'norm_drift': float(abs(norm_arr[-1] - norm_arr[0])),
        'final_energy': energy_arr[-1],
        'final_coherence': coherence_arr[-1],
        'coherence_converged': bool(coherence_arr[-1] > 0.8),
        'diffusion_frobenius': frobenius,
        'diffusion_block_ratio': block_ratio,
        'avg_cpu_norm': float(np.mean(norm_arr)),
    }


def run_robustness(n_seeds=1000, n_ticks=200):
    print("=" * 60)
    print(f"SEED ROBUSTNESS: {n_seeds} independent runs, {n_ticks} ticks each")
    print("=" * 60)

    t0 = time.time()
    results = []

    for i in range(n_seeds):
        seed = i * 137 + 42  # deterministic but spread out
        r = single_seed_run(seed, n_ticks=n_ticks)
        results.append(r)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{n_seeds} seeds done "
                  f"({elapsed:.1f}s, {rate:.0f} seeds/s)")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s ({n_seeds/elapsed:.0f} seeds/s)")

    # Aggregate statistics
    norms = [r['final_norm'] for r in results]
    energies = [r['final_energy'] for r in results]
    coherences = [r['final_coherence'] for r in results]
    frob = [r['diffusion_frobenius'] for r in results]
    block = [r['diffusion_block_ratio'] for r in results]

    print(f"\n--- Distribution Summary ---")
    print(f"  {'Metric':<25s} {'Mean':>8s} {'Std':>8s} "
          f"{'P5':>8s} {'P50':>8s} {'P95':>8s}")
    print(f"  {'-'*57}")

    for name, data in [
        ('final_norm', norms),
        ('norm_drift', [r['norm_drift'] for r in results]),
        ('final_energy', energies),
        ('coherence', coherences),
        ('diffusion_frobenius', frob),
        ('diffusion_block_ratio', block),
    ]:
        arr = np.array(data)
        print(f"  {name:<25s} {np.mean(arr):8.4f} {np.std(arr):8.4f} "
              f"{np.percentile(arr, 5):8.4f} {np.percentile(arr, 50):8.4f} "
              f"{np.percentile(arr, 95):8.4f}")

    # Categorize norm drift
    low_drift = sum(1 for r in results if r['norm_drift'] < 0.15)
    print(f"\n--- Norm Stability ---")
    print(f"  Low drift (<0.15): {low_drift}/{n_seeds} ({low_drift/n_seeds:.1%})")

    # Categorize coherence
    converged = sum(1 for c in coherences if c > 0.8)
    partial = sum(1 for c in coherences if 0.4 < c <= 0.8)
    dispersed = sum(1 for c in coherences if c <= 0.4)
    print(f"\n--- Coherence Distribution ---")
    print(f"  Converged (>0.8):  {converged}/{n_seeds} ({converged/n_seeds:.1%})")
    print(f"  Partial (0.4-0.8): {partial}/{n_seeds} ({partial/n_seeds:.1%})")
    print(f"  Dispersed (<0.4):  {dispersed}/{n_seeds} ({dispersed/n_seeds:.1%})")

    # Diffusion block structure consistency
    block_ratios = np.array(block)
    consistent = sum(1 for b in block_ratios if b > 1.0)
    print(f"\n--- Diffusion Structure ---")
    print(f"  Block ratio > 1.0 (within > between): {consistent}/{n_seeds} ({consistent/n_seeds:.1%})")
    print(f"  Mean block ratio: {np.mean(block_ratios):.3f}")

    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'robustness_results.json')
    with open(out, 'w') as f:
        json.dump({
            'n_seeds': n_seeds,
            'n_ticks': n_ticks,
            'elapsed_s': elapsed,
            'summary': {
                'norm_mean': float(np.mean(norms)),
                'norm_std': float(np.std(norms)),
                'norm_drift_mean': float(np.mean([r['norm_drift'] for r in results])),
                'coherence_mean': float(np.mean(coherences)),
                'coherence_std': float(np.std(coherences)),
                'coherence_converged_pct': sum(1 for c in coherences if c > 0.8) / n_seeds,
                'block_ratio_mean': float(np.mean(block)),
                'block_ratio_gt1_pct': consistent / n_seeds,
            },
            'results': results,
        }, f, indent=2)
    print(f"\nResults saved to {out}")

    return results


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run_robustness(n_seeds=n)
