#!/usr/bin/env python3
"""
B. Finite-Size Scaling — system size effects.

Sweeps system size (number of agents, field complexity)
and watches for qualitative transitions.

System sizes: 2, 5, 10, 25, 50, 100, 250, 500
Measures: consensus time, coherence, diversity, energy stability,
diffusion structure, critical point location.
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.field_evolution import FieldEvolver, FieldConfig
from substrate_echo.dynamics.conservation import ConservationFramework
from substrate_echo.dynamics.pillar_coupling import PillarCoupling
from substrate_echo.dynamics.diffusion import DiffusionTensor
from substrate_echo.core.multi_agent_dynamics import SocialField, SocialConfig


def run_size_experiment(n_agents, n_ticks=300, n_seeds=10):
    """Run one system size with multiple seeds."""
    results = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed * 137 + 42)

        # Field dynamics
        cfg = FieldConfig(dt=0.01, gamma=0.01, lambda_gl=1.0, temperature=0.001)
        evolver = FieldEvolver(cfg)
        conservation = ConservationFramework(enabled=True)
        coupling = PillarCoupling()

        field = rng.uniform(0.1, 0.9, 16)
        pillar = coupling.project_to_pillars(field)
        conservation.set_norm_target(float(np.linalg.norm(pillar)))

        # Social field
        sf = SocialField(SocialConfig(
            influence_strength=0.05,
            influence_range=1.5,
            consensus_threshold=0.7,
        ))
        roles = ['perceiver', 'analyzer', 'synthesizer', 'guardian', 'explorer']
        for i in range(n_agents):
            state = rng.uniform(0.2, 0.8, 16)
            sf.add_agent(f"a_{i}", roles[i % len(roles)], state)

        # Tracking
        coherence_history = []
        diversity_history = []
        conservation_pass = 0
        consensus_tick = None

        for tick in range(n_ticks):
            # Field evolution
            rhs = evolver.rhs(field, cfg.dt)
            field = field + cfg.dt * rhs
            field = np.clip(field, 0.0, 1.0)

            pillar = coupling.project_to_pillars(field)

            results_cons = conservation.check_all(pillar)
            if all(r.passed for r in results_cons):
                conservation_pass += 1

            # Agent dynamics
            for aid in sf.get_all_agents():
                influence = sf.compute_social_influence(aid)
                agent = sf.get_agent(aid)
                agent.state = np.clip(agent.state + influence, 0.0, 1.0)

            c = sf.compute_collective_coherence()
            d = sf.compute_collective_diversity()
            coherence_history.append(c)
            diversity_history.append(d)

            if consensus_tick is None and c > 0.8:
                consensus_tick = tick

        # Diffusion structure
        diff = DiffusionTensor()
        for _ in range(50):
            obs = rng.randn(16) * 0.1
            diff.update_from_observation(obs)
        off_diag = diff.tensor.copy()
        np.fill_diagonal(off_diag, 0)
        frobenius = float(np.linalg.norm(off_diag))

        results.append({
            'seed': seed,
            'n_agents': n_agents,
            'final_coherence': coherence_history[-1],
            'final_diversity': diversity_history[-1],
            'consensus_tick': consensus_tick if consensus_tick is not None else n_ticks,
            'conservation_rate': conservation_pass / n_ticks,
            'diffusion_frobenius': frobenius,
            'peak_diversity': max(diversity_history),
        })

    return results


def run_scaling():
    print("=" * 60)
    print("FINITE-SIZE SCALING")
    print("=" * 60)

    sizes = [2, 5, 10, 25, 50, 100]
    all_results = {}

    for n in sizes:
        print(f"\n--- n_agents = {n} ---")
        t0 = time.time()
        n_seeds = 5 if n <= 50 else 3
        n_ticks = 300 if n <= 50 else 200
        results = run_size_experiment(n, n_ticks=n_ticks, n_seeds=n_seeds)
        elapsed = time.time() - t0

        # Aggregate
        coherences = [r['final_coherence'] for r in results]
        diversities = [r['final_diversity'] for r in results]
        consensus_times = [r['consensus_tick'] for r in results]
        cons_rates = [r['conservation_rate'] for r in results]
        frobs = [r['diffusion_frobenius'] for r in results]
        peak_divs = [r['peak_diversity'] for r in results]

        summary = {
            'n_agents': n,
            'coherence_mean': float(np.mean(coherences)),
            'coherence_std': float(np.std(coherences)),
            'diversity_mean': float(np.mean(diversities)),
            'consensus_time_mean': float(np.mean(consensus_times)),
            'consensus_time_std': float(np.std(consensus_times)),
            'conservation_mean': float(np.mean(cons_rates)),
            'diffusion_frobenius_mean': float(np.mean(frobs)),
            'peak_diversity_mean': float(np.mean(peak_divs)),
            'elapsed_s': elapsed,
        }

        all_results[n] = summary

        print(f"  Coherence: {summary['coherence_mean']:.3f} "
              f"(+/-{summary['coherence_std']:.3f})")
        print(f"  Consensus time: {summary['consensus_time_mean']:.0f} "
              f"(+/-{summary['consensus_time_std']:.0f})")
        print(f"  Peak diversity: {summary['peak_diversity_mean']:.3f}")
        print(f"  Conservation: {summary['conservation_mean']:.1%}")
        print(f"  Diffusion F: {summary['diffusion_frobenius_mean']:.4f}")
        print(f"  ({elapsed:.1f}s)")

    # Scaling analysis
    print(f"\n--- Scaling Analysis ---")
    ns = sorted(all_results.keys())
    print(f"  {'n':>6s} {'coherence':>10s} {'cons_time':>10s} "
          f"{'diversity':>10s} {'diffusion':>10s}")
    for n in ns:
        s = all_results[n]
        print(f"  {n:>6d} {s['coherence_mean']:>10.3f} "
              f"{s['consensus_time_mean']:>10.1f} "
              f"{s['diversity_mean']:>10.3f} "
              f"{s['diffusion_frobenius_mean']:>10.4f}")

    # Check for qualitative transitions
    print(f"\n--- Qualitative Transitions ---")
    prev_coh = all_results[ns[0]]['coherence_mean']
    for n in ns[1:]:
        curr_coh = all_results[n]['coherence_mean']
        if abs(curr_coh - prev_coh) > 0.2:
            direction = "increased" if curr_coh > prev_coh else "decreased"
            print(f"  n={n}: Coherence {direction} significantly "
                  f"({prev_coh:.3f} -> {curr_coh:.3f})")
        prev_coh = curr_coh

    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'scaling_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")

    return all_results


if __name__ == '__main__':
    run_scaling()
