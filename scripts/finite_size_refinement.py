#!/usr/bin/env python3
"""
B2. Finite-Size Transition Refinement.

Scans n_agents in [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
and measures network-level properties to find the exact transition point.

Metrics:
- Network modularity (Q)
- Average path length
- Clustering coefficient
- Degree distribution (exponent)
- Consensus time
- Final coherence
"""
import sys, os, json, time, math
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.core.multi_agent_dynamics import SocialField, SocialConfig


def build_adjacency_matrix(sf):
    """Build binary adjacency matrix from communication topology."""
    agents = list(sf.get_all_agents().keys())
    n = len(agents)
    adj = np.zeros((n, n))

    topology = sf.get_communication_topology()
    for i, a_id in enumerate(agents):
        for b_id in topology.get(a_id, []):
            if b_id in agents:
                j = agents.index(b_id)
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    return adj, agents


def compute_modularity(adj):
    """Compute Newman modularity Q."""
    n = adj.shape[0]
    m = adj.sum() / 2
    if m < 1e-10:
        return 0.0

    degrees = adj.sum(axis=1)
    Q = 0.0
    for i in range(n):
        for j in range(n):
            expected = degrees[i] * degrees[j] / (2 * m)
            Q += adj[i, j] - expected

    return Q / (2 * m)


def compute_clustering_coefficient(adj):
    """Compute average clustering coefficient."""
    n = adj.shape[0]
    if n < 3:
        return 0.0

    coeffs = []
    for i in range(n):
        neighbors = list(np.where(adj[i] > 0)[0])
        k = len(neighbors)
        if k < 2:
            continue
        triangles = 0
        for a in range(len(neighbors)):
            for b in range(a + 1, len(neighbors)):
                if adj[neighbors[a], neighbors[b]] > 0:
                    triangles += 1
        coeffs.append(2 * triangles / (k * (k - 1)))

    return np.mean(coeffs) if coeffs else 0.0


def compute_average_path_length(adj):
    """Compute average shortest path length (BFS)."""
    n = adj.shape[0]
    total = 0
    count = 0

    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in range(n):
                if adj[node, neighbor] > 0 and dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)

        for d in dist:
            if d > 0:
                total += d
                count += 1

    return total / max(count, 1)


def compute_degree_distribution(adj):
    """Compute degree distribution exponent (power-law fit)."""
    degrees = adj.sum(axis=1).astype(int)
    degrees = degrees[degrees > 0]

    if len(degrees) < 5:
        return 0.0

    # Power-law fit: P(k) ~ k^(-gamma)
    counts = Counter(degrees)
    k_vals = np.array(sorted(counts.keys()), dtype=float)
    p_vals = np.array([counts[k] / len(degrees) for k in k_vals])

    # Linear regression in log-log space
    valid = (k_vals > 0) & (p_vals > 0)
    if valid.sum() < 3:
        return 0.0

    log_k = np.log(k_vals[valid])
    log_p = np.log(p_vals[valid])

    A = np.vstack([log_k, np.ones_like(log_k)]).T
    result = np.linalg.lstsq(A, log_p, rcond=None)
    gamma = -result[0][0]

    return float(gamma)


def run_single_size(n_agents, n_ticks=300, n_seeds=5):
    """Run one system size with multiple seeds."""
    all_results = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed * 137 + 42)

        sf = SocialField(SocialConfig(
            influence_strength=0.05,
            influence_range=1.5,
            max_communication_range=1.0,
        ))
        roles = ['perceiver', 'analyzer', 'synthesizer', 'guardian', 'explorer']
        for i in range(n_agents):
            state = rng.uniform(0.2, 0.8, 16)
            sf.add_agent(f"a_{i}", roles[i % len(roles)], state)

        coherence_history = []
        for tick in range(n_ticks):
            agents = list(sf.get_all_agents().keys())
            # Limit communication to first 3 partners per agent (O(n) not O(n^2))
            for a_id in agents[:min(len(agents), 50)]:
                partners = [b for b in agents if b != a_id][:3]
                for b_id in partners:
                    sf.send_message(a_id, b_id, "update", sf.get_agent(a_id).state)

            for aid in agents:
                influence = sf.compute_social_influence(aid)
                agent = sf.get_agent(aid)
                agent.state = np.clip(agent.state + influence, 0.0, 1.0)

            coherence_history.append(sf.compute_collective_coherence())

        # Network metrics
        adj, _ = build_adjacency_matrix(sf)
        modularity = compute_modularity(adj)
        clustering = compute_clustering_coefficient(adj)
        path_length = compute_average_path_length(adj)
        gamma = compute_degree_distribution(adj)

        # Consensus
        consensus_tick = None
        for t, c in enumerate(coherence_history):
            if c > 0.8:
                consensus_tick = t
                break

        all_results.append({
            'seed': seed,
            'final_coherence': coherence_history[-1],
            'consensus_tick': consensus_tick if consensus_tick is not None else n_ticks,
            'modularity': modularity,
            'clustering': clustering,
            'path_length': path_length,
            'degree_gamma': gamma,
            'n_edges': int(adj.sum() / 2),
            'n_possible': n_agents * (n_agents - 1) // 2,
            'density': float(adj.sum() / max(n_agents * (n_agents - 1), 1)),
        })

    return all_results


def run_refinement():
    print("=" * 60)
    print("FINITE-SIZE TRANSITION REFINEMENT")
    print("=" * 60)

    sizes = [20, 40, 60, 80, 100, 120, 150]
    all_data = {}

    for n in sizes:
        print(f"\n--- n_agents = {n} ---")
        t0 = time.time()
        n_seeds = 3
        n_ticks = 200 if n <= 100 else 150
        results = run_single_size(n, n_ticks=n_ticks, n_seeds=n_seeds)
        elapsed = time.time() - t0

        # Aggregate
        metrics = {}
        for key in ['final_coherence', 'consensus_tick', 'modularity',
                     'clustering', 'path_length', 'degree_gamma',
                     'n_edges', 'density']:
            vals = [r[key] for r in results if r[key] is not None]
            if vals:
                metrics[f'{key}_mean'] = float(np.mean(vals))
                metrics[f'{key}_std'] = float(np.std(vals))

        metrics['n_agents'] = n
        metrics['elapsed_s'] = elapsed
        all_data[n] = metrics

        print(f"  Coherence: {metrics['final_coherence_mean']:.3f} "
              f"(+/-{metrics.get('final_coherence_std', 0):.3f})")
        print(f"  Consensus tick: {metrics['consensus_tick_mean']:.0f} "
              f"(+/-{metrics.get('consensus_tick_std', 0):.0f})")
        print(f"  Modularity: {metrics['modularity_mean']:.3f}")
        print(f"  Clustering: {metrics['clustering_mean']:.3f}")
        print(f"  Path length: {metrics['path_length_mean']:.1f}")
        print(f"  Degree gamma: {metrics['degree_gamma_mean']:.2f}")
        print(f"  Density: {metrics['density_mean']:.3f}")
        print(f"  ({elapsed:.1f}s)")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'n':>5s} {'coherence':>10s} {'cons_time':>10s} {'modularity':>11s} "
          f"{'clustering':>11s} {'path_len':>9s} {'gamma':>7s} {'density':>8s}")
    print(f"{'-'*80}")
    for n in sizes:
        m = all_data[n]
        print(f"{n:>5d} {m['final_coherence_mean']:>10.3f} "
              f"{m['consensus_tick_mean']:>10.1f} "
              f"{m['modularity_mean']:>11.3f} "
              f"{m['clustering_mean']:>11.3f} "
              f"{m['path_length_mean']:>9.1f} "
              f"{m['degree_gamma_mean']:>7.2f} "
              f"{m['density_mean']:>8.3f}")

    # Find transition
    print(f"\n--- Transition Analysis ---")
    prev_coh = all_data[sizes[0]]['final_coherence_mean']
    for n in sizes[1:]:
        curr_coh = all_data[n]['final_coherence_mean']
        delta = curr_coh - prev_coh
        if abs(delta) > 0.05:
            print(f"  n={n}: coherence {delta:+.3f} "
                  f"({'BREAKDOWN' if delta < 0 else 'recovery'})")
        prev_coh = curr_coh

    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'finite_size_refinement.json')
    with open(out, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\nResults saved to {out}")

    return all_data


if __name__ == '__main__':
    run_refinement()
