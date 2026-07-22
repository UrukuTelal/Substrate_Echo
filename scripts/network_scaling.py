#!/usr/bin/env python3
"""
4. Adaptive Network Scaling Experiment.

Test whether scaling communication radius with agent count
maintains consensus at scale.

Uses vectorized DeGroot-style consensus — no SocialField overhead.
"""
import sys, os, json, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_scaling_experiment():
    print("=" * 60)
    print("ADAPTIVE NETWORK SCALING EXPERIMENT")
    print("=" * 60)
    
    agent_counts = [10, 20, 50, 100, 200, 500, 1000]
    n_ticks = 200
    dim = 16
    
    base_radius = 0.5
    target_density = 0.1
    
    scaling_strategies = {
        'fixed_sparse': lambda n: base_radius,
        'fixed_dense': lambda n: 2.0,
        'sqrt': lambda n: base_radius * np.sqrt(n / 10),
        'log': lambda n: base_radius * np.log(1 + n / 10) * 2,
        'density_target': lambda n: np.sqrt(target_density * n / math.pi) * 1.5,
        'percolation': lambda n: np.sqrt(math.log(n) / math.pi),  # critical connectivity
    }
    
    strategy_desc = {
        'fixed_sparse': f'Fixed r={base_radius} (sparse)',
        'fixed_dense': 'Fixed r=2.0 (dense)',
        'sqrt': 'r = base*sqrt(n/10)',
        'log': 'r = base*2*log(1+n/10)',
        'density_target': f'target rho={target_density}',
        'percolation': 'r = sqrt(log(n)/pi)',
    }
    
    all_results = {}
    
    for strategy_name, radius_fn in scaling_strategies.items():
        print(f"\n--- {strategy_desc[strategy_name]} ---")
        print(f"  {'n':>5s} {'radius':>7s} {'density':>8s} {'avg_deg':>8s} "
              f"{'cohesion':>9s} {'path_len':>9s} {'clusters':>9s}")
        
        strat_results = {}
        
        for n_agents in agent_counts:
            radius = radius_fn(n_agents)
            rng = np.random.RandomState(42)
            
            # Agent states: start near 0.5 ± small noise
            states = rng.uniform(0.3, 0.7, (n_agents, dim))
            
            # Build adjacency matrix (vectorized)
            diff = states[:, np.newaxis, :] - states[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            adj = (dists < radius).astype(np.float64)
            
            # Degree
            degrees = np.sum(adj, axis=1)
            avg_deg = float(np.mean(degrees))
            density = float(np.mean(degrees)) / max(n_agents - 1, 1)
            
            # Clustering coefficient (sample)
            sample_n = min(n_agents, 50)
            sample_idx = rng.choice(n_agents, sample_n, replace=False)
            clusterings = []
            for i in sample_idx:
                neighbors = np.where(adj[i] > 0)[0]
                k = len(neighbors)
                if k < 2:
                    clusterings.append(0.0)
                    continue
                # Edges among neighbors
                sub_adj = adj[np.ix_(neighbors, neighbors)]
                edges = np.sum(sub_adj) / 2
                possible = k * (k - 1) / 2
                clusterings.append(edges / possible if possible > 0 else 0)
            clustering = float(np.mean(clusterings))
            
            # Path length (BFS on sample)
            path_lengths = []
            for start in sample_idx[:min(15, sample_n)]:
                visited = {int(start): 0}
                queue = [int(start)]
                while queue:
                    node = queue.pop(0)
                    for nb in np.where(adj[node] > 0)[0]:
                        nb = int(nb)
                        if nb not in visited:
                            visited[nb] = visited[node] + 1
                            queue.append(nb)
                for nb in range(n_agents):
                    if nb != start and nb in visited:
                        path_lengths.append(visited[nb])
            avg_path = float(np.mean(path_lengths)) if path_lengths else float('inf')
            
            # Consensus: DeGroot dynamics (vectorized)
            # x(t+1) = (I - lr*L) x(t)  where L is graph Laplacian
            if avg_deg > 0:
                # Normalized Laplacian
                D_inv = np.diag(1.0 / np.maximum(degrees, 1))
                L = np.eye(n_agents) - D_inv @ adj
                lr = 0.5
                
                for _ in range(n_ticks):
                    states = states - lr * (L @ states)
                
                # Final spread
                mean_state = np.mean(states, axis=0)
                spread = float(np.mean(np.linalg.norm(states - mean_state, axis=1)))
                cohesion = 1.0 / (1.0 + spread)
            else:
                cohesion = 0.0
            
            strat_results[n_agents] = {
                'radius': float(radius),
                'density': density,
                'avg_degree': avg_deg,
                'clustering': clustering,
                'avg_path_length': avg_path,
                'cohesion': cohesion,
                'n_clusters': len(set(np.round(states[:, 0], 1))),  # rough
            }
            
            print(f"  {n_agents:>5d} {radius:>7.2f} {density:>8.4f} "
                  f"{avg_deg:>8.1f} {cohesion:>9.4f} {avg_path:>9.2f} "
                  f"{clustering:>9.4f}")
        
        all_results[strategy_name] = strat_results
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SCALING COMPARISON")
    print(f"{'='*60}")
    
    print(f"\nCohesion at n=1000 (higher=agents more aligned):")
    for s, data in all_results.items():
        if 1000 in data:
            print(f"  {strategy_desc[s]:<35s}: {data[1000]['cohesion']:.4f} "
                  f"(deg={data[1000]['avg_degree']:.0f}, "
                  f"r={data[1000]['radius']:.2f})")
    
    print(f"\nCohesion scaling trend:")
    for s, data in all_results.items():
        vals = [(n, data[n]['cohesion']) for n in sorted(data.keys())]
        if len(vals) >= 2:
            first, last = vals[0][1], vals[-1][1]
            trend = 'stable' if abs(last - first) < 0.05 else \
                    ('improving' if last > first else 'degrading')
            print(f"  {strategy_desc[s]:<35s}: {first:.3f} -> {last:.3f} ({trend})")
    
    print(f"\nDensity stability (does it stay constant?):")
    for s, data in all_results.items():
        densities = [data[n]['density'] for n in sorted(data.keys())]
        if len(densities) >= 2:
            d_range = max(densities) - min(densities)
            print(f"  {strategy_desc[s]:<35s}: [{min(densities):.3f}, {max(densities):.3f}] "
                  f"(range={d_range:.3f})")
    
    out = os.path.join(os.path.dirname(__file__), '..', 'scaling_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")
    
    return all_results


if __name__ == '__main__':
    run_scaling_experiment()
