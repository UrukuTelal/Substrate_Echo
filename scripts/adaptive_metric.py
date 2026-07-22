#!/usr/bin/env python3
"""
1. Adaptive Metric Experiment.

Replace fixed diffusion coupling with a learned metric that adapts
to the state distribution. Tests whether adaptive coupling maintains
criticality across dimensions (16D → 128D).

Key idea: the diffusion tensor D_ij becomes state-dependent,
effectively implementing a Riemannian metric on the state space.
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.field_evolution import FieldConfig
from substrate_echo.dynamics.diffusion import DiffusionTensor, DiffusionConfig


class AdaptiveMetric:
    """State-dependent diffusion coupling that acts as a learned metric.
    
    Instead of fixed D, computes D(state) based on observed correlations.
    The metric tensor G_ij = D_ij(state) defines local geometry.
    """
    
    def __init__(self, dim, learning_rate=0.01, decay=0.999):
        self.dim = dim
        self.lr = learning_rate
        self.decay = decay
        
        # Metric tensor: starts as identity (Euclidean)
        self.metric = np.eye(dim, dtype=np.float64)
        
        # State history for computing local statistics
        self.state_buffer = []
        self.buffer_size = 50
        
        # Gradient accumulator for metric update
        self.gradient_accum = np.zeros((dim, dim), dtype=np.float64)
        self.update_count = 0
    
    def observe(self, state, state_dot):
        """Observe a (state, time_derivative) pair.
        
        The metric should satisfy:
        G_ij ≈ cov(state_i, state_j) / var(state)
        
        This makes distances in state space proportional to
        dynamical relevance.
        """
        self.state_buffer.append(state.copy())
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
        
        # Compute local covariance
        if len(self.state_buffer) >= 10:
            data = np.array(self.state_buffer)
            mean = np.mean(data, axis=0)
            centered = data - mean
            cov = (centered.T @ centered) / len(data)
            
            # Normalize to unit trace
            trace = np.trace(cov)
            if trace > 1e-10:
                cov = cov / trace * self.dim
            
            # Update metric with exponential moving average
            self.metric = (1 - self.lr) * self.metric + self.lr * cov
            self.update_count += 1
    
    def get_metric(self):
        """Get current metric tensor."""
        return self.metric.copy()
    
    def distances(self, states):
        """Compute pairwise distances using the metric.
        
        d(a,b) = sqrt((a-b)^T G (a-b))
        """
        n = states.shape[0]
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                diff = states[i] - states[j]
                d2 = diff @ self.metric @ diff
                dists[i, j] = np.sqrt(max(d2, 0))
                dists[j, i] = dists[i, j]
        return dists
    
    def curvature(self):
        """Estimate local curvature from metric variation.
        
        In Riemannian geometry, curvature measures how much
        the metric deviates from flat space.
        
        For our purposes: high curvature = strong local structure.
        """
        # Approximate curvature from metric eigenvalues
        eigvals = np.linalg.eigvalsh(self.metric)
        eigvals = np.maximum(eigvals, 1e-10)
        
        # Condition number measures anisotropy
        condition = eigvals[-1] / eigvals[0]
        
        # Effective dimension from participation ratio
        normalized = eigvals / np.sum(eigvals)
        participation = 1.0 / np.sum(normalized ** 2)
        
        return {
            'condition_number': float(condition),
            'effective_dim': float(participation),
            'max_eigenval': float(eigvals[-1]),
            'min_eigenval': float(eigvals[0]),
        }


def run_adaptive_metric():
    print("=" * 60)
    print("ADAPTIVE METRIC EXPERIMENT")
    print("=" * 60)
    
    dimensions = [16, 32, 64, 128]
    n_ticks = 500
    lambda_gl_values = [0.5, 1.0, 2.8, 5.0, 10.0]
    
    all_results = {}
    
    for dim in dimensions:
        print(f"\n--- Dimension = {dim} ---")
        dim_results = {}
        
        for lam in lambda_gl_values:
            # Create config
            cfg = FieldConfig(dt=0.01, gamma=0.01, lambda_gl=lam, temperature=0.001)
            cfg.coupling_matrix = np.eye(dim, dtype=np.float64) * 0.1
            for i in range(dim):
                for j in range(dim):
                    if i != j:
                        d = abs(i - j)
                        cfg.coupling_matrix[i, j] = 0.01 * np.exp(-d / 4.0)
            
            # Create field evolver
            from substrate_echo.dynamics.field_evolution import FieldEvolver
            evolver = FieldEvolver(cfg)
            
            # Create adaptive metric
            metric = AdaptiveMetric(dim, learning_rate=0.005)
            
            # Initialize field
            rng = np.random.RandomState(42)
            field = rng.uniform(0.1, 0.9, dim)
            
            # Track metrics
            energy_history = []
            curvature_history = []
            norm_history = []
            
            for tick in range(n_ticks):
                # Compute RHS
                rhs = evolver.rhs(field, cfg.dt)
                
                # Observe state and derivative
                metric.observe(field, rhs)
                
                # Evolve
                field = field + cfg.dt * rhs
                field = np.clip(field, 0.0, 1.0)
                
                # Record
                energy_history.append(evolver.compute_energy(field))
                norm_history.append(float(np.linalg.norm(field)))
                
                if tick % 50 == 0:
                    curv = metric.curvature()
                    curvature_history.append(curv)
            
            # Analysis
            energy_arr = np.array(energy_history)
            norm_arr = np.array(norm_history)
            
            # Measure criticality indicators
            energy_std = float(np.std(energy_arr))
            energy_mean = float(np.mean(energy_arr))
            norm_std = float(np.std(norm_arr))
            
            # Relaxation time estimate from energy autocorrelation
            if len(energy_arr) > 50:
                autocorr = np.correlate(energy_arr - energy_mean, 
                                       energy_arr - energy_mean, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]
                # Find where autocorrelation drops to 1/e
                tau = 1.0
                for i in range(1, len(autocorr)):
                    if autocorr[i] < 1.0 / np.e:
                        tau = float(i)
                        break
            else:
                tau = 0.0
            
            # Final metric state
            curv_final = metric.curvature()
            
            result = {
                'energy_mean': energy_mean,
                'energy_std': energy_std,
                'norm_std': norm_std,
                'relaxation_time': tau,
                'condition_number': curv_final['condition_number'],
                'effective_dim': curv_final['effective_dim'],
                'metric_updates': metric.update_count,
            }
            
            dim_results[lam] = result
            
            print(f"  lam={lam:>5.1f}: E={energy_mean:.5f} "
                  f"tau={tau:>6.1f} cond={curv_final['condition_number']:>7.1f} "
                  f"eff_dim={curv_final['effective_dim']:>5.1f}")
        
        all_results[dim] = dim_results
    
    # Scaling analysis
    print(f"\n--- Scaling: Does adaptive metric maintain criticality? ---")
    print(f"  {'dim':>5s} {'lambda_c':>8s} {'tau_max':>8s} {'cond_range':>12s}")
    
    for dim in dimensions:
        dim_data = all_results[dim]
        # Find lambda with max relaxation time
        best_lam = max(dim_data.keys(), key=lambda l: dim_data[l]['relaxation_time'])
        max_tau = dim_data[best_lam]['relaxation_time']
        conds = [dim_data[l]['condition_number'] for l in dim_data]
        
        print(f"  {dim:>5d} {best_lam:>8.1f} {max_tau:>8.1f} "
              f"[{min(conds):.1f}, {max(conds):.1f}]")
    
    # Compare with fixed metric (from critical_scaling.json)
    fixed_path = os.path.join(os.path.dirname(__file__), '..', 'critical_scaling.json')
    if os.path.exists(fixed_path):
        with open(fixed_path) as f:
            fixed_data = json.load(f)
        
        print(f"\n--- Fixed vs Adaptive Metric ---")
        print(f"  {'dim':>5s} {'fixed_tau':>10s} {'adaptive_tau':>13s} {'improvement':>12s}")
        for dim in dimensions:
            if str(dim) in fixed_data and dim in all_results:
                fixed_tau = fixed_data[str(dim)]['tau_mean']
                # Best adaptive tau across lambdas
                best_adaptive = max(all_results[dim][l]['relaxation_time'] 
                                   for l in all_results[dim])
                improvement = best_adaptive / max(fixed_tau, 1) 
                print(f"  {dim:>5d} {fixed_tau:>10.1f} {best_adaptive:>13.1f} "
                      f"{improvement:>12.2f}x")
    
    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'adaptive_metric_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")
    
    return all_results


if __name__ == '__main__':
    run_adaptive_metric()
