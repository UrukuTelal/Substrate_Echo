#!/usr/bin/env python3
"""
C2. Critical Phenomena Scaling — increase field dimensions.

Tests whether the correlation length ξ grows with system size.
If yes: critical behavior is real within the model.
If no: finite-size artifact.

Field dimensions: 16, 32, 64, 128
Measures: ξ, τ, χ, order parameter at the critical point.
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.field_evolution import FieldEvolver, FieldConfig


def make_field_config(dim, lam=2.8, dt=0.01):
    """Create config for arbitrary dimension."""
    cfg = FieldConfig(dt=dt, gamma=0.01, lambda_gl=lam, temperature=0.001)
    # Override the coupling matrix for arbitrary dimension
    cfg.coupling_matrix = np.eye(dim, dtype=np.float64) * 0.1
    # Add off-diagonal coupling
    for i in range(dim):
        for j in range(dim):
            if i != j:
                d = abs(i - j)
                cfg.coupling_matrix[i, j] = 0.01 * np.exp(-d / 4.0)
    return cfg


def measure_correlation_length(field):
    """Measure spatial correlation length."""
    n = len(field)
    correlations = []
    for lag in range(1, min(n // 2, 32)):
        c = 0.0
        count = 0
        for i in range(n - lag):
            c += field[i] * field[i + lag]
            count += 1
        if count > 0:
            correlations.append(c / count)
        else:
            correlations.append(0.0)

    if not correlations or max(correlations) < 1e-10:
        return 1.0

    c_arr = np.array(correlations)
    c_arr = np.maximum(c_arr, 1e-10)
    k_arr = np.arange(1, len(c_arr) + 1, dtype=float)
    ln_c = np.log(c_arr)

    if len(k_arr) >= 2:
        A = np.vstack([k_arr, np.ones_like(k_arr)]).T
        result = np.linalg.lstsq(A, ln_c, rcond=None)
        slope = result[0][0]
        xi = -1.0 / slope if slope < -1e-10 else float(n)
    else:
        xi = float(n)

    return float(np.clip(xi, 0.1, float(n)))


def measure_relaxation_time(evolver, field, dt, max_steps=500, threshold=0.05):
    """Measure relaxation time after perturbation."""
    # Equilibrate
    for _ in range(200):
        rhs = evolver.rhs(field, dt)
        field = field + dt * rhs
        field = np.clip(field, 0.0, 1.0)

    baseline = field.copy()
    energy_eq = evolver.compute_energy(field)

    # Perturb
    field = field + np.random.randn(len(field)) * 0.1
    field = np.clip(field, 0.0, 1.0)

    for step in range(max_steps):
        rhs = evolver.rhs(field, dt)
        field = field + dt * rhs
        field = np.clip(field, 0.0, 1.0)

        energy = evolver.compute_energy(field)
        deviation = abs(energy - energy_eq) / max(abs(energy_eq), 1e-10)

        if deviation < threshold:
            return step + 1

    return max_steps


def measure_susceptibility(evolver, field, dt, perturbation=0.01):
    """Measure linear response."""
    for _ in range(200):
        rhs = evolver.rhs(field, dt)
        field = field + dt * rhs
        field = np.clip(field, 0.0, 1.0)

    delta = np.random.randn(len(field)) * perturbation
    field_pert = np.clip(field + delta, 0.0, 1.0)

    rhs_eq = evolver.rhs(field, dt)
    rhs_pert = evolver.rhs(field_pert, dt)

    response = np.linalg.norm(rhs_pert - rhs_eq) / max(np.linalg.norm(delta), 1e-10)
    return float(response)


def run_scaling():
    print("=" * 60)
    print("CRITICAL PHENOMENA SCALING")
    print("=" * 60)

    dimensions = [16, 32, 64, 128]
    critical_lambda = 2.8

    all_results = {}

    for dim in dimensions:
        print(f"\n--- Dimension = {dim} ---")
        t0 = time.time()

        cfg = make_field_config(dim, lam=critical_lambda)
        evolver = FieldEvolver(cfg)

        xi_vals = []
        tau_vals = []
        chi_vals = []
        order_vals = []

        n_seeds = 3
        for seed in range(n_seeds):
            np.random.seed(seed * 137 + 42)
            field = np.random.uniform(0.1, 0.9, dim)

            # Evolve to develop spatial structure before measuring
            for _ in range(300):
                rhs = evolver.rhs(field, cfg.dt)
                field = field + cfg.dt * rhs
                field = np.clip(field, 0.0, 1.0)

            xi = measure_correlation_length(field)
            tau = measure_relaxation_time(evolver, field.copy(), cfg.dt)
            chi = measure_susceptibility(evolver, field.copy(), cfg.dt)

            mean_val = np.mean(field)
            variance = np.var(field)
            order = 1.0 - variance / max(mean_val ** 2, 1e-10)

            xi_vals.append(xi)
            tau_vals.append(tau)
            chi_vals.append(chi)
            order_vals.append(order)

        elapsed = time.time() - t0

        results = {
            'dimension': dim,
            'xi_mean': float(np.mean(xi_vals)),
            'xi_std': float(np.std(xi_vals)),
            'tau_mean': float(np.mean(tau_vals)),
            'tau_std': float(np.std(tau_vals)),
            'chi_mean': float(np.mean(chi_vals)),
            'chi_std': float(np.std(chi_vals)),
            'order_mean': float(np.mean(order_vals)),
            'elapsed_s': elapsed,
        }

        all_results[dim] = results

        print(f"  xi = {results['xi_mean']:.2f} (+/-{results['xi_std']:.2f})")
        print(f"  tau = {results['tau_mean']:.1f} (+/-{results['tau_std']:.1f})")
        print(f"  chi = {results['chi_mean']:.3f} (+/-{results['chi_std']:.3f})")
        print(f"  order = {results['order_mean']:.3f}")
        print(f"  ({elapsed:.1f}s)")

    # Scaling analysis
    print(f"\n--- Scaling Analysis ---")
    print(f"  {'dim':>5s} {'xi':>8s} {'tau':>8s} {'chi':>8s} {'order':>8s}")
    for dim in dimensions:
        r = all_results[dim]
        print(f"  {dim:>5d} {r['xi_mean']:>8.2f} {r['tau_mean']:>8.1f} "
              f"{r['chi_mean']:>8.3f} {r['order_mean']:>8.3f}")

    # Check if xi grows with dimension
    xis = [all_results[d]['xi_mean'] for d in dimensions]
    if xis[-1] > xis[0] * 1.5:
        print(f"\n  xi GROWS with dimension ({xis[0]:.1f} -> {xis[-1]:.1f})")
        print(f"  Critical behavior is likely REAL within the model")
    elif xis[-1] < xis[0] * 1.2:
        print(f"\n  xi is BOUNDED ({xis[0]:.1f} -> {xis[-1]:.1f})")
        print(f"  May be finite-size artifact")
    else:
        print(f"\n  xi shows moderate growth ({xis[0]:.1f} -> {xis[-1]:.1f})")
        print(f"  Inconclusive — need larger dimensions")

    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'critical_scaling.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")

    return all_results


if __name__ == '__main__':
    run_scaling()
