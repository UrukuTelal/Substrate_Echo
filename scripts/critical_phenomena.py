#!/usr/bin/env python3
"""
C. Critical Phenomena — measures near the phase boundary.

Measures:
- Correlation length (pillar-pillar spatial correlations)
- Relaxation time (time to return to equilibrium after perturbation)
- Susceptibility (response to small field perturbation)
- Order parameter (coherence as order parameter)
- Critical slowing down near phase boundary
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.field_evolution import FieldEvolver, FieldConfig
from substrate_echo.dynamics.conservation import ConservationFramework
from substrate_echo.dynamics.pillar_coupling import PillarCoupling


def measure_correlation_length(field, n_samples=100):
    """Measure spatial correlation length in the 16D pillar state.
    
    Uses the pillar coupling to map field → pillars, then measures
    how correlations decay with "distance" (pillar index difference).
    """
    coupling = PillarCoupling()
    pillar = coupling.project_to_pillars(field)

    # Compute pillar-pillar correlations via the WHT matrix
    # Correlation at lag k = average <pillar_i * pillar_{i+k}>
    correlations = []
    for lag in range(1, 8):
        if lag >= 16:
            break
        c = 0.0
        count = 0
        for i in range(16 - lag):
            c += pillar[i] * pillar[i + lag]
            count += 1
        if count > 0:
            correlations.append(c / count)
        else:
            correlations.append(0.0)

    if not correlations or max(correlations) < 1e-10:
        return 1.0  # minimum correlation length

    # Exponential decay fit: C(k) ~ exp(-k/xi)
    # ln(C) = -k/xi + const
    c_arr = np.array(correlations)
    c_arr = np.maximum(c_arr, 1e-10)  # avoid log(0)

    # Fit using least squares
    k_arr = np.arange(1, len(c_arr) + 1, dtype=float)
    ln_c = np.log(c_arr)

    # Simple linear regression: ln(C) = a*k + b
    if len(k_arr) >= 2:
        A = np.vstack([k_arr, np.ones_like(k_arr)]).T
        result = np.linalg.lstsq(A, ln_c, rcond=None)
        slope = result[0][0]
        xi = -1.0 / slope if slope < -1e-10 else 16.0
    else:
        xi = 16.0

    return float(np.clip(xi, 0.1, 16.0))


def measure_relaxation_time(evolver, field, dt, perturbation=0.1,
                             max_steps=500, threshold=0.01):
    """Measure time to relax back to equilibrium after perturbation."""
    # Evolve to equilibrium
    for _ in range(200):
        rhs = evolver.rhs(field, dt)
        field = field + dt * rhs
        field = np.clip(field, 0.0, 1.0)

    equilibrium = field.copy()
    energy_eq = evolver.compute_energy(field)

    # Perturb
    field = field + np.random.randn(16) * perturbation
    field = np.clip(field, 0.0, 1.0)

    # Measure relaxation
    for step in range(max_steps):
        rhs = evolver.rhs(field, dt)
        field = field + dt * rhs
        field = np.clip(field, 0.0, 1.0)

        energy = evolver.compute_energy(field)
        deviation = abs(energy - energy_eq)

        if deviation < threshold * energy_eq:
            return step + 1, field

    return max_steps, field


def measure_susceptibility(evolver, field, dt, perturbation=0.01):
    """Measure linear response to small perturbation."""
    # Evolve to equilibrium
    for _ in range(200):
        rhs = evolver.rhs(field, dt)
        field = field + dt * rhs
        field = np.clip(field, 0.0, 1.0)

    energy_eq = evolver.compute_energy(field)

    # Apply perturbation
    delta = np.random.randn(16) * perturbation
    field_pert = np.clip(field + delta, 0.0, 1.0)

    # One step evolution
    rhs_eq = evolver.rhs(field, dt)
    rhs_pert = evolver.rhs(field_pert, dt)

    # Response = change in RHS (linear response)
    response = np.linalg.norm(rhs_pert - rhs_eq) / max(np.linalg.norm(delta), 1e-10)

    return float(response)


def measure_order_parameter(field):
    """Compute order parameter (field uniformity)."""
    mean = np.mean(field)
    if mean < 1e-10:
        return 0.0
    variance = np.var(field)
    # Order parameter: 1 - normalized variance
    return float(1.0 - variance / max(mean ** 2, 1e-10))


def run_critical_phenomena():
    print("=" * 60)
    print("CRITICAL PHENOMENA MEASUREMENTS")
    print("=" * 60)

    coupling = PillarCoupling()

    # Sweep lambda_gl across the critical region
    lambdas = np.logspace(-1, 1.5, 20)
    results_per_lambda = []

    print(f"\n{'lambda':>8s} {'xi':>6s} {'tau':>6s} {'chi':>6s} "
          f"{'order':>6s} {'energy':>8s} {'regime':>10s}")
    print("-" * 60)

    for lam in lambdas:
        cfg = FieldConfig(dt=0.01, gamma=0.01, lambda_gl=lam, temperature=0.001)
        evolver = FieldEvolver(cfg)

        # Multiple seeds for statistics
        xi_vals = []
        tau_vals = []
        chi_vals = []
        order_vals = []
        energy_vals = []

        for seed in range(5):
            rng = np.random.RandomState(seed * 137 + 42)
            field = rng.uniform(0.1, 0.9, 16)

            # Evolve
            for _ in range(100):
                rhs = evolver.rhs(field, 0.01)
                field = field + 0.01 * rhs
                field = np.clip(field, 0.0, 1.0)

            xi = measure_correlation_length(field)
            tau, _ = measure_relaxation_time(evolver, field.copy(), 0.01)
            chi = measure_susceptibility(evolver, field.copy(), 0.01)
            order = measure_order_parameter(field)
            energy = evolver.compute_energy(field)

            xi_vals.append(xi)
            tau_vals.append(tau)
            chi_vals.append(chi)
            order_vals.append(order)
            energy_vals.append(energy)

        xi_mean = np.mean(xi_vals)
        tau_mean = np.mean(tau_vals)
        chi_mean = np.mean(chi_vals)
        order_mean = np.mean(order_vals)
        energy_mean = np.mean(energy_vals)

        # Classify regime
        if energy_mean < 0.001:
            regime = 'collapse'
        elif order_mean > 0.9:
            regime = 'ordered'
        elif tau_mean > 400:
            regime = 'critical'
        else:
            regime = 'disordered'

        print(f"  {lam:8.3f} {xi_mean:6.2f} {tau_mean:6.1f} {chi_mean:6.3f} "
              f"{order_mean:6.3f} {energy_mean:8.5f} {regime:>10s}")

        results_per_lambda.append({
            'lambda_gl': lam,
            'xi': xi_mean,
            'tau': tau_mean,
            'chi': chi_mean,
            'order': order_mean,
            'energy': energy_mean,
            'regime': regime,
            'xi_std': float(np.std(xi_vals)),
            'tau_std': float(np.std(tau_vals)),
            'chi_std': float(np.std(chi_vals)),
        })

    # Find critical point: max relaxation time
    taus = [r['tau'] for r in results_per_lambda]
    crit_idx = np.argmax(taus)
    crit = results_per_lambda[crit_idx]

    print(f"\n--- Critical Point Estimate ---")
    print(f"  lambda_gl^c = {crit['lambda_gl']:.3f}")
    print(f"  xi^c = {crit['xi']:.2f}")
    print(f"  tau^c = {crit['tau']:.1f} (max)")
    print(f"  chi^c = {crit['chi']:.3f}")
    print(f"  order^c = {crit['order']:.3f}")

    # Check for power-law scaling near critical point
    # xi ~ |lambda - lambda_c|^(-nu)
    # tau ~ |lambda - lambda_c|^(-z*nu)
    near_crit = [r for r in results_per_lambda
                 if 0.5 < abs(r['lambda_gl'] / crit['lambda_gl'] - 1) < 0.5]
    if len(near_crit) >= 3:
        print(f"\n--- Scaling Near Critical Point ---")
        for r in near_crit:
            dx = abs(r['lambda_gl'] - crit['lambda_gl'])
            if dx > 0.01:
                print(f"  |dlambda|={dx:.3f}: xi={r['xi']:.2f} tau={r['tau']:.1f} chi={r['chi']:.3f}")

    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'critical_phenomena.json')
    with open(out, 'w') as f:
        json.dump({
            'results': results_per_lambda,
            'critical_point': crit,
        }, f, indent=2)
    print(f"\nResults saved to {out}")

    return results_per_lambda


if __name__ == '__main__':
    run_critical_phenomena()
