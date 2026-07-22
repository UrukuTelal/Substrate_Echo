#!/usr/bin/env python3
"""
GL Parameter Sweep — Phase Diagram Generator

Sweeps λ_gl, η, D (diffusion), γ (dissipation) to identify
regimes: stable, oscillatory, collapse, chaotic, critical slowing.
"""
import sys, os, time, json
import numpy as np
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.field_evolution import FieldEvolver, FieldConfig


def classify_behavior(energy_history, norm_history, n_transient=50):
    """Classify dynamical regime from time series."""
    if len(energy_history) < n_transient + 10:
        return 'insufficient'

    e = np.array(energy_history[n_transient:])
    n = np.array(norm_history[n_transient:])

    e_mean = np.mean(e)
    e_std = np.std(e)
    n_mean = np.mean(n)
    n_std = np.std(n)

    # Collapse: energy near zero
    if e_mean < 1e-6:
        return 'collapse'

    # Chaos: high variance relative to mean
    cv_energy = e_std / max(e_mean, 1e-12)
    cv_norm = n_std / max(n_mean, 1e-12)

    if cv_energy > 0.5 or cv_norm > 0.5:
        return 'chaotic'

    # Oscillatory: periodic behavior (high std but bounded, non-decaying)
    if cv_energy > 0.1 and e_std > 0.01:
        # Check for oscillation: count zero crossings of mean-subtracted
        e_centered = e - e_mean
        crossings = np.sum(np.abs(np.diff(np.sign(e_centered))) > 0)
        if crossings > len(e) * 0.1:
            return 'oscillatory'

    # Stable: low variance, non-zero energy
    if cv_energy < 0.05 and cv_norm < 0.05:
        return 'stable'

    # Critical: moderate variance, slow drift
    return 'critical'


def sweep_parameter(param_name, param_range, base_config, n_ticks=500):
    """Sweep a single parameter and classify behavior."""
    results = []
    for val in param_range:
        cfg = FieldConfig(
            dt=base_config.dt,
            gamma=base_config.gamma if param_name != 'gamma' else val,
            lambda_gl=base_config.lambda_gl if param_name != 'lambda_gl' else val,
            temperature=base_config.temperature if param_name != 'temperature' else val,
        )

        evolver = FieldEvolver(cfg)
        field = np.full(16, 0.5)

        energy_hist = []
        norm_hist = []

        for tick in range(n_ticks):
            rhs = evolver.rhs(field, cfg.dt)
            field = field + cfg.dt * rhs
            field = np.clip(field, 0.0, 1.0)

            if tick % 5 == 0:
                energy_hist.append(evolver.compute_energy(field))
                norm_hist.append(float(np.linalg.norm(field)))

        regime = classify_behavior(energy_hist, norm_hist)
        results.append({
            'param': param_name,
            'value': val,
            'regime': regime,
            'final_energy': energy_hist[-1] if energy_hist else 0,
            'final_norm': norm_hist[-1] if norm_hist else 0,
            'energy_range': [float(np.min(energy_hist)), float(np.max(energy_hist))] if energy_hist else [0, 0],
        })

        print(f"  {param_name}={val:.4f} -> {regime} "
              f"(E={energy_hist[-1]:.6f}, N={norm_hist[-1]:.6f})" if energy_hist else
              f"  {param_name}={val:.4f} -> {regime}")

    return results


def run_sweep(n_ticks=500):
    print("=" * 60)
    print("GL PARAMETER SWEEP")
    print("=" * 60)

    base = FieldConfig(dt=0.01, gamma=0.01, lambda_gl=1.0, temperature=0.001)

    all_results = {}

    # Sweep lambda_gl (coupling strength)
    print("\n--- lambda_gl sweep ---")
    lambdas = np.logspace(-2, 2, 15)
    all_results['lambda_gl'] = sweep_parameter('lambda_gl', lambdas, base, n_ticks)

    # Sweep gamma (dissipation)
    print("\n--- gamma sweep ---")
    gammas = np.logspace(-3, 1, 15)
    all_results['gamma'] = sweep_parameter('gamma', gammas, base, n_ticks)

    # Sweep temperature (noise)
    print("\n--- temperature sweep ---")
    temps = np.logspace(-4, 0, 15)
    all_results['temperature'] = sweep_parameter('temperature', temps, base, n_ticks)

    # 2D sweep: lambda_gl x gamma
    print("\n--- 2D sweep: lambda_gl x gamma ---")
    lambdas_2d = np.logspace(-1, 1, 8)
    gammas_2d = np.logspace(-2, 0, 8)
    grid = []

    for lam, gam in product(lambdas_2d, gammas_2d):
        cfg = FieldConfig(dt=0.01, gamma=gam, lambda_gl=lam, temperature=0.001)
        evolver = FieldEvolver(cfg)
        field = np.full(16, 0.5)

        energy_hist = []
        norm_hist = []
        for tick in range(n_ticks):
            rhs = evolver.rhs(field, cfg.dt)
            field = field + cfg.dt * rhs
            field = np.clip(field, 0.0, 1.0)
            if tick % 5 == 0:
                energy_hist.append(evolver.compute_energy(field))
                norm_hist.append(float(np.linalg.norm(field)))

        regime = classify_behavior(energy_hist, norm_hist)
        grid.append({
            'lambda_gl': lam,
            'gamma': gam,
            'regime': regime,
            'final_energy': energy_hist[-1] if energy_hist else 0,
        })

    all_results['grid'] = grid

    # Print 2D grid summary
    regimes = {}
    for g in grid:
        key = (round(g['lambda_gl'], 2), round(g['gamma'], 2))
        regimes[key] = g['regime']

    print("\n  Phase Diagram (lambda_gl vs gamma):")
    print(f"  {'gamma/lambda':>12s}", end='')
    for lam in lambdas_2d:
        print(f"  {lam:>6.2f}", end='')
    print()

    for gam in gammas_2d:
        print(f"  {gam:>10.2f}", end='')
        for lam in lambdas_2d:
            r = regimes.get((round(lam, 2), round(gam, 2)), '?')
            symbol = {'stable': 'S', 'collapse': 'C', 'oscillatory': 'O',
                      'chaotic': 'X', 'critical': 'K', 'insufficient': '?'}.get(r, '?')
            print(f"     {symbol}", end='')
        print()

    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'sweep_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")

    # Regime counts
    for key, results in all_results.items():
        if key == 'grid':
            counts = {}
            for g in results:
                counts[g['regime']] = counts.get(g['regime'], 0) + 1
            print(f"  Grid: {counts}")
        elif isinstance(results, list):
            counts = {}
            for r in results:
                counts[r['regime']] = counts.get(r['regime'], 0) + 1
            print(f"  {key}: {counts}")

    return all_results


if __name__ == '__main__':
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    run_sweep(n_ticks=ticks)
