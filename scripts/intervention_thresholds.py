#!/usr/bin/env python3
"""
E2+E3. Intervention Thresholds + Energy Injection Across Regimes.

E2: Find the recovery boundary for pillar weakening
E3: Compare energy injection half-life across stable/critical/chaotic regimes
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.field_evolution import FieldEvolver, FieldConfig
from substrate_echo.dynamics.pillar_coupling import PillarCoupling


def measure_recovery_time(baseline, trajectory, threshold=0.1):
    """Measure when trajectory returns within threshold of baseline."""
    for i, val in enumerate(trajectory):
        deviation = abs(val - baseline) / max(abs(baseline), 1e-10)
        if deviation < threshold:
            return i
    return len(trajectory)


def run_pillar_weakening():
    """E2: Find recovery boundary."""
    print("\n--- Pillar Weakening Threshold ---")
    coupling = PillarCoupling()

    weakening_values = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99]

    results = {}
    for weak in weakening_values:
        # Equilibrate
        cfg = FieldConfig(dt=0.01, gamma=0.01, lambda_gl=1.0, temperature=0.001)
        evolver = FieldEvolver(cfg)
        field = np.full(16, 0.5)
        for _ in range(300):
            rhs = evolver.rhs(field, cfg.dt)
            field = field + cfg.dt * rhs
            field = np.clip(field, 0.0, 1.0)

        baseline_energy = evolver.compute_energy(field)
        baseline_pillar5 = field[5]

        # Weaken pillar 5
        field[5] = weak
        weakened_energy = evolver.compute_energy(field)

        # Track recovery
        energy_trajectory = []
        pillar5_trajectory = []
        for step in range(500):
            rhs = evolver.rhs(field, cfg.dt)
            field = field + cfg.dt * rhs
            field = np.clip(field, 0.0, 1.0)
            energy_trajectory.append(evolver.compute_energy(field))
            pillar5_trajectory.append(field[5])

        energy_recovery = measure_recovery_time(baseline_energy, energy_trajectory)
        pillar5_recovery = measure_recovery_time(baseline_pillar5, pillar5_trajectory)
        final_pillar5 = field[5]

        recovered = final_pillar5 > baseline_pillar5 * 0.5
        results[weak] = {
            'weakened_to': weak,
            'energy_recovery_tick': energy_recovery,
            'pillar5_recovery_tick': pillar5_recovery,
            'final_pillar5': float(final_pillar5),
            'recovered': bool(recovered),
        }

        print(f"  weaken={weak:.3f}: "
              f"energy_recovery={energy_recovery:>4d} "
              f"pillar5_recovery={pillar5_recovery:>4d} "
              f"final={final_pillar5:.4f} "
              f"{'OK' if recovered else 'FAIL'}")

    # Find critical threshold
    recovered_vals = [(w, r['recovered']) for w, r in results.items()]
    boundary = None
    for i in range(len(recovered_vals) - 1):
        if recovered_vals[i][1] and not recovered_vals[i+1][1]:
            boundary = (recovered_vals[i][0] + recovered_vals[i+1][0]) / 2
            break

    if boundary:
        print(f"\n  Recovery boundary: ~{boundary:.3f}")
    else:
        all_recovered = all(r['recovered'] for r in results.values())
        if all_recovered:
            print(f"\n  Recovery: ALL values recover (no failure point found)")
        else:
            print(f"\n  Recovery: fails at all tested values")

    return results


def run_energy_injection_across_regimes():
    """E3: Compare half-life across regimes."""
    print("\n--- Energy Injection Across Regimes ---")

    regimes = {
        'stable':   {'lambda_gl': 10.0, 'gamma': 0.01},
        'critical': {'lambda_gl': 2.8,  'gamma': 0.01},
        'chaotic':  {'lambda_gl': 0.5,  'gamma': 0.001},
    }

    results = {}
    for regime_name, params in regimes.items():
        cfg = FieldConfig(dt=0.01, gamma=params['gamma'],
                         lambda_gl=params['lambda_gl'], temperature=0.001)
        evolver = FieldEvolver(cfg)

        # Equilibrate
        field = np.full(16, 0.5)
        for _ in range(300):
            rhs = evolver.rhs(field, cfg.dt)
            field = field + cfg.dt * rhs
            field = np.clip(field, 0.0, 1.0)

        baseline_energy = evolver.compute_energy(field)

        # Inject energy
        field = np.clip(field + 0.3, 0.0, 1.0)
        injected_energy = evolver.compute_energy(field)

        # Track dissipation
        energy_trajectory = []
        for step in range(500):
            rhs = evolver.rhs(field, cfg.dt)
            field = field + cfg.dt * rhs
            field = np.clip(field, 0.0, 1.0)
            energy_trajectory.append(evolver.compute_energy(field))

        # Half-life
        half_energy = (injected_energy + baseline_energy) / 2
        half_life = None
        for i, e in enumerate(energy_trajectory):
            if e <= half_energy:
                half_life = i
                break

        # Recovery time
        recovery_time = measure_recovery_time(baseline_energy, energy_trajectory)

        results[regime_name] = {
            'lambda_gl': params['lambda_gl'],
            'gamma': params['gamma'],
            'baseline_energy': float(baseline_energy),
            'injected_energy': float(injected_energy),
            'half_life': half_life,
            'recovery_time': recovery_time,
            'final_energy': energy_trajectory[-1],
        }

        print(f"  {regime_name:>10s}: "
              f"baseline={baseline_energy:.5f} "
              f"injected={injected_energy:.5f} "
              f"half_life={half_life} "
              f"recovery={recovery_time}")

    # Comparison
    print(f"\n--- Half-Life Comparison ---")
    for name, r in results.items():
        hl = r['half_life']
        print(f"  {name:>10s}: half_life = {hl}")

    critical_hl = results['critical']['half_life']
    stable_hl = results['stable']['half_life']
    chaotic_hl = results['chaotic']['half_life']

    if critical_hl and stable_hl and critical_hl > stable_hl:
        print(f"\n  Critical regime has LONGER half-life ({critical_hl} vs {stable_hl})")
        print(f"  Consistent with critical slowing down")
    else:
        print(f"\n  Half-life comparison: critical={critical_hl}, stable={stable_hl}")

    return results


def run_all():
    print("=" * 60)
    print("INTERVENTION THRESHOLDS + REGIME COMPARISON")
    print("=" * 60)

    results = {}
    results['pillar_weakening'] = run_pillar_weakening()
    results['energy_regimes'] = run_energy_injection_across_regimes()

    out = os.path.join(os.path.dirname(__file__), '..', 'intervention_thresholds.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    return results


if __name__ == '__main__':
    run_all()
