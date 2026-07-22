#!/usr/bin/env python3
"""
Long-duration stability test with metrics logging.

Tracks: conservation drift, energy stability, norm drift growth,
CPU time per tick, memory count.
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.field_evolution import FieldEvolver, FieldConfig
from substrate_echo.dynamics.conservation import ConservationFramework
from substrate_echo.dynamics.pillar_coupling import PillarCoupling
from substrate_echo.dynamics.topology_transitions import TopologyManager


def run_stability(num_ticks=5000, log_interval=100):
    cfg = FieldConfig(dt=0.01, gamma=0.01, lambda_gl=1.0, temperature=0.001)
    evolver = FieldEvolver(cfg)
    conservation = ConservationFramework(enabled=True)
    coupling = PillarCoupling()
    topology = TopologyManager()

    field = np.full(16, 0.5)
    pillar = coupling.project_to_pillars(field)
    conservation.set_norm_target(float(np.linalg.norm(pillar)))

    history = {
        'ticks': [], 'norm': [], 'energy': [], 'norm_drift': [],
        'cpu_ms': [], 'conservation_pass': [], 'topo_events': [],
    }

    prev_norm = float(np.linalg.norm(pillar))
    topo_count = 0
    t0 = time.time()

    for tick in range(num_ticks):
        tick_start = time.time()

        # Field evolution
        rhs = evolver.rhs(field, cfg.dt)
        field = field + cfg.dt * rhs
        field = np.clip(field, 0.0, 1.0)

        # Projection
        pillar = coupling.project_to_pillars(field)

        # Conservation
        results = conservation.check_all(pillar)
        passed = all(r.passed for r in results)

        # Topology transitions
        events = topology.check_transitions(field)
        topo_count += len(events)

        # Metrics
        norm = float(np.linalg.norm(pillar))
        energy = evolver.compute_energy(field)
        drift = abs(norm - prev_norm)
        cpu_ms = (time.time() - tick_start) * 1000

        if tick % log_interval == 0:
            history['ticks'].append(tick)
            history['norm'].append(norm)
            history['energy'].append(energy)
            history['norm_drift'].append(drift)
            history['cpu_ms'].append(cpu_ms)
            history['conservation_pass'].append(passed)
            history['topo_events'].append(topo_count)

            if tick % (log_interval * 10) == 0:
                print(f"  tick {tick:>5d}: norm={norm:.6f} energy={energy:.6f} "
                      f"drift={drift:.2e} cpu={cpu_ms:.2f}ms cons={passed} topo={topo_count}")

        prev_norm = norm

    elapsed = time.time() - t0

    # Summary
    norms = np.array(history['norm'])
    energies = np.array(history['energy'])
    drifts = np.array(history['norm_drift'])

    print(f"\n{'='*60}")
    print(f"STABILITY: {num_ticks} ticks in {elapsed:.1f}s "
          f"({elapsed/num_ticks*1000:.2f}ms/tick)")
    print(f"  Norm:  initial={norms[0]:.6f} final={norms[-1]:.6f} "
          f"drift={abs(norms[-1]-norms[0]):.6f}")
    print(f"  Norm drift: avg={np.mean(drifts):.2e} max={np.max(drifts):.2e} "
          f"std={np.std(drifts):.2e}")
    print(f"  Energy: initial={energies[0]:.6f} final={energies[-1]:.6f} "
          f"range=[{np.min(energies):.6f}, {np.max(energies):.6f}]")
    print(f"  Conservation pass: {sum(history['conservation_pass'])}/{len(history['conservation_pass'])} "
          f"({sum(history['conservation_pass'])/len(history['conservation_pass']):.1%})")
    print(f"  Topo events: {topo_count}")
    print(f"  Avg cpu/tick: {np.mean(history['cpu_ms']):.3f}ms")

    # Drift growth analysis
    if len(norms) > 10:
        mid = len(norms) // 2
        first_half_drift = np.mean(drifts[:mid])
        second_half_drift = np.mean(drifts[mid:])
        print(f"  Drift growth: first_half={first_half_drift:.2e} "
              f"second_half={second_half_drift:.2e} "
              f"ratio={second_half_drift/first_half_drift:.2f}x")

    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'stability_results.json')
    with open(out, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Results saved to {out}")

    return history


if __name__ == '__main__':
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    run_stability(num_ticks=ticks)
