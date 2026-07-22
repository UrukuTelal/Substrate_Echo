#!/usr/bin/env python3
"""
E. Intervention Experiments — controlled perturbations.

Tests:
1. Remove a vortex → measure field recovery
2. Weaken a pillar → measure restoration
3. Isolate an agent → measure social reconnection
4. Inject field energy → measure dissipation

Each intervention measures recovery trajectory and time.
"""
import sys, os, json, time, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.field_evolution import FieldEvolver, FieldConfig
from substrate_echo.dynamics.conservation import ConservationFramework
from substrate_echo.dynamics.pillar_coupling import PillarCoupling
from substrate_echo.dynamics.vortex_dynamics import VortexDynamics, VortexConfig, Vortex
from substrate_echo.core.multi_agent_dynamics import SocialField, SocialConfig


def measure_recovery(baseline_metric, trajectory, threshold=0.1):
    """Measure recovery time and quality.
    
    Returns (recovery_tick, final_deviation).
    """
    if not trajectory:
        return 0, 0.0

    for i, val in enumerate(trajectory):
        deviation = abs(val - baseline_metric) / max(abs(baseline_metric), 1e-10)
        if deviation < threshold:
            return i, deviation

    return len(trajectory), abs(trajectory[-1] - baseline_metric) / max(abs(baseline_metric), 1e-10)


def experiment_vortex_removal():
    """Test: remove a vortex, measure field recovery."""
    print("\n--- Vortex Removal Intervention ---")
    vd = VortexDynamics(VortexConfig(amplitude_threshold=0.5))

    # Create field with vortices
    grid_size = 32
    amp = np.ones((grid_size, grid_size))
    phase = np.zeros((grid_size, grid_size))

    # Add vortices
    vortex_positions = [(10, 10, 1), (22, 22, -1)]
    for vx, vy, w in vortex_positions:
        for y in range(grid_size):
            for x in range(grid_size):
                dx, dy = x - vx, y - vy
                r = math.sqrt(dx*dx + dy*dy)
                if r > 0.01:
                    amp[y, x] *= math.tanh(r / 3.0)
                    phase[y, x] += w * math.atan2(dy, dx)

    # Detect vortices
    detected = vd.identify_vortices(amp, phase)
    initial_count = len(detected)
    initial_energy = float(np.mean(amp**2))

    # "Remove" one vortex by modifying the field
    if detected:
        target = detected[0]
        tx, ty = int(target.position[0]), int(target.position[1])
        # Smooth out the vortex core
        for y in range(max(0, ty-3), min(grid_size, ty+4)):
            for x in range(max(0, tx-3), min(grid_size, tx+4)):
                r = math.sqrt((x-tx)**2 + (y-ty)**2)
                if r < 3.0:
                    amp[y, x] = min(amp[y, x] + 0.5, 1.0)
                    phase[y, x] *= 0.3

    # Measure recovery: does the field re-organize?
    energy_trajectory = []
    for step in range(100):
        # Simple diffusion relaxation
        amp_new = amp.copy()
        for y in range(1, grid_size-1):
            for x in range(1, grid_size-1):
                laplacian = (amp[y+1,x] + amp[y-1,x] + amp[y,x+1] + amp[y,x-1] - 4*amp[y,x])
                amp_new[y, x] = amp[y, x] + 0.01 * laplacian
        amp = np.clip(amp_new, 0.0, 1.0)
        energy_trajectory.append(float(np.mean(amp**2)))

    remaining = vd.identify_vortices(amp, phase)
    recovery_tick, deviation = measure_recovery(initial_energy, energy_trajectory)

    print(f"  Initial vortices: {initial_count}")
    print(f"  Removed: 1")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Energy recovery: tick={recovery_tick}, deviation={deviation:.4f}")

    return {
        'initial_vortices': initial_count,
        'remaining_vortices': len(remaining),
        'recovery_tick': recovery_tick,
        'deviation': deviation,
    }


def experiment_pillar_weakening():
    """Test: weaken one pillar, measure restoration."""
    print("\n--- Pillar Weakening Intervention ---")
    cfg = FieldConfig(dt=0.01, gamma=0.01, lambda_gl=1.0, temperature=0.001)
    evolver = FieldEvolver(cfg)
    coupling = PillarCoupling()
    conservation = ConservationFramework(enabled=True)

    # Establish equilibrium
    field = np.full(16, 0.5)
    for _ in range(200):
        rhs = evolver.rhs(field, cfg.dt)
        field = field + cfg.dt * rhs
        field = np.clip(field, 0.0, 1.0)

    baseline_norm = float(np.linalg.norm(field))
    baseline_energy = evolver.compute_energy(field)
    conservation.set_norm_target(float(np.linalg.norm(coupling.project_to_pillars(field))))

    # Weaken pillar 5 (Integrity) — set to near zero
    field[5] = 0.01
    weakened_norm = float(np.linalg.norm(field))
    weakened_energy = evolver.compute_energy(field)

    # Track recovery
    norm_trajectory = []
    energy_trajectory = []
    for step in range(300):
        rhs = evolver.rhs(field, cfg.dt)
        field = field + cfg.dt * rhs
        field = np.clip(field, 0.0, 1.0)
        norm_trajectory.append(float(np.linalg.norm(field)))
        energy_trajectory.append(evolver.compute_energy(field))

    norm_recovery, norm_dev = measure_recovery(baseline_norm, norm_trajectory)
    energy_recovery, energy_dev = measure_recovery(baseline_energy, energy_trajectory)
    final_pillar5 = field[5]

    print(f"  Baseline norm: {baseline_norm:.4f}")
    print(f"  Weakened norm: {weakened_norm:.4f}")
    print(f"  Norm recovery: tick={norm_recovery}, dev={norm_dev:.4f}")
    print(f"  Energy recovery: tick={energy_recovery}, dev={energy_dev:.4f}")
    print(f"  Pillar 5 restored: {final_pillar5:.4f}")

    return {
        'baseline_norm': baseline_norm,
        'weakened_norm': weakened_norm,
        'norm_recovery_tick': norm_recovery,
        'energy_recovery_tick': energy_recovery,
        'pillar5_restored': final_pillar5,
    }


def experiment_agent_isolation():
    """Test: isolate an agent, measure social reconnection."""
    print("\n--- Agent Isolation Intervention ---")
    sf = SocialField(SocialConfig(
        influence_strength=0.05,
        influence_range=1.5,
        max_communication_range=1.0,
    ))

    n_agents = 8
    roles = ['perceiver', 'analyzer', 'synthesizer', 'guardian'] * 2
    for i in range(n_agents):
        state = np.random.uniform(0.3, 0.7, 16)
        sf.add_agent(f"a_{i}", roles[i], state)

    # Establish communication
    for _ in range(50):
        agents = list(sf.get_all_agents().keys())
        for i, a_id in enumerate(agents):
            for b_id in agents[i+1:i+3]:
                sf.send_message(a_id, b_id, "update", sf.get_agent(a_id).state)
        for aid in agents:
            influence = sf.compute_social_influence(aid)
            agent = sf.get_agent(aid)
            agent.state = np.clip(agent.state + influence, 0.0, 1.0)

    baseline_coherence = sf.compute_collective_coherence()

    # Isolate agent 0: set its influence range to 0 and move far away
    target = sf.get_agent("a_0")
    original_state = target.state.copy()
    target.state = np.random.uniform(0.0, 0.1, 16)  # move far away

    # Disable its communication by moving outside range
    sf.config.max_communication_range = 0.1

    coherence_trajectory = []
    for step in range(200):
        # Restore communication range halfway through
        if step == 100:
            sf.config.max_communication_range = 1.0
            target.state = original_state.copy()  # restore state partially

        agents = list(sf.get_all_agents().keys())
        for i, a_id in enumerate(agents):
            for b_id in agents[i+1:i+3]:
                sf.send_message(a_id, b_id, "update", sf.get_agent(a_id).state)

        for aid in agents:
            influence = sf.compute_social_influence(aid)
            agent = sf.get_agent(aid)
            agent.state = np.clip(agent.state + influence, 0.0, 1.0)

        coherence_trajectory.append(sf.compute_collective_coherence())

    recovery_tick, dev = measure_recovery(baseline_coherence, coherence_trajectory)

    print(f"  Baseline coherence: {baseline_coherence:.3f}")
    print(f"  Isolated coherence: {coherence_trajectory[0]:.3f}")
    print(f"  Recovery tick: {recovery_tick}")
    print(f"  Final coherence: {coherence_trajectory[-1]:.3f}")

    return {
        'baseline_coherence': baseline_coherence,
        'recovery_tick': recovery_tick,
        'final_coherence': coherence_trajectory[-1],
    }


def experiment_energy_injection():
    """Test: inject field energy, measure dissipation."""
    print("\n--- Energy Injection Intervention ---")
    cfg = FieldConfig(dt=0.01, gamma=0.01, lambda_gl=1.0, temperature=0.001)
    evolver = FieldEvolver(cfg)

    # Establish equilibrium
    field = np.full(16, 0.5)
    for _ in range(200):
        rhs = evolver.rhs(field, cfg.dt)
        field = field + cfg.dt * rhs
        field = np.clip(field, 0.0, 1.0)

    baseline_energy = evolver.compute_energy(field)

    # Inject energy: boost all pillars
    field = np.clip(field + 0.3, 0.0, 1.0)
    injected_energy = evolver.compute_energy(field)

    # Track dissipation
    energy_trajectory = []
    for step in range(500):
        rhs = evolver.rhs(field, cfg.dt)
        field = field + cfg.dt * rhs
        field = np.clip(field, 0.0, 1.0)
        energy_trajectory.append(evolver.compute_energy(field))

    recovery_tick, dev = measure_recovery(baseline_energy, energy_trajectory)
    half_life = None
    for i, e in enumerate(energy_trajectory):
        if e <= (injected_energy + baseline_energy) / 2:
            half_life = i
            break

    print(f"  Baseline energy: {baseline_energy:.5f}")
    print(f"  Injected energy: {injected_energy:.5f}")
    print(f"  Delta: {injected_energy - baseline_energy:.5f}")
    print(f"  Recovery tick: {recovery_tick}")
    print(f"  Half-life: {half_life}")
    print(f"  Final energy: {energy_trajectory[-1]:.5f}")

    return {
        'baseline_energy': baseline_energy,
        'injected_energy': injected_energy,
        'recovery_tick': recovery_tick,
        'half_life': half_life,
        'final_energy': energy_trajectory[-1],
    }


def run_all():
    print("=" * 60)
    print("INTERVENTION EXPERIMENTS")
    print("=" * 60)

    results = {}
    results['vortex_removal'] = experiment_vortex_removal()
    results['pillar_weakening'] = experiment_pillar_weakening()
    results['agent_isolation'] = experiment_agent_isolation()
    results['energy_injection'] = experiment_energy_injection()

    print("\n--- Summary ---")
    for name, r in results.items():
        recovered = all(v > 0 for v in r.values() if isinstance(v, (int, float)))
        print(f"  {name}: {'OK' if recovered else 'CHECK'}")

    out = os.path.join(os.path.dirname(__file__), '..', 'intervention_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")

    return results


if __name__ == '__main__':
    run_all()
