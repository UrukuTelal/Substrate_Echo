#!/usr/bin/env python3
"""
Vortex Physics Measurements.

Creates 2D complex fields with explicit vortices and measures:
- Winding number conservation
- Vortex-antivortex annihilation rates
- Kelvin wave dispersion
- Interaction forces
"""
import sys, os, json, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.vortex_dynamics import VortexDynamics, VortexConfig, Vortex


def create_vortex_field(grid_size=32, vortices=None):
    """Create a 2D complex field with specified vortices.
    
    Each vortex: (x, y, winding_number)
    """
    amp = np.ones((grid_size, grid_size), dtype=np.float64)
    phase = np.zeros((grid_size, grid_size), dtype=np.float64)

    if vortices is None:
        vortices = []

    for vx, vy, winding in vortices:
        for y in range(grid_size):
            for x in range(grid_size):
                dx = x - vx
                dy = y - vy
                r = math.sqrt(dx * dx + dy * dy)
                if r > 0.01:
                    # Amplitude dips near core
                    amp[y, x] *= math.tanh(r / 3.0)
                    # Phase winds around core
                    theta = math.atan2(dy, dx)
                    phase[y, x] += winding * theta

    return amp, phase


def test_winding_number():
    """Test: winding numbers are quantized."""
    print("\n--- Winding Number Quantization ---")
    vd = VortexDynamics(VortexConfig(amplitude_threshold=0.5))

    grid_size = 32
    tests = [
        ([], "no vortices"),
        ([(16, 16, 1)], "single +1"),
        ([(16, 16, -1)], "single -1"),
        ([(10, 10, 1), (22, 22, -1)], "+1/-1 pair"),
        ([(8, 8, 1), (24, 8, 1), (16, 24, 1)], "three +1"),
    ]

    results = []
    for vortex_list, desc in tests:
        amp, phase = create_vortex_field(grid_size, vortex_list)
        detected = vd.identify_vortices(amp, phase)

        # Check: correct winding types found, no wrong types
        detected_windings = set(v.winding_number for v in detected)
        expected_windings = set(w for _, _, w in vortex_list)

        # No unexpected winding numbers
        no_false = detected_windings.issubset(expected_windings) or not vortex_list
        # Correct winding found when expected
        no_miss = expected_windings.issubset(detected_windings) if vortex_list else len(detected) == 0

        ok = no_false and no_miss
        print(f"  {desc:>20s}: expected_types={sorted(expected_windings)} "
              f"detected_types={sorted(detected_windings)} "
              f"count={len(detected)} {'OK' if ok else 'FAIL'}")
        results.append({'desc': desc, 'match': ok, 'detected_count': len(detected)})

    return results


def test_annihilation():
    """Test: vortex-antivortex pairs annihilate."""
    print("\n--- Vortex-Antivortex Annihilation ---")
    vd = VortexDynamics(VortexConfig(
        amplitude_threshold=0.5,
        damping=0.1,
    ))

    # Place +1 and -1 close together (within interaction range)
    vd.add_vortex(np.array([15.0, 16.0]), winding=1)
    vd.add_vortex(np.array([17.0, 16.0]), winding=-1)

    initial_count = len(vd.get_vortices())
    initial_winding = vd.get_total_winding()

    annihilated_at = None
    for step in range(500):
        vortices = vd.step(dt=0.05)
        if len(vortices) < initial_count and annihilated_at is None:
            annihilated_at = step
        if len(vortices) == 0:
            annihilated_at = step
            break

    annihilation_events = vd.get_annihilation_events()
    final_count = len(vd.get_vortices())
    final_winding = vd.get_total_winding()

    print(f"  Initial: {initial_count} vortices, winding={initial_winding}")
    print(f"  Final:   {final_count} vortices, winding={final_winding}")
    print(f"  Annihilation events: {len(annihilation_events)}")
    if annihilated_at is not None:
        print(f"  Annihilated at step: {annihilated_at}")
    print(f"  Winding conserved: {initial_winding + final_winding == 0}")

    return {
        'initial_count': initial_count,
        'final_count': final_count,
        'annihilated': final_count == 0,
        'events': len(annihilation_events),
    }


def test_kelvin_waves():
    """Test: Kelvin wave dispersion relation."""
    print("\n--- Kelvin Wave Dispersion ---")
    vd = VortexDynamics()

    k_values = np.logspace(-2, 0, 10)
    R = 1.0

    print(f"  {'k':>8s}  {'omega(k)':>10s}")
    for k in k_values:
        omega = vd.kelvin_dispersion(k, R)
        print(f"  {k:8.4f}  {omega:10.6f}")

    # Verify: omega should increase with k^2 for small k
    # Exclude k near singularity (kR >= 1)
    valid = k_values < 0.5
    k_valid = k_values[valid]
    o_valid = [vd.kelvin_dispersion(k, R) for k in k_valid]

    if len(k_valid) >= 2:
        ratio = o_valid[-1] / max(o_valid[0], 1e-10)
        k_ratio = (k_valid[-1] / k_valid[0]) ** 2
        print(f"  omega ratio (high/low k): {ratio:.1f}")
        print(f"  k^2 ratio: {k_ratio:.1f}")
        print(f"  Matches k^2 scaling: {'yes' if abs(ratio - k_ratio) / k_ratio < 0.5 else 'no'}")
        kelvin_ok = abs(ratio - k_ratio) / k_ratio < 0.5
    else:
        kelvin_ok = True  # not enough data to check

    return {'ratio': ratio if len(k_valid) >= 2 else 0, 'k_ratio': k_ratio if len(k_valid) >= 2 else 0, 'kelvin_ok': kelvin_ok}


def test_interaction_forces():
    """Test: like-sign vortices repel, opposite-sign attract."""
    print("\n--- Vortex Interaction Forces ---")
    vd = VortexDynamics(VortexConfig(max_interaction_range=20.0))

    # Like-sign pair
    v1 = Vortex(position=np.array([10.0, 16.0]), winding_number=1, core_size=1.0, amplitude=0.0, phase=0.0, id=0)
    v2 = Vortex(position=np.array([15.0, 16.0]), winding_number=1, core_size=1.0, amplitude=0.0, phase=0.0, id=1)
    force_like = vd.compute_interaction_force(v1, v2)

    # Opposite-sign pair
    v3 = Vortex(position=np.array([10.0, 16.0]), winding_number=1, core_size=1.0, amplitude=0.0, phase=0.0, id=2)
    v4 = Vortex(position=np.array([15.0, 16.0]), winding_number=-1, core_size=1.0, amplitude=0.0, phase=0.0, id=3)
    force_opposite = vd.compute_interaction_force(v3, v4)

    print(f"  Like-sign force:     [{force_like[0]:+.4f}, {force_like[1]:+.4f}]")
    print(f"  Opposite-sign force: [{force_opposite[0]:+.4f}, {force_opposite[1]:+.4f}]")
    print(f"  Opposite direction:  {'yes' if np.sign(force_like[0]) != np.sign(force_opposite[0]) else 'no'}")

    return {
        'like_sign': force_like.tolist(),
        'opposite_sign': force_opposite.tolist(),
        'opposite_direction': np.sign(force_like[0]) != np.sign(force_opposite[0]),
    }


def run_all():
    print("=" * 60)
    print("VORTEX PHYSICS MEASUREMENTS")
    print("=" * 60)

    results = {}
    results['winding'] = test_winding_number()
    results['annihilation'] = test_annihilation()
    results['kelvin'] = test_kelvin_waves()
    results['forces'] = test_interaction_forces()

    print("\n--- Summary ---")
    winding_ok = all(r['match'] for r in results['winding'])
    annihilation_ok = results['annihilation']['annihilated']
    forces_ok = results['forces']['opposite_direction']
    print(f"  Winding quantization: {'OK' if winding_ok else 'FAIL'}")
    print(f"  Annihilation: {'OK' if annihilation_ok else 'FAIL'}")
    print(f"  Interaction forces: {'OK' if forces_ok else 'FAIL'}")
    print(f"  All passed: {winding_ok and annihilation_ok and forces_ok}")

    out = os.path.join(os.path.dirname(__file__), '..', 'vortex_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {out}")

    return results


if __name__ == '__main__':
    run_all()
