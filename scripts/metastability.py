#!/usr/bin/env python3
"""
3. Metastability Experiment.

Replace single-basin GL potential V(ℱ)=λ(|ℱ|²-η²)² with multi-basin
landscape. Measure: state switching, hysteresis, memory effects.

This tests whether multiple attractors create genuine memory
(persistence of state after perturbation) vs the current
single-basin homeostatic behavior.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.field_evolution import FieldEvolver, FieldConfig
from substrate_echo.dynamics.pillar_coupling import PillarCoupling
from substrate_echo.core.ontological_field import OntologicalField
from substrate_echo.core.attractor_memory import AttractorMemory
from substrate_echo.models.experience import Experience, ExperienceType


class MultiBasinFieldEvolver:
    """Field evolver with multi-basin potential landscape.
    
    V(ℱ) = λ * Σ_b |ℱ - ℱ_b|^2 * (|ℱ - ℱ_b|^2 - r_b^2)
    
    where ℱ_b are basin centers and r_b are basin radii.
    
    This creates multiple stable attractors with barriers between them.
    """
    
    def __init__(self, cfg, basins, basin_weights=None):
        self.cfg = cfg
        self.basins = basins  # List of (center, radius, weight)
        self.n_basins = len(basins)
        
    def rhs(self, field, dt):
        """Multi-basin dynamics with GL damping."""
        gradient = np.zeros_like(field)
        
        for center, radius, weight in self.basins:
            diff = field - center
            r2 = np.sum(diff ** 2)
            
            # Gradient of double-well for this basin
            # ∂V/∂x = 2λ * (x-c) * (2(x-c)² - r²)
            grad = 2 * self.cfg.lambda_gl * diff * (2 * r2 - radius ** 2)
            gradient += weight * grad
        
        # GL damping
        damping = self.cfg.gamma * (self.cfg.temperature - field)
        
        # Noise
        noise = np.sqrt(2 * self.cfg.temperature * dt) * np.random.randn(*field.shape)
        
        return -gradient + damping + noise / dt


def compute_basin_trajectory(field, basins):
    """Which basin is the field closest to?"""
    distances = []
    for center, radius, _ in basins:
        d = np.sqrt(np.sum((field - center) ** 2))
        distances.append(d / radius)  # Normalize by basin radius
    return int(np.argmin(distances)), distances


def run_metastability():
    print("=" * 60)
    print("METASTABILITY EXPERIMENT")
    print("=" * 60)
    
    dim = 16
    
    # Create 3 basins in state space
    rng = np.random.RandomState(42)
    basins_raw = [rng.uniform(0.2, 0.8, dim) for _ in range(3)]
    
    # Make basins well-separated
    for i in range(3):
        basins_raw[i] = basins_raw[i] * 0.5 + 0.25
        basins_raw[i][i % dim] += 0.3  # Push apart
    
    basin_radii = [0.15, 0.15, 0.15]
    basin_weights = [1.0, 1.0, 1.0]
    basins = list(zip(basins_raw, basin_radii, basin_weights))
    
    print(f"\nBasin centers (first 4 dims):")
    for i, (c, r, w) in enumerate(basins):
        print(f"  Basin {i}: center={c[:4].round(3)}, radius={r}, weight={w}")
    
    # ---- Test 1: Basin stability (no noise) ----
    print(f"\n--- Test 1: Basin Stability (no noise) ---")
    cfg = FieldConfig(dt=0.01, gamma=0.1, lambda_gl=5.0, temperature=0.0)
    evolver = MultiBasinFieldEvolver(cfg, basins)
    
    # Start near each basin
    for b_idx in range(3):
        field = basins_raw[b_idx] + rng.randn(dim) * 0.02
        field = np.clip(field, 0.01, 0.99)
        
        trajectory = []
        basin_visits = []
        
        for tick in range(500):
            rhs = evolver.rhs(field, cfg.dt)
            field = field + cfg.dt * rhs
            field = np.clip(field, 0.01, 0.99)
            
            if tick % 10 == 0:
                b, dists = compute_basin_trajectory(field, basins)
                trajectory.append(field.copy())
                basin_visits.append(b)
        
        # Check: does it stay in the same basin?
        final_basin = basin_visits[-1]
        time_in_basin = sum(1 for b in basin_visits if b == final_basin) / len(basin_visits)
        
        print(f"  Start basin {b_idx}: stayed in basin {final_basin} "
              f"{time_in_basin:.1%} of time")
    
    # ---- Test 2: Barrier crossing with noise ----
    print(f"\n--- Test 2: Barrier Crossing (noise sweep) ---")
    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    n_ticks = 2000
    
    barrier_results = {}
    for noise in noise_levels:
        cfg_noisy = FieldConfig(dt=0.01, gamma=0.1, lambda_gl=5.0, temperature=noise)
        evolver_noisy = MultiBasinFieldEvolver(cfg_noisy, basins)
        
        field = basins_raw[0] + rng.randn(dim) * 0.01
        field = np.clip(field, 0.01, 0.99)
        
        basin_visits = []
        visit_lengths = []
        current_basin = 0
        current_run = 0
        
        for tick in range(n_ticks):
            rhs = evolver_noisy.rhs(field, cfg_noisy.dt)
            field = field + cfg_noisy.dt * rhs
            field = np.clip(field, 0.01, 0.99)
            
            b, _ = compute_basin_trajectory(field, basins)
            basin_visits.append(b)
            
            if b == current_basin:
                current_run += 1
            else:
                visit_lengths.append(current_run)
                current_basin = b
                current_run = 1
        
        visit_lengths.append(current_run)
        
        n_switches = sum(1 for i in range(1, len(basin_visits)) 
                        if basin_visits[i] != basin_visits[i-1])
        avg_visit = np.mean(visit_lengths) if visit_lengths else 0
        basin_distribution = [basin_visits.count(i) / len(basin_visits) for i in range(3)]
        
        barrier_results[noise] = {
            'switches': n_switches,
            'switches_per_1000': n_switches / n_ticks * 1000,
            'avg_visit_length': float(avg_visit),
            'basin_distribution': basin_distribution,
        }
        
        print(f"  noise={noise:<6.3f}: switches={n_switches:>4d} "
              f"(={n_switches/n_ticks*1000:.1f}/1k), avg_visit={avg_visit:>7.1f} "
              f"distribution={[f'{d:.2f}' for d in basin_distribution]}")
    
    # ---- Test 3: Hysteresis (drive through basins) ----
    print(f"\n--- Test 3: Hysteresis ---")
    
    # Create a time-varying potential that shifts basin centers
    # This creates a "moving landscape" that tests memory
    
    cfg_hyst = FieldConfig(dt=0.01, gamma=0.1, lambda_gl=3.0, temperature=0.005)
    n_ticks = 1500
    field = basins_raw[0].copy()
    
    hyst_states = []
    hyst_targets = []
    
    for tick in range(n_ticks):
        # Shift basin centers cyclically
        shift_angle = tick / n_ticks * 2 * np.pi
        shifted_basins = []
        for i, (c, r, w) in enumerate(basins):
            new_center = c.copy()
            new_center[0] += 0.2 * np.sin(shift_angle + i * 2 * np.pi / 3)
            new_center[1] += 0.2 * np.cos(shift_angle + i * 2 * np.pi / 3)
            shifted_basins.append((new_center, r, w))
        
        evolver_hyst = MultiBasinFieldEvolver(cfg_hyst, shifted_basins)
        
        rhs = evolver_hyst.rhs(field, cfg_hyst.dt)
        field = field + cfg_hyst.dt * rhs
        field = np.clip(field, 0.01, 0.99)
        
        b, dists = compute_basin_trajectory(field, shifted_basins)
        hyst_states.append(b)
        
        if tick % 200 == 0:
            print(f"  tick {tick:>5d}: basin={b}, dists={[f'{d:.2f}' for d in dists]}")
    
    # Hysteresis: does the field follow the moving minimum, or lag behind?
    lag = 0
    for i in range(1, len(hyst_states)):
        if hyst_states[i] != hyst_states[i-1]:
            lag = i
    print(f"  Last basin switch at tick {lag} (field lags landscape)")
    
    # ---- Test 4: Memory encoding in multi-basin ----
    print(f"\n--- Test 4: Multi-Basin Memory Encoding ---")
    
    cfg_mem = FieldConfig(dt=0.01, gamma=0.1, lambda_gl=5.0, temperature=0.003)
    evolver_mem = MultiBasinFieldEvolver(cfg_mem, basins)
    
    ont = OntologicalField()
    memory = AttractorMemory(field=ont)
    coupling = PillarCoupling()
    
    field = basins_raw[0].copy()
    
    # Phase 1: Learn basin 0 (500 ticks)
    print(f"  Phase 1: Learning basin 0...")
    for tick in range(500):
        rhs = evolver_mem.rhs(field, cfg_mem.dt)
        field = field + cfg_mem.dt * rhs
        field = np.clip(field, 0.01, 0.99)
        
        if tick % 10 == 0:
            pillar = coupling.project_to_pillars(field)
            try:
                exp = Experience(
                    experience_id=f"b0_{tick}",
                    experience_type=ExperienceType.LEARNING,
                    psv_snapshot=pillar.tolist(),
                    description=f"Basin 0 learning",
                )
                memory.encode(exp)
            except Exception:
                pass
    
    b0_final = field.copy()
    b0, _ = compute_basin_trajectory(field, basins)
    print(f"  After learning basin 0: in basin {b0}")
    
    # Phase 2: Shift to basin 2 (perturbation)
    print(f"  Phase 2: Perturbation toward basin 2...")
    field = basins_raw[2] + rng.randn(dim) * 0.05
    field = np.clip(field, 0.01, 0.99)
    
    for tick in range(500):
        rhs = evolver_mem.rhs(field, cfg_mem.dt)
        field = field + cfg_mem.dt * rhs
        field = np.clip(field, 0.01, 0.99)
    
    b_after_perturb, _ = compute_basin_trajectory(field, basins)
    print(f"  After perturbation: in basin {b_after_perturb}")
    
    # Phase 3: Test memory recall
    pillar = coupling.project_to_pillars(field)
    recalls = memory.recall(pillar, k=5)
    
    if recalls:
        recall_basins = []
        for t in recalls:
            if t.attractor_center is not None:
                center_arr = np.array(t.attractor_center)
                b, _ = compute_basin_trajectory(center_arr, basins)
                recall_basins.append(b)
        
        print(f"  Memory recalls basin: {recall_basins}")
        if recall_basins:
            most_common = max(set(recall_basins), key=recall_basins.count)
            print(f"  Most recalled: basin {most_common} "
                  f"(should be 0 if memory works)")
    
    # Summary
    results = {
        'basins': [{'center': c.tolist(), 'radius': r} for c, r, _ in basins],
        'barrier_crossing': barrier_results,
        'hysteresis': {
            'basin_switches': sum(1 for i in range(1, len(hyst_states)) 
                                  if hyst_states[i] != hyst_states[i-1]),
            'last_switch_tick': lag,
        },
    }
    
    out = os.path.join(os.path.dirname(__file__), '..', 'metastability_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")
    
    return results


if __name__ == '__main__':
    run_metastability()
