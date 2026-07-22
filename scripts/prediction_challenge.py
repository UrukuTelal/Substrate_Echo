#!/usr/bin/env python3
"""
2. Memory Prediction Challenge.

Create an environment where persistence FAILS:
- Moving attractor (oscillating field target)
- Topology changes (periodic perturbations)
- Non-stationary dynamics

Measure: memory prediction, reaction delay, adaptation speed.
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.field_evolution import FieldEvolver, FieldConfig
from substrate_echo.dynamics.pillar_coupling import PillarCoupling
from substrate_echo.core.attractor_memory import AttractorMemory
from substrate_echo.core.ontological_field import OntologicalField
from substrate_echo.models.experience import Experience, ExperienceType


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def make_moving_target(tick, dim=16, speed=0.5, n_basins=3):
    """Create a time-varying target state.
    
    The target oscillates between n_basins in state space,
    making persistence a poor predictor.
    """
    # Create basin centers
    rng = np.random.RandomState(123)
    basins = [rng.uniform(0.3, 0.7, dim) for _ in range(n_basins)]
    
    # Smooth interpolation between basins
    period = 200  # ticks per full cycle
    phase = (tick % period) / period * n_basins
    basin_idx = int(phase) % n_basins
    next_idx = (basin_idx + 1) % n_basins
    alpha = phase - int(phase)
    
    # Smoothstep interpolation
    alpha = alpha * alpha * (3 - 2 * alpha)
    
    target = (1 - alpha) * basins[basin_idx] + alpha * basins[next_idx]
    return target, basins


def run_prediction_challenge():
    print("=" * 60)
    print("MEMORY PREDICTION CHALLENGE")
    print("=" * 60)
    
    scenarios = {
        'stationary': {'speed': 0.0, 'perturb_period': 0, 'description': 'Static target'},
        'slow_oscillation': {'speed': 0.3, 'perturb_period': 0, 'description': 'Slow moving target (period=667)'},
        'fast_oscillation': {'speed': 1.0, 'perturb_period': 0, 'description': 'Fast moving target (period=200)'},
        'perturbations': {'speed': 0.0, 'perturb_period': 100, 'description': 'Static with periodic shocks'},
        'combined': {'speed': 0.5, 'perturb_period': 150, 'description': 'Moving target + shocks'},
    }
    
    horizons = [5, 10, 20, 50]
    n_ticks = 2000
    dim = 16
    
    all_results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"\n--- Scenario: {params['description']} ---")
        
        cfg = FieldConfig(dt=0.01, gamma=0.05, lambda_gl=1.0, temperature=0.001)
        evolver = FieldEvolver(cfg)
        coupling = PillarCoupling()
        
        ont_field = OntologicalField()
        memory = AttractorMemory(field=ont_field)
        
        rng = np.random.RandomState(42)
        field = rng.uniform(0.3, 0.7, dim)
        
        prediction_scores = {h: {'memory': [], 'baseline': [], 'target': []} for h in horizons}
        field_history = []
        target_history = []
        reaction_delays = []
        
        # For measuring adaptation: track how quickly error decreases after a shift
        error_windows = []
        
        for tick in range(n_ticks):
            # Get moving target
            target, basins = make_moving_target(tick, dim=dim, speed=params['speed'])
            target_history.append(target.copy())
            
            # Apply field dynamics toward target (with noise)
            rhs = evolver.rhs(field, cfg.dt)
            
            # Add attraction toward target
            attraction = 0.1 * (target - field)
            field = field + cfg.dt * (rhs + attraction)
            
            # Periodic perturbations
            if params['perturb_period'] > 0 and tick % params['perturb_period'] == 0 and tick > 0:
                field += rng.randn(dim) * 0.15
                error_windows.append(tick)
            
            field = np.clip(field, 0.0, 1.0)
            pillar = coupling.project_to_pillars(field)
            field_history.append(pillar.copy())
            
            # Encode memory
            if tick % 5 == 0:
                try:
                    exp = Experience(
                        experience_id=f"exp_{tick}",
                        experience_type=ExperienceType.LEARNING,
                        psv_snapshot=pillar.tolist(),
                        description=f"Tick {tick} target={target[:3]}",
                    )
                    memory.encode(exp)
                except Exception:
                    pass
            
            # Evaluate prediction
            if tick >= 50 and tick % 10 == 0:
                true_state = target_history[-1]  # What the target IS now
                
                for h in horizons:
                    if len(target_history) > h:
                        # Baseline: target from h steps ago (persistence)
                        baseline_pred = target_history[-(h+1)]
                        baseline_score = cosine_similarity(true_state, baseline_pred)
                        
                        # Memory prediction: weighted recall
                        recalls = memory.recall(pillar, k=5)
                        if recalls:
                            weights = [t.strength for t in recalls]
                            patterns = [np.array(t.attractor_center) for t in recalls
                                       if t.attractor_center is not None 
                                       and len(t.attractor_center) == dim]
                            if patterns:
                                total_w = sum(weights[:len(patterns)])
                                if total_w > 1e-10:
                                    memory_pred = sum(w * p for w, p in
                                                    zip(weights[:len(patterns)], patterns)) / total_w
                                    memory_score = cosine_similarity(true_state, memory_pred)
                                else:
                                    memory_score = 0.0
                            else:
                                memory_score = 0.0
                        else:
                            memory_score = 0.0
                        
                        # Target-based prediction (oracle: know the target function)
                        if len(target_history) > h + 5:
                            # Extrapolate from recent target movement
                            recent_targets = np.array(target_history[-5:])
                            target_velocity = np.mean(np.diff(recent_targets, axis=0), axis=0)
                            oracle_pred = target_history[-1] + h * target_velocity * 0.1
                            oracle_score = cosine_similarity(true_state, np.clip(oracle_pred, 0, 1))
                        else:
                            oracle_score = 0.0
                        
                        prediction_scores[h]['memory'].append(memory_score)
                        prediction_scores[h]['baseline'].append(baseline_score)
                        prediction_scores[h]['target'].append(oracle_score)
            
            if tick % 500 == 0:
                avg_mem = np.mean([np.mean(prediction_scores[h]['memory'][-5:])
                                   for h in horizons 
                                   if prediction_scores[h]['memory']])
                avg_base = np.mean([np.mean(prediction_scores[h]['baseline'][-5:])
                                    for h in horizons 
                                    if prediction_scores[h]['baseline']])
                print(f"  tick {tick:>5d}: memory={avg_mem:.3f} baseline={avg_base:.3f}")
        
        # Results
        print(f"\n  Prediction Accuracy:")
        print(f"  {'Horizon':>8s} {'Memory':>8s} {'Baseline':>9s} {'Oracle':>8s} {'Mem wins':>9s}")
        
        scenario_results = {}
        for h in horizons:
            mem = prediction_scores[h]['memory']
            base = prediction_scores[h]['baseline']
            oracle = prediction_scores[h]['target']
            
            if mem and base:
                mem_mean = float(np.mean(mem))
                base_mean = float(np.mean(base))
                oracle_mean = float(np.mean(oracle)) if oracle else 0.0
                wins = sum(1 for m, b in zip(mem, base) if m > b)
                win_rate = wins / len(mem)
                
                print(f"  dt={h:>4d}    {mem_mean:8.4f} {base_mean:9.4f} "
                      f"{oracle_mean:8.4f} {win_rate:8.1%}")
                
                scenario_results[h] = {
                    'memory_mean': mem_mean,
                    'baseline_mean': base_mean,
                    'oracle_mean': oracle_mean,
                    'memory_wins': win_rate,
                    'improvement_over_baseline': mem_mean - base_mean,
                }
        
        # Overall
        avg_improvement = np.mean([r['improvement_over_baseline'] 
                                   for r in scenario_results.values()])
        avg_wins = np.mean([r['memory_wins'] for r in scenario_results.values()])
        
        print(f"\n  Avg improvement over baseline: {avg_improvement:+.4f}")
        print(f"  Avg memory win rate: {avg_wins:.1%}")
        
        if avg_wins > 0.5:
            print(f"  >> Memory IS useful in this scenario")
        else:
            print(f"  >> Memory is NOT useful (baseline still wins)")
        
        all_results[scenario_name] = {
            'params': params,
            'horizons': scenario_results,
            'avg_improvement': avg_improvement,
            'avg_win_rate': avg_wins,
        }
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SCENARIO COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Scenario':<25s} {'Improvement':>12s} {'Win Rate':>10s} {'Verdict':>10s}")
    for name, r in all_results.items():
        verdict = 'USEFUL' if r['avg_win_rate'] > 0.5 else 'NOT USEFUL'
        print(f"  {name:<25s} {r['avg_improvement']:+12.4f} "
              f"{r['avg_win_rate']:>9.1%} {verdict:>10s}")
    
    out = os.path.join(os.path.dirname(__file__), '..', 'prediction_challenge_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")
    
    return all_results


if __name__ == '__main__':
    run_prediction_challenge()
