#!/usr/bin/env python3
"""
D2. Memory Prediction Experiment.

Can memory(t) predict field(t+Δt)?

Measures:
- Prediction accuracy at various horizons (Δt = 10, 20, 50, 100)
- Compares memory-based prediction vs baseline (constant field)
- Tests whether memory becomes a useful internal model
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


def run_prediction_experiment(n_ticks=2000, horizons=[10, 20, 50, 100]):
    print("=" * 60)
    print("MEMORY PREDICTION EXPERIMENT")
    print("=" * 60)

    cfg = FieldConfig(dt=0.01, gamma=0.01, lambda_gl=1.0, temperature=0.001)
    evolver = FieldEvolver(cfg)
    coupling = PillarCoupling()

    # Two field instances: one for memory, one as "ground truth future"
    # Start with diverse initial conditions to avoid trivial uniformity
    rng = np.random.RandomState(42)
    field_memory = rng.uniform(0.2, 0.8, 16)
    field_truth = field_memory.copy()

    # Memory system
    ont_field = OntologicalField()
    memory = AttractorMemory(field=ont_field)

    # History for baseline comparison
    field_history = []

    # Prediction scores: memory-based vs baseline (last-observation)
    prediction_scores = {h: {'memory': [], 'baseline': []} for h in horizons}

    # Periodic perturbation schedule to keep dynamics interesting
    perturb_schedule = set(range(100, n_ticks, 200))

    for tick in range(n_ticks):
        # Evolve both fields identically
        rhs_mem = evolver.rhs(field_memory, cfg.dt)
        field_memory = field_memory + cfg.dt * rhs_mem
        field_memory = np.clip(field_memory, 0.0, 1.0)

        rhs_tru = evolver.rhs(field_truth, cfg.dt)
        field_truth = field_truth + cfg.dt * rhs_tru
        field_truth = np.clip(field_truth, 0.0, 1.0)

        # Periodic perturbations to create non-trivial dynamics
        if tick in perturb_schedule:
            field_memory += rng.randn(16) * 0.05
            field_memory = np.clip(field_memory, 0.0, 1.0)
            field_truth += rng.randn(16) * 0.05
            field_truth = np.clip(field_truth, 0.0, 1.0)

    for tick in range(n_ticks):
        # Evolve both fields identically
        rhs_mem = evolver.rhs(field_memory, cfg.dt)
        field_memory = field_memory + cfg.dt * rhs_mem
        field_memory = np.clip(field_memory, 0.0, 1.0)

        rhs_tru = evolver.rhs(field_truth, cfg.dt)
        field_truth = field_truth + cfg.dt * rhs_tru
        field_truth = np.clip(field_truth, 0.0, 1.0)

        pillar = coupling.project_to_pillars(field_memory)
        field_history.append(pillar.copy())

        # Encode memory every 10 ticks
        if tick % 10 == 0:
            try:
                exp = Experience(
                    experience_id=f"exp_{tick}",
                    experience_type=ExperienceType.LEARNING,
                    psv_snapshot=pillar.tolist(),
                    description=f"Tick {tick}",
                )
                memory.encode(exp)
            except Exception:
                pass

        # Evaluate prediction at specific horizons
        for h in horizons:
            if tick >= h and tick % 10 == 0:
                # Ground truth: field state h steps ago (what actually happened)
                true_state = field_history[-1]

                # Baseline prediction: current state (persistence model)
                baseline_pred = field_history[-(h+1)] if len(field_history) > h else field_history[0]
                baseline_score = cosine_similarity(true_state, baseline_pred)

                # Memory prediction: recall with current state as cue
                recalls = memory.recall(pillar, k=3)
                if recalls:
                    # Weighted average of recalled patterns
                    weights = [t.strength for t in recalls]
                    patterns = [np.array(t.attractor_center) for t in recalls
                                if t.attractor_center is not None and len(t.attractor_center) == 16]
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

                prediction_scores[h]['memory'].append(memory_score)
                prediction_scores[h]['baseline'].append(baseline_score)

        if tick % 500 == 0:
            avg_mem = np.mean([prediction_scores[h]['memory'][-10:]
                               for h in horizons if prediction_scores[h]['memory']])
            print(f"  tick {tick:>5d}: avg memory score = {avg_mem:.4f}")

    # Results
    print(f"\n--- Prediction Accuracy by Horizon ---")
    print(f"  {'Horizon':>8s} {'Memory':>8s} {'Baseline':>9s} {'Improvement':>12s}")
    print(f"  {'-'*40}")

    results = {}
    for h in horizons:
        mem_scores = prediction_scores[h]['memory']
        base_scores = prediction_scores[h]['baseline']

        if mem_scores and base_scores:
            mem_mean = float(np.mean(mem_scores))
            base_mean = float(np.mean(base_scores))
            improvement = mem_mean - base_mean

            print(f"  dt={h:>4d}    {mem_mean:8.4f} {base_mean:9.4f} {improvement:+12.4f}")

            results[h] = {
                'memory_mean': mem_mean,
                'memory_std': float(np.std(mem_scores)),
                'baseline_mean': base_mean,
                'baseline_std': float(np.std(base_scores)),
                'improvement': improvement,
                'memory_beats_baseline': mem_mean > base_mean,
            }

    # Overall assessment
    improvements = [r['improvement'] for r in results.values()]
    beats = sum(1 for r in results.values() if r['memory_beats_baseline'])
    avg_improvement = np.mean(improvements) if improvements else 0

    print(f"\n--- Assessment ---")
    print(f"  Memory beats baseline: {beats}/{len(horizons)} horizons")
    print(f"  Average improvement: {avg_improvement:+.4f}")

    if avg_improvement > 0.05:
        print(f"  Memory IS a useful internal model")
    elif avg_improvement > 0:
        print(f"  Memory shows weak predictive value")
    else:
        print(f"  Memory is NOT predictive (just recording)")

    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'prediction_results.json')
    with open(out, 'w') as f:
        json.dump({
            'config': {'n_ticks': n_ticks, 'horizons': horizons},
            'results': results,
            'assessment': {
                'beats_baseline': beats,
                'total_horizons': len(horizons),
                'avg_improvement': avg_improvement,
            },
        }, f, indent=2)
    print(f"\nResults saved to {out}")

    return results


if __name__ == '__main__':
    run_prediction_experiment()
