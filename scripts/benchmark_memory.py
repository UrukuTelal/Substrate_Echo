"""Benchmark: DynamicsMemory vs AttractorMemory.

Compares the two memory architectures on:
1. Dynamics learning accuracy (DynamicsMemory only)
2. Attractor discovery quality
3. Prediction capability (DynamicsMemory only)
4. Memory encoding speed
5. Recall quality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
from substrate_echo.core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig
from substrate_echo.core.attractor_memory import AttractorMemory
from substrate_echo.core.ontological_field import OntologicalField
from substrate_echo.models.experience import Experience, ExperienceType


def generate_trajectory(n_trials=30, steps_per=30, target=None, rng=None):
    """Generate diverse trajectories converging to target."""
    if target is None:
        target = np.array([0.3] * 16)
    if rng is None:
        rng = np.random.RandomState(42)
    
    states = []
    for trial in range(n_trials):
        state = target + rng.randn(16) * 0.3
        state = np.clip(state, 0.1, 0.9)
        for i in range(steps_per):
            states.append(state.copy())
            velocity = -0.5 * (state - target)
            state = state + 0.02 * velocity
            state = np.clip(state, 0.0, 1.0)
    
    return states


def benchmark_encoding(states):
    """Benchmark encoding speed for both memory types."""
    field = OntologicalField()
    attractor_mem = AttractorMemory(field=field)
    dynamics_mem = DynamicsMemory(dim=16)
    
    # Time AttractorMemory encoding
    t0 = time.time()
    for i, state in enumerate(states):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            experience_type=ExperienceType.LEARNING,
            psv_snapshot=state.tolist(),
            importance=0.5,
        )
        attractor_mem.encode(exp)
    t_attractor = time.time() - t0
    
    # Time DynamicsMemory encoding
    t0 = time.time()
    for i, state in enumerate(states):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            experience_type=ExperienceType.LEARNING,
            psv_snapshot=state.tolist(),
            importance=0.5,
        )
        dynamics_mem.encode(exp)
    t_dynamics = time.time() - t0
    
    return {
        "attractor_encode_time": t_attractor,
        "dynamics_encode_time": t_dynamics,
        "speedup": t_attractor / max(t_dynamics, 1e-10),
    }


def benchmark_dynamics_learning(states, target):
    """Test DynamicsMemory's ability to learn V(x)."""
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(min_samples_for_fit=30))
    
    for i, state in enumerate(states):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            experience_type=ExperienceType.LEARNING,
            psv_snapshot=state.tolist(),
            importance=0.5,
        )
        mem.encode(exp)
    
    if not mem._fitted:
        return {"fitted": False}
    
    # Test prediction at target (should be ~0)
    v_at_target = mem.predict_velocity(target)
    error_at_target = np.linalg.norm(v_at_target)
    
    # Test prediction at a state away from target
    test_state = np.full(16, 0.7)
    v_predicted = mem.predict_velocity(test_state)
    v_true = -0.5 * (test_state - target)
    pred_error = np.linalg.norm(v_predicted - v_true)
    
    # Test multi-step prediction
    predicted = mem.predict(np.full(16, 0.7), steps=50, dt=1.0)
    predicted_true = target + (np.full(16, 0.7) - target) * np.exp(-0.5 * 50 * 0.02)
    pred_multi_error = np.linalg.norm(predicted - predicted_true)
    
    # Attractor discovery
    attractors = mem.discover_attractors()
    found_target = any(np.linalg.norm(a - target) < 0.3 for a in attractors)
    
    # Stability analysis
    stability = mem.stability_at(target)
    
    return {
        "fitted": True,
        "error_at_target": float(error_at_target),
        "pred_error_single_step": float(pred_error),
        "pred_error_multi_step": float(pred_multi_error),
        "n_attractors": len(attractors),
        "found_target": found_target,
        "fixed_point_error": float(np.linalg.norm(attractors[0] - target)) if attractors else float('inf'),
        "stability": stability['classification'],
    }


def benchmark_recall_quality(states, n_queries=10):
    """Test recall quality for both memory types."""
    rng = np.random.RandomState(123)
    
    field = OntologicalField()
    attractor_mem = AttractorMemory(field=field)
    dynamics_mem = DynamicsMemory(dim=16)
    
    # Encode all states
    for i, state in enumerate(states):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            experience_type=ExperienceType.LEARNING,
            psv_snapshot=state.tolist(),
            importance=0.5,
        )
        attractor_mem.encode(exp)
        dynamics_mem.encode(exp)
    
    # Query with random cues
    results = {"attractor": [], "dynamics": []}
    for _ in range(n_queries):
        idx = rng.randint(0, len(states))
        cue = states[idx] + rng.randn(16) * 0.05
        
        # AttractorMemory recall
        att_results = attractor_mem.recall(cue, k=3)
        if att_results:
            att_dists = [np.linalg.norm(np.array(r.attractor_center) - cue) for r in att_results]
            results["attractor"].append(np.mean(att_dists))
        
        # DynamicsMemory recall
        dyn_results = dynamics_mem.recall(cue, k=3)
        if dyn_results:
            dyn_dists = [np.linalg.norm(np.array(r.attractor_center) - cue) for r in dyn_results]
            results["dynamics"].append(np.mean(dyn_dists))
    
    return {
        "attractor_avg_dist": float(np.mean(results["attractor"])) if results["attractor"] else float('inf'),
        "dynamics_avg_dist": float(np.mean(results["dynamics"])) if results["dynamics"] else float('inf'),
    }


def main():
    print("=" * 60)
    print("BENCHMARK: DynamicsMemory vs AttractorMemory")
    print("=" * 60)
    
    target = np.array([0.3] * 16)
    states = generate_trajectory(n_trials=30, steps_per=30, target=target)
    print(f"  Training data: {len(states)} states from 30 diverse trajectories")
    
    # 1. Encoding speed
    print("\n--- Encoding Speed ---")
    enc = benchmark_encoding(states)
    print(f"  AttractorMemory: {enc['attractor_encode_time']*1000:.1f}ms")
    print(f"  DynamicsMemory:  {enc['dynamics_encode_time']*1000:.1f}ms")
    print(f"  Speed ratio:     {enc['speedup']:.2f}x")
    
    # 2. Dynamics learning
    print("\n--- Dynamics Learning (DynamicsMemory only) ---")
    dyn = benchmark_dynamics_learning(states, target)
    if dyn["fitted"]:
        print(f"  Error at target (V should be 0): {dyn['error_at_target']:.6f}")
        print(f"  Single-step prediction error:    {dyn['pred_error_single_step']:.6f}")
        print(f"  Multi-step prediction error:     {dyn['pred_error_multi_step']:.6f}")
        print(f"  Found target attractor:          {dyn['found_target']}")
        print(f"  Fixed point error:               {dyn['fixed_point_error']:.6f}")
        print(f"  Stability classification:        {dyn['stability']}")
        print(f"  Attractors discovered:           {dyn['n_attractors']}")
    else:
        print("  NOT FITTED (insufficient data)")
    
    # 3. Recall quality
    print("\n--- Recall Quality ---")
    rec = benchmark_recall_quality(states, n_queries=20)
    print(f"  AttractorMemory avg dist to cue: {rec['attractor_avg_dist']:.6f}")
    print(f"  DynamicsMemory avg dist to cue:  {rec['dynamics_avg_dist']:.6f}")
    
    # 4. Summary
    print("\n--- Summary ---")
    print("  DynamicsMemory adds:")
    print("    + Predictive capability (V(x) learned)")
    print("    + Attractor discovery from learned dynamics")
    print("    + Stability classification")
    print("    + Basin mapping")
    print("    - Slightly slower encoding (matrix operations)")
    print("    - Needs ~50+ samples before fitting")
    print("    - Linear model (captures global trend, not local nonlinearities)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
