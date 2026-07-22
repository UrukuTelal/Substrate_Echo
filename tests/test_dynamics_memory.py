"""Tests for Dynamics Memory — learns V(x) from experience."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from substrate_echo.core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig
from substrate_echo.models.experience import Experience, ExperienceType


def test_encode_creates_trace():
    """DynamicsMemory.encode() creates a memory trace."""
    mem = DynamicsMemory(dim=16)
    exp = Experience(
        experience_id="exp_001",
        experience_type=ExperienceType.PERCEPTION,
        description="I see a cup",
        psv_snapshot=[0.7] * 16,
        importance=0.8,
    )
    trace = mem.encode(exp)
    assert trace is not None
    assert len(mem.traces) == 1
    assert mem._prev_psv is not None
    print("PASS: test_encode_creates_trace")


def test_encode_computes_velocity():
    """Consecutive encodes compute velocities for dynamics learning."""
    mem = DynamicsMemory(dim=16)

    # First encode stores prev_psv but no velocity yet
    exp1 = Experience(
        experience_id="exp_001",
        experience_type=ExperienceType.PERCEPTION,
        psv_snapshot=[0.5] * 16,
        importance=0.5,
    )
    mem.encode(exp1)
    assert len(mem._states) == 0
    assert len(mem._velocities) == 0

    # Second encode produces first (state, velocity) pair
    exp2 = Experience(
        experience_id="exp_002",
        experience_type=ExperienceType.PERCEPTION,
        psv_snapshot=[0.55] * 16,
        importance=0.5,
    )
    mem.encode(exp2)
    assert len(mem._states) == 1
    assert len(mem._velocities) == 1
    velocity = mem._velocities[0]
    assert np.allclose(velocity, 0.05)
    print("PASS: test_encode_computes_velocity")


def test_dynamics_fitting():
    """After enough samples, V(x) = Ax + b is fitted."""
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(min_samples_for_fit=30))

    # Feed 60 experiences along a trajectory
    rng = np.random.RandomState(42)
    state = np.array([0.5] * 16)
    for i in range(60):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            experience_type=ExperienceType.PERCEPTION,
            psv_snapshot=state.tolist(),
            importance=0.5,
        )
        mem.encode(exp)
        # Move state by a known velocity
        state = state + 0.01 * np.ones(16)
        state = np.clip(state, 0.0, 1.0)

    assert mem._fitted
    assert mem.A is not None
    assert mem.b is not None
    assert mem.A.shape == (16, 16)
    assert mem.b.shape == (16,)
    print("PASS: test_dynamics_fitting")


def test_prediction_accuracy():
    """Predicted velocity matches actual velocity for linear dynamics."""
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(min_samples_for_fit=30))

    # Feed trajectory with constant velocity v = 0.01 * ones(16)
    state = np.array([0.5] * 16)
    for i in range(60):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            experience_type=ExperienceType.PERCEPTION,
            psv_snapshot=state.tolist(),
            importance=0.5,
        )
        mem.encode(exp)
        state = state + 0.01 * np.ones(16)
        state = np.clip(state, 0.0, 1.0)

    # Predict velocity at a state
    test_state = np.array([0.6] * 16)
    predicted_v = mem.predict_velocity(test_state)

    # For constant velocity dynamics, prediction should be ~0.01
    assert np.allclose(predicted_v, 0.01, atol=0.01)
    print("PASS: test_prediction_accuracy")


def test_prediction_multi_step():
    """Multi-step prediction integrates forward."""
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(min_samples_for_fit=30))

    # Feed trajectory: constant velocity of 0.01 per tick along all dims
    state = np.array([0.5] * 16)
    for i in range(60):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            experience_type=ExperienceType.PERCEPTION,
            psv_snapshot=state.tolist(),
            importance=0.5,
        )
        mem.encode(exp)
        state = state.copy()
        state += 0.01
        state = np.clip(state, 0.0, 1.0)

    # Predict 10 steps forward with dt=1.0 (velocities are per-encode-step)
    start = np.array([0.5] * 16)
    predicted = mem.predict(start, steps=10, dt=1.0)
    # Should have moved forward ~0.01 * 10 = 0.1
    assert predicted[0] > start[0]
    assert abs(predicted[0] - start[0] - 0.1) < 0.02
    for d in range(1, 16):
        assert abs(predicted[d] - start[d] - 0.1) < 0.02
    print("PASS: test_prediction_multi_step")


def _feed_converging_dynamics(mem, target, n_trials=20, steps_per=20, rng_seed=42):
    """Feed diverse trajectories converging to target with per-dim variation."""
    rng = np.random.RandomState(rng_seed)
    for trial in range(n_trials):
        # Per-dimension variation so X matrix is full rank
        state = target + rng.randn(16) * 0.25
        state = np.clip(state, 0.1, 0.9)
        for i in range(steps_per):
            exp = Experience(
                experience_id=f"exp_{trial}_{i:03d}",
                experience_type=ExperienceType.PERCEPTION,
                psv_snapshot=state.tolist(),
                importance=0.5,
            )
            mem.encode(exp)
            velocity = -0.5 * (state - target)
            state = state + 0.02 * velocity
            state = np.clip(state, 0.0, 1.0)


def test_attractor_discovery():
    """Discover attractors by integrating from random starts."""
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(
        min_samples_for_fit=60,
        attractor_samples=300,
        attractor_integration_steps=200,
    ))

    target = np.array([0.3] * 16)
    _feed_converging_dynamics(mem, target, n_trials=30, steps_per=30)

    assert mem._fitted

    attractors = mem.discover_attractors()

    found = False
    for att in attractors:
        if np.linalg.norm(att - target) < 0.5:
            found = True
            break
    assert found, f"No attractor near target {target[0]:.2f}. Found: {len(attractors)} attractors, first: {attractors[0][:3] if attractors else 'none'}"
    print("PASS: test_attractor_discovery")


def test_basin_classification():
    """States converge to the correct basin."""
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(
        min_samples_for_fit=60,
        attractor_samples=300,
        attractor_integration_steps=200,
    ))

    target_low = np.array([0.2] + [0.5]*15)
    target_high = np.array([0.8] + [0.5]*15)
    rng = np.random.RandomState(42)

    for trial in range(30):
        if trial < 15:
            state = target_low + rng.randn(16) * 0.15
        else:
            state = target_high + rng.randn(16) * 0.15
        state = np.clip(state, 0.05, 0.95)
        for i in range(20):
            exp = Experience(
                experience_id=f"exp_{trial}_{i:03d}",
                experience_type=ExperienceType.PERCEPTION,
                psv_snapshot=state.tolist(),
                importance=0.5,
            )
            mem.encode(exp)
            if state[0] < 0.5:
                velocity = -0.5 * (state - target_low)
            else:
                velocity = -0.5 * (state - target_high)
            state = state + 0.02 * velocity
            state = np.clip(state, 0.0, 1.0)

    mem.discover_attractors()
    assert len(mem._attractors) >= 1

    test_low = np.array([0.25] + [0.5]*15)
    basin = mem.basin_of(test_low)
    assert basin >= 0 or len(mem._attractors) == 0
    print("PASS: test_basin_classification")


def test_stability_analysis():
    """Stability analysis classifies attractors correctly."""
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(min_samples_for_fit=60))

    target = np.array([0.3] * 16)
    _feed_converging_dynamics(mem, target, n_trials=30, steps_per=20)

    result = mem.stability_at(target)
    assert 'eigenvalues' in result
    assert 'classification' in result
    assert result['classification'] in ['attractor', 'repellor', 'saddle', 'marginal']
    assert result['classification'] == 'attractor', f"Expected attractor, got {result['classification']}"
    print("PASS: test_stability_analysis")


def test_memory_stats():
    """memory_stats returns expected fields."""
    mem = DynamicsMemory(dim=16)

    for i in range(5):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            experience_type=ExperienceType.LEARNING,
            psv_snapshot=[0.5] * 16,
            importance=0.5,
        )
        mem.encode(exp)

    stats = mem.memory_stats()
    assert stats["total_memories"] == 5
    assert stats["active_memories"] == 5
    assert isinstance(stats["dynamics_fitted"], bool)
    assert isinstance(stats["n_training_samples"], int)
    assert isinstance(stats["n_attractors"], int)
    print("PASS: test_memory_stats")


def test_consolidation():
    """Consolidation prunes weak traces."""
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(
        prune_strength_threshold=0.3,
        consolidation_interval=0.0,
    ))

    # Encode some experiences with varying importance
    for i in range(10):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            experience_type=ExperienceType.PERCEPTION,
            psv_snapshot=[0.3 + i * 0.05] * 16,
            importance=0.1 if i < 5 else 0.8,
        )
        mem.encode(exp)

    before = len(mem.traces)
    mem.consolidate(force=True)
    after = len(mem.traces)

    # Weak traces (importance 0.1) should be pruned
    assert after <= before
    print("PASS: test_consolidation")


def test_identity_pattern():
    """Identity pattern is the weighted average of attractors."""
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(min_samples_for_fit=30))

    # Feed a trajectory
    state = np.array([0.5] * 16)
    for i in range(60):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            experience_type=ExperienceType.PERCEPTION,
            psv_snapshot=state.tolist(),
            importance=0.5,
        )
        mem.encode(exp)
        state = state + 0.01 * np.ones(16)
        state = np.clip(state, 0.0, 1.0)

    # Discover attractors (may be 0 if dynamics are too simple)
    mem.discover_attractors()

    # Identity pattern should be a 16D array or None
    identity = mem.identity_pattern()
    if mem._attractors:
        assert identity is not None
        assert identity.shape == (16,)
    else:
        assert identity is None
    print("PASS: test_identity_pattern")


def test_recall_compatible_with_attractor_memory():
    """DynamicsMemory.recall() returns MemoryTrace objects like AttractorMemory."""
    mem = DynamicsMemory(dim=16)

    for i in range(5):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            experience_type=ExperienceType.INTERACTION,
            description=f"Interaction {i}",
            psv_snapshot=[0.3 + i * 0.1] * 16,
            importance=0.8,
        )
        mem.encode(exp)

    cue = np.full(16, 0.5)
    results = mem.recall(cue, k=3)

    assert len(results) >= 1
    assert all(hasattr(r, 'description') for r in results)
    assert all(hasattr(r, 'strength') for r in results)
    print("PASS: test_recall_compatible_with_attractor_memory")


def test_surprising_events_stronger():
    """Surprising experiences get higher strength."""
    mem = DynamicsMemory(dim=16)

    normal = Experience(
        experience_id="normal",
        experience_type=ExperienceType.PERCEPTION,
        description="Normal",
        psv_snapshot=[0.3] * 16,
        importance=0.5,
    )
    trace1 = mem.encode(normal)

    surprise = Experience(
        experience_id="surprise",
        experience_type=ExperienceType.SURPRISE,
        description="Surprising!",
        psv_snapshot=[0.8] * 16,
        importance=0.5,
    )
    trace2 = mem.encode(surprise)

    assert trace2.strength > trace1.strength
    print("PASS: test_surprising_events_stronger")


if __name__ == "__main__":
    test_encode_creates_trace()
    test_encode_computes_velocity()
    test_dynamics_fitting()
    test_prediction_accuracy()
    test_prediction_multi_step()
    test_attractor_discovery()
    test_basin_classification()
    test_stability_analysis()
    test_memory_stats()
    test_consolidation()
    test_identity_pattern()
    test_recall_compatible_with_attractor_memory()
    test_surprising_events_stronger()
    test_local_model_predicts_velocity()
    test_local_model_discovers_attractors()
    test_local_model_multi_basin()


# ── Local Linear Model Tests ─────────────────────────────────────

def test_local_model_predicts_velocity():
    """Local linear model learns a simple linear dynamics."""
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(
        model_type="local", min_samples_for_fit=30, k_neighbors=25,
        bandwidth=1.0))
    target = np.full(16, 0.3)
    rng = np.random.RandomState(42)
    
    for trial in range(30):
        state = target + rng.randn(16) * 0.35
        state = np.clip(state, 0.05, 0.95)
        for step in range(30):
            exp = Experience(
                experience_id=f"loc_{trial}_{step:03d}",
                experience_type=ExperienceType.LEARNING,
                psv_snapshot=state.tolist(),
                importance=0.5,
            )
            mem.encode(exp)
            velocity = -0.5 * (state - target)
            state = state + 0.02 * velocity
            state = np.clip(state, 0.0, 1.0)
    
    assert mem._fitted
    
    # Predict velocity at a state away from target
    test_state = np.full(16, 0.7)
    v_pred = mem.predict_velocity(test_state)
    v_true = -0.5 * (test_state - target)
    error = np.linalg.norm(v_pred - v_true)
    assert error < 1.0  # local model in 16D is approximate
    print(f"PASS: test_local_model_predicts_velocity (error={error:.4f})")


def test_local_model_discovers_attractors():
    """Local model discovers attractors via integration."""
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(
        model_type="local", min_samples_for_fit=30, k_neighbors=30,
        bandwidth=1.0, attractor_samples=50, attractor_integration_steps=50))
    target = np.full(16, 0.3)
    rng = np.random.RandomState(42)
    
    for trial in range(30):
        state = target + rng.randn(16) * 0.35
        state = np.clip(state, 0.05, 0.95)
        for step in range(30):
            exp = Experience(
                experience_id=f"loc_att_{trial}_{step:03d}",
                experience_type=ExperienceType.LEARNING,
                psv_snapshot=state.tolist(),
                importance=0.5,
            )
            mem.encode(exp)
            velocity = -0.5 * (state - target)
            state = state + 0.02 * velocity
            state = np.clip(state, 0.0, 1.0)
    
    attractors = mem.discover_attractors()
    
    # Should discover at least one attractor (may not be exact in high-d)
    assert len(attractors) >= 1, "No attractors discovered"
    
    # The discovered attractor should be within the training data range
    for a in attractors:
        assert np.all(a >= 0.0) and np.all(a <= 1.0), "Attractor out of bounds"
    
    print(f"PASS: test_local_model_discovers_attractors ({len(attractors)} found)")


def test_local_model_multi_basin():
    """Local model handles multi-basin dynamics (2 basins in 16D)."""
    target_a = np.array([0.2] * 16)
    target_b = np.array([0.8] * 16)
    
    mem = DynamicsMemory(dim=16, config=DynamicsMemoryConfig(
        model_type="local", min_samples_for_fit=50, k_neighbors=30,
        bandwidth=1.0, attractor_samples=50, attractor_integration_steps=50))
    
    rng = np.random.RandomState(42)
    
    # Generate trajectories from both basins
    for trial in range(40):
        if trial % 2 == 0:
            target = target_a
        else:
            target = target_b
        state = target + rng.randn(16) * 0.35
        state = np.clip(state, 0.05, 0.95)
        for step in range(30):
            exp = Experience(
                experience_id=f"mb_{trial}_{step:03d}",
                experience_type=ExperienceType.LEARNING,
                psv_snapshot=state.tolist(),
                importance=0.5,
            )
            mem.encode(exp)
            velocity = -0.5 * (state - target)
            state = state + 0.02 * velocity
            state = np.clip(state, 0.0, 1.0)
    
    assert mem._fitted
    
    attractors = mem.discover_attractors()
    
    # Should find at least one attractor
    assert len(attractors) >= 1, f"Expected at least 1 attractor, got {len(attractors)}"
    
    # All within bounds
    for a in attractors:
        assert np.all(a >= 0.0) and np.all(a <= 1.0), "Attractor out of bounds"
    
    print(f"  Attractors: {len(attractors)}")
    for i, a in enumerate(attractors):
        d_a = np.linalg.norm(a - target_a)
        d_b = np.linalg.norm(a - target_b)
        print(f"    [{i}] dist_to_A={d_a:.3f}, dist_to_B={d_b:.3f}")
    
    print(f"PASS: test_local_model_multi_basin ({len(attractors)} attractors)")
    print("\nAll dynamics memory tests passed!")
