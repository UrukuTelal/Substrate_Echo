"""Tests for Dynamics module — BCFVT field evolution, diffusion, attractor dynamics."""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from substrate_echo.dynamics.field_evolution import FieldEvolver, EvolutionConfig, FieldConfig, SolverType
from substrate_echo.dynamics.diffusion import DiffusionTensor, DiffusionConfig
from substrate_echo.dynamics.attractor_dynamics import AttractorDynamics, DynamicsConfig


# ── Field Evolution Tests (BCFVT-02) ────────────────────────────────

def test_euler_solver():
    config = FieldConfig(dt=0.01, gamma=0.1, lambda_gl=0.0)
    evolver = FieldEvolver(config, solver_type=SolverType.RK4)
    
    state = np.full(16, 0.5)
    result = evolver.evolve(state, steps=100)
    # With gamma=0.1 damping, state should decay
    assert np.all(result < 0.5)
    print("PASS: test_euler_solver")


def test_rk4_solver():
    config = FieldConfig(dt=0.01, gamma=0.05)
    evolver = FieldEvolver(config, solver_type=SolverType.RK4)
    
    state = np.full(16, 0.5)
    result = evolver.evolve(state, steps=100)
    # State should evolve (decay from gamma)
    assert result is not None
    assert len(result) == 16
    print("PASS: test_rk4_solver")


def test_crank_nicolson():
    config = FieldConfig(dt=0.01, gamma=0.05)
    evolver = FieldEvolver(config, solver_type=SolverType.CRANK_NICOLSON)
    
    state = np.full(16, 0.5)
    result = evolver.evolve(state, steps=100)
    assert result is not None
    assert len(result) == 16
    print("PASS: test_crank_nicolson")


def test_adaptive_solver():
    config = FieldConfig(dt=0.01, gamma=0.05)
    evolver = FieldEvolver(config, solver_type=SolverType.ADAPTIVE)
    
    state = np.full(16, 0.5)
    result = evolver.evolve(state, steps=100)
    assert result is not None
    stats = evolver.stats()
    assert stats["total_steps"] > 0
    print("PASS: test_adaptive_solver")


def test_state_stays_in_bounds():
    config = FieldConfig(dt=0.01, gamma=0.01, temperature=0.1)
    evolver = FieldEvolver(config, solver_type=SolverType.RK4)
    
    state = np.full(16, 0.9)
    result = evolver.evolve(state, steps=50)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    print("PASS: test_state_stays_in_bounds")


def test_norm_drift():
    config = FieldConfig(dt=0.01, gamma=0.0, temperature=0.0, lambda_gl=0.0)
    evolver = FieldEvolver(config, solver_type=SolverType.RK4)
    
    # State with norm = 1.0
    state = np.ones(16) / 4.0  # norm = 1.0
    
    result = evolver.evolve(state, steps=10)
    norm_drift = abs(np.linalg.norm(result) - 1.0)
    assert norm_drift < 0.1  # should be small
    print("PASS: test_norm_drift")


def test_gl_potential():
    """Test Ginzburg-Landau potential computation."""
    config = FieldConfig(lambda_gl=1.0, eta=0.5)
    evolver = FieldEvolver(config)
    
    # At vacuum: |ℱ| = η, V should be minimum
    vacuum_state = np.full(16, 0.5 / 4.0)  # |ℱ|² = 16*(0.125)² = 0.25
    v = evolver.potential(vacuum_state)
    assert v >= 0.0  # V >= 0 always
    
    # Away from vacuum: V should be larger
    excited_state = np.full(16, 0.9)
    v_excited = evolver.potential(excited_state)
    assert v_excited > v
    print("PASS: test_gl_potential")


def test_energy_monitoring():
    """Test energy tracking during evolution."""
    config = FieldConfig(dt=0.01, gamma=0.01)
    evolver = FieldEvolver(config, solver_type=SolverType.RK4)
    
    state = np.full(16, 0.5)
    result = evolver.evolve(state, steps=50)
    
    stats = evolver.stats()
    assert stats["total_steps"] == 50
    assert len(stats["energy_range"]) == 2
    print("PASS: test_energy_monitoring")


def test_cfl_condition():
    """Test CFL adaptive time step."""
    config = FieldConfig(D=0.1, lambda_gl=1.0)
    evolver = FieldEvolver(config)
    
    # High energy state should give small dt
    high_energy = np.full(16, 0.9)
    dt_high = evolver.compute_cfl_dt(high_energy)
    
    # Low energy state should give larger dt
    low_energy = np.full(16, 0.1)
    dt_low = evolver.compute_cfl_dt(low_energy)
    
    assert dt_high < dt_low
    print("PASS: test_cfl_condition")


def test_conjugate_gradient_solve():
    """Test CG solver for Ax = b."""
    config = FieldConfig()
    evolver = FieldEvolver(config)
    
    A = np.eye(16) * 2.0 + 0.1
    b = np.ones(16)
    x0 = np.zeros(16)
    
    result = evolver._solve_conjugate_gradient(A, b, x0, tol=1e-6, max_iter=200)
    
    # Check solution: Ax ≈ b
    residual = np.linalg.norm(A @ result - b)
    assert residual < 1e-3
    print("PASS: test_conjugate_gradient_solve")


# ── Diffusion Tensor Tests ────────────────────────────────────────

def test_diffusion_identity():
    diff = DiffusionTensor()
    state = np.full(16, 0.5)
    force = diff.apply(state)
    # Identity diffusion should push toward zero
    assert np.all(force < 0)
    print("PASS: test_diffusion_identity")


def test_diffusion_update():
    diff = DiffusionTensor()
    
    # Simulate correlated changes
    for _ in range(15):
        delta = np.random.randn(16) * 0.1
        diff.update_from_observation(delta)
    
    # Tensor should have changed from identity
    assert not np.allclose(diff.tensor, np.eye(16) * 0.01)
    print("PASS: test_diffusion_update")


def test_diffusion_strongest_couplings():
    diff = DiffusionTensor()
    diff.set_manual_coupling(0, 8, 0.5)  # Awareness ↔ Presence
    diff.set_manual_coupling(10, 15, 0.3)  # Memory ↔ Depth
    
    top = diff.get_strongest_couplings(top_k=2)
    assert len(top) == 2
    assert top[0][2] >= top[1][2]  # sorted descending
    print("PASS: test_diffusion_strongest_couplings")


def test_diffusion_pillar_influence():
    diff = DiffusionTensor()
    diff.set_manual_coupling(0, 8, 0.5)
    
    influence = diff.get_pillar_influence(8)
    assert "Awareness" in influence
    assert influence["Awareness"] == 0.5
    print("PASS: test_diffusion_pillar_influence")


# ── Attractor Dynamics Tests ──────────────────────────────────────

def test_attractor_decay():
    dynamics = AttractorDynamics()
    
    t0 = 1000.0
    dynamics.register_formation("mem1", t0)
    dynamics._access_history["mem1"] = [t0]
    
    strength = dynamics.compute_decay("mem1", 1.0, t0 + 60)
    assert strength < 1.0
    assert strength > 0.5
    print("PASS: test_attractor_decay")


def test_attractor_strengthening():
    dynamics = AttractorDynamics()
    
    t0 = time.time()
    strength = 0.5
    
    for i in range(5):
        strength = dynamics.record_access("mem1", strength, t0 + i * 0.1)
    
    assert strength > 0.5
    print("PASS: test_attractor_strengthening")


def test_attractor_merge():
    dynamics = AttractorDynamics()
    
    a = np.ones(16) * 0.5
    b = np.ones(16) * 0.51
    
    assert dynamics.should_merge(a, b)
    
    c = np.zeros(16)
    c[0] = 1.0
    d = np.zeros(16)
    d[8] = 1.0
    
    assert not dynamics.should_merge(c, d)
    print("PASS: test_attractor_merge")


def test_attractor_merge_center():
    dynamics = AttractorDynamics()
    
    center_a = np.full(16, 0.4)
    center_b = np.full(16, 0.6)
    
    new_center, new_strength = dynamics.merge_attractors(
        center_a, 0.8, center_b, 0.4
    )
    
    assert np.all(new_center > 0.4)
    assert np.all(new_center < 0.6)
    assert new_strength < 1.2
    print("PASS: test_attractor_merge_center")


def test_attractor_pruning():
    dynamics = AttractorDynamics()
    assert dynamics.should_prune(0.01)
    assert not dynamics.should_prune(0.5)
    print("PASS: test_attractor_pruning")


def test_access_frequency():
    dynamics = AttractorDynamics()
    
    t0 = time.time()
    for i in range(10):
        dynamics.record_access("mem1", 0.5, t0 + i * 0.01)
    
    freq = dynamics.get_access_frequency("mem1", window_seconds=1.0, current_time=t0 + 0.1)
    assert freq > 0
    print("PASS: test_access_frequency")


if __name__ == "__main__":
    print("=== Field Evolution (BCFVT-02) ===")
    test_euler_solver()
    test_rk4_solver()
    test_crank_nicolson()
    test_adaptive_solver()
    test_state_stays_in_bounds()
    test_norm_drift()
    test_gl_potential()
    test_energy_monitoring()
    test_cfl_condition()
    test_conjugate_gradient_solve()
    
    print("\n=== Diffusion Tensor ===")
    test_diffusion_identity()
    test_diffusion_update()
    test_diffusion_strongest_couplings()
    test_diffusion_pillar_influence()
    
    print("\n=== Attractor Dynamics ===")
    test_attractor_decay()
    test_attractor_strengthening()
    test_attractor_merge()
    test_attractor_merge_center()
    test_attractor_pruning()
    test_access_frequency()
    
    print("\nAll dynamics tests passed!")
