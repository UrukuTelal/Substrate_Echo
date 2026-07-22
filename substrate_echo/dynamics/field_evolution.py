"""BCFVT-02 Field Theory Core — Ginzburg-Landau dynamics for 16D pillar state.

Implements the continuous field dynamics: ∂ℱ/∂t = D(ℱ) + I(ℱ) + T(ℱ)

Where:
- D(ℱ) = D∇²ℱ (diffusion term, D = diffusion coefficient)
- I(ℱ) = -∂V/∂ℱ* (interaction term from Ginzburg-Landau potential)
- T(ℱ) = -γℱ + η (dissipation + noise term)

Ginzburg-Landau potential: V(ℱ) = λ(|ℱ|² - η²)²

The solver uses semi-implicit Crank-Nicolson for stiff terms:
(I - Δt/2·A(ℱ_n))ℱ_{n+1} = (I + Δt/2·A(ℱ_n))ℱ_n + Δt·T(ℱ_n)

Vortices emerge naturally as zeros of ℱ where phase winds by 2πn.
They are energetically favorable when Ginzburg-Landau parameter κ > 1/√2 (Type-II).

References:
- BCFVT Implementation Plan, BCFVT-02
- Adversarial findings 02-1 (Crank-Nicolson details), 02-2 (explicit terms), 02-3 (vortex emergence)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np
import math


@dataclass
class FieldConfig:
    """Configuration for BCFVT field dynamics."""
    
    # Ginzburg-Landau parameters
    lambda_gl: float = 1.0          # GL coupling constant
    eta: float = 0.5                # vacuum expectation value
    kappa: float = 1.0              # GL parameter (κ = λ₂/λ₁)
    
    # Diffusion coefficient
    D: float = 0.1                  # diffusion coefficient
    
    # Dissipation
    gamma: float = 0.01             # dissipation coefficient
    
    # Noise
    temperature: float = 0.001      # noise amplitude (T)
    
    # Numerical parameters
    dt: float = 0.01                # time step
    max_dt: float = 0.1             # maximum adaptive dt
    min_dt: float = 1e-6            # minimum adaptive dt
    cfl_factor: float = 0.9         # CFL safety factor
    
    # Solver parameters
    cg_tolerance: float = 1e-8      # conjugate gradient tolerance
    cg_max_iter: int = 100          # max CG iterations
    max_iterations: int = 10000     # max field evolution steps
    
    # Validation thresholds
    energy_tolerance: float = 0.01  # 1% energy drift allowed
    norm_tolerance: float = 1e-10   # norm conservation tolerance


class SolverType(Enum):
    """Available numerical solvers."""
    RK4 = auto()                    # Runge-Kutta 4th order
    CRANK_NICOLSON = auto()         # Semi-implicit Crank-Nicolson
    ADAPTIVE = auto()               # Adaptive time stepping


# Backward-compatible alias
EvolutionConfig = FieldConfig


class FieldEvolver:
    """BCFVT-02 Field Theory Core.
    
    Evolves the 16D pillar state ℱ under Ginzburg-Landau dynamics:
    ∂ℱ/∂t = D∇²ℱ - ∂V/∂ℱ* - γℱ + η
    
    Where V(ℱ) = λ(|ℱ|² - η²)²
    
    Supports:
    - RK4 for non-stiff terms
    - Semi-implicit Crank-Nicolson for stiff terms
    - Adaptive time stepping with CFL condition
    - Energy monitoring and conservation checks
    """
    
    def __init__(self, config: Optional[FieldConfig] = None,
                 solver_type: SolverType = SolverType.CRANK_NICOLSON):
        self.config = config or FieldConfig()
        self.solver_type = solver_type
        
        # State tracking
        self._current_state: Optional[np.ndarray] = None
        self._energy_history: list[float] = []
        self._step_count: int = 0
        
        # Precompute coupling matrix
        self._coupling_matrix = self._build_coupling_matrix()
    
    def _build_coupling_matrix(self) -> np.ndarray:
        """Build 16×16 inter-pillar coupling matrix."""
        n = 16
        coupling = np.eye(n, dtype=np.float64)
        
        # Add off-diagonal coupling (adjacent pillars)
        for i in range(n):
            # Couple to neighboring pillars
            left = (i - 1) % n
            right = (i + 1) % n
            coupling[i, left] = 0.1
            coupling[i, right] = 0.1
        
        return coupling
    
    # ── Ginzburg-Landau Potential ─────────────────────────────────
    
    def potential(self, state: np.ndarray) -> float:
        """Compute Ginzburg-Landau potential: V(ℱ) = λ(|ℱ|² - η²)²
        
        This is the double-well potential that drives the system
        toward the vacuum expectation value η.
        """
        norm_sq = float(np.dot(state, state))
        return self.config.lambda_gl * (norm_sq - self.config.eta ** 2) ** 2
    
    def potential_gradient(self, state: np.ndarray) -> np.ndarray:
        """Compute ∂V/∂ℱ* = 2λ(|ℱ|² - η²)ℱ
        
        Gradient of the GL potential with respect to ℱ.
        """
        norm_sq = float(np.dot(state, state))
        return 2.0 * self.config.lambda_gl * (norm_sq - self.config.eta ** 2) * state
    
    # ── Diffusion Term ────────────────────────────────────────────
    
    def diffusion(self, state: np.ndarray) -> np.ndarray:
        """Compute D∇²ℱ (diffusion term).
        
        Discrete Laplacian on 16D pillar vector with cyclic coupling.
        """
        n = len(state)
        result = np.zeros_like(state)
        
        for i in range(n):
            left = (i - 1) % n
            right = (i + 1) % n
            # Second derivative: f''(x) ≈ f(x+h) - 2f(x) + f(x-h)
            result[i] = state[left] - 2.0 * state[i] + state[right]
        
        return self.config.D * result
    
    # ── Dissipation + Noise ───────────────────────────────────────
    
    def dissipation(self, state: np.ndarray) -> np.ndarray:
        """Compute T(ℱ) = -γℱ (dissipation term)."""
        return -self.config.gamma * state
    
    def noise(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Compute η (noise term).
        
        Amplitude scales with temperature and √dt for proper
        statistical mechanics (fluctuation-dissipation theorem).
        """
        amplitude = self.config.temperature * np.sqrt(dt)
        return amplitude * np.random.randn(*state.shape)
    
    # ── Full RHS ──────────────────────────────────────────────────
    
    def rhs(self, state: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Compute full right-hand side: D(ℱ) + I(ℱ) + T(ℱ)
        
        ∂ℱ/∂t = D∇²ℱ - ∂V/∂ℱ* - γℱ + η
        """
        diffusion_term = self.diffusion(state)
        interaction_term = -self.potential_gradient(state)
        dissipation_term = self.dissipation(state)
        noise_term = self.noise(state, dt)
        
        return diffusion_term + interaction_term + dissipation_term + noise_term
    
    # ── Solvers ───────────────────────────────────────────────────
    
    def step_rk4(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Runge-Kutta 4th order step.
        
        Good for non-stiff parts of the dynamics.
        """
        k1 = self.rhs(state, dt)
        k2 = self.rhs(state + 0.5 * dt * k1, dt)
        k3 = self.rhs(state + 0.5 * dt * k2, dt)
        k4 = self.rhs(state + dt * k3, dt)
        
        return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    
    def step_crank_nicolson(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Semi-implicit Crank-Nicolson step.
        
        Linearizes V(ℱ) around ℱ_n:
        (I - Δt/2·A(ℱ_n))ℱ_{n+1} = (I + Δt/2·A(ℱ_n))ℱ_n + Δt·T(ℱ_n)
        
        Where A is the linearized operator combining:
        - Diffusion: -D∇²
        - GL interaction: -∂²V/∂ℱ*∂ℱ
        
        Uses conjugate gradient for the implicit solve.
        """
        n = len(state)
        
        # Build linearized operator A
        A = self._build_linearized_operator(state, n)
        
        # RHS: (I + Δt/2·A)ℱ_n + Δt·T(ℱ_n)
        I = np.eye(n, dtype=np.float64)
        rhs_matrix = I + 0.5 * dt * A
        rhs_vec = rhs_matrix @ state + dt * self.dissipation(state)
        
        # Add noise
        rhs_vec += dt * self.noise(state, dt)
        
        # LHS: (I - Δt/2·A)ℱ_{n+1} = rhs_vec
        lhs_matrix = I - 0.5 * dt * A
        
        # Solve using conjugate gradient
        new_state = self._solve_conjugate_gradient(
            lhs_matrix, rhs_vec, state, 
            self.config.cg_tolerance, 
            self.config.cg_max_iter
        )
        
        return new_state
    
    def _build_linearized_operator(self, state: np.ndarray, n: int) -> np.ndarray:
        """Build linearized operator A for Crank-Nicolson.
        
        A combines:
        - Diffusion: -D∇² (negative Laplacian)
        - GL interaction: -∂²V/∂ℱ*∂ℱ
        """
        # Diffusion part: -D∇² (Laplacian)
        A_diffusion = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            left = (i - 1) % n
            right = (i + 1) % n
            A_diffusion[i, i] = 2.0 * self.config.D
            A_diffusion[i, left] = -self.config.D
            A_diffusion[i, right] = -self.config.D
        
        # GL interaction part: -∂²V/∂ℱ*∂ℱ
        # ∂²V/∂ℱ*∂ℱ = 2λ(|ℱ|² - η²)I + 4λℱ⊗ℱ
        norm_sq = float(np.dot(state, state))
        A_interaction = -2.0 * self.config.lambda_gl * (
            norm_sq - self.config.eta ** 2
        ) * np.eye(n, dtype=np.float64)
        
        # Add 4λℱ⊗ℱ term
        A_interaction -= 4.0 * self.config.lambda_gl * np.outer(state, state)
        
        return A_diffusion + A_interaction
    
    def _solve_conjugate_gradient(self, A: np.ndarray, b: np.ndarray,
                                  x0: np.ndarray, tol: float,
                                  max_iter: int) -> np.ndarray:
        """Solve Ax = b using conjugate gradient method."""
        x = x0.copy()
        r = b - A @ x
        p = r.copy()
        rsold = float(np.dot(r, r))
        
        for _ in range(max_iter):
            Ap = A @ p
            alpha = rsold / max(float(np.dot(p, Ap)), 1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = float(np.dot(r, r))
            
            if np.sqrt(rsnew) < tol:
                break
            
            beta = rsnew / max(rsold, 1e-12)
            p = r + beta * p
            rsold = rsnew
        
        return x
    
    # ── CFL Condition ─────────────────────────────────────────────
    
    def compute_cfl_dt(self, state: np.ndarray) -> float:
        """Compute maximum stable time step from CFL condition.
        
        For diffusion: Δt < Δx²/(2D)
        For GL potential: Δt < 1/(2λ|ℱ|²)
        """
        norm_sq = float(np.dot(state, state))
        
        # Diffusion constraint
        dt_diffusion = 0.5 / max(self.config.D, 1e-12)
        
        # GL constraint
        dt_gl = 1.0 / max(2.0 * self.config.lambda_gl * norm_sq, 1e-12)
        
        # Take minimum
        dt_max = min(dt_diffusion, dt_gl)
        
        return self.config.cfl_factor * min(dt_max, self.config.max_dt)
    
    # ── Energy Monitoring ─────────────────────────────────────────
    
    def compute_energy(self, state: np.ndarray) -> float:
        """Compute total energy: E[ℱ] = ½|∇ℱ|² + V(ℱ)
        
        Kinetic energy from gradients + GL potential energy.
        """
        # Kinetic energy: ½|∇ℱ|²
        kinetic = 0.0
        for i in range(len(state)):
            left = (i - 1) % len(state)
            right = (i + 1) % len(state)
            grad = 0.5 * (state[right] - state[left])
            kinetic += 0.5 * grad * grad
        
        # Potential energy
        potential = self.potential(state)
        
        return kinetic + potential
    
    def check_energy_stability(self, energy_history: list[float]) -> tuple[bool, str]:
        """Check Lyapunov stability: energy should be non-increasing.
        
        Returns (passed, message).
        """
        if len(energy_history) < 2:
            return True, "Baseline set"
        
        current = energy_history[-1]
        previous = energy_history[-2]
        
        if previous < 1e-12:
            return True, "Baseline too small"
        
        relative_change = (current - previous) / abs(previous)
        
        passed = relative_change < self.config.energy_tolerance
        return passed, f"ΔE/E = {relative_change:.6f}"
    
    # ── Main Evolution Loop ───────────────────────────────────────
    
    def evolve(self, initial_state: np.ndarray, 
               steps: Optional[int] = None,
               callback: Optional[callable] = None) -> np.ndarray:
        """Evolve the field for N steps.
        
        Args:
            initial_state: Initial 16D pillar state
            steps: Number of steps (default: config.max_iterations)
            callback: Optional callback(state, step, energy) called each step
        
        Returns:
            Final evolved state
        """
        if steps is None:
            steps = self.config.max_iterations
        
        state = initial_state.copy()
        self._current_state = state
        self._energy_history = []
        self._step_count = 0
        
        for step in range(steps):
            # Adaptive time step
            if self.solver_type == SolverType.ADAPTIVE:
                dt = self.compute_cfl_dt(state)
            else:
                dt = self.config.dt
            
            # Solve one step
            if self.solver_type == SolverType.RK4:
                new_state = self.step_rk4(state, dt)
            else:  # CRANK_NICOLSON or ADAPTIVE
                new_state = self.step_crank_nicolson(state, dt)
            
            # Enforce bounds [0, 1]
            new_state = np.clip(new_state, 0.0, 1.0)
            
            # Update state
            state = new_state
            self._current_state = state
            self._step_count += 1
            
            # Track energy
            energy = self.compute_energy(state)
            self._energy_history.append(energy)
            
            # Callback
            if callback:
                callback(state, step, energy)
            
            # Check convergence
            if step > 100 and step % 100 == 0:
                passed, msg = self.check_energy_stability(self._energy_history)
                if not passed:
                    # Reduce time step
                    self.config.dt *= 0.5
                    if self.config.dt < self.config.min_dt:
                        self.config.dt = self.config.min_dt
        
        return state
    
    # ── Statistics ─────────────────────────────────────────────────
    
    def stats(self) -> dict:
        """Get evolution statistics."""
        energy_history = self._energy_history
        
        return {
            "total_steps": self._step_count,
            "solver_type": self.solver_type.name,
            "current_energy": energy_history[-1] if energy_history else 0.0,
            "energy_range": (
                min(energy_history), max(energy_history)
            ) if energy_history else (0.0, 0.0),
            "energy_stable": self.check_energy_stability(energy_history)[0],
            "current_dt": self.config.dt,
        }
