"""BCFVT-03: Vortex Dynamics — topological defects in the BCFVT field.

Vortices are zeros of ℱ where the phase winds by 2πn around the zero.
In Ginzburg-Landau theory, vortices are energetically favorable when
the GL parameter κ = λ₂/λ₁ > 1/√2 (Type-II superconductor).

Key algorithms:
- Winding number: W = (1/2π)∮∇θ·dl (quantized to integer)
- Vortex core identification: |ℱ| < λ_GL (GL coherence length)
- Interaction forces: logarithmic potential in 2D
- Kelvin waves: helical displacements on vortex filaments

References:
- BCFVT Implementation Plan, BCFVT-03
- Adversarial findings 03-1 (algorithm detail), 03-2 (identification criteria),
  03-3 (Kelvin wave dispersion)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import math


@dataclass
class VortexConfig:
    """Configuration for vortex dynamics."""
    # Identification thresholds
    winding_tolerance: float = 0.01    # |W - n| < this for integer W
    min_core_size: float = 0.1         # minimum vortex core size (λ_GL)
    amplitude_threshold: float = 0.1   # |ℱ| < this to be "in core"
    
    # Interaction parameters
    interaction_strength: float = 1.0  # logarithmic interaction strength
    max_interaction_range: float = 5.0 # max range for pairwise forces
    
    # Kelvin wave parameters
    kelvin_omega0: float = 1.0        # base frequency
    kelvin_coupling: float = 0.1      # coupling strength
    
    # Dynamics
    dt: float = 0.01                  # time step
    damping: float = 0.01             # vortex motion damping


@dataclass
class Vortex:
    """A single vortex (topological defect)."""
    position: np.ndarray              # 2D/3D position
    winding_number: int               # quantized winding (±1, ±2, ...)
    core_size: float                  # vortex core radius
    amplitude: float                  # field amplitude at core
    phase: float                      # overall phase
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    age: float = 0.0                  # time since formation
    id: int = 0                       # unique identifier
    
    @property
    def energy(self) -> float:
        """Estimate vortex energy: E ∝ π|W|ln(R/λ_GL)"""
        return math.pi * abs(self.winding_number) * math.log(
            max(self.core_size * 10, self.core_size + 0.1)
        )


class VortexDynamics:
    """BCFVT-03 Vortex Dynamics.
    
    Extracts and tracks vortices from BCFVT field evolution.
    Vortices are topological defects in the order parameter field ℱ.
    
    Features:
    - Vortex identification via winding number
    - Pairwise interaction forces (logarithmic)
    - Kelvin wave excitations on filaments
    - Vortex-antivortex annihilation
    """
    
    def __init__(self, config: Optional[VortexConfig] = None):
        self.config = config or VortexConfig()
        self._vortices: list[Vortex] = []
        self._next_id: int = 0
        self._annihilation_events: list[dict] = []
    
    # ── Winding Number Computation ────────────────────────────────
    
    def compute_winding_number(self, field_amplitude: np.ndarray,
                                field_phase: np.ndarray,
                                center: tuple[int, int],
                                radius: int = 3) -> float:
        """Compute winding number around a point.
        
        W = (1/2π) ∮ ∇θ · dl
        
        Discrete version: sum phase differences along a loop
        around the center point.
        
        Args:
            field_amplitude: 2D array of |ℱ|
            field_phase: 2D array of arg(ℱ)
            center: (row, col) center point
            radius: loop radius in grid units
        
        Returns:
            Winding number (should be integer for true vortex)
        """
        cy, cx = center
        n_points = max(8, 4 * radius)  # number of sample points
        
        total_phase_winding = 0.0
        
        for i in range(n_points):
            angle = 2.0 * math.pi * i / n_points
            next_angle = 2.0 * math.pi * ((i + 1) % n_points) / n_points
            
            # Sample points on the loop
            y1 = int(cy + radius * math.sin(angle))
            x1 = int(cx + radius * math.cos(angle))
            y2 = int(cy + radius * math.sin(next_angle))
            x2 = int(cx + radius * math.cos(next_angle))
            
            # Bounds check
            h, w = field_amplitude.shape
            if not (0 <= y1 < h and 0 <= x1 < w and 0 <= y2 < h and 0 <= x2 < w):
                continue
            
            # Phase difference
            dtheta = field_phase[y2, x2] - field_phase[y1, x1]
            
            # Wrap to [-π, π]
            dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
            
            total_phase_winding += dtheta
        
        return total_phase_winding / (2.0 * math.pi)
    
    # ── Vortex Identification ─────────────────────────────────────
    
    def identify_vortices(self, field_amplitude: np.ndarray,
                          field_phase: np.ndarray) -> list[Vortex]:
        """Scan field for vortices.
        
        A vortex exists where:
        1. |ℱ| < amplitude_threshold (in the core)
        2. Winding number W ≈ integer (quantized)
        
        Returns list of detected vortices.
        """
        h, w = field_amplitude.shape
        detected = []
        
        # Scan with stride to avoid double-counting
        stride = max(1, self.config.min_core_size)
        
        for y in range(0, h, max(1, int(stride))):
            for x in range(0, w, max(1, int(stride))):
                # Check if in vortex core
                if field_amplitude[y, x] > self.config.amplitude_threshold:
                    continue
                
                # Compute winding number
                W = self.compute_winding_number(
                    field_amplitude, field_phase, (y, x)
                )
                
                # Check quantization
                nearest_int = round(W)
                if abs(W - nearest_int) < self.config.winding_tolerance and nearest_int != 0:
                    # Found a vortex!
                    vortex = Vortex(
                        position=np.array([float(x), float(y)]),
                        winding_number=int(nearest_int),
                        core_size=self.config.min_core_size,
                        amplitude=float(field_amplitude[y, x]),
                        phase=float(field_phase[y, x]),
                        id=self._next_id,
                    )
                    self._next_id += 1
                    detected.append(vortex)
        
        return detected
    
    # ── Interaction Forces ────────────────────────────────────────
    
    def compute_interaction_force(self, vortex_a: Vortex,
                                  vortex_b: Vortex) -> np.ndarray:
        """Compute interaction force between two vortices.
        
        Logarithmic potential in 2D:
        F = -∇V, V = -2π|W_a W_b| ln(r/λ_GL)
        
        Like-sign vortices repel, opposite-sign attract.
        """
        r_vec = vortex_b.position - vortex_a.position
        r = np.linalg.norm(r_vec)
        
        if r < 1e-6 or r > self.config.max_interaction_range:
            return np.zeros(2)
        
        # Direction unit vector
        r_hat = r_vec / r
        
        # Force magnitude: F ∝ W_a * W_b / r (logarithmic potential gradient)
        force_magnitude = (
            self.config.interaction_strength *
            vortex_a.winding_number * vortex_b.winding_number / r
        )
        
        # Like-sign repel (force along r_hat), opposite-sign attract
        return -force_magnitude * r_hat
    
    def compute_all_interactions(self) -> np.ndarray:
        """Compute net interaction force on each vortex."""
        n = len(self._vortices)
        forces = [np.zeros(2) for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                f_ij = self.compute_interaction_force(
                    self._vortices[i], self._vortices[j]
                )
                forces[i] += f_ij
                forces[j] -= f_ij  # Newton's third law
        
        return forces
    
    # ── Kelvin Waves ──────────────────────────────────────────────
    
    def kelvin_dispersion(self, k: float, R: float = 1.0) -> float:
        """Kelvin wave dispersion relation.
        
        ω(k) = (Ω₀ k² / ln(1/kR)) * (1 + ...)
        
        For helical displacements on a vortex filament.
        
        Args:
            k: wavenumber
            R: vortex core radius
        """
        if abs(k) < 1e-10:
            return 0.0
        
        kR = abs(k) * R
        if kR >= 1.0:
            kR = 0.999  # avoid log singularity
        
        return (
            self.config.kelvin_omega0 * k * k / math.log(1.0 / kR)
        )
    
    def apply_kelvin_wave(self, vortex: Vortex, k: float,
                           amplitude: float, dt: float) -> Vortex:
        """Apply Kelvin wave excitation to a vortex filament.
        
        Displaces the vortex core helically.
        """
        omega = self.kelvin_dispersion(k)
        
        # Helical displacement
        phase = omega * vortex.age
        displacement = amplitude * np.array([
            math.cos(k * vortex.position[0] + phase),
            math.sin(k * vortex.position[1] + phase),
        ])
        
        new_position = vortex.position + displacement * dt
        
        return Vortex(
            position=new_position,
            winding_number=vortex.winding_number,
            core_size=vortex.core_size,
            amplitude=vortex.amplitude,
            phase=vortex.phase,
            velocity=vortex.velocity,
            age=vortex.age + dt,
            id=vortex.id,
        )
    
    # ── Vortex Dynamics Step ──────────────────────────────────────
    
    def step(self, dt: Optional[float] = None) -> list[Vortex]:
        """Evolve vortex positions for one time step.
        
        1. Compute interaction forces
        2. Update velocities (with damping)
        3. Update positions
        4. Check for annihilation
        """
        if dt is None:
            dt = self.config.dt
        
        if not self._vortices:
            return []
        
        # Compute forces
        forces = self.compute_all_interactions()
        
        # Update vortices
        new_vortices = []
        for i, vortex in enumerate(self._vortices):
            # Velocity update with damping
            new_velocity = (
                vortex.velocity * (1 - self.config.damping) +
                forces[i] * dt
            )
            
            # Position update
            new_position = vortex.position + new_velocity * dt
            
            new_vortex = Vortex(
                position=new_position,
                winding_number=vortex.winding_number,
                core_size=vortex.core_size,
                amplitude=vortex.amplitude,
                phase=vortex.phase,
                velocity=new_velocity,
                age=vortex.age + dt,
                id=vortex.id,
            )
            new_vortices.append(new_vortex)
        
        # Check for annihilation (opposite-sign vortices close together)
        self._vortices = self._check_annihilation(new_vortices)
        
        return self._vortices
    
    def _check_annihilation(self, vortices: list[Vortex]) -> list[Vortex]:
        """Check for vortex-antivortex annihilation."""
        if len(vortices) < 2:
            return vortices
        
        alive = [True] * len(vortices)
        
        for i in range(len(vortices)):
            if not alive[i]:
                continue
            for j in range(i + 1, len(vortices)):
                if not alive[j]:
                    continue
                
                # Opposite-sign and close together
                if (vortices[i].winding_number + vortices[j].winding_number == 0):
                    r = np.linalg.norm(vortices[i].position - vortices[j].position)
                    if r < self.config.min_core_size:
                        alive[i] = False
                        alive[j] = False
                        self._annihilation_events.append({
                            "vortex_a_id": vortices[i].id,
                            "vortex_b_id": vortices[j].id,
                            "position": (vortices[i].position + vortices[j].position) / 2,
                            "winding": vortices[i].winding_number,
                        })
        
        return [v for v, a in zip(vortices, alive) if a]
    
    # ── Management ────────────────────────────────────────────────
    
    def add_vortex(self, position: np.ndarray, winding: int = 1) -> Vortex:
        """Manually add a vortex."""
        vortex = Vortex(
            position=np.asarray(position, dtype=np.float64),
            winding_number=winding,
            core_size=self.config.min_core_size,
            amplitude=0.0,
            phase=0.0,
            id=self._next_id,
        )
        self._next_id += 1
        self._vortices.append(vortex)
        return vortex
    
    def get_vortices(self) -> list[Vortex]:
        """Get all active vortices."""
        return list(self._vortices)
    
    def get_total_winding(self) -> int:
        """Get total winding number (conserved topological charge)."""
        return sum(v.winding_number for v in self._vortices)
    
    def get_annihilation_events(self) -> list[dict]:
        """Get history of annihilation events."""
        return list(self._annihilation_events)
    
    # ── Statistics ─────────────────────────────────────────────────
    
    def stats(self) -> dict:
        """Get vortex statistics."""
        n_vortices = len(self._vortices)
        
        if n_vortices == 0:
            return {
                "n_vortices": 0,
                "total_winding": 0,
                "annihilations": len(self._annihilation_events),
            }
        
        windings = [v.winding_number for v in self._vortices]
        energies = [v.energy for v in self._vortices]
        
        return {
            "n_vortices": n_vortices,
            "total_winding": sum(windings),
            "positive_winding": sum(1 for w in windings if w > 0),
            "negative_winding": sum(1 for w in windings if w < 0),
            "avg_energy": sum(energies) / len(energies),
            "total_energy": sum(energies),
            "annihilations": len(self._annihilation_events),
            "avg_age": sum(v.age for v in self._vortices) / n_vortices,
        }
