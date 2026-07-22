"""BCFVT-05: 16D Pillar Coupling — maps BCFVT field ℱ to 16D pillar state.

Implements the projection and back-reaction between the continuous
BCFVT order parameter field and the discrete 16D pillar state vector.

Projection: ℱ → ψ ∈ ℝ¹⁶ (pillar decomposition)
Back-reaction: ψ evolution affects ℱ through source terms

WHT (Walsh-Hadamard Transform) connection:
- Pillars represent WHT basis modes of the field
- Each pillar i corresponds to a Walsh function w_i(x)
- The field decomposition ℱ(x) = Σ_i ψ_i · w_i(x)

References:
- BCFVT Implementation Plan, BCFVT-05
- VNES-Lab psv_core.py PillarState
- DeveloperConsole bloch_sphere.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


PILLAR_NAMES = [
    "Awareness", "Willpower", "Force", "Influence",
    "Resistance", "Integrity", "Cohesion", "Relation",
    "Presence", "Warmth", "Memory", "Attraction",
    "Harm", "Distortion", "Flux", "Depth",
]


@dataclass
class PillarCouplingConfig:
    """Configuration for pillar-field coupling."""
    # Coupling strength between field and pillars
    coupling_strength: float = 0.1
    
    # Back-reaction strength (how much pillar evolution affects field)
    back_reaction_strength: float = 0.01
    
    # WHT basis normalization
    normalize_wht: bool = True
    
    # Conservation enforcement
    enforce_norm: bool = True
    norm_target: float = 1.0


class PillarCoupling:
    """BCFVT-05 16D Pillar Coupling.
    
    Maps between:
    - BCFVT field ℱ (continuous, complex order parameter)
    - 16D pillar state ψ (discrete, real-valued in [0,1])
    
    The mapping uses Walsh-Hadamard Transform (WHT) as the basis:
    - Forward: ℱ → ψ (projection onto WHT basis modes)
    - Backward: ψ → ℱ (reconstruction from basis modes)
    
    Each pillar i represents the amplitude of Walsh function w_i.
    """
    
    def __init__(self, config: Optional[PillarCouplingConfig] = None):
        self.config = config or PillarCouplingConfig()
        
        # Precompute WHT matrix (16×16 Hadamard)
        self._wht_matrix = self._build_wht_matrix()
        self._wht_inverse = np.linalg.inv(self._wht_matrix)
        
        # State tracking
        self._coupling_history: list[dict] = []
    
    def _build_wht_matrix(self) -> np.ndarray:
        """Build 16×16 Walsh-Hadamard Transform matrix.
        
        The WHT matrix H satisfies: H @ H^T = nI (orthogonal up to scaling)
        Normalized: H @ H^T = I
        """
        # Build recursively: H_{2n} = [[H_n, H_n], [H_n, -H_n]]
        H = np.array([[1.0]], dtype=np.float64)
        
        while H.shape[0] < 16:
            n = H.shape[0]
            H_new = np.zeros((2*n, 2*n), dtype=np.float64)
            H_new[:n, :n] = H
            H_new[:n, n:] = H
            H_new[n:, :n] = H
            H_new[n:, n:] = -H
            H = H_new
        
        if self.config.normalize_wht:
            H = H / np.sqrt(16.0)
        
        return H
    
    # ── Projection: ℱ → ψ ────────────────────────────────────────
    
    def project_to_pillars(self, field_state: np.ndarray) -> np.ndarray:
        """Project BCFVT field onto 16D pillar state.
        
        psi = H^T @ field (WHT projection)
        
        For a 16-component field, this decomposes into WHT basis modes.
        """
        if len(field_state) == 16:
            psi = self._wht_matrix.T @ field_state
        else:
            psi = self._wht_matrix.T @ field_state[:16]
        
        # Normalize to [0, 1] range
        psi = self._normalize_to_unit(psi)
        
        return psi
    
    def _normalize_to_unit(self, psi: np.ndarray) -> np.ndarray:
        """Normalize pillar values to [0, 1] range."""
        min_val = np.min(psi)
        max_val = np.max(psi)
        
        if max_val - min_val < 1e-10:
            return np.full_like(psi, 0.5)
        
        return (psi - min_val) / (max_val - min_val)
    
    # ── Reconstruction: ψ → ℱ ────────────────────────────────────
    
    def reconstruct_field(self, pillar_state: np.ndarray) -> np.ndarray:
        """Reconstruct BCFVT field from 16D pillar state.
        
        ℱ = H @ ψ (WHT reconstruction)
        """
        # Denormalize from [0, 1] to [-1, 1]
        field_range = 2.0 * pillar_state - 1.0
        
        # Reconstruct via WHT
        if self.config.normalize_wht:
            field = self._wht_matrix @ field_range
        else:
            field = self._wht_matrix @ field_range
        
        return field
    
    # ── Coupling: Field ↔ Pillars ─────────────────────────────────
    
    def compute_coupling_force(self, field_state: np.ndarray,
                                pillar_state: np.ndarray) -> np.ndarray:
        """Compute coupling force between field and pillars.
        
        F_coupling = -λ(ℱ_reconstructed - ℱ_actual)
        
        This force drives the field toward the pillar representation.
        """
        # Reconstruct field from pillars
        field_from_pillars = self.reconstruct_field(pillar_state)
        
        # Coupling force: push field toward pillar representation
        coupling = self.config.coupling_strength * (
            field_from_pillars - field_state
        )
        
        return coupling
    
    def compute_back_reaction(self, field_state: np.ndarray,
                               pillar_state: np.ndarray) -> np.ndarray:
        """Compute back-reaction of field on pillars.
        
        Δψ = γ(H^T @ (ℱ - ℱ_from_pillars))
        
        This updates pillars based on field evolution.
        """
        # Current field from pillars
        field_from_pillars = self.reconstruct_field(pillar_state)
        
        # Field difference
        field_diff = field_state - field_from_pillars
        
        # Project back to pillar space
        delta_psi = self._wht_matrix.T @ field_diff
        
        # Scale by back-reaction strength
        return self.config.back_reaction_strength * delta_psi
    
    # ── Full Coupling Step ────────────────────────────────────────
    
    def coupling_step(self, field_state: np.ndarray,
                      pillar_state: np.ndarray,
                      dt: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        """Perform one coupling step: update both field and pillars.
        
        Returns:
            (new_field, new_pillars)
        """
        # Compute forces
        field_force = self.compute_coupling_force(field_state, pillar_state)
        pillar_force = self.compute_back_reaction(field_state, pillar_state)
        
        # Update field
        new_field = field_state + dt * field_force
        
        # Update pillars
        new_pillars = pillar_state + dt * pillar_force
        
        # Enforce bounds
        new_pillars = np.clip(new_pillars, 0.0, 1.0)
        
        # Enforce norm conservation
        if self.config.enforce_norm:
            new_pillars = self._enforce_norm(new_pillars)
        
        # Track
        self._coupling_history.append({
            "field_force_norm": float(np.linalg.norm(field_force)),
            "pillar_force_norm": float(np.linalg.norm(pillar_force)),
        })
        
        return new_field, new_pillars
    
    def _enforce_norm(self, pillars: np.ndarray) -> np.ndarray:
        """Enforce norm conservation on pillar state."""
        current_norm = np.linalg.norm(pillars)
        if current_norm < 1e-10:
            return pillars
        
        return pillars * (self.config.norm_target / current_norm)
    
    # ── WHT Utilities ─────────────────────────────────────────────
    
    def get_wht_matrix(self) -> np.ndarray:
        """Get the WHT matrix."""
        return self._wht_matrix.copy()
    
    def get_pillar_name(self, index: int) -> str:
        """Get name of pillar at given index."""
        if 0 <= index < 16:
            return PILLAR_NAMES[index]
        return f"Pillar_{index}"
    
    def get_dominant_pillars(self, pillar_state: np.ndarray,
                              top_k: int = 3) -> list[tuple[int, str, float]]:
        """Get the most active pillars."""
        indices = np.argsort(pillar_state)[::-1][:top_k]
        return [
            (int(i), self.get_pillar_name(int(i)), float(pillar_state[i]))
            for i in indices
        ]
    
    # ── Statistics ─────────────────────────────────────────────────
    
    def stats(self) -> dict:
        """Get coupling statistics."""
        if not self._coupling_history:
            return {"coupling_steps": 0}
        
        field_forces = [h["field_force_norm"] for h in self._coupling_history]
        pillar_forces = [h["pillar_force_norm"] for h in self._coupling_history]
        
        return {
            "coupling_steps": len(self._coupling_history),
            "avg_field_force": sum(field_forces) / len(field_forces),
            "avg_pillar_force": sum(pillar_forces) / len(pillar_forces),
            "wht_matrix_shape": self._wht_matrix.shape,
        }
