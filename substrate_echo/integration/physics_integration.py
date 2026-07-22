"""S9: Physics Integration — syncs BCFVT field dynamics with engine entities.

Wires the BCFVT field evolution, pillar coupling, and conservation
framework into a unified physics tick that syncs with the engine.

Integration points:
- Engine entities provide initial state (PSV)
- BCFVT field evolves the state (GL dynamics + pillar coupling)
- Conservation framework enforces invariants
- Updated state is written back to entities

References:
- PLAN.md Phase S9: Physics Integration
- BCFVT-00, BCFVT-02, BCFVT-05 implementations
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time


@dataclass
class PhysicsConfig:
    """Configuration for physics integration."""
    # Tick rate
    target_tick_rate: float = 60.0   # Hz
    max_tick_dt: float = 0.1         # max time step
    
    # BCFVT parameters
    diffusion_coefficient: float = 0.1
    gl_coupling: float = 1.0
    gl_eta: float = 0.5
    dissipation: float = 0.01
    
    # Conservation
    enforce_conservation: bool = True
    norm_target: float = 1.0
    
    # Performance
    max_entities: int = 100
    field_resolution: int = 16       # 16D pillar space


class PhysicsIntegration:
    """S9 Physics Integration.
    
    Unified physics tick that:
    1. Reads entity states from engine
    2. Evolves BCFVT field dynamics
    3. Enforces conservation laws
    4. Writes updated states back to entities
    
    This is the bridge between the mathematical BCFVT framework
    and the real-time game engine loop.
    """
    
    def __init__(self, config: Optional[PhysicsConfig] = None):
        self.config = config or PhysicsConfig()
        
        # Initialize BCFVT components
        from ..dynamics.field_evolution import FieldEvolver, FieldConfig
        from ..dynamics.conservation import ConservationFramework
        from ..dynamics.pillar_coupling import PillarCoupling
        
        field_config = FieldConfig(
            D=self.config.diffusion_coefficient,
            lambda_gl=self.config.gl_coupling,
            eta=self.config.gl_eta,
            gamma=self.config.dissipation,
            dt=1.0 / self.config.target_tick_rate,
        )
        
        self.field_evolver = FieldEvolver(field_config)
        self.conservation = ConservationFramework(
            enabled=self.config.enforce_conservation,
            lambda_gl=self.config.gl_coupling,
            eta=self.config.gl_eta,
        )
        self.pillar_coupling = PillarCoupling()
        
        # Entity state cache
        self._entity_states: dict[str, np.ndarray] = {}
        self._entity_field_states: dict[str, np.ndarray] = {}
        
        # Performance tracking
        self._tick_times: list[float] = []
        self._tick_count: int = 0
        self._conservation_violations: int = 0
    
    # ── Entity Management ─────────────────────────────────────────
    
    def register_entity(self, entity_id: str, initial_psv: np.ndarray) -> None:
        """Register an entity for physics simulation."""
        if len(self._entity_states) >= self.config.max_entities:
            return
        
        self._entity_states[entity_id] = initial_psv.copy()
        
        # Initialize field state from PSV
        field_state = self.pillar_coupling.reconstruct_field(initial_psv)
        self._entity_field_states[entity_id] = field_state
    
    def unregister_entity(self, entity_id: str) -> None:
        """Remove an entity from physics simulation."""
        self._entity_states.pop(entity_id, None)
        self._entity_field_states.pop(entity_id, None)
    
    def get_entity_state(self, entity_id: str) -> Optional[np.ndarray]:
        """Get current PSV state for an entity."""
        return self._entity_states.get(entity_id)
    
    # ── Physics Tick ──────────────────────────────────────────────
    
    def tick(self, dt: Optional[float] = None,
             external_forces: Optional[dict[str, np.ndarray]] = None) -> dict:
        """Run one physics tick for all entities.
        
        Args:
            dt: time step (default: 1/target_tick_rate)
            external_forces: optional forces applied to entities
        
        Returns:
            dict with updated states and statistics
        """
        tick_start = time.time()
        
        if dt is None:
            dt = 1.0 / self.config.target_tick_rate
        
        if external_forces is None:
            external_forces = {}
        
        updated = {}
        conservation_results = []
        
        for entity_id, field_state in self._entity_field_states.items():
            # Get external force if any
            force = external_forces.get(entity_id, np.zeros(16))
            
            # Evolve field
            new_field = self._evolve_entity_field(entity_id, field_state, dt, force)
            
            # Project back to pillars
            new_pillars = self.pillar_coupling.project_to_pillars(new_field)
            
            # Conservation check
            if self.config.enforce_conservation:
                results = self.conservation.check_all(new_pillars)
                conservation_results.extend(results)
                
                # Check if any failed
                for r in results:
                    if not r.passed:
                        self._conservation_violations += 1
                        # Correct norm if needed
                        new_pillars = self.conservation.correct_norm(new_pillars)
            
            # Update states
            self._entity_field_states[entity_id] = new_field
            self._entity_states[entity_id] = new_pillars
            
            updated[entity_id] = new_pillars
        
        # Track performance
        tick_time = time.time() - tick_start
        self._tick_times.append(tick_time)
        if len(self._tick_times) > 100:
            self._tick_times.pop(0)
        
        self._tick_count += 1
        
        return {
            "updated_entities": updated,
            "tick_time_ms": tick_time * 1000,
            "conservation_results": conservation_results,
            "stats": self.stats(),
        }
    
    def _evolve_entity_field(self, entity_id: str,
                              field_state: np.ndarray,
                              dt: float,
                              external_force: np.ndarray) -> np.ndarray:
        """Evolve field for a single entity."""
        # GL dynamics
        rhs = self.field_evolver.rhs(field_state, dt)
        
        # Add external force
        rhs += external_force
        
        # Euler step
        new_field = field_state + dt * rhs
        
        # Enforce bounds
        new_field = np.clip(new_field, 0.0, 1.0)
        
        return new_field
    
    # ── Batch Operations ──────────────────────────────────────────
    
    def sync_from_engine(self, entity_data: list[dict]) -> None:
        """Sync entity states from engine data.
        
        Args:
            entity_data: list of dicts with 'id' and 'psv' keys
        """
        for data in entity_data:
            entity_id = data.get('id', 'unknown')
            psv = np.array(data.get('psv', [0.5] * 16), dtype=np.float64)
            
            if entity_id not in self._entity_states:
                self.register_entity(entity_id, psv)
            else:
                self._entity_states[entity_id] = psv
    
    def sync_to_engine(self) -> list[dict]:
        """Export current states for engine consumption.
        
        Returns list of dicts with 'id' and 'psv' keys.
        """
        return [
            {"id": eid, "psv": psv.tolist()}
            for eid, psv in self._entity_states.items()
        ]
    
    # ── Query ─────────────────────────────────────────────────────
    
    def get_entity_field(self, entity_id: str) -> Optional[np.ndarray]:
        """Get the BCFVT field state for an entity."""
        return self._entity_field_states.get(entity_id)
    
    def get_all_entities(self) -> dict[str, np.ndarray]:
        """Get all entity PSV states."""
        return {k: v.copy() for k, v in self._entity_states.items()}
    
    def get_similarity(self, entity_a: str, entity_b: str) -> float:
        """Compute similarity between two entities."""
        a = self._entity_states.get(entity_a)
        b = self._entity_states.get(entity_b)
        
        if a is None or b is None:
            return 0.0
        
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    # ── Statistics ─────────────────────────────────────────────────
    
    def stats(self) -> dict:
        """Get physics integration statistics."""
        avg_tick = sum(self._tick_times) / len(self._tick_times) if self._tick_times else 0
        
        return {
            "tick_count": self._tick_count,
            "entity_count": len(self._entity_states),
            "avg_tick_ms": round(avg_tick * 1000, 3),
            "max_tick_ms": round(max(self._tick_times) * 1000, 3) if self._tick_times else 0,
            "conservation_violations": self._conservation_violations,
            "conservation_rate": (
                self._conservation_violations / max(1, self._tick_count)
            ),
        }
    
    def reset(self) -> None:
        """Reset all state."""
        self._entity_states.clear()
        self._entity_field_states.clear()
        self._tick_times.clear()
        self._tick_count = 0
        self._conservation_violations = 0
