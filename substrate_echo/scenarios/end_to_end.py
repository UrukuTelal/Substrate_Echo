"""S12: End-to-End Scenario — full simulation with agents, physics, memory, learning.

Demonstrates the complete Substrate_Echo system working together:
1. Multiple cognitive agents with different roles
2. BCFVT field dynamics (Ginzburg-Landau)
3. Attractor memory formation and recall
4. Social field interactions and reputation
5. Conservation law enforcement
6. Vortex formation and tracking

This is a runnable scenario that exercises all components.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time


@dataclass
class ScenarioConfig:
    """Configuration for the end-to-end scenario."""
    # Simulation
    num_ticks: int = 100
    dt: float = 0.1
    
    # Agents
    num_agents: int = 5
    agent_roles: list[str] = field(default_factory=lambda: [
        "perception", "memory", "planning", "creativity", "environment"
    ])
    
    # Field
    initial_field_energy: float = 0.5
    noise_amplitude: float = 0.01
    
    # Memory
    memory_capacity: int = 50
    memory_type: str = "dynamics"  # "attractor" or "dynamics"
    
    # Social
    communication_probability: float = 0.3


class EndToEndScenario:
    """S12 End-to-End Scenario.
    
    Runs a complete simulation demonstrating all Substrate_Echo components:
    - BCFVT field dynamics with conservation
    - Cognitive agent ecology
    - Attractor memory formation
    - Social field interactions
    - Vortex detection
    """
    
    def __init__(self, config: Optional[ScenarioConfig] = None):
        self.config = config or ScenarioConfig()
        
        # Initialize all components
        self._setup_components()
        
        # State tracking
        self._tick_log: list[dict] = []
        self._final_stats: Optional[dict] = None
    
    def _setup_components(self) -> None:
        """Initialize all Substrate_Echo components."""
        # Field dynamics
        from ..dynamics.field_evolution import FieldEvolver, FieldConfig
        from ..dynamics.conservation import ConservationFramework
        from ..dynamics.pillar_coupling import PillarCoupling
        from ..dynamics.vortex_dynamics import VortexDynamics
        
        field_config = FieldConfig(
            dt=self.config.dt,
            gamma=0.01,
            lambda_gl=1.0,
            temperature=self.config.noise_amplitude,
        )
        
        self.field_evolver = FieldEvolver(field_config)
        self.conservation = ConservationFramework(enabled=True)
        self.pillar_coupling = PillarCoupling()
        self.vortex_dynamics = VortexDynamics()
        
        # Cognitive agents
        from ..core.cognitive_agents import AgentEcology
        self.agent_ecology = AgentEcology()
        
        # Memory
        from ..core.ontological_field import OntologicalField
        self._ontological_field = OntologicalField()
        
        if self.config.memory_type == "dynamics":
            from ..core.dynamics_memory import DynamicsMemory
            self.memory = DynamicsMemory(dim=16)
        else:
            from ..core.attractor_memory import AttractorMemory
            self.memory = AttractorMemory(field=self._ontological_field)
        
        # Social field
        from ..core.multi_agent_dynamics import SocialField
        self.social_field = SocialField()
        
        # Spatial world
        from ..core.spatial_world import SpatialWorldModel
        self.world_model = SpatialWorldModel()
        
        # Physics integration
        from ..integration.physics_integration import PhysicsIntegration
        self.physics = PhysicsIntegration()
        
        # Visualization
        from ..visualization.field_renderer import FieldRenderer
        self.renderer = FieldRenderer()
        
        # Planning stack (for dynamics memory)
        self._planner = None
        self._intent_generator = None
        if self.config.memory_type == "dynamics":
            from ..core.world_model import WorldModel as DynWorldModel
            from ..core.simulator import Simulator
            from ..core.evaluator import Evaluator, UtilityWeights
            from ..core.controller import Controller
            from ..core.planner import Planner
            from ..core.intent_generator import IntentGenerator, AgentPersonality
            
            dyn_wm = DynWorldModel(self.memory)
            simulator = Simulator(dyn_wm)
            evaluator = Evaluator(UtilityWeights(), dyn_wm)
            controller = Controller()
            self._planner = Planner(simulator, evaluator, controller)
            self._intent_generator = IntentGenerator(AgentPersonality(), dyn_wm)
        
        # State
        self._field_state = np.full(16, self.config.initial_field_energy)
        self._pillar_state = np.full(16, 0.5)
        
        # Project initial state and set norm target from projected norm
        self._pillar_state = self.pillar_coupling.project_to_pillars(self._field_state)
        self.conservation.set_norm_target(float(np.linalg.norm(self._pillar_state)))
        
        # Create agents
        for i in range(self.config.num_agents):
            role = self.config.agent_roles[i % len(self.config.agent_roles)]
            agent_id = f"agent_{i}"
            
            # Each agent has slightly different initial state
            agent_state = np.full(16, 0.5) + np.random.randn(16) * 0.1
            agent_state = np.clip(agent_state, 0.0, 1.0)
            
            self.social_field.add_agent(agent_id, role, agent_state)
    
    # ── Simulation Loop ───────────────────────────────────────────
    
    def run(self) -> dict:
        """Run the complete end-to-end scenario."""
        mode = "predictive" if self._planner else "reactive"
        print(f"=== Starting End-to-End Scenario ({self.config.num_ticks} ticks, {mode} mode) ===")
        
        for tick in range(self.config.num_ticks):
            tick_data = self._run_tick(tick)
            self._tick_log.append(tick_data)
            
            # Progress logging
            if tick % 20 == 0:
                intent_str = tick_data.get("intent", "")
                print(f"  Tick {tick:>4d}: "
                      f"energy={tick_data['field_energy']:.4f}, "
                      f"vortices={tick_data['vortex_count']}, "
                      f"agents={tick_data['active_agents']}, "
                      f"intent={intent_str}, "
                      f"conservation={tick_data['conservation_passed']}")
        
        # Final summary
        self._final_stats = self._compute_final_stats()
        
        print("\n=== Scenario Complete ===")
        print(f"  Total ticks: {self.config.num_ticks}")
        print(f"  Final energy: {self._final_stats['final_energy']:.4f}")
        print(f"  Conservation rate: {self._final_stats['conservation_rate']:.2%}")
        print(f"  Vortices formed: {self._final_stats['total_vortices']}")
        print(f"  Memories formed: {self._final_stats['memories_formed']}")
        
        return self._final_stats
    
    def _run_tick(self, tick: int) -> dict:
        """Run one simulation tick."""
        # 1. Field evolution (GL dynamics)
        rhs = self.field_evolver.rhs(self._field_state, self.config.dt)
        self._field_state = self._field_state + self.config.dt * rhs
        self._field_state = np.clip(self._field_state, 0.0, 1.0)
        
        # 2. Project to pillars
        self._pillar_state = self.pillar_coupling.project_to_pillars(self._field_state)
        
        # 3. Conservation check
        conservation_results = self.conservation.check_all(self._pillar_state)
        conservation_passed = all(r.passed for r in conservation_results)
        
        # 4. Memory encoding (every 10 ticks)
        memories_formed = 0
        if tick % 10 == 0:
            try:
                from ..models.experience import Experience, ExperienceType
                exp = Experience(
                    experience_id=f"exp_{tick}",
                    experience_type=ExperienceType.LEARNING,
                    psv_snapshot=self._pillar_state.tolist(),
                    description=f"Tick {tick} observation",
                )
                self.memory.encode(exp)
                memories_formed = 1
            except Exception:
                pass
        
        # 5. Agent ecology tick (always run for perception)
        responses = self.agent_ecology.tick(
            self._pillar_state,
            world_model=self.world_model,
            memory=self.memory,
        )
        active_agents = len(responses)
        
        # 5b. Action selection: predictive or reactive
        intent = None
        if self._planner is not None and self._intent_generator is not None:
            try:
                intent_proposal = self._intent_generator.generate_intent(self._pillar_state)
                plan = self._planner.plan(self._pillar_state, intent_proposal)
                intent = intent_proposal.intent.name if plan.actions else None
            except Exception:
                intent = None
        
        # 6. Social field tick
        social_stats = self.social_field.tick(self.config.dt)
        
        # 7. Vortex detection (simplified: check for low-amplitude points)
        vortex_count = 0
        if np.min(self._field_state) < 0.1:
            vortex_count = 1
        
        # 8. Physics integration
        self.physics.register_entity(f"main", self._pillar_state)
        physics_tick = self.physics.tick(self.config.dt)
        
        # Compute field energy
        field_energy = float(np.dot(self._field_state, self._field_state))
        
        return {
            "tick": tick,
            "field_energy": field_energy,
            "vortex_count": vortex_count,
            "active_agents": active_agents,
            "conservation_passed": conservation_passed,
            "conservation_results": conservation_results,
            "memories_formed": memories_formed,
            "intent": intent,
            "social_coherence": social_stats.get("coherence", 0),
            "social_diversity": social_stats.get("diversity", 0),
        }
    
    def _compute_final_stats(self) -> dict:
        """Compute final statistics."""
        if not self._tick_log:
            return {}
        
        energies = [t["field_energy"] for t in self._tick_log]
        conservation_results = []
        for t in self._tick_log:
            conservation_results.extend(t.get("conservation_results", []))
        
        passed = sum(1 for r in conservation_results if r.passed)
        total = len(conservation_results)
        
        return {
            "final_energy": energies[-1],
            "min_energy": min(energies),
            "max_energy": max(energies),
            "energy_stable": abs(energies[-1] - energies[0]) < 0.1,
            "conservation_rate": passed / max(1, total),
            "total_vortices": sum(t["vortex_count"] for t in self._tick_log),
            "memories_formed": sum(t["memories_formed"] for t in self._tick_log),
            "avg_agents_active": np.mean([t["active_agents"] for t in self._tick_log]),
            "avg_social_coherence": np.mean([t["social_coherence"] for t in self._tick_log]),
            "avg_social_diversity": np.mean([t["social_diversity"] for t in self._tick_log]),
            "tick_log": self._tick_log,
        }
    
    # ── Rendering ─────────────────────────────────────────────────
    
    def render_dashboard(self) -> str:
        """Render complete dashboard of scenario state."""
        return self.renderer.render_full_dashboard(
            field_state=self._field_state,
            attractors=[],
            vortices=self.vortex_dynamics.get_vortices(),
            agents=self.social_field.get_all_agents(),
            conservation_results=self.conservation.check_all(self._pillar_state) if self._pillar_state is not None else [],
            physics_stats=self.physics.stats(),
        )
    
    # ── Query ─────────────────────────────────────────────────────
    
    def get_field_state(self) -> np.ndarray:
        """Get current field state."""
        return self._field_state.copy()
    
    def get_pillar_state(self) -> np.ndarray:
        """Get current pillar state."""
        return self._pillar_state.copy()
    
    def get_tick_log(self) -> list[dict]:
        """Get the tick log."""
        return list(self._tick_log)
    
    def get_final_stats(self) -> Optional[dict]:
        """Get final statistics."""
        return self._final_stats


def run_scenario(num_ticks: int = 100, verbose: bool = True) -> dict:
    """Convenience function to run the end-to-end scenario."""
    config = ScenarioConfig(num_ticks=num_ticks)
    scenario = EndToEndScenario(config)
    
    results = scenario.run()
    
    if verbose:
        print("\n" + scenario.render_dashboard())
    
    return results
