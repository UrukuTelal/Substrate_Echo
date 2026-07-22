"""S8: Agent Cognitive Loop — integrates memory, dynamics, and action.

The cognitive loop creates a continuous cycle:
1. Memory → Dynamics: attractor memories influence field evolution
2. Dynamics → Action: field state drives agent decisions
3. Action → Learning: agent actions create experiences that update memory

This closes the perception-action-learning loop, enabling
adaptive behavior through continuous self-organization.

Supports two action selection modes:
- Reactive (default): agent ecology consensus, no look-ahead
- Predictive (optimal): intent→plan→control→act via full planning stack

References:
- PLAN.md Phase S8: Agent Cognitive Loop Integration
- BCFVT: field evolution drives cognition through pillar coupling
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time


@dataclass
class CognitiveLoopConfig:
    """Configuration for the cognitive loop."""
    # Loop timing
    dt: float = 0.01                  # time step
    max_steps_per_tick: int = 10      # max field evolution steps per tick
    
    # Memory integration
    memory_influence_strength: float = 0.1  # how much memories affect field
    memory_recall_threshold: float = 0.3    # minimum similarity to recall
    
    # Learning
    learning_rate: float = 0.01       # learning rate for experience encoding
    experience_decay: float = 0.001   # how fast experiences fade
    
    # Energy
    cognitive_energy_cost: float = 0.01  # energy per cognitive cycle
    
    # Conservation
    enforce_conservation: bool = True
    norm_target: float = 1.0
    
    # Planning mode
    use_planner: bool = False         # use planning stack instead of reactive


class CognitiveLoop:
    """S8 Agent Cognitive Loop.
    
    Orchestrates the continuous perception-action-learning cycle:
    
    1. RECALL: Memory provides attractor patterns
    2. EVOLVE: Field evolves under GL dynamics + memory influence
    3. PERCEIVE: Agents evaluate field state
    4. PLAN (if planner available): simulate futures, evaluate, select
       OR ACT (reactive): agent ecology consensus
    5. LEARN: Actions create experiences, update memory
    
    This creates a self-organizing system where:
    - Memories shape perception
    - Perception drives action
    - Action creates new memories
    - The whole system adapts through experience
    """
    
    def __init__(self, config: Optional[CognitiveLoopConfig] = None):
        self.config = config or CognitiveLoopConfig()
        
        # State
        self._field_state: Optional[np.ndarray] = None
        self._pillar_state: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._tick_count: int = 0
        
        # Energy
        self._energy: float = 1.0
        
        # Experience buffer
        self._recent_experiences: list[dict] = []
        
        # Performance tracking
        self._action_history: list[dict] = []
        
        # Planning stack (optional)
        self._planner = None
        self._intent_generator = None
        self._controller = None
        self._current_intent = None
        
        # Developmental cognition modules (optional)
        self._intent_translator = None
        self._goal_manager = None
        self._comm_detector = None
    
    def set_planning_stack(self, planner, intent_generator=None, controller=None):
        """Set the planning stack for predictive action selection.
        
        When set, the cognitive loop uses intent→plan→control→act
        instead of reactive agent ecology consensus.
        """
        self._planner = planner
        self._intent_generator = intent_generator
        self._controller = controller
        if planner is not None:
            self.config.use_planner = True
    
    def set_developmental_modules(self, intent_translator=None,
                                   goal_manager=None,
                                   comm_detector=None):
        """Set developmental cognition modules.
        
        These add perception→cognition translation:
        - IntentTranslator: translates entities into affordances/PSV deltas
        - GoalManager: tracks per-agent goal states
        - CommunicativeIntentDetector: detects communicative behavior
        
        When set, the tick() method automatically:
        1. Translates perceived entities into affordances
        2. Updates goal tracking for observed agents
        3. Detects communicative intent
        4. Applies affordance PSV deltas to pillar state
        """
        self._intent_translator = intent_translator
        self._goal_manager = goal_manager
        self._comm_detector = comm_detector
    
    def initialize(self, initial_field: np.ndarray,
                   initial_pillars: np.ndarray) -> None:
        """Initialize the cognitive loop with starting state."""
        self._field_state = initial_field.copy()
        self._pillar_state = initial_pillars.copy()
        self._step_count = 0
        self._tick_count = 0
    
    # ── Step 1: Memory Recall ─────────────────────────────────────
    
    def recall_memories(self, memory_system,
                        current_pillars: np.ndarray,
                        top_k: int = 3) -> list[np.ndarray]:
        """Recall relevant memories based on current pillar state.
        
        Uses attractor memory to find memories similar to current state.
        Returns list of recalled memory patterns.
        """
        if memory_system is None:
            return []
        
        recalled = []
        
        try:
            # Try to use attractor memory's recall method
            results = memory_system.recall_by_cue(current_pillars, top_k=top_k)
            
            for result in results:
                if hasattr(result, 'center'):
                    # AttractorMemory result
                    if result.strength >= self.config.memory_recall_threshold:
                        recalled.append(result.center)
                elif isinstance(result, np.ndarray):
                    # Direct pattern
                    recalled.append(result)
        except (AttributeError, TypeError):
            # Fallback: use memory stats to get general influence
            pass
        
        return recalled
    
    def compute_memory_influence(self, recalled_patterns: list[np.ndarray],
                                  field_state: np.ndarray) -> np.ndarray:
        """Compute the influence of recalled memories on the field.
        
        M(ℱ) = Σ_i w_i · (ψ_i - ℱ)
        
        Where ψ_i are recalled memory patterns and w_i are weights.
        """
        if not recalled_patterns:
            return np.zeros_like(field_state)
        
        influence = np.zeros_like(field_state)
        
        for pattern in recalled_patterns:
            # Compute influence: push field toward memory pattern
            delta = pattern - field_state
            influence += delta * self.config.memory_influence_strength
        
        # Normalize by number of memories
        influence /= len(recalled_patterns)
        
        return influence
    
    # ── Step 2: Field Evolution ───────────────────────────────────
    
    def evolve_field(self, field_evolver,
                     memory_influence: np.ndarray,
                     dt: Optional[float] = None,
                     steps: int = 1) -> np.ndarray:
        """Evolve field with memory influence.
        
        Combines GL dynamics with memory influence:
        ∂ℱ/∂t = D∇²ℱ - ∂V/∂ℱ* - γℱ + η + M(ℱ)
        
        Where M(ℱ) is the memory influence term.
        """
        if dt is None:
            dt = self.config.dt
        
        if self._field_state is None:
            return np.zeros(16)
        
        # Evolve field with memory influence
        for _ in range(steps):
            # GL dynamics (without memory)
            gl_rhs = field_evolver.rhs(self._field_state, dt)
            
            # Add memory influence
            total_rhs = gl_rhs + memory_influence
            
            # Euler step
            self._field_state = self._field_state + dt * total_rhs
            
            # Enforce bounds
            self._field_state = np.clip(self._field_state, 0.0, 1.0)
            
            self._step_count += 1
        
        return self._field_state
    
    # ── Step 3: Agent Perception ──────────────────────────────────
    
    def perceive(self, agent_ecology,
                 world_model=None,
                 memory=None) -> list:
        """Agents evaluate current field state.
        
        Each active agent produces a response based on
        its pillar affinity and the current field state.
        """
        if agent_ecology is None or self._pillar_state is None:
            return []
        
        # Run cognitive cycle
        responses = agent_ecology.tick(
            self._pillar_state,
            world_model=world_model,
            memory=memory,
        )
        
        return responses
    
    # ── Step 3.5: Developmental Perception ──────────────────────────
    
    def _developmental_perception(self, responses, world_model, memory_system):
        """Translate perceived entities into affordances and PSV deltas.
        
        This is the Stage 2-5 developmental layer:
        - Stage 2 (Objects): Affordance evaluation
        - Stage 3 (Agents): Entity type classification
        - Stage 4 (Goals): Goal tracking per agent
        - Stage 5 (Intentionality): Communicative intent detection
        
        Returns dict with affordance results for logging.
        """
        result = {
            "affordances": [],
            "goal_states": {},
            "comm_signals": [],
            "total_psv_magnitude": 0.0,
        }
        
        if self._pillar_state is None:
            return result
        
        # Collect observed entities from responses
        observed_entities = self._extract_observed_entities(
            responses, world_model)
        
        if not observed_entities:
            return result
        
        # Translate each entity through developmental layers
        for entity in observed_entities:
            entity_id = entity.get("id", 0)
            entity_type = entity.get("type", "unknown")
            position = entity.get("position", np.zeros(3))
            properties = entity.get("properties", None)
            
            # Stage 2-3: Affordance translation
            if self._intent_translator is not None:
                from .affordance import EntityType
                type_map = {
                    "human": EntityType.HUMAN,
                    "animal": EntityType.ANIMAL,
                    "plant": EntityType.PLANT,
                    "mycelium": EntityType.MYCELIUM,
                    "object": EntityType.OBJECT,
                    "terrain": EntityType.TERRAIN,
                    "agent": EntityType.AGENT,
                    "unknown": EntityType.UNKNOWN,
                }
                etype = type_map.get(entity_type, EntityType.UNKNOWN)
                
                aff = self._intent_translator.translate(
                    entity_id=entity_id,
                    entity_type=etype,
                    position=position,
                    properties=properties,
                )
                result["affordances"].append(aff)
                result["total_psv_magnitude"] += aff.total_psv_magnitude
                
                # Stage 4: Goal tracking
                if self._goal_manager is not None:
                    ts = float(self._tick_count) * self.config.dt
                    social = aff.social_intent
                    self._goal_manager.update(
                        entity_id=entity_id,
                        position=position,
                        timestamp=ts,
                        social_intent=social,
                    )
                    goal_state = self._goal_manager.get_state(entity_id)
                    if goal_state is not None:
                        result["goal_states"][entity_id] = {
                            "phase": goal_state.phase.name,
                            "description": goal_state.estimated_goal_desc,
                            "social_intent": goal_state.social_intent,
                        }
                
                # Stage 5: Communicative intent detection
                if self._comm_detector is not None and etype == EntityType.HUMAN:
                    from .communicative_intent import BehavioralSignals
                    signals = BehavioralSignals(
                        distance=properties.distance if properties else 1.0,
                        speech_level=0.3 if social > 0.3 else 0.0,
                        facing_toward_me=social > 0.4,
                        gesture_speed=0.0,
                    )
                    comm = self._comm_detector.analyze(
                        signals, entity_position=position)
                    if comm.is_directed_at_me:
                        result["comm_signals"].append({
                            "entity_id": entity_id,
                            "intent": comm.intent.name,
                            "confidence": comm.confidence,
                            "requires_response": comm.requires_response,
                        })
        
        # Apply accumulated PSV deltas to pillar state
        if result["affordances"] and self._intent_translator is not None:
            self._pillar_state = self._intent_translator.apply_deltas(
                self._pillar_state, result["affordances"])
        
        return result
    
    def _extract_observed_entities(self, responses, world_model) -> list[dict]:
        """Extract entity observations from agent responses and world model.
        
        Returns list of dicts with keys: id, type, position, properties
        """
        entities = []
        
        # From world model if available
        if world_model is not None and hasattr(world_model, 'get_tracked_agents'):
            try:
                tracked = world_model.get_tracked_agents()
                for agent_id, info in tracked.items():
                    entities.append({
                        "id": agent_id,
                        "type": info.get("type", "unknown"),
                        "position": np.asarray(info.get("position", np.zeros(3))),
                        "properties": info.get("properties"),
                    })
            except (AttributeError, TypeError):
                pass
        
        # From responses (agent ecology outputs)
        if responses:
            for i, resp in enumerate(responses):
                if hasattr(resp, 'agent_role'):
                    entities.append({
                        "id": 1000 + i,  # agent ecology IDs start at 1000
                        "type": "agent",
                        "position": np.zeros(3),
                        "properties": None,
                    })
        
        return entities
    
    def _curiosity_signal(self, affordance_info, world_model, memory_system):
        """Compute curiosity signal from world model prediction error.

        For each observed entity, measures how well the dynamics model
        predicts its movement. High prediction error = model doesn't
        understand this region = curiosity/exploration signal.
        
        Returns dict with per-entity prediction errors and aggregate metrics.
        """
        result = {
            "entity_errors": {},
            "mean_error": 0.0,
            "max_error": 0.0,
            "high_uncertainty_regions": 0,
        }
        
        if world_model is None or not hasattr(world_model, 'memory'):
            return result
        
        dm = world_model.memory
        if not hasattr(dm, 'predict_velocity') or not hasattr(dm, '_fitted'):
            return result
        if not dm._fitted:
            return result
        
        errors = []
        for entity_id, goal_state in affordance_info.get("goal_states", {}).items():
            # Get the entity's position from goal tracker
            if self._goal_manager is not None:
                state = self._goal_manager.get_state(entity_id)
                if state is not None and state.position is not None:
                    pos = state.position
                    # Get predicted velocity
                    vel_dir = state.velocity.direction
                    vel_speed = state.velocity.speed
                    actual_vel = vel_dir * vel_speed
                    
                    # Compute prediction error
                    error = dm.prediction_error(pos, actual_vel)
                    errors.append(error)
                    result["entity_errors"][entity_id] = error
                    
                    # Also check region uncertainty
                    uncertainty = dm.region_uncertainty(pos, n_samples=10)
                    if uncertainty > 0.01:
                        result["high_uncertainty_regions"] += 1
        
        if errors:
            result["mean_error"] = float(np.mean(errors))
            result["max_error"] = float(np.max(errors))
        
        return result
    
    # ── Step 4a: Reactive Action Selection ────────────────────────
    
    def select_action_reactive(self, responses: list,
                               agent_ecology=None) -> dict:
        """Select action from agent responses (reactive mode).
        
        Uses weighted consensus to select the best action.
        """
        if not responses:
            return {"action": "none", "confidence": 0.0, "mode": "reactive"}
        
        # Get consensus
        if agent_ecology:
            consensus = agent_ecology.get_consensus(responses)
        else:
            # Simple: highest confidence
            consensus = max(responses, key=lambda r: r.confidence)
        
        if consensus is None:
            return {"action": "none", "confidence": 0.0, "mode": "reactive"}
        
        action = {
            "action": getattr(consensus, 'proposed_action', 'none') or 'none',
            "confidence": consensus.confidence,
            "agent": consensus.agent_role.name if hasattr(consensus, 'agent_role') else "unknown",
            "reasoning": consensus.reasoning if hasattr(consensus, 'reasoning') else "",
            "mode": "reactive",
        }
        
        self._action_history.append(action)
        
        return action
    
    # ── Step 4b: Predictive Action Selection ──────────────────────
    
    def select_action_predictive(self, memory_system) -> dict:
        """Select action via planning stack (predictive mode).
        
        1. Generate intent from situation assessment
        2. Plan: simulate candidate futures, evaluate, select best
        3. Control: translate plan to feasible delta
        """
        if self._planner is None or self._pillar_state is None:
            return {"action": "none", "confidence": 0.0, "mode": "predictive_fallback"}
        
        # 1. Generate intent
        if self._intent_generator is not None:
            intent = self._intent_generator.generate_intent(self._pillar_state)
        else:
            from .intent import Intent, IntentProposal
            intent = IntentProposal(
                intent=Intent.EXPLORE,
                priority=0.5,
                confidence=0.5,
                reasoning="default intent (no intent generator)",
            )
        
        self._current_intent = intent
        
        # 2. Plan
        try:
            plan = self._planner.plan(self._pillar_state, intent)
        except Exception:
            return {"action": "none", "confidence": 0.0, "mode": "predictive_error"}
        
        if not plan.actions:
            return {"action": "none", "confidence": 0.0, "mode": "predictive_no_plan"}
        
        # 3. Get the best action's delta as the action dict
        best_action = plan.actions[0]
        
        action = {
            "action": "plan",
            "delta": best_action.delta.tolist(),
            "description": best_action.description,
            "confidence": plan.confidence,
            "intent": intent.intent.name,
            "intent_priority": intent.priority,
            "plan_utility": plan.total_utility,
            "mode": "predictive",
        }
        
        self._action_history.append(action)
        
        return action
    
    # ── Step 4: Action Selection (dispatch) ───────────────────────
    
    def select_action(self, responses: list,
                      agent_ecology=None,
                      memory_system=None) -> dict:
        """Select action using the configured mode.
        
        If use_planner is True and planner is set, uses predictive mode.
        Otherwise falls back to reactive mode.
        """
        if self.config.use_planner and self._planner is not None:
            return self.select_action_predictive(memory_system)
        return self.select_action_reactive(responses, agent_ecology)
    
    # ── Step 5: Learning ──────────────────────────────────────────
    
    def learn(self, memory_system,
              experience: Optional[np.ndarray] = None,
              action_result: Optional[dict] = None) -> None:
        """Create experience and update memory.
        
        Encodes the current state-action-outcome as a new experience
        and stores it in attractor memory.
        """
        if memory_system is None:
            return
        
        # Create experience from current state
        if experience is None and self._pillar_state is not None:
            experience = self._pillar_state.copy()
        
        if experience is None:
            return
        
        # Store experience in memory
        try:
            # Try AttractorMemory's encode method
            memory_system.encode(experience)
        except (AttributeError, TypeError):
            # Fallback: just record the experience
            self._recent_experiences.append({
                "state": experience.copy(),
                "action": action_result,
                "time": time.time(),
            })
            
            # Keep buffer bounded
            if len(self._recent_experiences) > 100:
                self._recent_experiences.pop(0)
    
    # ── Full Tick ─────────────────────────────────────────────────
    
    def tick(self, field_evolver,
             memory_system,
             agent_ecology,
             world_model=None) -> dict:
        """Run one complete cognitive cycle.
        
        Returns dict with:
        - field_state: current field
        - pillar_state: current pillars
        - action: selected action (with mode: reactive or predictive)
        - stats: loop statistics
        """
        # Check energy
        if self._energy < self.config.cognitive_energy_cost:
            return {
                "field_state": self._field_state,
                "pillar_state": self._pillar_state,
                "action": {"action": "none", "confidence": 0.0},
                "stats": {"energy_depleted": True},
            }
        
        # 1. Recall memories
        recalled = self.recall_memories(memory_system, self._pillar_state)
        memory_influence = self.compute_memory_influence(recalled, self._field_state)
        
        # 2. Evolve field
        self.evolve_field(field_evolver, memory_influence,
                         steps=self.config.max_steps_per_tick)
        
        # 3. Update pillars from field (via projection)
        self._pillar_state = self._project_to_pillars(self._field_state)
        
        # 4. Perceive (always, for both modes — reactive needs responses, predictive uses for intent)
        responses = self.perceive(agent_ecology, world_model, memory_system)
        
        # 4b. Developmental perception: translate entities → affordances → PSV deltas
        affordance_info = self._developmental_perception(
            responses, world_model, memory_system)
        
        # 4c. Curiosity signal: prediction error for observed entities
        curiosity_info = self._curiosity_signal(
            affordance_info, world_model, memory_system)
        
        # 5. Act (mode-dependent)
        action = self.select_action(responses, agent_ecology, memory_system)
        
        # 5b. Apply planned delta to pillar state (predictive mode)
        if action.get("mode") == "predictive" and "delta" in action:
            delta = np.array(action["delta"], dtype=np.float64)
            self._pillar_state = np.clip(self._pillar_state + delta, 0.0, 1.0)
            self._field_state = self._pillar_state.copy()
        
        # 6. Learn
        self.learn(memory_system, action_result=action)
        
        # Consume energy
        self._energy -= self.config.cognitive_energy_cost
        
        self._tick_count += 1
        
        return {
            "field_state": self._field_state,
            "pillar_state": self._pillar_state,
            "action": action,
            "recalled_memories": len(recalled),
            "active_agents": len(responses),
            "intent": self._current_intent.intent.name if self._current_intent else None,
            "developmental": affordance_info,
            "curiosity": curiosity_info,
            "stats": self.stats(),
        }
    
    def _project_to_pillars(self, field_state: np.ndarray) -> np.ndarray:
        """Project field state to pillar representation."""
        if field_state is None:
            return np.zeros(16)
        
        # Simple projection: take first 16 components
        if len(field_state) >= 16:
            pillars = field_state[:16]
        else:
            pillars = np.zeros(16)
            pillars[:len(field_state)] = field_state
        
        # Normalize to [0, 1]
        min_val = np.min(pillars)
        max_val = np.max(pillars)
        if max_val - min_val > 1e-10:
            pillars = (pillars - min_val) / (max_val - min_val)
        else:
            pillars = np.full(16, 0.5)
        
        return pillars
    
    # ── Energy Management ─────────────────────────────────────────
    
    def recharge(self, amount: float = 0.1) -> None:
        """Add energy to the cognitive pool."""
        self._energy = min(1.0, self._energy + amount)
    
    # ── Statistics ─────────────────────────────────────────────────
    
    def stats(self) -> dict:
        """Get cognitive loop statistics."""
        return {
            "ticks": self._tick_count,
            "field_steps": self._step_count,
            "energy": round(self._energy, 3),
            "recent_experiences": len(self._recent_experiences),
            "action_count": len(self._action_history),
            "last_action": self._action_history[-1] if self._action_history else None,
            "mode": "predictive" if (self.config.use_planner and self._planner) else "reactive",
            "has_planner": self._planner is not None,
            "has_intent_translator": self._intent_translator is not None,
            "has_goal_manager": self._goal_manager is not None,
            "has_comm_detector": self._comm_detector is not None,
        }
    
    def reset(self) -> None:
        """Reset the cognitive loop."""
        self._field_state = None
        self._pillar_state = None
        self._step_count = 0
        self._tick_count = 0
        self._energy = 1.0
        self._recent_experiences.clear()
        self._action_history.clear()
        self._current_intent = None
        # Note: developmental modules are NOT reset (persistent state)
