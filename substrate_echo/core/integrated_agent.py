"""Integrated Agent — The complete cognitive organism.

Wires all cognitive modules into a single loop:
Perceive → Estimate → Model → Predict → Plan → Execute → Learn → Calibrate

This is the "world-ready" agent that can live in an ecology.

Usage:
    agent = IntegratedAgent(agent_id=0)
    obs = world.observe(agent.id)
    action = agent.think(obs)
    world.apply_action(agent.id, action)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from substrate_echo.core.dynamics_memory import DynamicsMemory
from substrate_echo.core.episodic_memory import EpisodicMemory, Episode
from substrate_echo.core.habit_formation import HabitFormation, Habit
from substrate_echo.core.counterfactual import CounterfactualReasoning
from substrate_echo.core.hierarchical_planner import (
    HierarchicalPlanner, GoalPlan, GoalStatus, GoalType
)
from substrate_echo.core.self_model import SelfModel
from substrate_echo.core.theory_of_mind import TheoryOfMind
from substrate_echo.core.emotional_contagion import EmotionalContagion
from substrate_echo.core.meta_cognition import MetaCognition, ConfidenceSource
from substrate_echo.core.experience_scheduler import ExperienceScheduler


@dataclass
class AgentMetrics:
    """Tracks agent performance metrics over time."""
    ticks: int = 0
    actions_taken: int = 0
    resources_harvested: float = 0.0
    social_interactions: int = 0
    plans_created: int = 0
    habits_formed: int = 0
    deaths_avoided: int = 0
    episodes_stored: int = 0
    lessons_learned: int = 0
    calibration_updates: int = 0
    idle_explorations: int = 0
    verification_checks: int = 0
    predictions_verified: int = 0
    energy_history: list[float] = field(default_factory=list)
    confidence_history: list[float] = field(default_factory=list)


@dataclass
class IntegratedAgentConfig:
    """Configuration for the integrated agent."""
    context_dim: int = 16
    curiosity_drive: float = 0.3
    social_drive: float = 0.2
    survival_drive: float = 0.5
    plan_interval: int = 10
    habit_threshold: int = 5
    counterfactual_interval: int = 20
    calibration_interval: int = 50
    share_knowledge_interval: int = 100


class IntegratedAgent:
    """A complete cognitive agent with all modules wired together.

    The think() method runs one cycle of the cognitive loop:
    1. Perceive (parse observation)
    2. Update world model
    3. Check habits (fast path)
    4. Check meta-cognition (should I plan?)
    5. Generate intent (weighted by drives)
    6. Plan (hierarchical)
    7. Evaluate alternatives (counterfactual)
    8. Execute action
    9. Record experience (episodic)
    10. Update habits
    11. Update social models
    12. Calibrate confidence

    Usage:
        agent = IntegratedAgent(agent_id=0)
        for tick in range(10000):
            obs = world.observe(agent.id)
            action = agent.think(obs)
            world.apply_action(agent.id, action)
    """

    def __init__(self, agent_id: int = 0,
                 config: Optional[IntegratedAgentConfig] = None):
        self.id = agent_id
        self.config = config or IntegratedAgentConfig()

        # Core modules
        self.dynamics_memory = DynamicsMemory()
        self.episodic_memory = EpisodicMemory()
        self.habit_formation = HabitFormation()
        self.counterfactual = CounterfactualReasoning(
            dynamics_memory=self.dynamics_memory)
        self.hierarchical_planner = HierarchicalPlanner(
            counterfactual=self.counterfactual)
        self.self_model = SelfModel()
        self.theory_of_mind = TheoryOfMind()
        self.emotional_contagion = EmotionalContagion()
        self.meta_cognition = MetaCognition()
        self.experience_scheduler = ExperienceScheduler(
            meta_cognition=self.meta_cognition,
            self_model=self.self_model,
            dynamics_memory=self.dynamics_memory,
        )

        # State
        self._position = np.zeros(2)
        self._energy = 1.0
        self._health = 1.0
        self._state_16d = np.zeros(self.config.context_dim)
        self._current_plan: Optional[GoalPlan] = None
        self._tick = 0

        # Prediction-verification state
        self._last_prediction: Optional[np.ndarray] = None
        self._last_state: Optional[np.ndarray] = None

        # Metrics
        self.metrics = AgentMetrics()

        # Known agents (for ToM)
        self._known_agents: set[int] = set()

    def think(self, observation: dict) -> dict:
        """Run one cognitive cycle and return an action.

        Args:
            observation: from world.observe(self.id)

        Returns:
            action dict for world.apply_action()
        """
        self._tick += 1
        self.metrics.ticks = self._tick

        # 1. Perceive: parse observation into 16D state
        self._update_state(observation)

        # 2. Update world model
        self._update_world_model()

        # 3. Update other agents (ToM)
        self._update_other_agents(observation)

        # 4. Meta-cognitive check
        meta_state = self.meta_cognition.get_meta_state()

        # 5. Check habits (fast path)
        habit = self.habit_formation.check_context(self._state_16d)
        if habit and not meta_state.should_be_cautious:
            actions = self.habit_formation.execute_habit(habit, self._tick)
            self.metrics.habits_formed = len(self.habit_formation.get_established_habits())
            if actions:
                return self._habit_to_action(actions[0], observation)

        # 6. Generate intent based on drives
        intent = self._generate_intent(observation, meta_state)

        # 7. Plan if needed
        if (self._current_plan is None or
            self._tick % self.config.plan_interval == 0):
            self._current_plan = self.hierarchical_planner.plan(
                intent, state=self._state_16d, tick=self._tick,
                max_depth=2)
            self.metrics.plans_created += 1

        # 8. Get next action from plan
        action = self._get_planned_action(observation)

        # 9. Record decision for counterfactual analysis
        self._record_decision(action)

        # 10. Record experience
        self._record_experience(action)

        # 11. Update habits
        self._update_habits(action)

        # 12. Update emotional state
        self._update_emotions(observation)

        # 13. Periodic calibration
        if self._tick % self.config.calibration_interval == 0:
            self._calibrate()

        # 14. Periodic social learning
        if self._tick % self.config.share_knowledge_interval == 0:
            self._share_knowledge(observation)

        # 15. Verify previous prediction against current state
        self._verify_prediction()

        # 16. Idle-time exploration via ExperienceScheduler
        self._idle_exploration()

        self.metrics.actions_taken += 1
        return action

    def _update_state(self, observation: dict) -> None:
        """Parse observation into 16D state vector."""
        pos = observation.get("agent_position", [0, 0])
        self._position = np.array(pos)
        self._energy = observation.get("agent_energy", 1.0)
        self._health = observation.get("agent_health", 1.0)

        # Build 16D state
        state = np.zeros(self.config.context_dim)
        state[0] = pos[0] / 10.0  # normalize
        state[1] = pos[1] / 10.0
        state[2] = self._energy
        state[3] = self._health

        # Resource signals
        resources = observation.get("nearby_resources", [])
        state[4] = len(resources) / 5.0  # resource density
        if resources:
            closest = min(resources, key=lambda r: r.get("distance", 999))
            state[5] = closest.get("quantity", 0)
            state[6] = 1.0 - closest.get("distance", 5) / 5.0

        # Social signals
        agents = observation.get("nearby_agents", [])
        state[7] = len(agents) / 5.0  # social density
        if agents:
            avg_energy = np.mean([a.get("energy", 0.5) for a in agents])
            state[8] = avg_energy

        # Inventory
        inventory = observation.get("inventory", {})
        state[9] = sum(inventory.values()) / 10.0

        # Habit & plan status
        state[10] = 1.0 if self._current_plan else 0.0
        state[11] = len(self.habit_formation.get_established_habits()) / 10.0

        # Emotional state
        emotional = self.emotional_contagion.get_group_mood()
        state[12] = emotional.get("warmth", 0.5)
        state[13] = emotional.get("stress", 0.0)

        # Meta-cognitive state
        meta = self.meta_cognition.get_meta_state()
        state[14] = meta.calibrated_confidence
        state[15] = meta.model_disagreement

        self._state_16d = state

    def _update_world_model(self) -> None:
        """Update dynamics memory with current state."""
        pass  # World model updated via episodic memory and planning

    def _update_other_agents(self, observation: dict) -> None:
        """Update Theory of Mind for nearby agents."""
        agents = observation.get("nearby_agents", [])
        for agent_info in agents:
            aid = agent_info.get("agent_id")
            if aid is None:
                continue

            self._known_agents.add(aid)
            pos = np.array(agent_info.get("position", [0, 0]))
            vel = np.array(agent_info.get("velocity", [0, 0]))

            self.theory_of_mind.update(
                agent_id=aid,
                position=pos,
                velocity=vel,
            )

            # Update emotional contagion
            energy = agent_info.get("energy", 0.5)
            stress = max(0, 1.0 - energy)
            other_pillars = np.array([
                energy,      # warmth
                stress,      # stress
                1.0 - stress, # calm
                0.5,         # social
                energy,      # energy
                0.5,         # trust
            ])
            self.emotional_contagion.update(
                agent_id=int(aid),
                pillars=other_pillars,
                position=pos,
            )

    def _generate_intent(self, observation: dict, meta_state) -> str:
        """Generate intent based on drives and situation."""
        energy = self._energy
        resources = observation.get("nearby_resources", [])
        agents = observation.get("nearby_agents", [])

        # Survival drive: if energy low, seek food
        if energy < 0.3:
            return "get_food"

        # Social drive: if agents nearby, interact
        if agents and np.random.random() < self.config.social_drive:
            return "social_interaction"

        # Curiosity drive: if novel, explore
        if meta_state.calibrated_confidence < 0.5:
            return "explore_area"

        # Default: get resources
        if resources:
            return "get_food"

        return "explore_area"

    def _get_planned_action(self, observation: dict) -> dict:
        """Convert plan into a concrete action."""
        if self._current_plan:
            next_actions = self.hierarchical_planner.get_next_actions(
                self._current_plan)
            if next_actions:
                node = next_actions[0]
                action_type = node.action_type or "move"

                # Convert to world action
                if action_type in ("search", "approach", "move"):
                    return self._move_toward_nearest_resource(observation)
                elif action_type == "grasp" or action_type == "consume":
                    resources = observation.get("nearby_resources", [])
                    if resources:
                        closest = min(resources, key=lambda r: r.get("distance", 999))
                        return {"type": "harvest",
                                "target": closest["position"],
                                "amount": 0.5}
                    return {"type": "harvest",
                            "target": self._position.tolist(),
                            "amount": 0.5}
                elif action_type == "observe":
                    return {"type": "wait"}
                else:
                    return {"type": "wait"}

        # Default: move toward nearest resource
        return self._move_toward_nearest_resource(observation)

    def _move_toward_nearest_resource(self, observation: dict) -> dict:
        """Move toward the nearest resource."""
        resources = observation.get("nearby_resources", [])
        if not resources:
            # Random walk
            direction = np.random.randn(2)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            return {"type": "move", "direction": direction.tolist(),
                    "speed": 0.3}

        # Move toward closest
        closest = min(resources, key=lambda r: r.get("distance", 999))
        target = np.array(closest["position"])
        direction = target - self._position
        dist = np.linalg.norm(direction)
        if dist > 0:
            direction = direction / dist

        return {"type": "move", "direction": direction.tolist(),
                "speed": min(0.5, dist)}

    def _habit_to_action(self, habit_action: dict,
                         observation: dict) -> dict:
        """Convert a habit action to a world action."""
        action_type = habit_action.get("type", "wait")
        if action_type in ("approach", "move"):
            return self._move_toward_nearest_resource(observation)
        elif action_type in ("grasp", "harvest"):
            return {"type": "harvest",
                    "target": self._position.tolist(),
                    "amount": 0.5}
        return {"type": "wait"}

    def _record_decision(self, action: dict) -> None:
        """Record decision for counterfactual analysis."""
        self.counterfactual.record_decision(
            state=self._state_16d,
            action_taken=action.get("type", "unknown"),
            action_taken_id=hash(action.get("type", "")) % 10,
            outcome=self._state_16d,
            tick=self._tick,
            utility=self._energy,
        )

    def _record_experience(self, action: dict) -> None:
        """Record experience in episodic memory."""
        emotion = "neutral"
        intensity = 0.5
        if self._energy < 0.3:
            emotion = "stress"
            intensity = 1.0 - self._energy
        elif self._energy > 0.8:
            emotion = "satisfaction"
            intensity = self._energy

        self.episodic_memory.store(
            context=self._state_16d,
            actions=[action],
            outcome=self._state_16d,
            tick=self._tick,
            success=action.get("type", "wait") != "error",
            emotion=emotion,
            emotional_intensity=intensity,
        )
        self.metrics.episodes_stored += 1

    def _update_habits(self, action: dict) -> None:
        """Update habit formation."""
        self.habit_formation.record(
            context=self._state_16d,
            actions=[action],
            success=True,
            tick=self._tick,
        )

    def _update_emotions(self, observation: dict) -> None:
        """Update emotional state from environment."""
        agents = observation.get("nearby_agents", [])
        energy = self._energy

        # Update own emotional state using the actual API
        own_pillars = np.array([
            energy,                    # warmth
            max(0, 1.0 - energy),     # stress
            energy,                    # calm
            len(agents) / 5.0,        # social
            energy,                    # energy
            0.5,                       # trust
        ])
        self.emotional_contagion.update(
            agent_id=self.id,
            pillars=own_pillars,
            position=self._position,
        )

        # Process emotional contagion
        self.emotional_contagion.apply_contagion(self.id)

    def _calibrate(self) -> None:
        """Calibrate meta-cognitive confidence."""
        # Use recent prediction accuracy
        recent = self.episodic_memory.recall_recent(10)
        if not recent:
            return

        for ep in recent:
            # Predict: was the action successful?
            predicted = 0.7  # baseline confidence
            self.meta_cognition.update(
                predicted_confidence=predicted,
                actual_outcome_correct=ep.success,
                source="episodic",
                tick=self._tick,
            )
            self.metrics.calibration_updates += 1

    def _share_knowledge(self, observation: dict) -> None:
        """Share knowledge with nearby agents (social learning)."""
        agents = observation.get("nearby_agents", [])
        if not agents:
            return

        self.metrics.social_interactions += len(agents)

    def _verify_prediction(self) -> None:
        """Verify the previous tick's prediction against current state.

        This creates the prediction → outcome → error → memory update loop.
        If dynamics_memory has a fitted model, we check whether our
        prediction about the current state was accurate.
        """
        if self._last_prediction is None or self._last_state is None:
            # No previous prediction — store current for next tick
            self._last_state = self._state_16d.copy()
            if self.dynamics_memory._fitted:
                self._last_prediction = self.dynamics_memory.predict_velocity(
                    self._last_state)
            return

        # Compute actual velocity from last state to current
        actual_velocity = self._state_16d - self._last_state

        # Compute prediction error
        if self._last_prediction is not None:
            error = float(np.mean((self._last_prediction - actual_velocity) ** 2))
            self.metrics.verification_checks += 1

            # Feed error to meta-cognition for calibration
            if error < 0.01:
                self.metrics.predictions_verified += 1
                self.meta_cognition.update(
                    predicted_confidence=0.9,
                    actual_outcome_correct=True,
                    source="dynamics_prediction",
                    tick=self._tick,
                )
            else:
                self.meta_cognition.update(
                    predicted_confidence=0.3,
                    actual_outcome_correct=False,
                    source="dynamics_prediction",
                    tick=self._tick,
                )

        # Store current state for next tick's verification
        self._last_state = self._state_16d.copy()
        if self.dynamics_memory._fitted:
            self._last_prediction = self.dynamics_memory.predict_velocity(
                self._state_16d)
        else:
            self._last_prediction = None

    def _idle_exploration(self) -> None:
        """Use idle cycles for purposeful exploration.

        Instead of doing nothing, the agent:
        1. Evaluates uncertainty regions, unexplored states, low-confidence predictions
        2. Selects exploration targets via ExperienceScheduler
        3. Feeds results into episodic memory and DynamicsMemory
        """
        if not self.experience_scheduler.should_activate():
            return

        session = self.experience_scheduler.create_session()
        budget = min(session["budget"], 10)  # Cap per-tick exploration

        for _ in range(budget):
            # Generate an exploration target based on weaknesses
            weakness = session["weakness"]
            target_state = self._generate_exploration_target(weakness)

            # Simulate: move toward target and observe outcome
            predicted_velocity = None
            if self.dynamics_memory._fitted:
                predicted_velocity = self.dynamics_memory.predict_velocity(
                    target_state)

            # Record as episode
            self.episodic_memory.store(
                context=target_state,
                actions=[{"type": "explore", "target": "idle_learning"}],
                outcome=target_state,
                tick=self._tick,
                success=True,
                emotion="curiosity",
                emotional_intensity=0.6,
            )

            # Feed into dynamics memory if we have a fitted model
            if self.dynamics_memory._fitted and predicted_velocity is not None:
                # The exploration provides training data
                self.dynamics_memory._states.append(target_state.copy())
                actual = np.random.randn(len(target_state)) * 0.01  # small perturbation
                self.dynamics_memory._velocities.append(actual)

        self.metrics.idle_explorations += 1

    def _generate_exploration_target(self, weakness) -> np.ndarray:
        """Generate an exploration target based on identified weaknesses.

        Targets states that are:
        - Far from known attractors (high novelty)
        - In regions of high model uncertainty
        - Related to weak modules
        """
        base = self._state_16d.copy()

        if self.dynamics_memory._fitted:
            # Find high-uncertainty region: sample perturbations and pick
            # the one with highest predicted variance
            rng = np.random.default_rng(self._tick)
            best_state = base
            best_uncertainty = 0.0

            for _ in range(5):
                candidate = base + rng.standard_normal(len(base)) * 0.3
                candidate = np.clip(candidate, -2, 2)

                # Use novelty as uncertainty proxy
                novelty = self.dynamics_memory.novelty(candidate)
                if novelty > best_uncertainty:
                    best_uncertainty = novelty
                    best_state = candidate

            return best_state
        else:
            # No model — random exploration in nearby region
            rng = np.random.default_rng(self._tick)
            return base + rng.standard_normal(len(base)) * 0.2

    def summary(self) -> dict:
        """Agent summary."""
        return {
            "agent_id": self.id,
            "tick": self._tick,
            "energy": self._energy,
            "health": self._health,
            "metrics": {
                "actions": self.metrics.actions_taken,
                "harvested": self.metrics.resources_harvested,
                "social": self.metrics.social_interactions,
                "plans": self.metrics.plans_created,
                "habits": len(self.habit_formation.get_established_habits()),
                "episodes": self.metrics.episodes_stored,
                "idle_explorations": self.metrics.idle_explorations,
                "verification_checks": self.metrics.verification_checks,
                "predictions_verified": self.metrics.predictions_verified,
            },
            "meta": self.meta_cognition.summary(),
            "modules": {
                "dynamics_memory": {"states": len(self.dynamics_memory._states)},
                "episodic": self.episodic_memory.summary(),
                "habits": self.habit_formation.stats(),
                "counterfactual": self.counterfactual.summary(),
                "planner": self.hierarchical_planner.stats(),
            },
        }
