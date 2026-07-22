"""Theory of Mind (ToM) — P7.5

Model of other agents' beliefs, desires, and knowledge states.

Core capabilities:
1. Belief tracking: what does agent X believe about the world?
2. Desire inference: what does agent X want?
3. Knowledge state: what has agent X observed?
4. False belief detection: does agent X's belief differ from reality?
5. Predictive modeling: what will agent X do next?

Theory of Mind is the foundation for:
- Deception (intentionally creating false beliefs)
- Teaching (correcting false beliefs)
- Cooperative planning (aligning beliefs and goals)
- Social prediction (anticipating behavior from beliefs)

Architecture:
- Each tracked agent gets a BeliefState
- BeliefState contains: world_model (their private copy), desires, observations
- update() ingests new observations about the agent
- predict_action() uses their beliefs + desires to predict behavior
- detect_false_belief() compares their belief to ground truth

Usage:
    tom = TheoryOfMind()
    tom.update(agent_id=1, observation=np.array([...]),
               desire="find_food", position=np.array([5,5,0]))
    predicted = tom.predict_action(agent_id=1)
    false_belief = tom.detect_false_belief(agent_id=1, reality=np.array([...]))
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .dynamics_memory import DynamicsMemory


@dataclass
class BeliefState:
    """What a specific agent believes about the world."""
    agent_id: int

    # World model: what they believe about dynamics
    world_model: DynamicsMemory = field(default_factory=lambda: DynamicsMemory(dim=16))

    # Desires: what they want (inferred or known)
    desires: dict[str, float] = field(default_factory=dict)  # desire_type -> strength
    primary_desire: str = "unknown"

    # Observations: what they've seen (their information state)
    observations: list[np.ndarray] = field(default_factory=list)
    observation_count: int = 0
    last_observation: Optional[np.ndarray] = None

    # Belief confidence: how sure are they about their world model
    confidence: float = 0.0

    # Position and trajectory
    position: Optional[np.ndarray] = None
    trajectory: list[np.ndarray] = field(default_factory=list)

    # Inferred goal
    inferred_goal: Optional[np.ndarray] = None
    goal_confidence: float = 0.0


@dataclass
class FalseBelief:
    """Detected discrepancy between agent's belief and reality."""
    agent_id: int
    belief_state: np.ndarray     # what the agent believes
    reality_state: np.ndarray    # what's actually true
    discrepancy: float           # magnitude of difference
    belief_type: str             # "position", "dynamics", "goal"


@dataclass
class TheoryOfMindConfig:
    """Configuration for Theory of Mind."""
    max_observations: int = 200       # per agent
    belief_decay: float = 0.01        # confidence decay per tick without observation
    desire_smoothing: float = 0.3     # EMA for desire inference
    false_belief_threshold: float = 0.3  # minimum discrepancy to report
    min_observations_to_model: int = 20  # minimum before fitting dynamics


class TheoryOfMind:
    """Models other agents' beliefs, desires, and knowledge.

    Maintains a separate BeliefState for each tracked agent.
    Each BeliefState contains a private DynamicsMemory representing
    what that agent believes about the world — which may differ
    from reality and from our own world model.

    Usage:
        tom = TheoryOfMind()

        # Each tick, update with observations about each agent
        tom.update(agent_id=1, observation=current_psv,
                   desire="explore", position=agent_pos)

        # Predict what they'll do
        action = tom.predict_action(agent_id=1)

        # Check if they have false beliefs
        false = tom.detect_false_belief(agent_id=1, reality=actual_state)
    """

    def __init__(self, config: Optional[TheoryOfMindConfig] = None,
                 dim: int = 16):
        self.config = config or TheoryOfMindConfig()
        self.dim = dim
        self._beliefs: dict[int, BeliefState] = {}

    def update(self, agent_id: int,
               observation: Optional[np.ndarray] = None,
               desire: Optional[str] = None,
               position: Optional[np.ndarray] = None,
               velocity: Optional[np.ndarray] = None) -> BeliefState:
        """Update belief state for an agent.

        Args:
            agent_id: unique identifier
            observation: what we observe their state to be (PSV)
            desire: inferred desire type (explore, social, avoid, etc.)
            position: their current position
            velocity: their current velocity

        Returns:
            Updated BeliefState
        """
        if agent_id not in self._beliefs:
            self._beliefs[agent_id] = BeliefState(
                agent_id=agent_id,
                world_model=DynamicsMemory(dim=self.dim),
            )

        state = self._beliefs[agent_id]

        # Record observation
        if observation is not None:
            obs = np.asarray(observation, dtype=np.float64)
            state.observations.append(obs.copy())
            if len(state.observations) > self.config.max_observations:
                state.observations.pop(0)
            state.observation_count += 1
            state.last_observation = obs.copy()
            state.confidence = min(1.0, state.confidence + 0.05)

            # Add to their world model
            if state.position is not None and len(state.observations) >= 2:
                prev = state.observations[-2]
                vel = obs - prev
                state.world_model._states.append(prev.copy())
                state.world_model._velocities.append(vel.copy())
                if len(state.world_model._states) >= self.config.min_observations_to_model:
                    state.world_model._fit_dynamics()

        # Record desire
        if desire is not None:
            current = state.desires.get(desire, 0.5)
            state.desires[desire] = current * (1 - self.config.desire_smoothing) + \
                                     1.0 * self.config.desire_smoothing
            # Update primary desire
            if state.desires:
                state.primary_desire = max(state.desires, key=state.desires.get)

        # Record position
        if position is not None:
            pos = np.asarray(position, dtype=np.float64)
            state.position = pos.copy()
            state.trajectory.append(pos.copy())
            if len(state.trajectory) > 50:
                state.trajectory.pop(0)

        # Decay confidence over time
        state.confidence *= (1 - self.config.belief_decay)

        return state

    def get_belief(self, agent_id: int) -> Optional[BeliefState]:
        """Get current belief state for an agent."""
        return self._beliefs.get(agent_id)

    def predict_action(self, agent_id: int) -> Optional[dict]:
        """Predict what an agent will do next based on their beliefs + desires.

        Uses their private world model to simulate forward, then
        selects the action aligned with their strongest desire.
        """
        state = self._beliefs.get(agent_id)
        if state is None or state.position is None:
            return None

        # Predict where they'll go based on their world model
        if state.world_model._fitted and state.last_observation is not None:
            predicted = state.world_model.predict(
                state.last_observation, steps=5, dt=1.0)
        else:
            # Fall back to trajectory extrapolation
            if len(state.trajectory) >= 2:
                vel = state.trajectory[-1] - state.trajectory[-2]
                predicted = state.position + vel * 5
            else:
                predicted = state.position.copy()

        return {
            "agent_id": agent_id,
            "predicted_position": predicted,
            "primary_desire": state.primary_desire,
            "desire_strength": state.desires.get(state.primary_desire, 0.0),
            "confidence": state.confidence,
            "inferred_goal": state.inferred_goal,
            "goal_confidence": state.goal_confidence,
        }

    def detect_false_belief(self, agent_id: int,
                             reality: np.ndarray) -> Optional[FalseBelief]:
        """Detect if an agent's belief differs from reality.

        Compares the agent's last observed state (what they believe)
        to the actual current state (reality).
        """
        state = self._beliefs.get(agent_id)
        if state is None or state.last_observation is None:
            return None

        reality = np.asarray(reality, dtype=np.float64)
        belief = state.last_observation

        discrepancy = float(np.linalg.norm(belief - reality))

        if discrepancy < self.config.false_belief_threshold:
            return None

        return FalseBelief(
            agent_id=agent_id,
            belief_state=belief,
            reality_state=reality,
            discrepancy=discrepancy,
            belief_type="state",
        )

    def infer_desire(self, agent_id: int,
                     action_type: Optional[str] = None,
                     target: Optional[np.ndarray] = None) -> str:
        """Infer what an agent wants from their behavior.

        Uses trajectory analysis and observed actions to infer desires.
        """
        state = self._beliefs.get(agent_id)
        if state is None:
            return "unknown"

        desires = {}

        # From trajectory: moving toward something = approach desire
        if len(state.trajectory) >= 3:
            recent = np.array(state.trajectory[-3:])
            velocity = recent[-1] - recent[-2]
            speed = np.linalg.norm(velocity)
            if speed > 0.05:
                desires["explore"] = desires.get("explore", 0) + 0.3
                desires["approach"] = desires.get("approach", 0) + 0.2

        # From action type
        if action_type is not None:
            action_desires = {
                "approach": "social",
                "communicate": "social",
                "retreat": "avoid",
                "observe": "explore",
                "investigate": "explore",
                "grasp": "acquire",
                "defend": "protect",
            }
            mapped = action_desires.get(action_type, "unknown")
            if mapped != "unknown":
                desires[mapped] = desires.get(mapped, 0) + 0.5

        # From target proximity
        if target is not None and state.position is not None:
            dist = np.linalg.norm(state.position - target)
            if dist < 2.0:
                desires["approach"] = desires.get("approach", 0) + 0.4

        # Update desire strengths
        for d, s in desires.items():
            current = state.desires.get(d, 0.0)
            state.desires[d] = current * 0.7 + s * 0.3

        if state.desires:
            state.primary_desire = max(state.desires, key=state.desires.get)
            return state.primary_desire

        return "unknown"

    def predict_trajectory(self, agent_id: int,
                           steps: int = 10) -> list[np.ndarray]:
        """Predict an agent's future trajectory based on their beliefs."""
        state = self._beliefs.get(agent_id)
        if state is None or state.position is None:
            return []

        if state.world_model._fitted and state.last_observation is not None:
            trajectory = state.world_model.predict_trajectory(
                state.last_observation, steps=steps)
            return trajectory

        # Fallback: linear extrapolation
        if len(state.trajectory) >= 2:
            vel = state.trajectory[-1] - state.trajectory[-2]
            trajectory = [state.position.copy()]
            for _ in range(steps):
                trajectory.append(trajectory[-1] + vel)
            return trajectory

        return [state.position.copy()]

    def get_all_beliefs(self) -> dict[int, BeliefState]:
        """Get all tracked belief states."""
        return dict(self._beliefs)

    def stats(self) -> dict:
        """Summary statistics."""
        beliefs = list(self._beliefs.values())
        return {
            "n_agents": len(beliefs),
            "avg_confidence": np.mean([b.confidence for b in beliefs]) if beliefs else 0,
            "avg_observations": np.mean([b.observation_count for b in beliefs]) if beliefs else 0,
            "agents": {
                bid: {
                    "observations": b.observation_count,
                    "confidence": b.confidence,
                    "primary_desire": b.primary_desire,
                    "world_model_fitted": b.world_model._fitted,
                }
                for bid, b in self._beliefs.items()
            },
        }
