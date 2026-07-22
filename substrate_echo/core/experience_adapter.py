"""Experience Adapter — Generic environment interface.

Converts any external environment into a learning substrate.
The agent should not know "this came from Minecraft" — it should
learn "this entity has affordances" and "this action has consequences."

Every environment passes through this adapter into universal
cognitive representation (PSV, HSV, BSV, ESV, DynamicsMemory).

Usage:
    adapter = ExperienceAdapter(env_name="minecraft")

    observation = adapter.observe(env_state)
    action_result = adapter.act(action)
    reward = adapter.reward_signal()
    experiences = adapter.export_experience()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np
from collections import defaultdict


@dataclass
class PerceptionFrame:
    """A single frame of perception from an environment."""
    tick: int
    state_vector: np.ndarray      # 16D universal representation
    entities: list[dict]          # perceived entities with properties
    spatial_map: Optional[np.ndarray] = None  # local spatial occupancy
    social_signals: list[dict] = field(default_factory=list)
    resource_signals: list[dict] = field(default_factory=list)
    raw_observation: Any = None    # environment-specific (for debugging)


@dataclass
class ActionFrame:
    """A cognitive action translated for the environment."""
    action_type: str              # universal: move, grasp, observe, etc.
    parameters: dict              # environment-specific parameters
    magnitude: float = 0.1
    duration: int = 1


@dataclass
class RewardSignal:
    """Outcome information from an action."""
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)
    success: bool = True
    state_change: Optional[np.ndarray] = None


@dataclass
class ExperienceRecord:
    """A complete experience for memory storage."""
    tick: int
    state_before: np.ndarray
    action: dict
    state_after: np.ndarray
    reward: float
    success: bool
    entities_observed: list[dict]
    context_hash: str = ""


@dataclass
class ExperienceAdapterConfig:
    """Configuration for the experience adapter."""
    state_dim: int = 16
    max_entity_count: int = 20
    history_size: int = 1000
    export_batch_size: int = 50


class ExperienceAdapter:
    """Generic adapter that converts environment interaction into
    universal cognitive experience.

    The agent interacts through this adapter. It translates:
    - Environment state → PerceptionFrame (universal)
    - Cognitive action → ActionFrame (environment-specific)
    - Environment feedback → RewardSignal (universal)
    - Interaction history → ExperienceRecords (for memory)

    No environment-specific knowledge leaks into cognition.

    Usage:
        adapter = ExperienceAdapter(env_name="minecraft")

        # Each tick
        frame = adapter.observe(env_state)
        result = adapter.act(cognitive_action)
        reward = adapter.reward_signal()

        # When ready to consolidate
        experiences = adapter.export_experience()
        for exp in experiences:
            episodic_memory.store(...)
            dynamics_memory.encode(...)
    """

    def __init__(self, env_name: str = "generic",
                 config: Optional[ExperienceAdapterConfig] = None):
        self.env_name = env_name
        self.config = config or ExperienceAdapterConfig()

        self._history: list[ExperienceRecord] = []
        self._tick = 0
        self._last_state: Optional[np.ndarray] = None
        self._last_action: Optional[dict] = None

        # Entity tracking
        self._known_entities: dict[str, dict] = {}

    def observe(self, env_state: Any,
                entities: Optional[list[dict]] = None,
                resources: Optional[list[dict]] = None,
                agents: Optional[list[dict]] = None) -> PerceptionFrame:
        """Convert environment state into universal perception.

        Args:
            env_state: raw environment state (any format)
            entities: list of entity dicts with position, type, properties
            resources: list of resource dicts
            agents: list of other agent dicts

        Returns:
            PerceptionFrame in universal representation
        """
        state_vector = self._encode_state(env_state, entities, resources, agents)

        entity_list = entities or []
        social = agents or []
        resource_sig = resources or []

        frame = PerceptionFrame(
            tick=self._tick,
            state_vector=state_vector,
            entities=entity_list,
            social_signals=social,
            resource_signals=resource_sig,
            raw_observation=env_state,
        )

        self._last_state = state_vector
        self._tick += 1

        return frame

    def act(self, action: dict) -> ActionFrame:
        """Translate a cognitive action into environment-specific form.

        Args:
            action: universal action dict with type, target, magnitude

        Returns:
            ActionFrame ready for environment execution
        """
        action_type = action.get("type", "wait")

        return ActionFrame(
            action_type=action_type,
            parameters=action,
            magnitude=action.get("magnitude", 0.1),
        )

    def record_outcome(self, action: dict, reward: float,
                       done: bool = False,
                       next_state: Optional[Any] = None,
                       success: bool = True,
                       entities: Optional[list[dict]] = None) -> ExperienceRecord:
        """Record an action-outcome pair for experience export.

        Args:
            action: the action taken
            reward: reward signal
            done: whether episode ended
            next_state: resulting environment state
            success: whether action succeeded
            entities: entities observed after action

        Returns:
            ExperienceRecord
        """
        state_after = self._encode_state(next_state, entities) if next_state is not None else (
            self._last_state.copy() if self._last_state is not None else np.zeros(self.config.state_dim))

        record = ExperienceRecord(
            tick=self._tick - 1,
            state_before=self._last_state.copy() if self._last_state is not None else np.zeros(self.config.state_dim),
            action=action,
            state_after=state_after,
            reward=reward,
            success=success,
            entities_observed=entities or [],
        )

        self._history.append(record)
        self._last_action = action

        # Maintain history size
        if len(self._history) > self.config.history_size:
            self._history = self._history[-self.config.history_size // 2:]

        return record

    def export_experience(self, n: Optional[int] = None) -> list[ExperienceRecord]:
        """Export recent experiences for memory integration.

        Args:
            n: number of records to export (None = batch size)

        Returns:
            list of ExperienceRecords ready for memory systems
        """
        if n is None:
            n = self.config.export_batch_size

        batch = self._history[-n:]
        return batch

    def clear_exported(self, n: Optional[int] = None) -> None:
        """Clear exported experiences from history."""
        if n is None:
            n = self.config.export_batch_size
        self._history = self._history[:-n] if n < len(self._history) else []

    def get_information_gap(self) -> dict[str, float]:
        """Estimate information gaps based on experience patterns.

        Returns dict of domain → uncertainty score (0-1, higher = more uncertain).
        """
        if not self._history:
            return {"general": 1.0}

        # Analyze reward variance per action type
        action_rewards: dict[str, list[float]] = defaultdict(list)
        for record in self._history:
            atype = record.action.get("type", "unknown")
            action_rewards[atype].append(record.reward)

        gaps = {}
        for action_type, rewards in action_rewards.items():
            if len(rewards) < 3:
                gaps[action_type] = 1.0  # very uncertain
            else:
                variance = float(np.var(rewards))
                mean_abs = float(np.mean(np.abs(rewards))) + 1e-8
                gaps[action_type] = min(1.0, variance / mean_abs)

        return gaps

    def stats(self) -> dict:
        """Adapter statistics."""
        return {
            "env_name": self.env_name,
            "tick": self._tick,
            "history_size": len(self._history),
            "n_entities_known": len(self._known_entities),
        }

    def _encode_state(self, env_state: Any,
                      entities: Optional[list[dict]] = None,
                      resources: Optional[list[dict]] = None,
                      agents: Optional[list[dict]] = None) -> np.ndarray:
        """Encode arbitrary environment state into 16D vector."""
        state = np.zeros(self.config.state_dim)

        # If env_state is already a vector, use it directly
        if isinstance(env_state, np.ndarray) and env_state.shape == (self.config.state_dim,):
            return env_state

        # If env_state is a dict, extract numeric values
        if isinstance(env_state, dict):
            vals = [v for v in env_state.values() if isinstance(v, (int, float))]
            for i, v in enumerate(vals[:self.config.state_dim]):
                state[i] = float(v)
        elif isinstance(env_state, (int, float)):
            state[0] = float(env_state)

        # Entity signals
        if entities:
            state[4] = min(1.0, len(entities) / self.config.max_entity_count)
            if entities:
                e = entities[0]
                pos = e.get("position", [0, 0])
                if isinstance(pos, (list, np.ndarray)) and len(pos) >= 2:
                    state[0] = pos[0] / 10.0
                    state[1] = pos[1] / 10.0

        # Resource signals
        if resources:
            state[5] = min(1.0, len(resources) / 10.0)
            closest = min(resources, key=lambda r: r.get("distance", 999))
            state[6] = closest.get("quantity", 0)
            state[7] = 1.0 - min(1.0, closest.get("distance", 5) / 5.0)

        # Social signals
        if agents:
            state[8] = min(1.0, len(agents) / 10.0)

        return state
