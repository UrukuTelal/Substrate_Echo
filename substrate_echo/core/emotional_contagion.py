"""Emotional Contagion — P7.6

Agent's PSV is influenced by nearby agents' emotional states.

Core idea: emotions spread through proximity and similarity.
- High-warmth agents nearby increase own warmth
- High-stress agents nearby increase own stress
- Calm agents can dampen others' stress
- Emotional state affects decision-making and social behavior

This creates emergent social dynamics:
- Group calm: when several calm agents are together
- Panic spread: stress propagates through groups
- Emotional regulation: agents can learn to dampen negative emotions
- Social mood: overall emotional tone of a group

Architecture:
- Each agent has an emotional state (subset of PSV pillars)
- Emotional influence is weighted by proximity and relationship
- EMA smoothing prevents instantaneous emotional jumps
- Emotional state feeds back into PSV and affects behavior

Usage:
    ec = EmotionalContagion()
    ec.update(agent_id=1, pillars=current_psv, position=agent_pos)
    ec.update(agent_id=2, pillars=nearby_psv, position=nearby_pos)
    modified = ec.apply_contagion(agent_id=1)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# Emotional dimensions mapped to pillar indices
EMOTIONAL_DIMS = {
    "warmth": 9,        # Warmth pillar
    "stress": 12,       # Harm pillar (high = stressed)
    "calm": 4,          # Resistance pillar (high = resilient/calm)
    "social": 7,        # Relation pillar
    "energy": 0,        # Awareness pillar (high = alert/energized)
    "trust": 5,         # Integrity pillar
}


@dataclass
class EmotionalState:
    """Emotional state of an agent."""
    warmth: float = 0.5
    stress: float = 0.1
    calm: float = 0.5
    social: float = 0.5
    energy: float = 0.5
    trust: float = 0.5

    def to_array(self) -> np.ndarray:
        return np.array([self.warmth, self.stress, self.calm,
                         self.social, self.energy, self.trust])

    @classmethod
    def from_pillars(cls, pillars: np.ndarray) -> 'EmotionalState':
        """Extract emotional state from full PSV."""
        pillars = np.asarray(pillars, dtype=np.float64)
        return cls(
            warmth=pillars[9] if len(pillars) > 9 else 0.5,
            stress=pillars[12] / 100.0 if len(pillars) > 12 else 0.1,
            calm=pillars[4] / 100.0 if len(pillars) > 4 else 0.5,
            social=pillars[7] / 100.0 if len(pillars) > 7 else 0.5,
            energy=pillars[0] / 100.0 if len(pillars) > 0 else 0.5,
            trust=pillars[5] / 100.0 if len(pillars) > 5 else 0.5,
        )


@dataclass
class EmotionalInfluence:
    """How one agent's emotions influenced another."""
    source_id: int
    target_id: int
    dimension: str        # which emotion was transmitted
    magnitude: float      # how much was transmitted
    distance: float       # distance between agents


@dataclass
class EmotionalContagionConfig:
    """Configuration for emotional contagion."""
    influence_radius: float = 5.0      # max distance for emotional influence
    proximity_decay: float = 2.0       # exponential decay rate
    similarity_weight: float = 0.3     # how much emotional similarity amplifies influence
    smoothing: float = 0.2             # EMA smoothing factor
    min_influence: float = 0.01        # minimum influence to apply
    stress_dampening: float = 0.05     # how much calm agents reduce others' stress
    max_stress_boost: float = 0.3      # maximum stress increase per tick


class EmotionalContagion:
    """Models emotional influence between agents.

    Each agent's emotional state is influenced by nearby agents'
    emotional states, weighted by proximity and emotional similarity.

    This creates emergent group dynamics without explicit rules:
    - Groups of calm agents stay calm
    - One stressed agent can spread stress
    - Warm agents create warm environments
    - Emotional regulation emerges from proximity to calm agents

    Usage:
        ec = EmotionalContagion()

        # Each tick, update agent positions and pillars
        ec.update(agent_id=1, pillars=my_psv, position=my_pos)
        ec.update(agent_id=2, pillars=other_psv, position=other_pos)

        # Apply contagion and get modified PSV
        modified_psv = ec.apply_contagion(agent_id=1)
    """

    def __init__(self, config: Optional[EmotionalContagionConfig] = None):
        self.config = config or EmotionalContagionConfig()
        self._agents: dict[int, dict] = {}
        self._influence_history: list[EmotionalInfluence] = []

    def update(self, agent_id: int,
               pillars: np.ndarray,
               position: np.ndarray) -> EmotionalState:
        """Update agent's emotional state from their PSV."""
        pillars = np.asarray(pillars, dtype=np.float64)
        position = np.asarray(position, dtype=np.float64)

        if agent_id not in self._agents:
            self._agents[agent_id] = {
                "emotional_state": EmotionalState.from_pillars(pillars),
                "raw_pillars": pillars.copy(),
                "position": position.copy(),
            }
        else:
            self._agents[agent_id]["emotional_state"] = EmotionalState.from_pillars(pillars)
            self._agents[agent_id]["raw_pillars"] = pillars.copy()
            self._agents[agent_id]["position"] = position.copy()

        return self._agents[agent_id]["emotional_state"]

    def apply_contagion(self, agent_id: int) -> Optional[np.ndarray]:
        """Apply emotional contagion from nearby agents.

        Returns modified PSV with emotional influence applied.
        """
        if agent_id not in self._agents:
            return None

        target = self._agents[agent_id]
        target_emotion = target["emotional_state"]
        target_pos = target["position"]
        modified = target["raw_pillars"].copy()

        influences = []

        for other_id, other in self._agents.items():
            if other_id == agent_id:
                continue

            other_pos = other["position"]
            distance = float(np.linalg.norm(target_pos - other_pos))

            if distance > self.config.influence_radius or distance < 1e-6:
                continue

            other_emotion = other["emotional_state"]

            # Proximity weight (exponential decay)
            proximity_weight = np.exp(-distance / self.config.proximity_decay)

            # Emotional similarity amplifies influence
            similarity = 1.0 - np.mean(np.abs(
                target_emotion.to_array() - other_emotion.to_array()))
            similarity_weight = 1.0 + self.config.similarity_weight * similarity

            total_weight = proximity_weight * similarity_weight

            # Influence on each dimension
            for dim_name, pillar_idx in EMOTIONAL_DIMS.items():
                other_val = getattr(other_emotion, dim_name)
                target_val = getattr(target_emotion, dim_name)

                # Stress is special: calm agents dampen stress
                if dim_name == "stress" and other_val < target_val:
                    magnitude = -self.config.stress_dampening * total_weight
                else:
                    magnitude = (other_val - target_val) * total_weight * 0.1

                # Clamp stress boost
                if dim_name == "stress" and magnitude > 0:
                    magnitude = min(magnitude, self.config.max_stress_boost)

                if abs(magnitude) > self.config.min_influence:
                    # Apply to PSV
                    if pillar_idx < len(modified):
                        new_val = modified[pillar_idx] + magnitude * 100
                        modified[pillar_idx] = np.clip(new_val, 0, 100)

                    influences.append(EmotionalInfluence(
                        source_id=other_id,
                        target_id=agent_id,
                        dimension=dim_name,
                        magnitude=magnitude,
                        distance=distance,
                    ))

        # Update emotional state from modified PSV
        target["emotional_state"] = EmotionalState.from_pillars(modified)
        self._influence_history.extend(influences)

        return modified

    def get_group_mood(self) -> dict:
        """Compute aggregate emotional state of all agents."""
        if not self._agents:
            return {"n_agents": 0}

        emotions = [a["emotional_state"] for a in self._agents.values()]
        return {
            "n_agents": len(emotions),
            "avg_warmth": np.mean([e.warmth for e in emotions]),
            "avg_stress": np.mean([e.stress for e in emotions]),
            "avg_calm": np.mean([e.calm for e in emotions]),
            "avg_social": np.mean([e.social for e in emotions]),
            "avg_energy": np.mean([e.energy for e in emotions]),
            "avg_trust": np.mean([e.trust for e in emotions]),
            "stress_variance": np.var([e.stress for e in emotions]),
        }

    def get_influence_history(self) -> list[EmotionalInfluence]:
        """Get recent emotional influences."""
        return list(self._influence_history)

    def stats(self) -> dict:
        """Summary statistics."""
        return {
            "n_agents": len(self._agents),
            "n_influences": len(self._influence_history),
            "group_mood": self.get_group_mood(),
        }
