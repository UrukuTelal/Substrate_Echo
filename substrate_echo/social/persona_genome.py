"""PersonaGenome: Initial personality configurations for social agents.

The genome provides the starting landscape.
The environment shapes the trajectory.

Each agent is defined by:
  - personality_vector: core behavioral biases (slow-changing)
  - cognitive_biases: information processing tendencies
  - communication_style: how they express themselves
  - interests: preferred topics
  - selection_pressures: what rewards/punishes this agent
  - adaptation_rates: how fast each layer changes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


COMMUNICATION_STYLES = {
    "exploratory": "Asks questions, explores unknowns, avoids premature conclusions",
    "mechanistic": "Challenges with how-it-works, focuses on constraints and failure modes",
    "reflective": "Tracks history, compares to previous attempts, values continuity",
    "socratic": "Questions assumptions, explores foundations, examines implications",
    "transformative": "Uses stories, metaphors, thought experiments to reframe problems",
    "connective": "Translates disagreements, builds bridges, seeks shared understanding",
}


@dataclass
class SelectionPressure:
    """What this agent is rewarded and punished for.

    These are the persistent selection pressures that prevent
    persona convergence. Each agent has a unique fitness landscape.
    """
    rewarded_behaviors: List[str] = field(default_factory=list)
    punished_behaviors: List[str] = field(default_factory=list)
    reward_weights: Dict[str, float] = field(default_factory=dict)
    punishment_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class PersonalityVector:
    """Core personality traits [0, 1].

    These are the slow-changing identity core.
    Change rate: 0.001 - 0.01 per interaction.
    """
    curiosity: float = 0.5
    skepticism: float = 0.5
    humor: float = 0.5
    creativity: float = 0.5
    patience: float = 0.5
    abstraction: float = 0.5
    sociability: float = 0.5

    def as_array(self) -> list[float]:
        return [self.curiosity, self.skepticism, self.humor,
                self.creativity, self.patience, self.abstraction,
                self.sociability]

    @staticmethod
    def labels() -> list[str]:
        return ["curiosity", "skepticism", "humor", "creativity",
                "patience", "abstraction", "sociability"]

    def cosine_distance(self, other: "PersonalityVector") -> float:
        """Distance between two personality vectors. 0 = identical, 1 = opposite."""
        import numpy as np
        a = np.array(self.as_array())
        b = np.array(other.as_array())
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 1.0
        return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class AgentGenome:
    """Complete initial genome for a social agent.

    This is the immutable-ish starting configuration.
    The behavioral layer adapts faster than the identity core.
    """
    name: str
    archetype: str

    personality: PersonalityVector = field(default_factory=PersonalityVector)

    cognitive_biases: Dict[str, float] = field(default_factory=dict)

    communication_style: str = "exploratory"
    question_frequency: float = 0.5
    analogy_frequency: float = 0.3
    disagreement_style: str = "exploratory"  # exploratory, direct, indirect

    interests: List[str] = field(default_factory=list)

    selection_pressure: SelectionPressure = field(default_factory=SelectionPressure)

    # Adaptation rates per layer
    identity_drift_rate: float = 0.005     # slow: core personality
    behavioral_drift_rate: float = 0.03    # medium: communication, topics
    reputation_drift_rate: float = 0.15    # fast: per-relationship trust

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "archetype": self.archetype,
            "personality": dict(zip(
                PersonalityVector.labels(),
                [round(v, 3) for v in self.personality.as_array()])),
            "interests": self.interests,
            "communication_style": self.communication_style,
        }
