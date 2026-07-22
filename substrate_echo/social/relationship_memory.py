"""RelationshipMemory: Per-agent relationship model.

Social identity emerges from relationships.
An agent should not behave identically with:
  - a trusted collaborator
  - a rival
  - a novice
  - a consistently incorrect agent
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RelationshipRecord:
    """Detailed relationship state with another agent.

    Attributes
    ----------
    other_agent_id : str
    interaction_count : int
    shared_topics : dict[str, int]
        Topic distribution of interactions.
    trust : float
        Current trust level [0, 1].
    conflict_history : int
        Number of disagreements/conflicts.
    successful_collaborations : int
        Number of productive exchanges.
    communication_style_model : dict
        Learned model of how the other agent communicates.
    last_interaction_tick : int
    emotional_valence : float
        Running average of interaction valence [-1, 1].
    domain_trust : dict[str, float]
        Per-domain trust in this agent's claims.
    """
    other_agent_id: str = ""
    interaction_count: int = 0
    shared_topics: Dict[str, int] = field(default_factory=dict)
    trust: float = 0.5
    conflict_history: int = 0
    successful_collaborations: int = 0
    communication_style_model: Dict[str, float] = field(default_factory=dict)
    last_interaction_tick: int = 0
    emotional_valence: float = 0.0
    domain_trust: Dict[str, float] = field(default_factory=dict)

    @property
    def relationship_quality(self) -> float:
        """Composite relationship quality [0, 1]."""
        if self.interaction_count == 0:
            return 0.5
        collab_ratio = self.successful_collaborations / max(self.interaction_count, 1)
        conflict_ratio = self.conflict_history / max(self.interaction_count, 1)
        return float(np.clip(
            0.5 * self.trust + 0.3 * collab_ratio + 0.2 * (1.0 - conflict_ratio),
            0, 1))

    def update_trust(self, signal: float, alpha: float = 0.1) -> None:
        """Update trust with EMA."""
        self.trust = float(np.clip(
            self.trust + alpha * (signal - self.trust), 0, 1))

    def record_interaction(self, topic: str, valence: float,
                           collaborated: bool, tick: int) -> None:
        self.interaction_count += 1
        self.shared_topics[topic] = self.shared_topics.get(topic, 0) + 1
        alpha = 0.1
        self.emotional_valence += alpha * (valence - self.emotional_valence)
        if collaborated:
            self.successful_collaborations += 1
        self.last_interaction_tick = tick

    def record_conflict(self) -> None:
        self.conflict_history += 1

    def to_dict(self) -> dict:
        return {
            "other": self.other_agent_id,
            "interactions": self.interaction_count,
            "trust": round(self.trust, 3),
            "quality": round(self.relationship_quality, 3),
            "collaborations": self.successful_collaborations,
            "conflicts": self.conflict_history,
            "valence": round(self.emotional_valence, 3),
            "top_topics": sorted(self.shared_topics.items(),
                                key=lambda x: -x[1])[:3],
        }


class RelationshipMemory:
    """Per-agent relationship store.

    Each agent maintains a map of relationship records
    keyed by the other agent's ID.
    """
    def __init__(self):
        self._relationships: Dict[str, RelationshipRecord] = {}

    def get_or_create(self, other_id: str) -> RelationshipRecord:
        if other_id not in self._relationships:
            self._relationships[other_id] = RelationshipRecord(
                other_agent_id=other_id)
        return self._relationships[other_id]

    def get_relationship(self, other_id: str) -> Optional[RelationshipRecord]:
        return self._relationships.get(other_id)

    def all_relationships(self) -> Dict[str, RelationshipRecord]:
        return dict(self._relationships)

    def top_trusted(self, n: int = 3) -> List[RelationshipRecord]:
        return sorted(self._relationships.values(),
                      key=lambda r: r.trust, reverse=True)[:n]

    def top_collaborators(self, n: int = 3) -> List[RelationshipRecord]:
        return sorted(self._relationships.values(),
                      key=lambda r: r.successful_collaborations,
                      reverse=True)[:n]

    def summary(self) -> dict:
        return {
            "count": len(self._relationships),
            "avg_trust": float(np.mean([r.trust for r in self._relationships.values()]))
                         if self._relationships else 0.5,
            "total_collaborations": sum(r.successful_collaborations
                                       for r in self._relationships.values()),
            "total_conflicts": sum(r.conflict_history
                                  for r in self._relationships.values()),
        }
