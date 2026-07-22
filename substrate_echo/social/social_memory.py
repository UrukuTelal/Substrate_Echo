"""SocialMemory: Stores social episodes from agent interactions.

Each episode records what happened, who was involved,
and what was learned — enabling long-term social learning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SocialEpisode:
    """A single social interaction record.

    Attributes
    ----------
    participants : list of str
        Agent IDs involved.
    topic : str
        Primary topic or domain.
    content_summary : str
        Brief summary of what was said.
    emotional_valence : float
        Positive/negative tone [-1, 1].
    knowledge_exchanged : float
        Amount of useful information exchanged [0, 1].
    reputation_change : dict[str, float]
        How each participant's reputation changed.
    outcome : str
        "cooperation", "disagreement", "innovation", "conflict", "learning"
    tick : int
        When this occurred.
    novelty_score : float
        How novel the content was [0, 1].
    """
    participants: List[str] = field(default_factory=list)
    topic: str = "general"
    content_summary: str = ""
    emotional_valence: float = 0.0
    knowledge_exchanged: float = 0.0
    reputation_change: Dict[str, float] = field(default_factory=dict)
    outcome: str = "learning"
    tick: int = 0
    novelty_score: float = 0.0


class SocialMemory:
    """Stores and retrieves social episodes.

    Provides查询 by participant, topic, outcome, and recency.
    """
    def __init__(self, max_episodes: int = 5000):
        self._episodes: List[SocialEpisode] = []
        self._max = max_episodes

    def store(self, episode: SocialEpisode) -> None:
        self._episodes.append(episode)
        if len(self._episodes) > self._max:
            self._episodes = self._episodes[-self._max:]

    def recall_by_participant(self, agent_id: str,
                              limit: int = 50) -> List[SocialEpisode]:
        return [e for e in self._episodes if agent_id in e.participants][-limit:]

    def recall_by_topic(self, topic: str,
                        limit: int = 50) -> List[SocialEpisode]:
        return [e for e in self._episodes if e.topic == topic][-limit:]

    def recall_by_outcome(self, outcome: str,
                          limit: int = 50) -> List[SocialEpisode]:
        return [e for e in self._episodes if e.outcome == outcome][-limit:]

    def recall_recent(self, limit: int = 50) -> List[SocialEpisode]:
        return self._episodes[-limit:]

    def compute_relationship_history(self, agent_a: str,
                                     agent_b: str) -> dict:
        """Compute interaction summary between two agents."""
        shared = [e for e in self._episodes
                  if agent_a in e.participants and agent_b in e.participants]
        if not shared:
            return {"count": 0, "topics": {}, "avg_valence": 0.0}

        topics = {}
        valences = []
        for e in shared:
            topics[e.topic] = topics.get(e.topic, 0) + 1
            valences.append(e.emotional_valence)

        return {
            "count": len(shared),
            "topics": topics,
            "avg_valence": float(np.mean(valences)) if valences else 0.0,
        }

    @property
    def total_episodes(self) -> int:
        return len(self._episodes)

    def summary(self) -> dict:
        outcomes = {}
        for e in self._episodes:
            outcomes[e.outcome] = outcomes.get(e.outcome, 0) + 1
        return {
            "total": len(self._episodes),
            "outcomes": outcomes,
        }
