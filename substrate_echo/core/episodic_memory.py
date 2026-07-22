"""Episodic Narrative Memory — P7.1

Structured event sequences with causal links, emotional tags,
and narrative organization for coherent memory recall.

Core idea: the agent doesn't just remember individual events — it
remembers them as narratives. "First I approached the object, then
I observed it, then I learned its shape." Each event is linked to
what caused it and what it caused, creating a web of causality.

This enables:
- Coherent recall: retrieve a whole sequence from one event
- Causal reasoning: "What caused this outcome?"
- Temporal reasoning: "What happened before/after X?"
- Emotional recall: "When did I feel like this before?"
- Narrative comprehension: "What's the story of my recent actions?"

Architecture:
- Episode: single event (context, actions, outcome, emotion, tick)
- CausalLink: connects cause → effect between episodes
- NarrativeChapter: coherent sequence of episodes with theme
- EpisodicMemory: stores episodes, builds causal links, retrieves by similarity/emotion/causality

Usage:
    em = EpisodicMemory()
    ep1 = em.store(context=state1, actions=[a1], outcome=o1, tick=t1)
    ep2 = em.store(context=state2, actions=[a2], outcome=o2, tick=t2)
    em.link_causal(ep1.episode_id, ep2.episode_id, strength=0.8)
    em.build_narratives()
    narrative = em.recall_by_emotion("curiosity")
    recent = em.recall_recent(n=5)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from collections import defaultdict


@dataclass
class Episode:
    """A single remembered event."""
    episode_id: int
    tick: int

    # What happened
    context: np.ndarray          # state when event occurred
    actions: list[dict]          # actions taken
    outcome: np.ndarray          # resulting state
    success: bool = True

    # Emotional tag
    emotion: str = "neutral"     # "curiosity", "satisfaction", "frustration", etc.
    emotional_intensity: float = 0.5  # 0-1

    # Metadata
    pillar_values: Optional[np.ndarray] = None  # PSV at time of event
    narrative_id: Optional[int] = None

    @property
    def delta(self) -> np.ndarray:
        """State change from this event."""
        return self.outcome - self.context


@dataclass
class CausalLink:
    """Directed link between two episodes."""
    from_id: int
    to_id: int
    strength: float = 1.0    # 0-1, how strong the causal connection is
    link_type: str = "direct"  # "direct", "enabling", "inhibiting"


@dataclass
class NarrativeChapter:
    """A coherent sequence of episodes forming a narrative."""
    narrative_id: int
    episodes: list[int] = field(default_factory=list)  # episode_ids in order
    theme: str = ""           # what this narrative is about
    start_tick: int = 0
    end_tick: int = 0

    @property
    def duration(self) -> int:
        return self.end_tick - self.start_tick

    @property
    def length(self) -> int:
        return len(self.episodes)


@dataclass
class EpisodicMemoryConfig:
    """Configuration for episodic memory."""
    max_episodes: int = 500
    max_narratives: int = 50
    causal_link_threshold: float = 0.3  # min similarity for auto-linking
    narrative_merge_threshold: float = 0.5  # min episodes to merge narratives
    emotional_decay: float = 0.01  # per tick
    context_dim: int = 16


class EpisodicMemory:
    """Stores and retrieves structured event sequences.

    Each episode captures what happened, what resulted, and how
    it felt. Episodes are linked causally, forming narratives —
    coherent stories of the agent's experience.

    Usage:
        em = EpisodicMemory()

        # Store events
        ep1 = em.store(context=s1, actions=[a1], outcome=s2, tick=t)
        ep2 = em.store(context=s2, actions=[a2], outcome=s3, tick=t+1)

        # Link causally
        em.link_causal(ep1.episode_id, ep2.episode_id)

        # Build narratives from causal chains
        em.build_narratives()

        # Retrieve
        recent = em.recall_recent(3)
        curious = em.recall_by_emotion("curiosity")
        from_ep = em.recall_by_causality(ep1.episode_id)
    """

    def __init__(self, config: Optional[EpisodicMemoryConfig] = None):
        self.config = config or EpisodicMemoryConfig()
        self._episodes: dict[int, Episode] = {}
        self._causal_links: list[CausalLink] = []
        self._narratives: dict[int, NarrativeChapter] = {}
        self._next_episode_id = 0
        self._next_narrative_id = 0

        # Indexes
        self._emotion_index: dict[str, list[int]] = defaultdict(list)
        self._tick_index: list[tuple[int, int]] = []  # (tick, episode_id)

    def store(self, context: np.ndarray, actions: list[dict],
              outcome: np.ndarray, tick: int,
              success: bool = True, emotion: str = "neutral",
              emotional_intensity: float = 0.5,
              pillar_values: Optional[np.ndarray] = None) -> Episode:
        """Store a new episode.

        Args:
            context: state when event occurred
            actions: actions taken
            outcome: resulting state
            tick: current tick
            success: whether event succeeded
            emotion: emotional tag
            emotional_intensity: 0-1
            pillar_values: PSV at time of event

        Returns:
            The created Episode
        """
        ep_id = self._next_episode_id
        self._next_episode_id += 1

        episode = Episode(
            episode_id=ep_id,
            tick=tick,
            context=np.asarray(context, dtype=np.float64),
            actions=actions,
            outcome=np.asarray(outcome, dtype=np.float64),
            success=success,
            emotion=emotion,
            emotional_intensity=emotional_intensity,
            pillar_values=pillar_values,
        )

        self._episodes[ep_id] = episode
        self._emotion_index[emotion].append(ep_id)
        self._tick_index.append((tick, ep_id))

        # Auto-link to previous episode if sequential (no duplicate links)
        if self._tick_index and len(self._tick_index) > 1:
            prev_tick, prev_id = self._tick_index[-2]
            if tick == prev_tick + 1:
                already_linked = any(
                    l.from_id == prev_id and l.to_id == ep_id
                    for l in self._causal_links
                )
                if not already_linked:
                    self.link_causal(prev_id, ep_id, strength=0.8, link_type="direct")

        # Maintain size
        if len(self._episodes) > self.config.max_episodes:
            self._evict_oldest()

        return episode

    def link_causal(self, from_id: int, to_id: int,
                    strength: float = 1.0,
                    link_type: str = "direct") -> CausalLink:
        """Create a causal link between two episodes.

        Deduplicates: if an identical link already exists, updates strength.
        """
        for link in self._causal_links:
            if link.from_id == from_id and link.to_id == to_id:
                link.strength = max(link.strength, strength)
                return link
        link = CausalLink(from_id=from_id, to_id=to_id,
                          strength=strength, link_type=link_type)
        self._causal_links.append(link)
        return link

    def get_causes(self, episode_id: int) -> list[Episode]:
        """Get all episodes that caused this one."""
        causes = []
        for link in self._causal_links:
            if link.to_id == episode_id and link.from_id in self._episodes:
                causes.append(self._episodes[link.from_id])
        return causes

    def get_effects(self, episode_id: int) -> list[Episode]:
        """Get all episodes caused by this one."""
        effects = []
        for link in self._causal_links:
            if link.from_id == episode_id and link.to_id in self._episodes:
                effects.append(self._episodes[link.to_id])
        return effects

    def get_causal_chain(self, episode_id: int) -> list[Episode]:
        """Get the full causal chain starting from this episode."""
        chain = []
        visited = set()
        current = episode_id

        while current not in visited and current in self._episodes:
            visited.add(current)
            chain.append(self._episodes[current])
            effects = self.get_effects(current)
            if effects:
                current = effects[0].episode_id
            else:
                break

        return chain

    def recall_recent(self, n: int = 5) -> list[Episode]:
        """Recall the N most recent episodes."""
        sorted_eps = sorted(self._episodes.values(),
                           key=lambda e: e.tick, reverse=True)
        return sorted_eps[:n]

    def recall_by_emotion(self, emotion: str,
                          max_results: int = 10) -> list[Episode]:
        """Recall episodes with a specific emotional tag."""
        ids = self._emotion_index.get(emotion, [])
        episodes = [self._episodes[eid] for eid in ids if eid in self._episodes]
        # Sort by emotional intensity
        episodes.sort(key=lambda e: e.emotional_intensity, reverse=True)
        return episodes[:max_results]

    def recall_by_context(self, context: np.ndarray,
                          max_results: int = 5) -> list[Episode]:
        """Recall episodes with similar context."""
        context = np.asarray(context, dtype=np.float64)
        scored = []
        for ep in self._episodes.values():
            sim = self._context_similarity(context, ep.context)
            scored.append((sim, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:max_results]]

    def recall_by_causality(self, episode_id: int) -> list[Episode]:
        """Recall the causal chain from this episode."""
        return self.get_causal_chain(episode_id)

    def build_narratives(self) -> list[NarrativeChapter]:
        """Build narrative chapters from causal chains.

        Groups causally-linked episodes into coherent narratives.
        """
        # Find root episodes (no causes)
        roots = set(self._episodes.keys())
        for link in self._causal_links:
            roots.discard(link.to_id)

        # Build chains from each root
        for root_id in sorted(roots):
            chain = self.get_causal_chain(root_id)
            if len(chain) < 2:
                continue

            # Check if already in a narrative
            if chain[0].narrative_id is not None:
                continue

            # Create narrative
            n_id = self._next_narrative_id
            self._next_narrative_id += 1

            narrative = NarrativeChapter(
                narrative_id=n_id,
                episodes=[ep.episode_id for ep in chain],
                theme=self._infer_theme(chain),
                start_tick=chain[0].tick,
                end_tick=chain[-1].tick,
            )

            self._narratives[n_id] = narrative

            for ep in chain:
                ep.narrative_id = n_id

        return list(self._narratives.values())

    def recall_narrative(self, narrative_id: int) -> Optional[NarrativeChapter]:
        """Recall a specific narrative."""
        return self._narratives.get(narrative_id)

    def recall_narratives_by_theme(self, theme: str) -> list[NarrativeChapter]:
        """Recall narratives matching a theme."""
        return [n for n in self._narratives.values()
                if theme.lower() in n.theme.lower()]

    def summary(self) -> dict:
        """Summary of episodic memory."""
        episodes = list(self._episodes.values())
        emotions = defaultdict(int)
        for ep in episodes:
            emotions[ep.emotion] += 1

        return {
            "n_episodes": len(episodes),
            "n_causal_links": len(self._causal_links),
            "n_narratives": len(self._narratives),
            "emotions": dict(emotions),
            "avg_emotional_intensity": (
                np.mean([ep.emotional_intensity for ep in episodes])
                if episodes else 0
            ),
        }

    def _infer_theme(self, chain: list[Episode]) -> str:
        """Infer narrative theme from episode actions and emotions."""
        emotions = [ep.emotion for ep in chain]
        most_common = max(set(emotions), key=emotions.count) if emotions else "neutral"

        action_types = []
        for ep in chain:
            for a in ep.actions:
                action_types.append(a.get("type", "unknown"))
        most_action = max(set(action_types), key=action_types.count) if action_types else "action"

        return f"{most_action}_{most_common}"

    def _context_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute context similarity (0-1)."""
        dist = np.linalg.norm(a - b)
        return float(np.exp(-dist / 0.5))

    def _evict_oldest(self) -> None:
        """Remove oldest episodes when at capacity."""
        if not self._tick_index:
            return
        # Remove oldest 10%
        n_remove = max(1, len(self._tick_index) // 10)
        self._tick_index.sort(key=lambda x: x[0])
        for _ in range(n_remove):
            if self._tick_index:
                _, ep_id = self._tick_index.pop(0)
                if ep_id in self._episodes:
                    del self._episodes[ep_id]
