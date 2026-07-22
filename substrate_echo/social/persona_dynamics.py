"""PersonaDynamics: Controls personality drift and behavioral adaptation.

Identity layer:   slow-changing (0.001-0.01/interaction)
Behavioral layer: medium-changing (0.01-0.1/interaction)
Reputation layer: fast-changing (0.1+/interaction)

new_trait = old_trait * stability + experience_signal * adaptation_rate
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from substrate_echo.social.persona_genome import (
    AgentGenome, PersonalityVector, SelectionPressure,
)
from substrate_echo.social.relationship_memory import RelationshipMemory
from substrate_echo.social.social_memory import SocialMemory, SocialEpisode


@dataclass
class PersonaState:
    """Current dynamic state of a persona.

    This is the live, evolving state derived from the genome.
    """
    # Current personality (drifts from genome starting point)
    personality: PersonalityVector = field(default_factory=PersonalityVector)

    # Behavioral tendencies (adapt faster)
    confidence: float = 0.5
    topic_expertise: Dict[str, float] = field(default_factory=dict)
    communication_adaptation: Dict[str, float] = field(default_factory=dict)

    # Social state
    relationship_memory: RelationshipMemory = field(default_factory=RelationshipMemory)
    social_memory: SocialMemory = field(default_factory=SocialMemory)

    # Innovation tracking
    concepts_proposed: int = 0
    concepts_verified: int = 0
    perspectives_generated: int = 0

    @property
    def expertise_entropy(self) -> float:
        """Measure of topic specialization. Low = specialized, High = generalist."""
        if not self.topic_expertise:
            return 1.0
        vals = np.array(list(self.topic_expertise.values()))
        vals = vals / (vals.sum() + 1e-10)
        return float(-np.sum(vals * np.log2(vals + 1e-10)))


class PersonaDynamics:
    """Controls how personas evolve through interaction.

    Applies the three-layer adaptation model:
    1. Identity core drifts slowly (personality traits)
    2. Behavioral layer adapts at medium speed (communication, topics)
    3. Reputation layer changes fast (per-relationship trust)
    """

    def __init__(self):
        self._states: Dict[str, PersonaState] = {}
        self._genomes: Dict[str, AgentGenome] = {}
        self._tick = 0

    def register(self, genome: AgentGenome) -> PersonaState:
        """Register a new agent genome and create its initial state."""
        self._genomes[genome.name] = genome
        state = PersonaState(
            personality=PersonalityVector(
                curiosity=genome.personality.curiosity,
                skepticism=genome.personality.skepticism,
                humor=genome.personality.humor,
                creativity=genome.personality.creativity,
                patience=genome.personality.patience,
                abstraction=genome.personality.abstraction,
                sociability=genome.personality.sociability,
            ),
        )
        self._states[genome.name] = state
        return state

    def get_state(self, agent_id: str) -> Optional[PersonaState]:
        return self._states.get(agent_id)

    def get_genome(self, agent_id: str) -> Optional[AgentGenome]:
        return self._genomes.get(agent_id)

    def process_interaction(self, agent_id: str,
                            episode: SocialEpisode) -> None:
        """Update persona state after a social interaction.

        This is where adaptation happens.
        """
        state = self._states.get(agent_id)
        genome = self._genomes.get(agent_id)
        if state is None or genome is None:
            return

        # 1. Identity layer: slow personality drift
        self._drift_personality(state, genome, episode)

        # 2. Behavioral layer: medium-speed adaptation
        self._adapt_behavior(state, genome, episode)

        # 3. Record social episode
        state.social_memory.store(episode)

        # 4. Update selection pressure fitness
        self._apply_selection_pressure(state, genome, episode)

    def _drift_personality(self, state: PersonaState, genome: AgentGenome,
                           episode: SocialEpisode) -> None:
        """Slow drift of core personality traits.

        Includes homeostatic pull toward genome starting values.
        Without this, all agents converge toward the same trait values
        because the environment rewards the same behaviors for everyone.
        """
        rate = genome.identity_drift_rate
        homeostasis = 0.003  # pull back toward genome each interaction
        p = state.personality
        g = genome.personality

        # Homeostatic pull toward genome values (identity anchor)
        p.curiosity = float(np.clip(
            p.curiosity + homeostasis * (g.curiosity - p.curiosity), 0, 1))
        p.skepticism = float(np.clip(
            p.skepticism + homeostasis * (g.skepticism - p.skepticism), 0, 1))
        p.humor = float(np.clip(
            p.humor + homeostasis * (g.humor - p.humor), 0, 1))
        p.creativity = float(np.clip(
            p.creativity + homeostasis * (g.creativity - p.creativity), 0, 1))
        p.patience = float(np.clip(
            p.patience + homeostasis * (g.patience - p.patience), 0, 1))
        p.abstraction = float(np.clip(
            p.abstraction + homeostasis * (g.abstraction - p.abstraction), 0, 1))
        p.sociability = float(np.clip(
            p.sociability + homeostasis * (g.sociability - p.sociability), 0, 1))

        # Experience-driven drift (genome-specific, not universal)
        # Only drift traits that this genome's selection pressure rewards
        if episode.emotional_valence > 0.3 and g.sociability > 0.7:
            # High-sociability agents get a small sociability boost from positive interactions
            p.sociability = float(np.clip(
                p.sociability + rate * 0.2, 0, 1))
        elif episode.emotional_valence > 0.3 and g.sociability < 0.5:
            # Low-sociability agents: positive interactions boost curiosity instead
            p.curiosity = float(np.clip(
                p.curiosity + rate * 0.15, 0, 1))

        if episode.novelty_score > 0.5:
            p.curiosity = float(np.clip(
                p.curiosity + rate * 0.2, 0, 1))

        if episode.outcome == "conflict":
            p.skepticism = float(np.clip(
                p.skepticism + rate * 0.15, 0, 1))

        if episode.outcome == "cooperation":
            p.patience = float(np.clip(
                p.patience + rate * 0.08, 0, 1))

        if episode.outcome == "innovation":
            p.creativity = float(np.clip(
                p.creativity + rate * 0.1, 0, 1))

    def _adapt_behavior(self, state: PersonaState, genome: AgentGenome,
                        episode: SocialEpisode) -> None:
        """Medium-speed behavioral adaptation."""
        rate = genome.behavioral_drift_rate

        # Topic expertise grows with knowledge exchange
        if episode.knowledge_exchanged > 0.3:
            current = state.topic_expertise.get(episode.topic, 0.3)
            state.topic_expertise[episode.topic] = float(np.clip(
                current + rate * episode.knowledge_exchanged, 0, 1))

        # Confidence adjusts based on outcome
        if episode.outcome in ("cooperation", "innovation"):
            state.confidence = float(np.clip(
                state.confidence + rate * 0.1, 0, 1))
        elif episode.outcome == "conflict":
            state.confidence = float(np.clip(
                state.confidence - rate * 0.05, 0, 1))

    def _apply_selection_pressure(self, state: PersonaState,
                                  genome: AgentGenome,
                                  episode: SocialEpisode) -> None:
        """Apply the agent's unique selection pressures."""
        sp = genome.selection_pressure

        # Check if episode contains rewarded/punished behaviors
        for behavior in sp.rewarded_behaviors:
            if behavior in episode.outcome or behavior in episode.content_summary:
                # Reward: boost confidence and relevant traits
                state.confidence = float(np.clip(
                    state.confidence + 0.02, 0, 1))

        for behavior in sp.punished_behaviors:
            if behavior in episode.outcome or behavior in episode.content_summary:
                # Punishment: reduce confidence
                state.confidence = float(np.clip(
                    state.confidence - 0.02, 0, 1))

    def get_all_personalities(self) -> Dict[str, PersonalityVector]:
        """Get current personality vectors for all agents."""
        return {aid: s.personality for aid, s in self._states.items()}

    def compute_personality_divergence(self) -> float:
        """Measure how distinct agents are from each other.

        Returns average pairwise cosine distance.
        Higher = more distinct.
        """
        agents = list(self._states.values())
        if len(agents) < 2:
            return 0.0

        distances = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                d = agents[i].personality.cosine_distance(agents[j].personality)
                distances.append(d)

        return float(np.mean(distances)) if distances else 0.0

    def compute_specialization(self) -> Dict[str, Dict[str, float]]:
        """Get topic expertise distribution per agent."""
        result = {}
        for aid, state in self._states.items():
            total = sum(state.topic_expertise.values()) + 1e-10
            result[aid] = {t: v / total
                          for t, v in state.topic_expertise.items()}
        return result

    def compute_innovation_rate(self) -> Dict[str, float]:
        """Concepts verified / concepts proposed per agent."""
        result = {}
        for aid, state in self._states.items():
            if state.concepts_proposed > 0:
                result[aid] = state.concepts_verified / state.concepts_proposed
            else:
                result[aid] = 0.0
        return result

    def tick(self) -> None:
        self._tick += 1

    def summary(self) -> dict:
        return {
            "tick": self._tick,
            "agents": len(self._states),
            "divergence": round(self.compute_personality_divergence(), 4),
            "innovation": self.compute_innovation_rate(),
        }
