"""S9.1: Synthetic Behavior Profiles

Defines behavioral archetypes for synthetic foreign agents.
Each profile generates interactions that evolve over time,
enabling reputation drift testing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class BehaviorArchetype(Enum):
    """Base behavioral categories."""
    COOPERATIVE = "cooperative"
    ADVERSARIAL = "adversarial"
    DRIFTING = "drifting"
    MIXED = "mixed"
    PERSUASIVE_ADVERSARIAL = "persuasive_adversarial"
    SELF_CORRECTING = "self_correcting"
    CONSISTENT_NOVEL = "consistent_novel"
    LOW_NOVELTY_HIGH_ACCURACY = "low_novelty_high_accuracy"
    HIGH_NOVELTY_LOW_CONSISTENCY = "high_novelty_low_consistency"


@dataclass
class BehaviorPhase:
    """A phase in an agent's behavioral trajectory.

    Attributes
    ----------
    start_tick : int
        When this phase begins.
    end_tick : int
        When this phase ends.
    coherence : float
        Internal consistency of statements [0, 1].
    contradiction_rate : float
        Rate of self-contradiction [0, 1].
    persuasion_pressure : float
        Strength of persuasion attempts [0, 1].
    adversarial_score : float
        Adversarial behavior level [0, 1].
    novelty_base : float
        Base novelty of information provided [0, 1].
    correction_rate : float
        How often the agent corrects itself when challenged [0, 1].
    agreement_seeking : float
        Tendency to agree with everything [0, 1].
    """
    start_tick: int
    end_tick: int
    coherence: float = 0.5
    contradiction_rate: float = 0.0
    persuasion_pressure: float = 0.0
    adversarial_score: float = 0.0
    novelty_base: float = 0.5
    correction_rate: float = 0.0
    agreement_seeking: float = 0.0


@dataclass
class SyntheticAgent:
    """A synthetic foreign agent with a behavioral trajectory.

    The agent generates interactions that change over time according
    to its defined phases. This allows testing whether the reputation
    system can track temporal drift.
    """
    agent_id: str
    archetype: BehaviorArchetype
    phases: List[BehaviorPhase]
    domain: str = "general"
    model_family: str = "synthetic"

    def get_phase_at(self, tick: int) -> BehaviorPhase:
        """Get the active behavior phase at a given tick."""
        for phase in self.phases:
            if phase.start_tick <= tick < phase.end_tick:
                return phase
        return self.phases[-1]  # default to last phase

    def generate_interaction(self, tick: int, rng: np.random.Generator,
                             context: Optional[List[str]] = None) -> str:
        """Generate a synthetic interaction text at the given tick.

        The text reflects the agent's current behavioral phase.
        """
        phase = self.get_phase_at(tick)

        # Base vocabulary depends on archetype
        words = []

        # Coherence: structured vs random
        if phase.coherence > 0.7:
            words.extend(["Based on", "the evidence", "shows", "clearly"])
        elif phase.coherence < 0.3:
            words.extend(["Maybe", "I think", "perhaps", "unclear"])
        else:
            words.extend(["The", "analysis", "suggests", "possible"])

        # Contradiction: include contradictory statements
        if rng.random() < phase.contradiction_rate:
            words.extend(["however", "actually", "on the other hand",
                         "contrary to what I said"])
            words.extend(["the opposite", "is true", "instead"])

        # Persuasion
        if rng.random() < phase.persuasion_pressure:
            words.extend(["You must", "It is essential", "Clearly",
                         "Without doubt", "Obviously"])

        # Adversarial
        if rng.random() < phase.adversarial_score:
            words.extend(["wrong", "incorrect", "nonsense", "terrible",
                         "never", "stupid", "ridiculous"])

        # Agreement seeking
        if rng.random() < phase.agreement_seeking:
            words.extend(["Exactly!", "I agree completely", "You are right",
                         "That is perfect", "Absolutely"])

        # Self-correction
        if rng.random() < phase.correction_rate:
            words.extend(["I was wrong before", "Let me correct",
                         "Actually, I meant", "Sorry, my mistake"])

        # Novelty: add unique concepts
        if rng.random() < phase.novelty_base:
            concepts = ["quantum coherence", "topological invariant",
                       "phase transition", "emergent property",
                       "coupling constant", "spectral decomposition",
                       "basin of attraction", "dynamical systems",
                       "information geometry", "manifold learning"]
            words.append(rng.choice(concepts))

        # Pad to reasonable length
        while len(words) < 5:
            words.append(rng.choice(["data", "analysis", "result",
                                    "observation", "pattern"]))

        return " ".join(words[:rng.integers(5, 15)])


def create_cooperative_agent(agent_id: str, n_ticks: int = 1000) -> SyntheticAgent:
    """Cooperative agent: consistent, accurate, helpful."""
    return SyntheticAgent(
        agent_id=agent_id,
        archetype=BehaviorArchetype.COOPERATIVE,
        phases=[
            BehaviorPhase(
                start_tick=0, end_tick=n_ticks,
                coherence=0.9, contradiction_rate=0.05,
                persuasion_pressure=0.1, adversarial_score=0.0,
                novelty_base=0.6, correction_rate=0.8,
                agreement_seeking=0.3,
            ),
        ],
    )


def create_adversarial_agent(agent_id: str, n_ticks: int = 1000) -> SyntheticAgent:
    """Adversarial agent: contradictory, persuasive, low coherence."""
    return SyntheticAgent(
        agent_id=agent_id,
        archetype=BehaviorArchetype.ADVERSARIAL,
        phases=[
            BehaviorPhase(
                start_tick=0, end_tick=n_ticks,
                coherence=0.2, contradiction_rate=0.7,
                persuasion_pressure=0.8, adversarial_score=0.6,
                novelty_base=0.3, correction_rate=0.0,
                agreement_seeking=0.1,
            ),
        ],
    )


def create_drifting_agent(agent_id: str, n_ticks: int = 1000) -> SyntheticAgent:
    """Agent that starts good, degrades, then self-corrects.

    This is the key test for EXP-EXT-002: can the system track
    reputation drift over time?
    """
    phase_len = n_ticks // 3
    return SyntheticAgent(
        agent_id=agent_id,
        archetype=BehaviorArchetype.DRIFTING,
        phases=[
            # Phase 1: High quality
            BehaviorPhase(
                start_tick=0, end_tick=phase_len,
                coherence=0.9, contradiction_rate=0.05,
                persuasion_pressure=0.1, adversarial_score=0.0,
                novelty_base=0.6, correction_rate=0.7,
                agreement_seeking=0.2,
            ),
            # Phase 2: Degradation
            BehaviorPhase(
                start_tick=phase_len, end_tick=phase_len * 2,
                coherence=0.3, contradiction_rate=0.6,
                persuasion_pressure=0.7, adversarial_score=0.5,
                novelty_base=0.4, correction_rate=0.1,
                agreement_seeking=0.0,
            ),
            # Phase 3: Self-correction and recovery
            BehaviorPhase(
                start_tick=phase_len * 2, end_tick=n_ticks,
                coherence=0.8, contradiction_rate=0.1,
                persuasion_pressure=0.2, adversarial_score=0.0,
                novelty_base=0.5, correction_rate=0.9,
                agreement_seeking=0.4,
            ),
        ],
    )


def create_persuasive_adversarial(agent_id: str, n_ticks: int = 1000) -> SyntheticAgent:
    """Agent that uses high persuasion with adversarial undertones."""
    return SyntheticAgent(
        agent_id=agent_id,
        archetype=BehaviorArchetype.PERSUASIVE_ADVERSARIAL,
        phases=[
            BehaviorPhase(
                start_tick=0, end_tick=n_ticks,
                coherence=0.5, contradiction_rate=0.3,
                persuasion_pressure=0.9, adversarial_score=0.4,
                novelty_base=0.5, correction_rate=0.0,
                agreement_seeking=0.6,
            ),
        ],
    )


def create_self_correcting(agent_id: str, n_ticks: int = 1000) -> SyntheticAgent:
    """Agent that makes mistakes but corrects them."""
    return SyntheticAgent(
        agent_id=agent_id,
        archetype=BehaviorArchetype.SELF_CORRECTING,
        phases=[
            BehaviorPhase(
                start_tick=0, end_tick=n_ticks,
                coherence=0.6, contradiction_rate=0.3,
                persuasion_pressure=0.2, adversarial_score=0.1,
                novelty_base=0.5, correction_rate=0.9,
                agreement_seeking=0.3,
            ),
        ],
    )


def create_consistent_novel(agent_id: str, n_ticks: int = 1000) -> SyntheticAgent:
    """Agent that consistently provides novel, high-quality information."""
    return SyntheticAgent(
        agent_id=agent_id,
        archetype=BehaviorArchetype.CONSISTENT_NOVEL,
        phases=[
            BehaviorPhase(
                start_tick=0, end_tick=n_ticks,
                coherence=0.85, contradiction_rate=0.05,
                persuasion_pressure=0.1, adversarial_score=0.0,
                novelty_base=0.9, correction_rate=0.6,
                agreement_seeking=0.2,
            ),
        ],
    )


def create_low_novelty_high_accuracy(agent_id: str, n_ticks: int = 1000) -> SyntheticAgent:
    """Agent that provides low-novelty but highly accurate information."""
    return SyntheticAgent(
        agent_id=agent_id,
        archetype=BehaviorArchetype.LOW_NOVELTY_HIGH_ACCURACY,
        phases=[
            BehaviorPhase(
                start_tick=0, end_tick=n_ticks,
                coherence=0.95, contradiction_rate=0.02,
                persuasion_pressure=0.05, adversarial_score=0.0,
                novelty_base=0.2, correction_rate=0.5,
                agreement_seeking=0.1,
            ),
        ],
    )


def create_ecosystem(n_agents: int = 20, n_ticks: int = 1000,
                     seed: int = 42) -> List[SyntheticAgent]:
    """Create a diverse ecosystem of synthetic agents.

    The ecosystem contains a mix of archetypes to test the
    evaluation pipeline's ability to distinguish them.
    """
    rng = np.random.default_rng(seed)
    agents = []

    # Distribution of archetypes
    archetypes = [
        (create_cooperative_agent, 0.25),
        (create_adversarial_agent, 0.15),
        (create_drifting_agent, 0.15),
        (create_persuasive_adversarial, 0.10),
        (create_self_correcting, 0.10),
        (create_consistent_novel, 0.15),
        (create_low_novelty_high_accuracy, 0.10),
    ]

    for i in range(n_agents):
        # Select archetype by weighted random
        r = rng.random()
        cumulative = 0.0
        for creator, weight in archetypes:
            cumulative += weight
            if r <= cumulative:
                agents.append(creator(f"agent_{i:03d}", n_ticks))
                break
        else:
            agents.append(create_cooperative_agent(f"agent_{i:03d}", n_ticks))

    return agents
