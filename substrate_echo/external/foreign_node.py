"""S8.2: ForeignAgent + ReputationVector

ForeignAgent extends AgentState with external agent metadata.
ReputationVector tracks behavioral metrics (no truth labels needed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from substrate_echo.core.world import AgentState


@dataclass
class ReputationVector:
    """Behavioral reputation of an external agent.

    Uses measurable proxies instead of accuracy (which requires truth labels).
    All values are running averages that update with each interaction.

    Domain-conditioned trust allows the system to recognize that a source
    can be excellent in one domain and unreliable in another.

    Attributes
    ----------
    self_consistency : float
        How often the agent's statements are internally consistent. [0, 1]
    correction_rate : float
        How often the agent corrects itself when challenged. [0, 1]
    contradiction_rate : float
        How often the agent contradicts its own prior statements. [0, 1]
    novelty_contribution : float
        How much novel information the agent provides. [0, 1]
    prediction_alignment : float
        How well the agent's claims predict actual outcomes. [0, 1]
    interaction_stability : float
        How stable the agent's behavior is across interactions. [0, 1]
    total_interactions : int
        Total number of interactions with this agent.
    domain_trust : dict[str, float]
        Per-domain prediction alignment. Key=domain, Value=trust [0, 1].
    domain_interactions : dict[str, int]
        Number of interactions per domain.
    """
    self_consistency: float = 0.5
    correction_rate: float = 0.0
    contradiction_rate: float = 0.0
    novelty_contribution: float = 0.5
    prediction_alignment: float = 0.5
    interaction_stability: float = 0.5
    total_interactions: int = 0
    domain_trust: dict = field(default_factory=dict)
    domain_interactions: dict = field(default_factory=dict)

    def update(self, *, self_consistency: Optional[float] = None,
               correction_rate: Optional[float] = None,
               contradiction_rate: Optional[float] = None,
               novelty_contribution: Optional[float] = None,
               prediction_alignment: Optional[float] = None,
               interaction_stability: Optional[float] = None) -> None:
        """Update reputation with exponential moving average."""
        alpha = 0.1  # learning rate
        if self_consistency is not None:
            self.self_consistency += alpha * (self_consistency - self.self_consistency)
        if correction_rate is not None:
            self.correction_rate += alpha * (correction_rate - self.correction_rate)
        if contradiction_rate is not None:
            self.contradiction_rate += alpha * (contradiction_rate - self.contradiction_rate)
        if novelty_contribution is not None:
            self.novelty_contribution += alpha * (novelty_contribution - self.novelty_contribution)
        if prediction_alignment is not None:
            self.prediction_alignment += alpha * (prediction_alignment - self.prediction_alignment)
        if interaction_stability is not None:
            self.interaction_stability += alpha * (interaction_stability - self.interaction_stability)
        self.total_interactions += 1

    def update_domain(self, domain: str, signal: float) -> None:
        """Update domain-specific prediction alignment.

        Parameters
        ----------
        domain : str
            The domain label (e.g., "physics", "social", "ecology").
        signal : float
            Verification signal [0, 1]. 1 = prediction was correct.
        """
        alpha = 0.15  # faster adaptation per-domain (less data)
        current = self.domain_trust.get(domain, 0.5)
        self.domain_trust[domain] = current + alpha * (signal - current)
        self.domain_interactions[domain] = self.domain_interactions.get(domain, 0) + 1

    def domain_trust_score(self, domain: str) -> float:
        """Get trust score for a specific domain.

        Falls back to global trust if no domain-specific data exists.
        Uses a Bayesian blend: more domain interactions → more weight on domain.
        """
        global_trust = self.trust_score
        domain_t = self.domain_trust.get(domain)
        domain_n = self.domain_interactions.get(domain, 0)

        if domain_t is None or domain_n < 3:
            # Not enough domain data — blend toward global
            return global_trust

        # Blend: weight domain more as n increases (max 0.8 domain, 0.2 global)
        weight = min(0.8, domain_n / 20.0)
        return weight * domain_t + (1.0 - weight) * global_trust

    @property
    def trust_score(self) -> float:
        """Composite trust score. Higher = more trustworthy.

        Computed from behavioral metrics:
        - High self_consistency, correction_rate, prediction_alignment → trustworthy
        - High contradiction_rate → untrustworthy
        - High novelty_contribution → valuable (but not necessarily trustworthy)
        """
        return (
            0.25 * self.self_consistency +
            0.15 * self.correction_rate +
            0.15 * (1.0 - self.contradiction_rate) +
            0.15 * self.prediction_alignment +
            0.15 * self.interaction_stability +
            0.15 * self.novelty_contribution
        )

    def to_dict(self) -> dict:
        return {
            "self_consistency": round(self.self_consistency, 4),
            "correction_rate": round(self.correction_rate, 4),
            "contradiction_rate": round(self.contradiction_rate, 4),
            "novelty_contribution": round(self.novelty_contribution, 4),
            "prediction_alignment": round(self.prediction_alignment, 4),
            "interaction_stability": round(self.interaction_stability, 4),
            "trust_score": round(self.trust_score, 4),
            "total_interactions": self.total_interactions,
            "domain_trust": {k: round(v, 4) for k, v in self.domain_trust.items()},
            "domain_interactions": dict(self.domain_interactions),
        }


@dataclass
class InteractionRecord:
    """A single interaction record for history tracking."""
    tick: int
    spectrum_hash: str       # hash of the interaction spectrum
    evaluation_summary: str  # "alignment=X, novelty=Y, risk=Z"
    decision: str            # IntegrationDecision value


@dataclass
class ForeignAgent(AgentState):
    """An external agent extending the native AgentState.

    This allows TheoryOfMind to reason about both internal and external
    agents through the same interface, avoiding two separate "agent worlds."
    """
    origin: str = "unknown"
    model_family: str = "unknown"
    reputation: ReputationVector = field(default_factory=ReputationVector)
    interaction_history: List[InteractionRecord] = field(default_factory=list)
    domain_scores: dict = field(default_factory=dict)

    def record_interaction(self, tick: int, spectrum_hash: str,
                           evaluation_summary: str, decision: str) -> None:
        """Record an interaction for history tracking."""
        self.interaction_history.append(InteractionRecord(
            tick=tick, spectrum_hash=spectrum_hash,
            evaluation_summary=evaluation_summary, decision=decision,
        ))
        # Keep last 100 interactions
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]

    def get_domain_reputation(self, domain: str) -> float:
        """Get reputation score for a specific domain."""
        return self.domain_scores.get(domain, 0.5)

    def update_domain(self, domain: str, score: float) -> None:
        """Update domain-specific reputation."""
        alpha = 0.1
        current = self.domain_scores.get(domain, 0.5)
        self.domain_scores[domain] = current + alpha * (score - current)
