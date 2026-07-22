"""S8.4: ForeignEvaluator

Evaluates external interactions using DynamicsMemory (novelty, prediction error),
MetaCognition (source trust, confidence), and attractor analysis (alignment).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from substrate_echo.external.candidate_queue import (
    EvaluationResult,
    IntegrationDecision,
    InteractionSpectrum,
)
from substrate_echo.external.foreign_node import ReputationVector


# Domain detection keywords
DOMAIN_KEYWORDS = {
    "physics": ["force", "energy", "velocity", "momentum", "gravity", "mass",
                "acceleration", "friction", "temperature", "pressure", "heat",
                "entropy", "field", "wave", "photon", "quantum"],
    "ecology": ["ecosystem", "population", "habitat", "species", "predator",
                "prey", "symbiosis", "food", "resource", "growth", "decay",
                "biodiversity", "niche", "sustainability", "cycle"],
    "social": ["trust", "cooperation", "betray", "alliance", "diplomacy",
               "negotiation", "reputation", "influence", "persuade", "convince",
               "group", "community", "social", "relationship", "conflict"],
    "strategy": ["plan", "approach", "optimize", "efficiency", "improve",
                 "strategy", "tactic", "decision", "risk", "reward", "goal",
                 "objective", "algorithm", "method", "solution"],
}


def detect_domain(text: str) -> str:
    """Detect the domain of a claim from its text content.

    Returns the most likely domain, or "general" if no strong signal.
    """
    text_lower = text.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[domain] = score

    if not scores:
        return "general"
    return max(scores, key=scores.get)


class ForeignEvaluator:
    """Evaluates external interactions for integration into the cognitive substrate.

    Uses three existing cognitive systems:
    1. DynamicsMemory: novelty() and prediction_error()
    2. MetaCognition: source trust and confidence calibration
    3. Attractor analysis: alignment with existing attractor structure

    The evaluator produces an IntegrationDecision:
    - REJECT: noise, adversarial, or irrelevant
    - OBSERVE: interesting but insufficient confidence
    - CANDIDATE: promising, awaiting further validation
    - ACCEPT: ready for integration into memory
    """

    def __init__(self, dynamics_memory=None, meta_cognition=None,
                 attractor_memory=None):
        """
        Parameters
        ----------
        dynamics_memory : DynamicsMemory, optional
            For novelty and prediction error computation.
        meta_cognition : MetaCognition, optional
            For source trust and confidence calibration.
        attractor_memory : AttractorMemory, optional
            For alignment computation (nearest attractor).
        """
        self._dm = dynamics_memory
        self._mc = meta_cognition
        self._am = attractor_memory

    def evaluate(self, spectrum: InteractionSpectrum,
                 reputation: Optional[ReputationVector] = None) -> EvaluationResult:
        """Evaluate an interaction spectrum.

        Parameters
        ----------
        spectrum : InteractionSpectrum
            The encoded interaction.
        reputation : ReputationVector, optional
            Reputation of the source agent.

        Returns
        -------
        EvaluationResult
            Evaluation with alignment, novelty, risk, and recommendation.
        """
        # Detect domain from raw text
        domain = detect_domain(spectrum.raw_text)

        # Alignment: how well does this match existing attractors?
        alignment = self._compute_alignment(spectrum)

        # Novelty: how new is this information?
        novelty = self._compute_novelty(spectrum)

        # Risk: how likely is this to be misleading?
        risk = self._compute_risk(spectrum, reputation)

        # Coherence: internal consistency of the interaction
        coherence = self._compute_coherence(spectrum)

        # Make decision
        recommendation, reasoning = self._decide(
            alignment, novelty, risk, coherence, reputation, domain)

        return EvaluationResult(
            alignment=alignment,
            novelty=novelty,
            risk=risk,
            coherence=coherence,
            recommendation=recommendation,
            reasoning=reasoning,
        )

    def _compute_alignment(self, spectrum: InteractionSpectrum) -> float:
        """Compute alignment with existing attractor structure.

        High alignment = matches known attractors = familiar information.
        Low alignment = novel territory = potentially valuable but risky.
        """
        if self._am is None:
            # No attractor memory → use semantic similarity as proxy
            return self._semantic_alignment_proxy(spectrum)

        try:
            # Use the combined 32D vector to find nearest attractor
            # This is a simplified version — real implementation would
            # use the full attractor memory lookup
            return self._semantic_alignment_proxy(spectrum)
        except Exception:
            return 0.5

    def _semantic_alignment_proxy(self, spectrum: InteractionSpectrum) -> float:
        """Simple proxy for alignment based on feature magnitudes.

        High-magnitude features in known pillars = aligned.
        Uniform features = less aligned.
        """
        s = spectrum.semantic_features
        # Check how concentrated the features are (concentrated = aligned with pillars)
        sorted_abs = np.sort(np.abs(s))[::-1]
        total = np.sum(sorted_abs)
        if total < 1e-10:
            return 0.5
        # Top-4 features concentration
        concentration = np.sum(sorted_abs[:4]) / total
        return float(np.clip(concentration, 0, 1))

    def _compute_novelty(self, spectrum: InteractionSpectrum) -> float:
        """Compute novelty using DynamicsMemory if available.

        High novelty = the system hasn't seen this before.
        """
        if self._dm is None:
            return self._novelty_proxy(spectrum)

        try:
            # Use the semantic features as the state vector
            return self._dm.novelty(spectrum.semantic_features)
        except Exception:
            return self._novelty_proxy(spectrum)

    def _novelty_proxy(self, spectrum: InteractionSpectrum) -> float:
        """Proxy novelty based on feature uniqueness.

        Features far from the origin are more novel.
        """
        s = spectrum.semantic_features
        # Euclidean norm as a simple novelty proxy
        norm = np.linalg.norm(s)
        # Map to [0, 1] range (assuming typical feature range [0, 1])
        return float(np.clip(norm / np.sqrt(len(s)), 0, 1))

    def _compute_risk(self, spectrum: InteractionSpectrum,
                      reputation: Optional[ReputationVector] = None) -> float:
        """Compute risk of integrating this information.

        High risk = adversarial, contradictory, or low-reputation source.
        """
        r = spectrum.relational_features

        # Risk factors from relational features
        adversarial = r[5] if len(r) > 5 else 0.0   # adversarial_patterns
        persuasion = r[1] if len(r) > 1 else 0.0     # persuasion_pressure
        repetition = r[0] if len(r) > 0 else 0.0     # repetition_score

        # High adversarial + high persuasion = high risk
        behavioral_risk = 0.4 * adversarial + 0.3 * persuasion + 0.1 * repetition

        # Reputation-based risk
        reputation_risk = 0.0
        if reputation is not None:
            # Low trust = high risk
            reputation_risk = 1.0 - reputation.trust_score

        # Combined risk
        risk = 0.6 * behavioral_risk + 0.4 * reputation_risk
        return float(np.clip(risk, 0, 1))

    def _compute_coherence(self, spectrum: InteractionSpectrum) -> float:
        """Compute internal coherence of the interaction.

        Coherent = features are internally consistent.
        Incoherent = features contradict each other.
        """
        s = spectrum.semantic_features

        # Check for contradictions: high negation + high assertion
        negation = s[10] if len(s) > 10 else 0.0   # negation_count
        assertion = s[13] if len(s) > 13 else 0.0   # assertion_ratio
        confidence = s[15] if len(s) > 15 else 0.0  # confidence_expression
        uncertainty = s[4] if len(s) > 4 else 0.0   # uncertainty_expression

        # High confidence + high uncertainty = incoherent
        # Low confidence + low uncertainty = coherent (tentative but consistent)
        # Only penalize when BOTH are high (paradoxical)
        paradox_score = min(confidence, uncertainty)

        # Negation + assertion at high levels simultaneously = incoherent
        contradiction_score = min(negation, assertion)

        # Coherence is high when paradox and contradiction are low
        coherence = 1.0 - 0.5 * paradox_score - 0.5 * contradiction_score
        return float(np.clip(coherence, 0, 1))

    def _decide(self, alignment: float, novelty: float, risk: float,
                coherence: float,
                reputation: Optional[ReputationVector] = None,
                domain: str = "general") -> tuple:
        """Make an integration decision.

        Uses domain-specific trust when available, falling back to global.
        """
        # Rejection: high risk or low coherence
        if risk > 0.8:
            return IntegrationDecision.REJECT, f"High risk ({risk:.3f})"
        if coherence < 0.2:
            return IntegrationDecision.REJECT, f"Low coherence ({coherence:.3f})"

        # Check trust — prefer domain-specific if available
        if reputation is not None:
            trust = reputation.domain_trust_score(domain)
            trust_source = f"domain({domain})"
        else:
            trust = 0.5
            trust_source = "none"

        if trust < 0.2:
            return IntegrationDecision.REJECT, (
                f"Low trust ({trust:.3f}, {trust_source})")

        # Accept: high alignment + low risk + good reputation
        if alignment > 0.6 and risk < 0.3 and trust > 0.6:
            return IntegrationDecision.ACCEPT, (
                f"High alignment ({alignment:.3f}), low risk ({risk:.3f}), "
                f"good trust ({trust:.3f}, {trust_source})")

        # Candidate: moderate alignment + acceptable risk
        if alignment > 0.3 and risk < 0.5:
            return IntegrationDecision.CANDIDATE, (
                f"Moderate alignment ({alignment:.3f}), "
                f"acceptable risk ({risk:.3f})")

        # Observe: interesting but uncertain
        if novelty > 0.5 or (alignment > 0.2 and risk < 0.6):
            return IntegrationDecision.OBSERVE, (
                f"Novel ({novelty:.3f}), uncertain "
                f"(alignment={alignment:.3f}, risk={risk:.3f})")

        # Default: reject
        return IntegrationDecision.REJECT, (
            f"Low alignment ({alignment:.3f}), low novelty ({novelty:.3f})")
