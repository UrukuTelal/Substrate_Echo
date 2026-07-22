"""S8.1: MemoryCandidate + CandidateQueue + IntegrationDecision

The information bottleneck for external agent interactions.
External interactions enter as candidates and are routed through
REJECT / OBSERVE / CANDIDATE / ACCEPT based on evaluation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class IntegrationDecision(Enum):
    """How to handle an incoming external interaction."""
    REJECT = "reject"       # Noise, adversarial, or irrelevant
    OBSERVE = "observe"     # Interesting but insufficient confidence
    CANDIDATE = "candidate" # Promising, awaiting further validation
    ACCEPT = "accept"       # Ready for integration into memory


class IntegrationMode(Enum):
    """Operational mode for the integration gate.

    Controls how much influence external interactions can have.
    """
    OBSERVATION_ONLY = "observation_only"  # Log only, no memory changes
    CANDIDATE_ONLY = "candidate_only"      # Queue fills, nothing accepted
    FULL = "full"                          # Normal operation


class CandidateStatus(Enum):
    """Lifecycle state of a memory candidate."""
    PENDING = "pending"       # Just submitted, awaiting evaluation
    OBSERVED = "observed"     # In observation queue, accumulating evidence
    CANDIDATE = "candidate"   # Passed initial evaluation, awaiting confirmation
    ACCEPTED = "accepted"     # Integrated into memory
    REJECTED = "rejected"     # Failed evaluation
    STALE = "stale"           # Expired without resolution


@dataclass
class InteractionSpectrum:
    """The encoded representation of an external interaction.

    Attributes
    ----------
    semantic_features : np.ndarray
        16D semantic feature vector (what is being said).
    relational_features : np.ndarray
        16D relational feature vector (how the agent is behaving).
    combined : np.ndarray
        32D combined vector after spectral normalization.
    raw_text : str
        Original interaction text (truncated, audit trail).
    raw_text_hash : str
        SHA-256 hash of the ORIGINAL untruncated text.
    original_length : int
        Character count of the original untruncated text.
    source_node : str
        ID of the originating foreign agent.
    tick : int
        Simulation tick when this interaction was received.
    """
    semantic_features: np.ndarray
    relational_features: np.ndarray
    combined: np.ndarray
    raw_text: str = ""
    raw_text_hash: str = ""
    original_length: int = 0
    source_node: str = ""
    tick: int = 0


@dataclass
class Provenance:
    """Tracks the origin and transformation history of accepted information.

    Once information is accepted, the most important question becomes:
    "Why does the system believe this?" -- not just "What does it believe?"
    """
    source_node: str = ""
    source_type: str = "external"
    first_seen_tick: int = 0
    transformations: List[str] = field(default_factory=list)
    evaluator_version: str = "1.0"
    acceptance_tick: int = 0
    verification_count: int = 0


@dataclass
class LatentIntegrationRecord:
    """Audit trail linking a latent vector back to its origin.

    Created when a candidate is accepted and WHT-encoded.
    Enables the trace: attractor → latent → candidate → verification → source.

    Attributes
    ----------
    candidate_id : str
        Unique key for this candidate in the queue.
    source_node : str
        ID of the foreign agent that provided the information.
    accepted_tick : int
        Simulation tick when acceptance occurred.
    observation_hash : str
        SHA-256 hash of the original raw text (immutable reference).
    latent_checksum : str
        SHA-256 of the latent vector bytes (detects in-place mutation).
    wht_seed : int
        The seed used for WHT encoding (reproducibility).
    semantic_hash : str
        Hash of the semantic feature vector.
    relational_hash : str
        Hash of the relational feature vector.
    verification_count : int
        How many times this knowledge was re-verified.
    """
    candidate_id: str
    source_node: str
    accepted_tick: int
    observation_hash: str
    latent_checksum: str
    wht_seed: int = 42
    semantic_hash: str = ""
    relational_hash: str = ""
    verification_count: int = 0

    @staticmethod
    def from_candidate(candidate: "MemoryCandidate",
                       candidate_id: str,
                       latent: np.ndarray,
                       wht_seed: int = 42) -> "LatentIntegrationRecord":
        """Create a record from an accepted MemoryCandidate."""
        latent_bytes = latent.tobytes()
        latent_checksum = hashlib.sha256(latent_bytes).hexdigest()
        sem_hash = hashlib.sha256(
            candidate.spectrum.semantic_features.tobytes()).hexdigest()
        rel_hash = hashlib.sha256(
            candidate.spectrum.relational_features.tobytes()).hexdigest()
        return LatentIntegrationRecord(
            candidate_id=candidate_id,
            source_node=candidate.spectrum.source_node,
            accepted_tick=candidate.tick_resolved or 0,
            observation_hash=candidate.spectrum.raw_text_hash,
            latent_checksum=latent_checksum,
            wht_seed=wht_seed,
            semantic_hash=sem_hash,
            relational_hash=rel_hash,
            verification_count=candidate.provenance.verification_count,
        )


@dataclass
class EvaluationResult:
    """Output from the ForeignEvaluator.

    Attributes
    ----------
    alignment : float
        Cosine similarity to nearest existing attractor. Range [0, 1].
    novelty : float
        Distance to nearest training sample. Higher = more novel.
    risk : float
        Prediction error under perturbation. Higher = riskier.
    coherence : float
        Internal consistency of the interaction. Range [0, 1].
    recommendation : IntegrationDecision
        What the evaluator recommends.
    reasoning : str
        Human-readable explanation.
    """
    alignment: float = 0.0
    novelty: float = 0.0
    risk: float = 0.0
    coherence: float = 0.0
    recommendation: IntegrationDecision = IntegrationDecision.REJECT
    reasoning: str = ""


@dataclass
class MemoryCandidate:
    """An external interaction routed through the integration pipeline.

    Lifecycle:
        PENDING -> OBSERVED -> CANDIDATE -> ACCEPTED
                          -> REJECTED
                          -> STALE (timeout)

    External information should have different persistence classes.
    The confidence_decay_rate controls how quickly acceptance confidence
    fades without verification.
    """
    spectrum: InteractionSpectrum
    evaluation: EvaluationResult
    status: CandidateStatus = CandidateStatus.PENDING
    tick_created: int = 0
    tick_resolved: Optional[int] = None
    observation_count: int = 0
    evidence: List[EvaluationResult] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    confidence_decay_rate: float = 0.01  # per tick, how fast confidence fades
    last_verified_tick: int = 0

    # Post-acceptance latent encoding (None until accepted)
    latent_vector: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def is_alive(self) -> bool:
        return self.status in (CandidateStatus.PENDING, CandidateStatus.OBSERVED,
                               CandidateStatus.CANDIDATE)

    @property
    def average_alignment(self) -> float:
        if not self.evidence:
            return self.evaluation.alignment
        return float(np.mean([e.alignment for e in self.evidence]))

    @property
    def average_novelty(self) -> float:
        if not self.evidence:
            return self.evaluation.novelty
        return float(np.mean([e.novelty for e in self.evidence]))

    @property
    def average_risk(self) -> float:
        if not self.evidence:
            return self.evaluation.risk
        return float(np.mean([e.risk for e in self.evidence]))

    def add_evidence(self, evaluation: EvaluationResult) -> None:
        """Record an additional evaluation observation.

        Caps evidence list at 20 entries to prevent unbounded growth.
        Resets the verification timer.
        """
        self.evidence.append(evaluation)
        self.observation_count += 1
        if len(self.evidence) > 20:
            self.evidence = self.evidence[-20:]

    def current_confidence(self, tick: int) -> float:
        """Compute current confidence with time decay.

        External knowledge should have different persistence.
        Without verification, confidence decays exponentially.
        Verification (new evidence) resets the decay clock.
        """
        base = self.average_alignment
        ticks_since_verify = max(0, tick - self.last_verified_tick)
        decay = np.exp(-self.confidence_decay_rate * ticks_since_verify)
        return float(base * decay)


@dataclass
class CandidateQueueConfig:
    """Configuration for the candidate queue."""
    max_pending: int = 100
    max_observed: int = 50
    max_candidate: int = 20
    max_all: int = 10000
    max_rejected: int = 5000
    observation_threshold: int = 3
    stale_timeout_ticks: int = 1000
    min_alignment_for_candidate: float = 0.3
    max_risk_for_accept: float = 0.7
    min_observations_for_accept: int = 2


class CandidateQueue:
    """Routes external interactions through the integration pipeline.

    The queue implements a multi-stage filter:

    1. PENDING: New interactions arrive here
    2. OBSERVED: Interesting but uncertain — accumulate more samples
    3. CANDIDATE: Passed evaluation — ready for final confirmation
    4. ACCEPTED: Integrated into memory
    5. REJECTED: Failed evaluation
    6. STALE: Expired without resolution

    The OBSERVE state is the key biological insight: a lot of valuable
    information initially looks noisy. The system says "I don't trust this
    enough to integrate, but I want more samples."
    """

    def __init__(self, config: Optional[CandidateQueueConfig] = None):
        self.config = config or CandidateQueueConfig()
        self._pending: List[MemoryCandidate] = []
        self._observed: List[MemoryCandidate] = []
        self._candidates: List[MemoryCandidate] = []
        self._accepted: List[MemoryCandidate] = []
        self._rejected: List[MemoryCandidate] = []
        self._all: Dict[str, MemoryCandidate] = {}
        self._tick = 0

    def submit(self, spectrum: InteractionSpectrum,
               evaluation: EvaluationResult) -> MemoryCandidate:
        """Submit a new external interaction for evaluation.

        Parameters
        ----------
        spectrum : InteractionSpectrum
            The encoded interaction.
        evaluation : EvaluationResult
            Initial evaluation result.

        Returns
        -------
        MemoryCandidate
            The newly created candidate.
        """
        candidate = MemoryCandidate(
            spectrum=spectrum,
            evaluation=evaluation,
            status=CandidateStatus.PENDING,
            tick_created=self._tick,
        )
        candidate.add_evidence(evaluation)

        key = f"{spectrum.source_node}:{spectrum.tick}:{len(self._all)}"
        self._all[key] = candidate
        self._pending.append(candidate)

        # Prune _all if too large (keep only live candidates)
        if len(self._all) > self.config.max_all:
            self._prune_all()

        # Prune _rejected if too large
        if len(self._rejected) > self.config.max_rejected:
            self._rejected = self._rejected[-self.config.max_rejected:]

        # Auto-route based on initial evaluation
        self._route(candidate)
        return candidate

    def _prune_all(self) -> None:
        """Remove terminal-status entries from _all to bound memory."""
        terminal = {CandidateStatus.REJECTED, CandidateStatus.STALE,
                    CandidateStatus.ACCEPTED}
        keys_to_remove = [
            k for k, c in self._all.items()
            if c.status in terminal
        ]
        for k in keys_to_remove:
            del self._all[k]

    def _route(self, candidate: MemoryCandidate) -> None:
        """Route a candidate based on its current evaluation."""
        rec = candidate.evaluation.recommendation

        if rec == IntegrationDecision.REJECT:
            self._move(candidate, CandidateStatus.REJECTED, self._rejected)
        elif rec == IntegrationDecision.OBSERVE:
            if len(self._observed) < self.config.max_observed:
                self._move(candidate, CandidateStatus.OBSERVED, self._observed)
            else:
                self._move(candidate, CandidateStatus.REJECTED, self._rejected)
        elif rec == IntegrationDecision.CANDIDATE:
            if len(self._candidates) < self.config.max_candidate:
                self._move(candidate, CandidateStatus.CANDIDATE, self._candidates)
            else:
                # Overflow → observe instead
                self._move(candidate, CandidateStatus.OBSERVED, self._observed)
        elif rec == IntegrationDecision.ACCEPT:
            self._accept(candidate)

    def _move(self, candidate: MemoryCandidate, new_status: CandidateStatus,
              target_list: List[MemoryCandidate]) -> None:
        """Move a candidate to a new status and list."""
        # Remove from current list
        for lst in [self._pending, self._observed, self._candidates]:
            if candidate in lst:
                lst.remove(candidate)
                break

        candidate.status = new_status
        target_list.append(candidate)

    def _accept(self, candidate: MemoryCandidate) -> None:
        """Accept a candidate into memory."""
        for lst in [self._pending, self._observed, self._candidates]:
            if candidate in lst:
                lst.remove(candidate)
                break
        candidate.status = CandidateStatus.ACCEPTED
        candidate.tick_resolved = self._tick
        self._accepted.append(candidate)

    def observe(self, key: str, evaluation: EvaluationResult) -> None:
        """Add an observation to an existing candidate.

        Parameters
        ----------
        key : str
            The candidate key (from submit()).
        evaluation : EvaluationResult
            New evaluation observation.
        """
        candidate = self._all.get(key)
        if candidate is None or not candidate.is_alive:
            return

        candidate.add_evidence(evaluation)

        # Check if we have enough observations to upgrade
        if (candidate.status == CandidateStatus.OBSERVED and
                candidate.observation_count >= self.config.observation_threshold):
            avg_alignment = candidate.average_alignment
            avg_risk = candidate.average_risk

            if avg_alignment >= self.config.min_alignment_for_candidate:
                if avg_risk <= self.config.max_risk_for_accept:
                    if (candidate.observation_count >=
                            self.config.min_observations_for_accept):
                        self._accept(candidate)
                        return
                # Not safe enough for accept, but good enough for candidate
                if len(self._candidates) < self.config.max_candidate:
                    self._move(candidate, CandidateStatus.CANDIDATE,
                               self._candidates)

    def tick(self) -> None:
        """Advance the simulation clock. Checks for stale candidates."""
        self._tick += 1
        stale = []
        for c in self._observed:
            if self._tick - c.tick_created > self.config.stale_timeout_ticks:
                stale.append(c)
        for c in stale:
            self._move(c, CandidateStatus.STALE, self._rejected)

    def get_accepted(self) -> List[MemoryCandidate]:
        """Return all accepted candidates (ready for memory integration)."""
        return list(self._accepted)

    def pop_accepted(self) -> List[MemoryCandidate]:
        """Return and clear accepted candidates."""
        result = list(self._accepted)
        self._accepted.clear()
        return result

    @property
    def stats(self) -> dict:
        """Current queue statistics."""
        return {
            "pending": len(self._pending),
            "observed": len(self._observed),
            "candidates": len(self._candidates),
            "accepted": len(self._accepted),
            "rejected": len(self._rejected),
            "total": len(self._all),
            "tick": self._tick,
        }
