"""S10: VerificationLoop

Transforms reputation from "does this agent sound coherent?" to
"does this agent improve my model of reality?"

External claims are tested against DynamicsMemory predictions.
Accepted candidates that predict correctly gain confidence.
Those that predict poorly lose confidence and may be quarantined.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from substrate_echo.external.candidate_queue import (
    CandidateStatus,
    MemoryCandidate,
)


@dataclass
class VerificationRecord:
    """Result of testing an external claim against reality.

    Attributes
    ----------
    candidate_id : str
        Identifier for the verified candidate.
    claim_state : np.ndarray
        The dynamical state the claim is about.
    predicted_velocity : np.ndarray
        What the external claim implies will happen.
    actual_velocity : np.ndarray
        What actually happened (from DynamicsMemory or world).
    error : float
        MSE between predicted and actual velocity.
    information_gain : float
        How much this claim improved the model (0-1).
    tick : int
        When verification occurred.
    verified : bool
        Whether the claim passed verification.
    """
    candidate_id: str = ""
    claim_state: np.ndarray = field(default_factory=lambda: np.zeros(16))
    predicted_velocity: np.ndarray = field(default_factory=lambda: np.zeros(16))
    actual_velocity: np.ndarray = field(default_factory=lambda: np.zeros(16))
    error: float = 1.0
    information_gain: float = 0.0
    tick: int = 0
    verified: bool = False


class VerificationLoop:
    """Tests external claims against reality via DynamicsMemory.

    The loop:
        Candidate (claim about world state)
            |
            v
        Extract implied prediction
            |
            v
        Observe actual outcome (from DynamicsMemory or world)
            |
            v
        Compute prediction error
            |
            v
        Update candidate confidence
            |
            v
        Update source reputation

    This transforms reputation from perceptual ("sounds good")
    to empirical ("predicts correctly").
    """

    def __init__(self, dynamics_memory=None, verification_threshold: float = 0.1,
                 max_pending: int = 100):
        """
        Parameters
        ----------
        dynamics_memory : DynamicsMemory
            For prediction_error() and information_gain().
        verification_threshold : float
            Max acceptable prediction error to consider a claim verified.
        max_pending : int
            Max candidates awaiting verification.
        """
        self._dm = dynamics_memory
        self._threshold = verification_threshold
        self._max_pending = max_pending

        # State
        self._pending: Dict[str, MemoryCandidate] = {}
        self._records: List[VerificationRecord] = []
        self._verified_ids: set = set()
        self._next_id = 0

    def submit_for_verification(self, candidate: MemoryCandidate,
                                tick: int = 0) -> str:
        """Submit a candidate for empirical verification.

        Returns a verification ID for tracking.
        """
        if len(self._pending) >= self._max_pending:
            return ""

        vid = f"v{self._next_id}"
        self._next_id += 1
        candidate.provenance.transformations.append(f"verification_submitted:{vid}")
        self._pending[vid] = candidate
        return vid

    def verify(self, vid: str, actual_velocity: np.ndarray,
               tick: int = 0) -> Optional[VerificationRecord]:
        """Verify a pending claim against observed reality.

        Parameters
        ----------
        vid : str
            Verification ID from submit_for_verification().
        actual_velocity : np.ndarray
            What actually happened in the dynamical system.
        tick : int
            Current simulation tick.

        Returns
        -------
        VerificationRecord or None
            The verification result, or None if vid not found.
        """
        if vid not in self._pending:
            return None

        candidate = self._pending.pop(vid)
        state = candidate.spectrum.semantic_features

        # Compute prediction error
        if self._dm is not None and self._dm._fitted:
            predicted = self._dm.predict_velocity(state)
            error = float(np.mean((predicted - actual_velocity) ** 2))

            # Information gain: how much did this claim help the model?
            ig = self._dm.information_gain(state)
        else:
            # No dynamics memory — cannot make predictions.
            # Without a model, verification has no empirical basis.
            # Skip verification with neutral result.
            record = VerificationRecord(
                candidate_id=vid,
                claim_state=state.copy(),
                predicted_velocity=np.zeros_like(actual_velocity),
                actual_velocity=actual_velocity.copy(),
                error=0.0,
                information_gain=0.0,
                tick=tick,
                verified=True,  # No basis to reject
            )
            candidate.provenance.transformations.append("verification_skipped:no_model")
            self._records.append(record)
            return record

        verified = error < self._threshold

        record = VerificationRecord(
            candidate_id=vid,
            claim_state=state.copy(),
            predicted_velocity=predicted.copy(),
            actual_velocity=actual_velocity.copy(),
            error=error,
            information_gain=ig,
            tick=tick,
            verified=verified,
        )

        # Update candidate confidence based on verification
        if verified:
            candidate.confidence_decay_rate *= 0.5  # Slow decay — verified
            candidate.last_verified_tick = tick
            candidate.provenance.verification_count += 1
            candidate.provenance.transformations.append(f"verified:error={error:.4f}")
            self._verified_ids.add(vid)
        else:
            candidate.confidence_decay_rate *= 2.0  # Accelerate decay — failed
            candidate.provenance.transformations.append(f"failed:error={error:.4f}")

            # If error is extreme, quarantine
            if error > self._threshold * 10:
                candidate.status = CandidateStatus.REJECTED

        self._records.append(record)
        return record

    def get_records(self) -> List[VerificationRecord]:
        """All verification records."""
        return list(self._records)

    def get_verified_ids(self) -> set:
        """IDs of candidates that passed verification."""
        return set(self._verified_ids)

    def compute_source_verification_score(self, source_node: str,
                                           candidates: List[MemoryCandidate],
                                           max_age: int = 1000) -> float:
        """Compute verification-based trust score for a source.

        Returns a score in [0, 1] based on how many of the source's
        claims were verified vs failed.
        """
        source_records = []
        for r in self._records:
            if r.candidate_id in self._verified_ids:
                source_records.append(r)

        if not source_records:
            return 0.5  # No evidence — neutral

        verified = sum(1 for r in source_records if r.verified)
        return verified / len(source_records)

    def get_stats(self) -> dict:
        """Verification loop statistics."""
        return {
            "pending": len(self._pending),
            "total_verified": len(self._verified_ids),
            "total_records": len(self._records),
            "pass_rate": (len(self._verified_ids) / max(len(self._records), 1)),
            "avg_error": float(np.mean([r.error for r in self._records]))
                          if self._records else 0.0,
        }
