"""Hypothesis Objects — Representing uncertainty and competing explanations.

Hypotheses are objects that represent possible explanations for observations.
They maintain their own evidence base and confidence, allowing multiple
competing explanations to coexist until evidence distinguishes them.

Architecture:
    Feature Set
         |
    Hypothesis Generation
         |
    [Hypothesis A, Hypothesis B, ...]
         |
    Evidence Accumulation
         |
    Confidence Update
         |
    Hypothesis Selection / Pruning
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import numpy as np
import time


class HypothesisStatus(Enum):
    """Lifecycle status of a hypothesis."""
    CANDIDATE = "candidate"       # Newly generated
    ACTIVE = "active"             # Under consideration
    SUPPORTED = "supported"       # Evidence supports it
    CONTRADICTED = "contradicted" # Evidence contradicts it
    CONFIRMED = "confirmed"       # Strongly validated
    REJECTED = "rejected"         # Discarded
    ARCHIVED = "archived"         # Retained for reference


@dataclass
class Evidence:
    """A piece of evidence that supports or contradicts a hypothesis."""
    description: str
    supports: bool                 # True = supports, False = contradicts
    strength: float                # How strong is this evidence [0, 1]
    source: str = ""               # Where this evidence came from
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """A prediction derived from a hypothesis.
    
    If the hypothesis is true, what should we expect to observe?
    """
    description: str
    expected_features: Dict[str, Any] = field(default_factory=dict)
    time_horizon: int = 10         # How many steps until prediction should be verified
    confidence: float = 0.5
    verified: Optional[bool] = None  # None = not yet checked
    actual_outcome: Optional[Dict[str, Any]] = None


@dataclass
class Hypothesis:
    """A possible explanation for observed phenomena.
    
    Hypotheses are not binary true/false. They accumulate evidence
    and maintain confidence levels that reflect uncertainty.
    """
    id: str
    description: str
    
    # Evidence
    supporting_evidence: List[Evidence] = field(default_factory=list)
    contradicting_evidence: List[Evidence] = field(default_factory=list)
    
    # Confidence model
    confidence: float = 0.5        # Prior confidence [0, 1]
    confidence_history: List[float] = field(default_factory=list)
    
    # Predictions generated from this hypothesis
    predictions: List[Prediction] = field(default_factory=list)
    
    # Lifecycle
    status: HypothesisStatus = HypothesisStatus.CANDIDATE
    created_at: float = 0.0
    updated_at: float = 0.0
    
    # Metadata
    source: str = ""               # What generated this hypothesis
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_evidence(self, evidence: Evidence):
        """Add evidence and update confidence."""
        if evidence.supports:
            self.supporting_evidence.append(evidence)
        else:
            self.contradicting_evidence.append(evidence)
        
        self._update_confidence()
        self.updated_at = time.time()
    
    def _update_confidence(self):
        """Update confidence based on evidence balance."""
        total_evidence = len(self.supporting_evidence) + len(self.contradicting_evidence)
        if total_evidence == 0:
            return
        
        # Weighted evidence balance
        support_weight = sum(e.strength for e in self.supporting_evidence)
        contra_weight = sum(e.strength for e in self.contradicting_evidence)
        
        # Bayesian-inspired update
        if support_weight + contra_weight > 0:
            evidence_ratio = support_weight / (support_weight + contra_weight)
        else:
            evidence_ratio = 0.5
        
        # Smooth update toward evidence ratio
        alpha = 0.3  # Learning rate
        self.confidence = self.confidence * (1 - alpha) + evidence_ratio * alpha
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        self.confidence_history.append(self.confidence)
    
    def get_support_score(self) -> float:
        """Get net support score (positive = more support)."""
        support = sum(e.strength for e in self.supporting_evidence)
        contra = sum(e.strength for e in self.contradicting_evidence)
        return support - contra
    
    def get_evidence_summary(self) -> Dict[str, Any]:
        """Get summary of evidence."""
        return {
            "supporting": len(self.supporting_evidence),
            "contradicting": len(self.contradicting_evidence),
            "support_strength": sum(e.strength for e in self.supporting_evidence),
            "contra_strength": sum(e.strength for e in self.contradicting_evidence),
            "net_support": self.get_support_score(),
        }
    
    def add_prediction(self, prediction: Prediction):
        """Add a prediction derived from this hypothesis."""
        self.predictions.append(prediction)
    
    def verify_predictions(self, actual_outcomes: List[Dict[str, Any]]) -> int:
        """Verify predictions against actual outcomes.
        
        Returns number of predictions verified.
        """
        verified_count = 0
        
        for prediction in self.predictions:
            if prediction.verified is not None:
                continue  # Already verified
            
            # Check if outcome matches prediction
            match = self._check_prediction_match(prediction, actual_outcomes)
            if match is not None:
                prediction.verified = match
                prediction.actual_outcome = actual_outcomes[0] if actual_outcomes else None
                
                # Add evidence based on verification
                evidence = Evidence(
                    description=f"Prediction '{prediction.description}' {'confirmed' if match else 'failed'}",
                    supports=match,
                    strength=0.7 if match else 0.8,  # Failures are stronger evidence
                    source="prediction_verification",
                    timestamp=time.time(),
                )
                self.add_evidence(evidence)
                
                if match:
                    verified_count += 1
        
        return verified_count
    
    def _check_prediction_match(self, prediction: Prediction,
                                actual_outcomes: List[Dict[str, Any]]) -> Optional[bool]:
        """Check if prediction matches outcomes."""
        if not actual_outcomes:
            return None
        
        # Simple feature matching
        for outcome in actual_outcomes:
            matches = 0
            total = 0
            
            for key, expected in prediction.expected_features.items():
                if key in outcome:
                    total += 1
                    actual = outcome[key]
                    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                        # Numeric comparison with tolerance
                        if abs(expected - actual) / max(abs(expected), 1e-6) < 0.2:
                            matches += 1
                    elif expected == actual:
                        matches += 1
            
            if total > 0 and matches / total >= 0.5:
                return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "description": self.description,
            "confidence": self.confidence,
            "status": self.status.value,
            "supporting": len(self.supporting_evidence),
            "contradicting": len(self.contradicting_evidence),
            "predictions": len(self.predictions),
            "predictions_verified": sum(1 for p in self.predictions if p.verified is True),
            "predictions_failed": sum(1 for p in self.predictions if p.verified is False),
        }


class HypothesisSpace:
    """Manages multiple competing hypotheses.
    
    The hypothesis space maintains a population of explanations,
    allowing evidence to accumulate and hypotheses to compete.
    """
    
    def __init__(self, max_hypotheses: int = 20):
        self._hypotheses: Dict[str, Hypothesis] = {}
        self._max_hypotheses = max_hypotheses
        self._next_id = 0
        self._generation_count = 0
    
    def generate(self, description: str, confidence: float = 0.5,
                 source: str = "", **kwargs) -> Hypothesis:
        """Generate a new hypothesis."""
        if len(self._hypotheses) >= self._max_hypotheses:
            self._prune_weakest()
        
        hypothesis_id = f"H{self._next_id:04d}"
        self._next_id += 1
        
        hypothesis = Hypothesis(
            id=hypothesis_id,
            description=description,
            confidence=confidence,
            source=source,
            created_at=time.time(),
            updated_at=time.time(),
            **kwargs,
        )
        
        self._hypotheses[hypothesis_id] = hypothesis
        self._generation_count += 1
        
        return hypothesis
    
    def get(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get hypothesis by ID."""
        return self._hypotheses.get(hypothesis_id)
    
    def get_active(self) -> List[Hypothesis]:
        """Get all active hypotheses."""
        return [
            h for h in self._hypotheses.values()
            if h.status in (HypothesisStatus.ACTIVE, HypothesisStatus.CANDIDATE)
        ]
    
    def get_by_confidence(self, min_confidence: float = 0.5) -> List[Hypothesis]:
        """Get hypotheses above confidence threshold."""
        return [
            h for h in self._hypotheses.values()
            if h.confidence >= min_confidence
        ]
    
    def get_best(self) -> Optional[Hypothesis]:
        """Get the highest confidence hypothesis."""
        active = self.get_active()
        if not active:
            return None
        return max(active, key=lambda h: h.confidence)
    
    def add_evidence(self, hypothesis_id: str, evidence: Evidence) -> bool:
        """Add evidence to a hypothesis."""
        hypothesis = self._hypotheses.get(hypothesis_id)
        if not hypothesis:
            return False
        
        hypothesis.add_evidence(evidence)
        
        # Update status based on confidence
        if hypothesis.confidence > 0.8:
            hypothesis.status = HypothesisStatus.CONFIRMED
        elif hypothesis.confidence > 0.6:
            hypothesis.status = HypothesisStatus.SUPPORTED
        elif hypothesis.confidence < 0.2:
            hypothesis.status = HypothesisStatus.REJECTED
        
        return True
    
    def prune_rejected(self):
        """Remove rejected hypotheses."""
        to_remove = [
            hid for hid, h in self._hypotheses.items()
            if h.status == HypothesisStatus.REJECTED
        ]
        for hid in to_remove:
            del self._hypotheses[hid]
    
    def _prune_weakest(self):
        """Remove weakest hypothesis to make room."""
        if not self._hypotheses:
            return
        
        # Don't prune confirmed hypotheses
        prunable = [
            h for h in self._hypotheses.values()
            if h.status != HypothesisStatus.CONFIRMED
        ]
        
        if not prunable:
            return
        
        weakest = min(prunable, key=lambda h: h.confidence)
        del self._hypotheses[weakest.id]
    
    def get_competition(self) -> List[Dict[str, Any]]:
        """Get competition status between hypotheses."""
        active = self.get_active()
        return [
            {
                "id": h.id,
                "description": h.description,
                "confidence": h.confidence,
                "status": h.status.value,
                "evidence": h.get_evidence_summary(),
            }
            for h in sorted(active, key=lambda h: h.confidence, reverse=True)
        ]
    
    def tick(self):
        """Update hypothesis space (decay inactive hypotheses)."""
        for h in self._hypotheses.values():
            # Decay confidence slightly over time without evidence
            time_since_update = time.time() - h.updated_at
            if time_since_update > 100:  # 100 seconds
                decay = 0.001 * (time_since_update / 100)
                h.confidence = max(0.0, h.confidence - decay)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total": len(self._hypotheses),
            "active": len(self.get_active()),
            "confirmed": sum(1 for h in self._hypotheses.values()
                           if h.status == HypothesisStatus.CONFIRMED),
            "hypotheses": [h.to_dict() for h in self._hypotheses.values()],
        }