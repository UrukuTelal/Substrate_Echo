"""Prediction Layer — Generating and verifying expectations.

The prediction layer generates expectations from hypotheses and
verifies them against actual outcomes. This creates a feedback
loop that updates beliefs based on reality.

Architecture:
    Hypothesis
         |
    Prediction Generation
         |
    Expected Outcome
         |
    Compare with Reality
         |
    Surprise / Confirmation
         |
    Belief Update
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import numpy as np
import time


class PredictionType(Enum):
    """Types of predictions."""
    FEATURE = "feature"           # Expected feature value
    PATTERN = "pattern"           # Expected pattern occurrence
    TEMPORAL = "temporal"         # Expected timing
    RELATIONAL = "relational"   # Expected relationship
    BEHAVIORAL = "behavioral"   # Expected behavior


class PredictionStatus(Enum):
    """Status of a prediction."""
    PENDING = "pending"           # Not yet verified
    CONFIRMED = "confirmed"       # Reality matched expectation
    FAILED = "failed"             # Reality contradicted expectation
    PARTIAL = "partial"           # Partially matched
    EXPIRED = "expired"           # Too old to verify


@dataclass
class PredictionRecord:
    """A prediction with its verification outcome."""
    id: str
    description: str
    prediction_type: PredictionType
    
    # What was expected
    expected: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    
    # When it was made and when it should be verified
    created_at: float = 0.0
    verify_at: int = 0           # Step at which to verify
    
    # What actually happened
    actual: Optional[Dict[str, Any]] = None
    status: PredictionStatus = PredictionStatus.PENDING
    
    # Metadata
    source_hypothesis: str = ""   # Which hypothesis generated this
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def verify(self, actual_outcome: Dict[str, Any],
               tolerance: float = 0.2) -> PredictionStatus:
        """Verify prediction against actual outcome.
        
        Returns the verification status.
        """
        self.actual = actual_outcome
        
        matches = 0
        total = 0
        
        for key, expected_val in self.expected.items():
            if key in actual_outcome:
                total += 1
                actual_val = actual_outcome[key]
                
                if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
                    # Numeric comparison with tolerance
                    if abs(expected_val - actual_val) / max(abs(expected_val), 1e-6) < tolerance:
                        matches += 1
                elif expected_val == actual_val:
                    matches += 1
        
        if total == 0:
            self.status = PredictionStatus.EXPIRED
        elif matches == total:
            self.status = PredictionStatus.CONFIRMED
        elif matches > 0:
            self.status = PredictionStatus.PARTIAL
        else:
            self.status = PredictionStatus.FAILED
        
        return self.status
    
    def get_accuracy(self) -> float:
        """Get accuracy of this prediction."""
        if self.actual is None:
            return 0.0
        
        matches = 0
        total = 0
        
        for key, expected_val in self.expected.items():
            if key in self.actual:
                total += 1
                actual_val = self.actual[key]
                
                if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
                    if abs(expected_val - actual_val) / max(abs(expected_val), 1e-6) < 0.2:
                        matches += 1
                elif expected_val == actual_val:
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "type": self.prediction_type.value,
            "confidence": self.confidence,
            "status": self.status.value,
            "accuracy": self.get_accuracy(),
            "source_hypothesis": self.source_hypothesis,
        }


class PredictionEngine:
    """Generates predictions from hypotheses and features.
    
    The prediction engine takes hypotheses and current features
    and generates expectations about what should happen next.
    """
    
    def __init__(self):
        self._prediction_count = 0
        self._generators: Dict[str, Callable] = {}
    
    def register_generator(self, name: str, generator: Callable):
        """Register a prediction generator function."""
        self._generators[name] = generator
    
    def generate_from_hypothesis(self, hypothesis: 'Hypothesis',
                                  features: 'FeatureSet',
                                  time_horizon: int = 10) -> List[PredictionRecord]:
        """Generate predictions from a hypothesis."""
        predictions = []
        
        # Generate predictions based on hypothesis description
        # This is a simplified version - real implementation would
        # parse hypothesis semantics
        
        prediction = PredictionRecord(
            id=f"P{self._prediction_count:06d}",
            description=f"Prediction from: {hypothesis.description[:50]}",
            prediction_type=PredictionType.FEATURE,
            expected=self._infer_expected_features(hypothesis, features),
            confidence=hypothesis.confidence,
            created_at=time.time(),
            verify_at=time_horizon,
            source_hypothesis=hypothesis.id,
        )
        
        predictions.append(prediction)
        self._prediction_count += 1
        
        return predictions
    
    def _infer_expected_features(self, hypothesis: 'Hypothesis',
                                  features: 'FeatureSet') -> Dict[str, Any]:
        """Infer expected features from hypothesis."""
        # Simplified inference - would be more sophisticated in practice
        expected = {}
        
        # Look for patterns in hypothesis description
        desc_lower = hypothesis.description.lower()
        
        if "increase" in desc_lower:
            for name, feature in features.features.items():
                if isinstance(feature.value, (int, float)):
                    expected[name] = feature.value * 1.1
        
        elif "decrease" in desc_lower:
            for name, feature in features.features.items():
                if isinstance(feature.value, (int, float)):
                    expected[name] = feature.value * 0.9
        
        elif "stable" in desc_lower:
            for name, feature in features.features.items():
                if isinstance(feature.value, (int, float)):
                    expected[name] = feature.value
        
        return expected
    
    def generate_from_features(self, features: 'FeatureSet',
                                trend: Optional[str] = None) -> List[PredictionRecord]:
        """Generate predictions from current features and trends."""
        predictions = []
        
        if trend == "increasing":
            for name, feature in features.features.items():
                if isinstance(feature.value, (int, float)):
                    predictions.append(PredictionRecord(
                        id=f"P{self._prediction_count:06d}",
                        description=f"Expect {name} to increase",
                        prediction_type=PredictionType.TEMPORAL,
                        expected={name: feature.value * 1.05},
                        confidence=0.6,
                        created_at=time.time(),
                        verify_at=10,
                    ))
                    self._prediction_count += 1
        
        elif trend == "decreasing":
            for name, feature in features.features.items():
                if isinstance(feature.value, (int, float)):
                    predictions.append(PredictionRecord(
                        id=f"P{self._prediction_count:06d}",
                        description=f"Expect {name} to decrease",
                        prediction_type=PredictionType.TEMPORAL,
                        expected={name: feature.value * 0.95},
                        confidence=0.6,
                        created_at=time.time(),
                        verify_at=10,
                    ))
                    self._prediction_count += 1
        
        return predictions


class PredictionMemory:
    """Memory of past predictions and their outcomes.
    
    Stores prediction history for accuracy tracking and
    learning from prediction errors.
    """
    
    def __init__(self, max_size: int = 500):
        self._predictions: List[PredictionRecord] = []
        self._max_size = max_size
    
    def record(self, prediction: PredictionRecord):
        """Record a prediction."""
        self._predictions.append(prediction)
        if len(self._predictions) > self._max_size:
            self._predictions.pop(0)
    
    def get_pending(self, current_step: int) -> List[PredictionRecord]:
        """Get predictions that need verification."""
        return [
            p for p in self._predictions
            if p.status == PredictionStatus.PENDING and p.verify_at <= current_step
        ]
    
    def get_confirmed(self) -> List[PredictionRecord]:
        """Get confirmed predictions."""
        return [
            p for p in self._predictions
            if p.status == PredictionStatus.CONFIRMED
        ]
    
    def get_failed(self) -> List[PredictionRecord]:
        """Get failed predictions."""
        return [
            p for p in self._predictions
            if p.status == PredictionStatus.FAILED
        ]
    
    def get_accuracy_stats(self) -> Dict[str, float]:
        """Get overall accuracy statistics."""
        verified = [
            p for p in self._predictions
            if p.status in (PredictionStatus.CONFIRMED, PredictionStatus.FAILED,
                           PredictionStatus.PARTIAL)
        ]
        
        if not verified:
            return {"accuracy": 0.0, "confirmed": 0, "failed": 0, "total": 0}
        
        confirmed = sum(1 for p in verified if p.status == PredictionStatus.CONFIRMED)
        failed = sum(1 for p in verified if p.status == PredictionStatus.FAILED)
        partial = sum(1 for p in verified if p.status == PredictionStatus.PARTIAL)
        
        # Weighted accuracy (partial counts as 0.5)
        accuracy = (confirmed + partial * 0.5) / len(verified)
        
        return {
            "accuracy": accuracy,
            "confirmed": confirmed,
            "failed": failed,
            "partial": partial,
            "total": len(verified),
            "pending": sum(1 for p in self._predictions if p.status == PredictionStatus.PENDING),
        }
    
    def get_recent_accuracy(self, n: int = 50) -> float:
        """Get accuracy of recent predictions."""
        recent = [
            p for p in self._predictions[-n:]
            if p.status in (PredictionStatus.CONFIRMED, PredictionStatus.FAILED,
                           PredictionStatus.PARTIAL)
        ]
        
        if not recent:
            return 0.0
        
        confirmed = sum(1 for p in recent if p.status == PredictionStatus.CONFIRMED)
        partial = sum(1 for p in recent if p.status == PredictionStatus.PARTIAL)
        
        return (confirmed + partial * 0.5) / len(recent)
    
    def detect_systematic_errors(self) -> List[Dict[str, Any]]:
        """Detect systematic prediction errors."""
        failed = self.get_failed()
        
        # Group by source hypothesis
        errors_by_source = {}
        for p in failed:
            source = p.source_hypothesis or "unknown"
            if source not in errors_by_source:
                errors_by_source[source] = []
            errors_by_source[source].append(p)
        
        # Find sources with high failure rates
        systematic_errors = []
        for source, errors in errors_by_source.items():
            if len(errors) >= 3:  # At least 3 failures
                systematic_errors.append({
                    "source": source,
                    "failure_count": len(errors),
                    "description": f"Hypothesis {source} has systematic prediction failures",
                })
        
        return systematic_errors