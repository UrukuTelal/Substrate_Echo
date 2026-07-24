"""Development Record — History of becoming.

The development record captures the agent's learning journey,
not just current state. Failed predictions are valuable data.
The agent should retain: "I believed X, predicted Y, reality
produced Z, therefore confidence changed."

Architecture:
    Observations
         |
    Emerging Patterns
         |
    Hypotheses
         |
    Predictions
         |
    Outcomes
         |
    Rule Learning
         |
    Adaptations
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time
import json
import numpy as np


class EventType(Enum):
    """Types of development events."""
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    PREDICTION = "prediction"
    OUTCOME = "outcome"
    RULE_LEARNED = "rule_learned"
    RULE_FAILED = "rule_failed"
    ADAPTATION = "adaptation"
    MILESTONE = "milestone"
    NOTE = "note"


@dataclass
class DevelopmentEvent:
    """A single event in the agent's development."""
    event_id: int
    event_type: EventType
    timestamp: float
    step: int
    
    # Event content
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Relationships
    parent_event: Optional[int] = None  # What triggered this event
    child_events: List[int] = field(default_factory=list)
    
    # Impact
    confidence_change: float = 0.0  # How this changed beliefs
    significance: float = 0.5       # How important is this event
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.event_type.value,
            "timestamp": self.timestamp,
            "step": self.step,
            "description": self.description,
            "confidence_change": self.confidence_change,
            "significance": self.significance,
        }


@dataclass
class BeliefSnapshot:
    """Snapshot of beliefs at a point in time."""
    step: int
    timestamp: float
    
    # Current beliefs
    hypotheses: Dict[str, float] = field(default_factory=dict)  # id -> confidence
    rules: Dict[str, float] = field(default_factory=dict)       # id -> confidence
    
    # Overall metrics
    total_confidence: float = 0.5
    prediction_accuracy: float = 0.0
    learning_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "hypotheses": self.hypotheses,
            "rules": self.rules,
            "total_confidence": self.total_confidence,
            "prediction_accuracy": self.prediction_accuracy,
        }


class DevelopmentRecord:
    """Persistent history of agent development.
    
    Captures the journey of learning, not just current state.
    Failed predictions are valuable data that should be retained.
    """
    
    def __init__(self, max_events: int = 10000):
        self._events: List[DevelopmentEvent] = []
        self._snapshots: List[BeliefSnapshot] = []
        self._max_events = max_events
        self._event_count = 0
        self._current_step = 0
    
    def record_event(self, event_type: EventType, description: str,
                     step: int, details: Optional[Dict[str, Any]] = None,
                     parent_event: Optional[int] = None,
                     confidence_change: float = 0.0,
                     significance: float = 0.5) -> DevelopmentEvent:
        """Record a development event."""
        event = DevelopmentEvent(
            event_id=self._event_count,
            event_type=event_type,
            timestamp=time.time(),
            step=step,
            description=description,
            details=details or {},
            parent_event=parent_event,
            confidence_change=confidence_change,
            significance=significance,
        )
        
        self._events.append(event)
        self._event_count += 1
        
        if parent_event is not None:
            for e in self._events:
                if e.event_id == parent_event:
                    e.child_events.append(event.event_id)
                    break
        
        if len(self._events) > self._max_events:
            self._events.pop(0)
        
        return event
    
    def record_observation(self, step: int, description: str,
                           details: Optional[Dict[str, Any]] = None) -> DevelopmentEvent:
        """Record an observation event."""
        return self.record_event(
            EventType.OBSERVATION, description, step, details,
            significance=0.3,
        )
    
    def record_hypothesis(self, step: int, description: str,
                          hypothesis_id: str, confidence: float) -> DevelopmentEvent:
        """Record hypothesis generation."""
        return self.record_event(
            EventType.HYPOTHESIS, description, step,
            details={"hypothesis_id": hypothesis_id, "confidence": confidence},
            confidence_change=confidence - 0.5,
            significance=0.5,
        )
    
    def record_prediction(self, step: int, description: str,
                          prediction_id: str, hypothesis_id: str) -> DevelopmentEvent:
        """Record prediction generation."""
        return self.record_event(
            EventType.PREDICTION, description, step,
            details={"prediction_id": prediction_id, "hypothesis_id": hypothesis_id},
            significance=0.4,
        )
    
    def record_outcome(self, step: int, prediction_id: str,
                       success: bool, actual: Dict[str, Any],
                       expected: Dict[str, Any]) -> DevelopmentEvent:
        """Record prediction outcome (success or failure)."""
        # This is the key method - failures are valuable data
        description = f"Prediction {prediction_id}: {'CONFIRMED' if success else 'FAILED'}"
        
        return self.record_event(
            EventType.OUTCOME, description, step,
            details={
                "prediction_id": prediction_id,
                "success": success,
                "actual": actual,
                "expected": expected,
            },
            confidence_change=0.1 if success else -0.2,
            significance=0.7 if not success else 0.5,  # Failures are more significant
        )
    
    def record_rule_learned(self, step: int, rule_id: str,
                            description: str) -> DevelopmentEvent:
        """Record a learned rule."""
        return self.record_event(
            EventType.RULE_LEARNED, description, step,
            details={"rule_id": rule_id},
            confidence_change=0.15,
            significance=0.8,
        )
    
    def record_adaptation(self, step: int, description: str,
                          details: Optional[Dict[str, Any]] = None) -> DevelopmentEvent:
        """Record an adaptation (change in strategy or belief)."""
        return self.record_event(
            EventType.ADAPTATION, description, step, details,
            significance=0.6,
        )
    
    def take_snapshot(self, step: int, hypotheses: Dict[str, float],
                      rules: Dict[str, float],
                      prediction_accuracy: float = 0.0) -> BeliefSnapshot:
        """Take a snapshot of current beliefs."""
        snapshot = BeliefSnapshot(
            step=step,
            timestamp=time.time(),
            hypotheses=hypotheses,
            rules=rules,
            total_confidence=np.mean(list(hypotheses.values()) + [0.5]) if hypotheses else 0.5,
            prediction_accuracy=prediction_accuracy,
        )
        self._snapshots.append(snapshot)
        return snapshot
    
    def get_events(self, event_type: Optional[EventType] = None,
                   since_step: Optional[int] = None) -> List[DevelopmentEvent]:
        """Get events, optionally filtered."""
        events = self._events
        
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        
        if since_step is not None:
            events = [e for e in events if e.step >= since_step]
        
        return events
    
    def get_failed_predictions(self) -> List[DevelopmentEvent]:
        """Get all failed predictions (valuable data)."""
        return [
            e for e in self._events
            if e.event_type == EventType.OUTCOME
            and e.details.get("success") is False
        ]
    
    def get_learning_trajectory(self) -> List[Dict[str, Any]]:
        """Get trajectory of learning over time."""
        return [
            {
                "step": s.step,
                "confidence": s.total_confidence,
                "accuracy": s.prediction_accuracy,
                "hypotheses": len(s.hypotheses),
                "rules": len(s.rules),
            }
            for s in self._snapshots
        ]
    
    def get_significant_events(self, min_significance: float = 0.7) -> List[DevelopmentEvent]:
        """Get significant events."""
        return [
            e for e in self._events
            if e.significance >= min_significance
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of development record."""
        return {
            "total_events": len(self._events),
            "snapshots": len(self._snapshots),
            "by_type": {
                et.value: sum(1 for e in self._events if e.event_type == et)
                for et in EventType
            },
            "failed_predictions": len(self.get_failed_predictions()),
            "significant_events": len(self.get_significant_events()),
        }
    
    def export_narrative(self) -> str:
        """Export development record as narrative text."""
        lines = ["Development Record\n", "=" * 50 + "\n"]
        
        for event in self._events:
            if event.significance >= 0.5:
                lines.append(f"[Step {event.step}] {event.description}")
                if event.confidence_change != 0:
                    direction = "increased" if event.confidence_change > 0 else "decreased"
                    lines.append(f"  Confidence {direction} by {abs(event.confidence_change):.2f}")
                lines.append("")
        
        return "\n".join(lines)