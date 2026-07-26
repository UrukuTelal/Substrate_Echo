"""Evidence and belief structures for the blackboard."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, List
from datetime import datetime
import uuid


class EvidenceType(Enum):
    OBSERVATION = "observation"
    CAPABILITY_TEST = "capability_test"
    PREDICTION_CONFIRMED = "prediction_confirmed"
    PREDICTION_FAILED = "prediction_failed"
    OPPONENT_ACTION = "opponent_action"
    GAME_EVENT = "game_event"
    REPLAY_LEARNING = "replay_learning"
    COUNCIL_INFERENCE = "council_inference"


class Confidence(Enum):
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CERTAIN = 1.0


@dataclass
class Evidence:
    evidence_type: EvidenceType
    source: str
    claim: str
    confidence: float
    
    tick: int = 0
    game_time: float = 0.0
    entities_involved: List[int] = field(default_factory=list)
    
    data: Dict[str, Any] = field(default_factory=dict)
    
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    
    def __lt__(self, other: "Evidence") -> bool:
        return self.confidence > other.confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "type": self.evidence_type.value,
            "source": self.source,
            "claim": self.claim,
            "confidence": self.confidence,
            "tick": self.tick,
            "game_time": self.game_time,
            "data": self.data,
            "tags": self.tags,
        }


@dataclass
class Belief:
    claim: str
    evidence: List[Evidence] = field(default_factory=list)
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    tick: int = 0
    
    supporting_evidence_count: int = 0
    contradicting_evidence_count: int = 0
    
    def add_evidence(self, evidence: Evidence):
        self.evidence.append(evidence)
        self._recalculate_confidence()
        self.last_updated = datetime.now()
        self.tick = evidence.tick
    
    def _recalculate_confidence(self):
        if not self.evidence:
            self.confidence = 0.0
            return
        
        supporting = [e for e in self.evidence if e.confidence > 0]
        contradicting = [e for e in self.evidence if e.confidence < 0]
        
        self.supporting_evidence_count = len(supporting)
        self.contradicting_evidence_count = len(contradicting)
        
        if supporting:
            total_weight = sum(e.confidence for e in supporting)
            self.confidence = total_weight / len(supporting)
        
        if contradicting:
            contradiction_weight = sum(abs(e.confidence) for e in contradicting) / len(contradicting)
            self.confidence *= (1.0 - contradiction_weight * 0.5)
        
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "confidence": self.confidence,
            "supporting": self.supporting_evidence_count,
            "contradicting": self.contradicting_evidence_count,
            "last_tick": self.tick,
            "evidence": [e.to_dict() for e in self.evidence[-10:]],
        }


@dataclass
class CapabilityModel:
    entity_type: str
    capabilities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    test_count: int = 0
    success_count: int = 0
    confidence: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.test_count)
    
    def record_test(self, ability_name: str, success: bool, observed_effects: Dict[str, float] = None):
        self.test_count += 1
        if success:
            self.success_count += 1
            if observed_effects:
                if ability_name in self.capabilities:
                    cap = self.capabilities[ability_name]
                    for k, v in observed_effects.items():
                        cap["effects"][k] = v
                else:
                    self.capabilities[ability_name] = {
                        "preconditions": {},
                        "effects": observed_effects or {},
                        "confidence": 0.5,
                    }
        
        if self.test_count > 0:
            self.confidence = min(1.0, self.success_rate * (1.0 - 0.5 / self.test_count))


@dataclass
class OpponentBelief(Belief):
    opponent_id: int = 0
    belief_type: str = ""
    location: Optional[tuple] = None
    estimated_time: Optional[float] = None
