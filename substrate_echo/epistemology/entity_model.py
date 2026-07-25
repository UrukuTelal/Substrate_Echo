"""Entity Model — Evidence-backed relationship hypotheses.

Entities are not labeled as "enemy" or "friend."
Relationships are hypotheses that update with evidence.

Architecture:
    Entity
    ├── Identity (UUID, embodiment, observed capabilities)
    ├── Relationship Hypotheses (evidence-backed)
    │     ├── competitive: 0.82
    │     ├── cooperative: 0.15
    │     └── neutral: 0.03
    ├── Goal Hypotheses (what they're trying to do)
    ├── Threat Hypothesis (will they harm my assets?)
    └── Predictive Model (what will they do next?)

Usage:
    model = EntityModel()

    # Create entity
    entity = model.create_entity("player_2", embodiment="sc2")

    # Record evidence
    entity.add_evidence(RelationshipType.COMPETITIVE,
                        "Contesting expansion location",
                        weight=0.8)

    # Update hypotheses
    entity.update_hypotheses(tick=500)

    # Get current belief
    belief = entity.get_relationship(RelationshipType.COMPETITIVE)
    # confidence: 0.82, evidence_count: 3, last_validated: 450

    # Portability: same model for SC2, robotics, social sim, etc.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import time
import uuid


class RelationshipType(Enum):
    """Types of entity relationships (not labels — hypotheses)."""
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"
    NEUTRAL = "neutral"
    COORDINATING = "coordinating"
    DEPENDENT = "dependent"
    ADVERSARIAL = "adversarial"
    UNKNOWN = "unknown"


class EvidenceType(Enum):
    """Types of evidence for relationship hypotheses."""
    OBSERVED_BEHAVIOR = "observed_behavior"
    RESOURCE_COMPETITION = "resource_competition"
    ATTACK = "attack"
    DEFENSE = "defense"
    COMMUNICATION = "communication"
    SHARED_GOAL = "shared_goal"
    COORDINATION = "coordination"
    DECEPTION = "deception"
    RETREAT = "retreat"
    EXPANSION = "expansion"


@dataclass
class Evidence:
    """A piece of evidence supporting or contradicting a hypothesis."""
    evidence_id: str
    evidence_type: EvidenceType
    description: str
    weight: float = 1.0  # how strong this evidence is
    tick: int = 0
    supports: Optional[RelationshipType] = None  # which hypothesis it supports
    contradicts: Optional[RelationshipType] = None  # which it contradicts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.evidence_id,
            "type": self.evidence_type.value,
            "description": self.description,
            "weight": round(self.weight, 3),
            "tick": self.tick,
            "supports": self.supports.value if self.supports else None,
            "contradicts": self.contradicts.value if self.contradicts else None,
        }


@dataclass
class RelationshipHypothesis:
    """A hypothesis about the entity's relationship to us."""
    relationship_type: RelationshipType
    confidence: float = 0.5  # 0-1, starts at neutral
    evidence_count: int = 0
    supporting_evidence: int = 0
    contradicting_evidence: int = 0
    last_validated_tick: int = 0
    created_tick: int = 0

    # Decay: confidence drifts toward 0.5 (neutral) without validation
    decay_rate: float = 0.001  # per tick without validation

    def update_confidence(self, evidence_delta: float, tick: int):
        """Update confidence based on new evidence."""
        old = self.confidence
        self.confidence = max(0.01, min(0.99, self.confidence + evidence_delta))
        self.evidence_count += 1
        if evidence_delta > 0:
            self.supporting_evidence += 1
        elif evidence_delta < 0:
            self.contradicting_evidence += 1
        self.last_validated_tick = tick

    def decay(self, current_tick: int):
        """Decay confidence toward neutral without validation."""
        ticks_since_validation = current_tick - self.last_validated_tick
        if ticks_since_validation > 100:
            decay = self.decay_rate * ticks_since_validation
            if self.confidence > 0.5:
                self.confidence = max(0.5, self.confidence - decay)
            elif self.confidence < 0.5:
                self.confidence = min(0.5, self.confidence + decay)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.relationship_type.value,
            "confidence": round(self.confidence, 4),
            "evidence_count": self.evidence_count,
            "supporting": self.supporting_evidence,
            "contradicting": self.contradicting_evidence,
            "last_validated_tick": self.last_validated_tick,
        }


@dataclass
class GoalHypothesis:
    """A hypothesis about what the entity is trying to achieve."""
    description: str
    confidence: float = 0.5
    evidence_count: int = 0
    created_tick: int = 0
    last_validated_tick: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "confidence": round(self.confidence, 4),
            "evidence_count": self.evidence_count,
        }


@dataclass
class PredictiveModel:
    """What we expect this entity to do next."""
    expected_actions: List[str] = field(default_factory=list)
    confidence: float = 0.5
    last_prediction_tick: int = 0
    predictions_correct: int = 0
    predictions_wrong: int = 0

    def to_dict(self) -> Dict[str, Any]:
        accuracy = 0.0
        total = self.predictions_correct + self.predictions_wrong
        if total > 0:
            accuracy = self.predictions_correct / total
        return {
            "expected_actions": self.expected_actions,
            "confidence": round(self.confidence, 4),
            "accuracy": round(accuracy, 4),
            "predictions_made": total,
        }


class Entity:
    """An entity with evidence-backed relationship hypotheses.

    The entity did not change.
    The evidence changed.
    """

    def __init__(self, entity_id: str, embodiment: str = "unknown",
                 created_tick: int = 0):
        self.entity_id = entity_id
        self.embodiment = embodiment
        self.created_tick = created_tick

        # Identity
        self.observed_capabilities: List[str] = []
        self.predicted_capabilities: List[str] = []

        # Relationship hypotheses
        self._relationships: Dict[RelationshipType, RelationshipHypothesis] = {}
        for rt in RelationshipType:
            self._relationships[rt] = RelationshipHypothesis(
                relationship_type=rt,
                confidence=0.5 if rt == RelationshipType.UNKNOWN else 0.1,
                created_tick=created_tick,
            )
        self._relationships[RelationshipType.UNKNOWN].confidence = 0.8

        # Goal hypotheses
        self._goal_hypotheses: List[GoalHypothesis] = []

        # Threat hypothesis
        self._threat_confidence: float = 0.0

        # Predictive model
        self.predictive_model = PredictiveModel()

        # Evidence log
        self._evidence: List[Evidence] = []

    def add_evidence(self, evidence_type: EvidenceType,
                     description: str,
                     supports: Optional[RelationshipType] = None,
                     contradicts: Optional[RelationshipType] = None,
                     weight: float = 1.0,
                     tick: int = 0):
        """Add evidence about this entity."""
        evidence = Evidence(
            evidence_id=str(uuid.uuid4())[:8],
            evidence_type=evidence_type,
            description=description,
            weight=weight,
            tick=tick,
            supports=supports,
            contradicts=contradicts,
        )
        self._evidence.append(evidence)

        # Update relationship hypotheses
        if supports:
            delta = 0.1 * weight
            self._relationships[supports].update_confidence(delta, tick)
        if contradicts:
            delta = -0.1 * weight
            self._relationships[contradicts].update_confidence(delta, tick)

        # Update threat
        self._update_threat(evidence_type, tick)

    def _update_threat(self, evidence_type: EvidenceType, tick: int):
        """Update threat hypothesis based on evidence."""
        threat_evidence = {
            EvidenceType.ATTACK: 0.3,
            EvidenceType.DECEPTION: 0.2,
            EvidenceType.RESOURCE_COMPETITION: 0.1,
            EvidenceType.DEFENSE: -0.1,
            EvidenceType.COORDINATION: -0.2,
            EvidenceType.COMMUNICATION: -0.05,
        }
        delta = threat_evidence.get(evidence_type, 0.0)
        self._threat_confidence = max(0.0, min(1.0,
                                                self._threat_confidence + delta))

    def add_goal_hypothesis(self, description: str,
                            confidence: float = 0.5,
                            tick: int = 0):
        """Add a hypothesis about what this entity is trying to do."""
        self._goal_hypotheses.append(GoalHypothesis(
            description=description,
            confidence=confidence,
            created_tick=tick,
        ))

    def update_hypotheses(self, tick: int):
        """Decay and validate hypotheses."""
        # Decay all relationships
        for rh in self._relationships.values():
            rh.decay(tick)

        # Remove low-confidence goal hypotheses
        self._goal_hypotheses = [
            g for g in self._goal_hypotheses if g.confidence > 0.2
        ]

    def get_relationship(self, rel_type: RelationshipType) -> RelationshipHypothesis:
        """Get current belief about a relationship."""
        return self._relationships[rel_type]

    def get_dominant_relationship(self) -> Tuple[RelationshipType, float]:
        """Get the strongest relationship hypothesis."""
        best_type = RelationshipType.UNKNOWN
        best_conf = 0.0
        for rt, rh in self._relationships.items():
            if rh.confidence > best_conf:
                best_conf = rh.confidence
                best_type = rt
        return best_type, best_conf

    def get_threat_level(self) -> float:
        """Get current threat hypothesis."""
        return self._threat_confidence

    def get_evidence_count(self) -> int:
        return len(self._evidence)

    def to_dict(self) -> Dict[str, Any]:
        dominant_type, dominant_conf = self.get_dominant_relationship()
        return {
            "entity_id": self.entity_id,
            "embodiment": self.embodiment,
            "dominant_relationship": dominant_type.value,
            "dominant_confidence": round(dominant_conf, 4),
            "threat_level": round(self._threat_confidence, 4),
            "relationships": {
                rt.value: rh.to_dict()
                for rt, rh in self._relationships.items()
            },
            "goal_hypotheses": [g.to_dict() for g in self._goal_hypotheses],
            "predictive_model": self.predictive_model.to_dict(),
            "evidence_count": len(self._evidence),
            "observed_capabilities": self.observed_capabilities,
        }

    def render(self) -> str:
        """Render entity state."""
        lines = []
        dominant_type, dominant_conf = self.get_dominant_relationship()
        lines.append(f"Entity: {self.entity_id} ({self.embodiment})")
        lines.append(f"  Dominant relationship: {dominant_type.value} ({dominant_conf:.3f})")
        lines.append(f"  Threat level: {self._threat_confidence:.3f}")
        lines.append(f"  Evidence: {len(self._evidence)} observations")
        lines.append("")

        # Relationships (only show non-trivial)
        lines.append("  Relationships:")
        for rt, rh in sorted(self._relationships.items(),
                             key=lambda x: x[1].confidence, reverse=True):
            if rh.confidence > 0.15:
                bar_len = int(rh.confidence * 30)
                bar = "#" * bar_len + "." * (30 - bar_len)
                lines.append(
                    f"    {rt.value:15s} {bar} {rh.confidence:.3f} "
                    f"(+{rh.supporting_evidence}/-{rh.contradicting_evidence})"
                )

        # Goals
        if self._goal_hypotheses:
            lines.append("")
            lines.append("  Goal hypotheses:")
            for g in self._goal_hypotheses[:3]:
                lines.append(f"    {g.description} ({g.confidence:.3f})")

        # Predictive model
        pm = self.predictive_model
        if pm.predictions_correct + pm.predictions_wrong > 0:
            lines.append("")
            accuracy = pm.predictions_correct / max(1, pm.predictions_correct + pm.predictions_wrong)
            lines.append(f"  Predictive accuracy: {accuracy:.1%}")

        return "\n".join(lines)


class EntityModel:
    """Model of all entities in the environment.

    Relationships are hypotheses, not labels.
    The entity did not change. The evidence changed.

    Usage:
        model = EntityModel()

        # Create entities
        model.create_entity("player_2", embodiment="sc2")

        # Record evidence
        model.add_evidence("player_2", EvidenceType.ATTACK,
                           "Attacked my expansion",
                           supports=RelationshipType.ADVERSARIAL,
                           tick=500)

        # Get current beliefs
        entity = model.get_entity("player_2")
        dominant, conf = entity.get_dominant_relationship()
    """

    def __init__(self):
        self._entities: Dict[str, Entity] = {}

    def create_entity(self, entity_id: str,
                      embodiment: str = "unknown",
                      created_tick: int = 0) -> Entity:
        """Create a new entity."""
        entity = Entity(entity_id, embodiment, created_tick)
        self._entities[entity_id] = entity
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def get_all_entities(self) -> List[Entity]:
        return list(self._entities.values())

    def add_evidence(self, entity_id: str,
                     evidence_type: EvidenceType,
                     description: str,
                     supports: Optional[RelationshipType] = None,
                     contradicts: Optional[RelationshipType] = None,
                     weight: float = 1.0,
                     tick: int = 0):
        """Add evidence about an entity."""
        entity = self._entities.get(entity_id)
        if entity:
            entity.add_evidence(evidence_type, description,
                                supports, contradicts, weight, tick)

    def update_all(self, tick: int):
        """Update all entity hypotheses."""
        for entity in self._entities.values():
            entity.update_hypotheses(tick)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all entities."""
        return {
            "total_entities": len(self._entities),
            "entities": {
                eid: e.to_dict()
                for eid, e in self._entities.items()
            },
        }

    def render(self) -> str:
        """Render all entities."""
        lines = []
        lines.append("=" * 60)
        lines.append("Entity Model")
        lines.append("=" * 60)
        lines.append(f"Total entities: {len(self._entities)}")
        lines.append("")

        for entity in self._entities.values():
            lines.append(entity.render())
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
