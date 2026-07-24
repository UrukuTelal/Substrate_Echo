"""Epistemic Trust — Separating cooperation from predictability.

The key insight: an agent can distrust another agent while still
being able to predict it accurately. This enables:

"I do not trust this entity.
I can still predict this entity."

Architecture:
    Interaction History
         |
    Cooperation Trust (Will they help me?)
         |
    Predictability (Can I model them?)
         |
    Domain-Specific Trust (What are they good at?)
         |
    Combined Assessment

Domain-Specific Epistemic Trust:
    Instead of "Agent A trusts Agent B", we get:
    "Agent A trusts Agent B's model of domain X"
    
    This enables:
    - Selective knowledge exchange based on demonstrated competence
    - Trust-informed information filtering
    - Cultural prior formation from validated discoveries
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import numpy as np
from collections import deque


class TrustDimension(Enum):
    """Dimensions of trust."""
    COOPERATION = "cooperation"           # Will they act in my interest?
    PREDICTABILITY = "predictability"     # Can I model their behavior?
    COMPETENCE = "competence"             # Can they accomplish goals?
    HONESTY = "honesty"                   # Will they share accurate information?
    INFORMATION_RELIABILITY = "information_reliability"  # Are their observations trustworthy?


@dataclass
class DomainTrust:
    """Trust in an entity's competence within a specific domain."""
    domain: str
    confidence: float = 0.5           # How much do we trust their model in this domain?
    prediction_accuracy: float = 0.5  # Historical accuracy in this domain
    validation_count: int = 0         # How many times validated
    last_updated: float = 0.0
    
    def record_validation(self, accuracy: float, timestamp: float):
        """Record a validation event."""
        self.validation_count += 1
        # Running average with recency bias
        alpha = 0.1
        self.prediction_accuracy = alpha * accuracy + (1 - alpha) * self.prediction_accuracy
        self.confidence = min(1.0, self.confidence + 0.02)
        self.last_updated = timestamp


@dataclass
class TrustVector:
    """Multi-dimensional trust state for an entity."""
    entity_id: str
    
    # Trust dimensions [0, 1]
    cooperation: float = 0.5
    predictability: float = 0.5
    competence: float = 0.5
    honesty: float = 0.5
    information_reliability: float = 0.5
    
    # Confidence in each dimension
    cooperation_confidence: float = 0.1
    predictability_confidence: float = 0.1
    competence_confidence: float = 0.1
    honesty_confidence: float = 0.1
    
    # Interaction history
    interaction_count: int = 0
    last_interaction: float = 0.0
    
    # Prediction tracking
    prediction_attempts: int = 0
    prediction_successes: int = 0
    
    # Domain-specific trust
    domain_trust: Dict[str, DomainTrust] = field(default_factory=dict)
    
    def get_domain_trust(self, domain: str) -> DomainTrust:
        """Get trust for a specific domain."""
        if domain not in self.domain_trust:
            self.domain_trust[domain] = DomainTrust(domain=domain)
        return self.domain_trust[domain]
    
    def get_domain_confidence(self, domain: str) -> float:
        """Get confidence in this entity's model of a domain."""
        return self.get_domain_trust(domain).confidence
    
    def get_overall_trust(self) -> float:
        """Get weighted overall trust score."""
        return (
            self.cooperation * 0.3 +
            self.predictability * 0.25 +
            self.competence * 0.2 +
            self.honesty * 0.15 +
            self.information_reliability * 0.1
        )
    
    def get_trust_level(self) -> str:
        """Get trust classification."""
        overall = self.get_overall_trust()
        if overall < 0.2:
            return "HOSTILE"
        elif overall < 0.4:
            return "DISTRUSTFUL"
        elif overall < 0.6:
            return "CAUTIOUS"
        elif overall < 0.8:
            return "TRUSTWORTHY"
        return "ALLIED"
    
    def get_prediction_accuracy(self) -> float:
        """Get prediction accuracy for this entity."""
        if self.prediction_attempts == 0:
            return 0.5
        return self.prediction_successes / self.prediction_attempts
    
    def record_interaction(self, cooperation_delta: float,
                           predicted_correctly: bool,
                           timestamp: float,
                           domain: Optional[str] = None):
        """Record an interaction and update trust."""
        self.interaction_count += 1
        self.last_interaction = timestamp
        
        # Update cooperation trust
        self.cooperation = np.clip(self.cooperation + cooperation_delta, 0.0, 1.0)
        self.cooperation_confidence = min(1.0, self.cooperation_confidence + 0.02)
        
        # Update predictability
        self.prediction_attempts += 1
        if predicted_correctly:
            self.prediction_successes += 1
        self.predictability = self.get_prediction_accuracy()
        self.predictability_confidence = min(1.0, self.predictability_confidence + 0.02)
        
        # Update domain-specific trust if provided
        if domain:
            domain_trust = self.get_domain_trust(domain)
            accuracy = 1.0 if predicted_correctly else 0.0
            domain_trust.record_validation(accuracy, timestamp)
    
    def decay(self, decay_rate: float = 0.001):
        """Decay trust toward neutral over time."""
        # Cooperation decays toward 0.5
        self.cooperation += (0.5 - self.cooperation) * decay_rate
        # Predictability is more stable (based on actual accuracy)
        # Competence and honesty decay slowly
        self.competence += (0.5 - self.competence) * decay_rate * 0.5
        self.honesty += (0.5 - self.honesty) * decay_rate * 0.5
        self.information_reliability += (0.5 - self.information_reliability) * decay_rate * 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "cooperation": round(self.cooperation, 3),
            "predictability": round(self.predictability, 3),
            "competence": round(self.competence, 3),
            "honesty": round(self.honesty, 3),
            "information_reliability": round(self.information_reliability, 3),
            "overall": round(self.get_overall_trust(), 3),
            "level": self.get_trust_level(),
            "prediction_accuracy": round(self.get_prediction_accuracy(), 3),
            "interactions": self.interaction_count,
            "domains": {
                d: {
                    "confidence": round(dt.confidence, 3),
                    "accuracy": round(dt.prediction_accuracy, 3),
                    "validations": dt.validation_count,
                }
                for d, dt in self.domain_trust.items()
            },
        }


class EpistemicTrustSystem:
    """Multi-dimensional trust system for swarm epistemology.
    
    Separates cooperation trust from predictability, enabling:
    - Accurate modeling of adversarial entities
    - Cooperation decisions independent of prediction ability
    - Trust-informed communication and information sharing
    - Domain-specific trust: "I trust Agent B's model of domain X"
    
    The feedback path:
        Individual experience
            ↓
        Local hypothesis formation
            ↓
        Validation
            ↓
        Compressed discovery
            ↓
        Swarm knowledge
            ↓
        Cultural prior
            ↓
        Future agent interpretation
    """
    
    def __init__(self):
        self._entities: Dict[str, TrustVector] = {}
        self._global_trust: float = 0.5
        
        # Domain expertise tracking
        self._domain_experts: Dict[str, Dict[str, float]] = {}  # domain -> entity_id -> confidence
    
    def register_entity(self, entity_id: str,
                        initial_cooperation: float = 0.5,
                        initial_predictability: float = 0.5):
        """Register a new entity with initial trust values."""
        self._entities[entity_id] = TrustVector(
            entity_id=entity_id,
            cooperation=initial_cooperation,
            predictability=initial_predictability,
        )
    
    def record_interaction(self, entity_id: str,
                           cooperation_delta: float,
                           predicted_correctly: bool,
                           timestamp: float,
                           domain: Optional[str] = None):
        """Record an interaction with an entity."""
        if entity_id not in self._entities:
            self.register_entity(entity_id)
        
        self._entities[entity_id].record_interaction(
            cooperation_delta, predicted_correctly, timestamp, domain
        )
        
        # Update domain expertise tracking
        if domain:
            if domain not in self._domain_experts:
                self._domain_experts[domain] = {}
            self._domain_experts[domain][entity_id] = \
                self._entities[entity_id].get_domain_trust(domain).confidence
    
    def get_trust(self, entity_id: str) -> Optional[TrustVector]:
        """Get trust vector for an entity."""
        return self._entities.get(entity_id)
    
    def get_cooperation_trust(self, entity_id: str) -> float:
        """Get cooperation trust for an entity."""
        entity = self._entities.get(entity_id)
        return entity.cooperation if entity else 0.5
    
    def get_predictability(self, entity_id: str) -> float:
        """Get predictability of an entity."""
        entity = self._entities.get(entity_id)
        return entity.predictability if entity else 0.5
    
    def get_domain_trust(self, entity_id: str, domain: str) -> float:
        """Get trust in an entity's model of a specific domain."""
        entity = self._entities.get(entity_id)
        if not entity:
            return 0.5
        return entity.get_domain_trust(domain).confidence
    
    def get_domain_expert(self, domain: str) -> Optional[str]:
        """Get the most trusted entity for a domain."""
        if domain not in self._domain_experts:
            return None
        experts = self._domain_experts[domain]
        if not experts:
            return None
        return max(experts, key=experts.get)
    
    def should_share_information(self, entity_id: str,
                                 info_value: float,
                                 info_type: str = "general") -> bool:
        """Determine if information should be shared with entity.
        
        Uses cooperation trust and info sensitivity.
        """
        entity = self._entities.get(entity_id)
        if not entity:
            return False
        
        # Higher value info requires higher trust
        threshold = 0.3 + (info_value * 0.4)
        
        # Sensitive info requires even higher trust
        if info_type in ("military", "strategic", "weakness"):
            threshold += 0.2
        
        return entity.cooperation >= threshold
    
    def should_trust_domain(self, entity_id: str, domain: str,
                            min_confidence: float = 0.6) -> bool:
        """Check if we should trust this entity's model of a domain."""
        entity = self._entities.get(entity_id)
        if not entity:
            return False
        
        domain_trust = entity.get_domain_trust(domain)
        return (domain_trust.confidence >= min_confidence and
                domain_trust.validation_count >= 5)
    
    def get_swarm_knowledge_value(self, discovery_domain: str,
                                  discovery_confidence: float) -> Dict[str, Any]:
        """Evaluate a discovery's value for swarm knowledge.
        
        Returns the best domain expert and whether the discovery
        should be integrated into cultural prior.
        """
        expert_id = self.get_domain_expert(discovery_domain)
        
        if expert_id:
            expert_trust = self.get_domain_trust(expert_id, discovery_domain)
            # Trust-weighted value
            value = discovery_confidence * expert_trust
        else:
            # No expert yet, use raw confidence
            value = discovery_confidence
            expert_id = "unknown"
        
        return {
            "value": value,
            "expert": expert_id,
            "should_integrate": value >= 0.7,
            "confidence": discovery_confidence,
        }
    
    def can_predict(self, entity_id: str) -> bool:
        """Check if we can reliably predict this entity."""
        entity = self._entities.get(entity_id)
        if not entity:
            return False
        
        return (entity.predictability >= 0.6 and 
                entity.predictability_confidence >= 0.3)
    
    def tick(self, current_time: float):
        """Update all trust states."""
        for entity in self._entities.values():
            # Decay old interactions
            time_since = current_time - entity.last_interaction
            if time_since > 100:
                entity.decay(0.001)
    
    def get_all_trusts(self) -> Dict[str, Dict[str, Any]]:
        """Get all entity trusts."""
        return {eid: t.to_dict() for eid, t in self._entities.items()}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get trust system summary."""
        if not self._entities:
            return {"entities": 0}
        
        cooperations = [e.cooperation for e in self._entities.values()]
        predictabilities = [e.predictability for e in self._entities.values()]
        
        return {
            "entities": len(self._entities),
            "avg_cooperation": float(np.mean(cooperations)),
            "avg_predictability": float(np.mean(predictabilities)),
            "highly_trustworthy": sum(1 for c in cooperations if c > 0.7),
            "highly_predictable": sum(1 for p in predictabilities if p > 0.7),
            "hostile": sum(1 for c in cooperations if c < 0.2),
            "domains": len(self._domain_experts),
        }