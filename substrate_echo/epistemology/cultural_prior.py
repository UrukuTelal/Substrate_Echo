"""Cultural Prior Engine — Making culture participate in cognition.

Before S20:
    Knowledge exists beside the process:
    
          Discoveries
              |
              |
    Observation → Hypothesis → Prediction

After S20:
    Knowledge shapes what the agent considers plausible:
    
                 Swarm Knowledge
                       |
                       v
              Cultural Prior Layer
                       |
                       v
    Observation → Hypothesis Generation
                       |
                       v
                 Prediction
                       |
                       v
                  Outcome
                       |
                       v
             Discovery Update

The difference:
    A discovery is no longer something the agent reads.
    It becomes something that shapes what the agent considers plausible.

Critical constraint:
    Priors cannot become dogma.
    
    A bad system:
        Swarm discovered X. Therefore X is true.
    
    A healthy epistemic system:
        Swarm discovered X. Therefore X deserves increased consideration.
    
    Lineage tracking prevents dogma by enabling evaluation of every prior.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import numpy as np
import time
import uuid


class PriorType(Enum):
    """Types of cultural priors."""
    HYPOTHESIS_BIAS = "hypothesis_bias"      # Bias toward certain hypotheses
    FEATURE_WEIGHT = "feature_weight"        # Weight certain features higher
    PATTERN_TEMPLATE = "pattern_template"    # Template for pattern matching
    EXPLANATION_FRAME = "explanation_frame"  # Frame for explaining observations


@dataclass
class CulturalPrior:
    """A validated discovery converted into a hypothesis prior.
    
    Critical constraint: Priors shape consideration, not conclusion.
    They deserve increased weight, not automatic acceptance.
    """
    prior_id: str
    prior_type: PriorType
    domain: str
    
    # Source tracking (lineage)
    source_discovery_id: str = ""
    source_agent_id: str = ""
    source_confidence: float = 0.5
    source_timestamp: float = 0.0
    
    # Prior parameters
    prior_weight: float = 0.5          # [0, 1] How much to weight this prior
    confidence: float = 0.5            # [0, 1] Confidence in this prior
    
    # Applicability conditions
    applicability_conditions: Dict[str, Any] = field(default_factory=dict)
    exclusion_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Validation history
    validation_count: int = 0
    contradiction_count: int = 0
    last_validated: float = 0.0
    
    # Usage tracking
    times_applied: int = 0
    times_helped: int = 0           # When hypothesis was correct
    times_hurt: int = 0             # When hypothesis was incorrect
    
    def get_effective_weight(self) -> float:
        """Get effective weight considering validation and usage."""
        # Base weight from source
        base = self.prior_weight * self.confidence
        
        # Boost from validation
        validation_boost = min(0.2, self.validation_count * 0.01)
        
        # Penalty from contradictions
        contradiction_penalty = min(0.3, self.contradiction_count * 0.05)
        
        # Usage feedback
        if self.times_applied > 0:
            usage_ratio = self.times_helped / self.times_applied
            usage_adjustment = (usage_ratio - 0.5) * 0.2
        else:
            usage_adjustment = 0.0
        
        effective = base + validation_boost - contradiction_penalty + usage_adjustment
        return max(0.0, min(1.0, effective))
    
    def is_applicable(self, observation: Dict[str, Any]) -> bool:
        """Check if this prior applies to the current observation."""
        # If always_apply is set, this prior always applies
        if self.applicability_conditions.get("always_apply", False):
            return True
        
        # If no applicability conditions, always applicable
        if not self.applicability_conditions:
            return True
        
        # Check applicability conditions - lenient matching
        applicable_count = 0
        checked_count = 0
        total_conditions = len(self.applicability_conditions)
        
        for key, value in self.applicability_conditions.items():
            if key == "always_apply":
                continue
            
            if key in observation:
                checked_count += 1
                if isinstance(value, (int, float)):
                    if abs(observation[key] - value) <= 0.5:  # Lenient threshold
                        applicable_count += 1
                elif observation[key] == value:
                    applicable_count += 1
            # If key not in observation, skip (don't fail)
        
        # Prior is applicable if:
        # - At least one condition matches, OR
        # - No conditions could be checked (observation doesn't have the keys)
        return applicable_count > 0 or checked_count == 0
    
    def get_applicability_score(self, observation: Dict[str, Any]) -> float:
        """Get how applicable this prior is to the observation [0, 1]."""
        if self.applicability_conditions.get("always_apply", False):
            return 1.0
        
        if not self.applicability_conditions:
            return 1.0
        
        matches = 0
        total = 0
        
        for key, value in self.applicability_conditions.items():
            if key == "always_apply":
                continue
            total += 1
            
            if key in observation:
                if isinstance(value, (int, float)):
                    if abs(observation[key] - value) <= 0.5:
                        matches += 1
                elif observation[key] == value:
                    matches += 1
        
        return matches / total if total > 0 else 0.5
    
    def record_application(self, helped: bool):
        """Record an application event."""
        self.times_applied += 1
        if helped:
            self.times_helped += 1
        else:
            self.times_hurt += 1
    
    def record_contradiction(self):
        """Record a contradiction event."""
        self.contradiction_count += 1
    
    def record_validation(self, confidence: float):
        """Record a validation event."""
        self.validation_count += 1
        self.last_validated = time.time()
        # Update confidence with recency bias
        alpha = 0.1
        self.confidence = alpha * confidence + (1 - alpha) * self.confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.prior_id,
            "type": self.prior_type.value,
            "domain": self.domain,
            "weight": round(self.get_effective_weight(), 3),
            "confidence": round(self.confidence, 3),
            "source": self.source_discovery_id,
            "validations": self.validation_count,
            "contradictions": self.contradiction_count,
            "applied": self.times_applied,
            "helped": self.times_helped,
            "hurt": self.times_hurt,
        }


@dataclass
class PriorApplication:
    """Record of a prior being applied."""
    application_id: str
    prior_id: str
    observation_hash: str
    
    # What happened
    hypothesis_generated: str = ""
    was_correct: bool = False
    
    # Timing
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class CulturalPriorEngine:
    """Converts validated swarm discoveries into hypothesis priors.
    
    The engine finds relevant discoveries and converts them into
    priors that shape hypothesis generation without forcing conclusions.
    
    Architecture:
        Swarm Knowledge
              |
              v
        Find Relevant Discoveries
              |
              v
        Convert to Cultural Priors
              |
              v
        Apply to Hypothesis Generation
              |
              v
        Track Usage and Update
    """
    
    def __init__(self):
        self._priors: Dict[str, CulturalPrior] = {}
        self._domain_index: Dict[str, List[str]] = {}  # domain -> prior_ids
        self._application_history: List[PriorApplication] = []
        
        # Configuration
        self._min_confidence_for_prior: float = 0.6
        self._max_priors_per_domain: int = 20
        self._decay_rate: float = 0.001
    
    def ingest_discovery(self, discovery: Any, 
                         lineage: Optional[Any] = None) -> Optional[CulturalPrior]:
        """Convert a discovery into a cultural prior.
        
        Args:
            discovery: The compressed discovery
            lineage: Optional lineage information
        
        Returns:
            CulturalPrior if successfully created, None otherwise
        """
        # Check if discovery meets threshold
        if discovery.confidence < self._min_confidence_for_prior:
            return None
        
        # Extract domain
        domain = discovery.pattern.get("domain", "general")
        
        # Create prior
        prior = CulturalPrior(
            prior_id=str(uuid.uuid4()),
            prior_type=self._determine_prior_type(discovery),
            domain=domain,
            source_discovery_id=discovery.discovery_id,
            source_agent_id=discovery.discovered_by,
            source_confidence=discovery.confidence,
            source_timestamp=discovery.discovered_at,
            prior_weight=self._calculate_initial_weight(discovery),
            confidence=discovery.confidence,
            applicability_conditions=self._extract_conditions(discovery),
        )
        
        # Add to storage
        self._priors[prior.prior_id] = prior
        
        # Update domain index
        if domain not in self._domain_index:
            self._domain_index[domain] = []
        self._domain_index[domain].append(prior.prior_id)
        
        # Enforce max per domain
        self._enforce_domain_limit(domain)
        
        return prior
    
    def _determine_prior_type(self, discovery: Any) -> PriorType:
        """Determine what type of prior this discovery should become."""
        discovery_type = discovery.discovery_type.value
        
        if "pattern" in discovery_type:
            return PriorType.PATTERN_TEMPLATE
        elif "rule" in discovery_type:
            return PriorType.HYPOTHESIS_BIAS
        elif "prediction" in discovery_type:
            return PriorType.FEATURE_WEIGHT
        else:
            return PriorType.EXPLANATION_FRAME
    
    def _calculate_initial_weight(self, discovery: Any) -> float:
        """Calculate initial weight for a prior."""
        # Base weight from confidence
        base = discovery.confidence * 0.5
        
        # Boost from evidence count
        evidence_boost = min(0.3, discovery.evidence_count * 0.01)
        
        return min(1.0, base + evidence_boost)
    
    def _extract_conditions(self, discovery: Any) -> Dict[str, Any]:
        """Extract applicability conditions from discovery."""
        conditions = {}
        
        # Extract from pattern - include both numeric and string values
        for key, value in discovery.pattern.items():
            if key != "domain":
                conditions[key] = value
        
        # If no conditions, make it always applicable
        if not conditions:
            conditions["always_apply"] = True
        
        return conditions
    
    def is_applicable(self, observation: Dict[str, Any]) -> bool:
        """Check if this prior applies to the current observation."""
        # If always_apply is set, this prior always applies
        if self.applicability_conditions.get("always_apply", False):
            return True
        
        # If no applicability conditions, always applicable
        if not self.applicability_conditions:
            return True
        
        # Check applicability conditions - lenient matching
        applicable_count = 0
        total_conditions = len(self.applicability_conditions)
        
        for key, value in self.applicability_conditions.items():
            if key == "always_apply":
                continue
            
            if key in observation:
                if isinstance(value, (int, float)):
                    if abs(observation[key] - value) <= 0.5:  # Lenient threshold
                        applicable_count += 1
                elif observation[key] == value:
                    applicable_count += 1
            # If key not in observation, skip (don't fail)
        
        # Prior is applicable if at least one condition matches
        # or if no conditions could be checked
        return applicable_count > 0 or total_conditions == 0
    
    def get_applicability_score(self, observation: Dict[str, Any]) -> float:
        """Get how applicable this prior is to the observation [0, 1]."""
        if self.applicability_conditions.get("always_apply", False):
            return 1.0
        
        if not self.applicability_conditions:
            return 1.0
        
        matches = 0
        total = 0
        
        for key, value in self.applicability_conditions.items():
            if key == "always_apply":
                continue
            total += 1
            
            if key in observation:
                if isinstance(value, (int, float)):
                    if abs(observation[key] - value) <= 0.5:
                        matches += 1
                elif observation[key] == value:
                    matches += 1
        
        return matches / total if total > 0 else 0.5
    
    def _enforce_domain_limit(self, domain: str):
        """Keep only the top priors per domain."""
        if domain not in self._domain_index:
            return
        
        prior_ids = self._domain_index[domain]
        if len(prior_ids) <= self._max_priors_per_domain:
            return
        
        # Sort by effective weight
        priors = [(pid, self._priors[pid].get_effective_weight()) 
                  for pid in prior_ids if pid in self._priors]
        priors.sort(key=lambda x: -x[1])
        
        # Keep only top N
        keep_ids = [pid for pid, _ in priors[:self._max_priors_per_domain]]
        remove_ids = [pid for pid in prior_ids if pid not in keep_ids]
        
        for pid in remove_ids:
            del self._priors[pid]
        
        self._domain_index[domain] = keep_ids
    
    def get_priors_for_observation(self, observation: Dict[str, Any],
                                   domain: str) -> List[CulturalPrior]:
        """Get relevant priors for an observation in a domain.
        
        Returns priors sorted by effective weight (highest first).
        """
        if domain not in self._domain_index:
            return []
        
        applicable = []
        for prior_id in self._domain_index[domain]:
            prior = self._priors.get(prior_id)
            if prior and prior.is_applicable(observation):
                applicable.append(prior)
        
        # Sort by effective weight
        applicable.sort(key=lambda p: -p.get_effective_weight())
        
        return applicable
    
    def apply_priors_to_hypotheses(self, observation: Dict[str, Any],
                                   domain: str,
                                   hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply cultural priors to modify hypothesis probabilities.
        
        This is the core integration point where culture shapes cognition.
        
        Args:
            observation: Current observation
            domain: Knowledge domain
            hypotheses: List of hypotheses with initial probabilities
        
        Returns:
            Modified hypotheses with prior-adjusted probabilities
        """
        priors = self.get_priors_for_observation(observation, domain)
        
        if not priors:
            return hypotheses  # No priors, return unchanged
        
        # Apply priors to modify hypothesis weights
        modified = []
        for hyp in hypotheses:
            hyp_copy = hyp.copy()
            
            # Calculate prior adjustment
            adjustment = 0.0
            applied_priors = []
            
            for prior in priors:
                # Check if this prior is relevant to this hypothesis
                if self._prior_relevant_to_hypothesis(prior, hyp):
                    weight = prior.get_effective_weight()
                    adjustment += weight * 0.3  # Scale factor
                    applied_priors.append(prior.prior_id)
            
            # Apply adjustment
            hyp_copy["probability"] = hyp.get("probability", 0.33) * (1 + adjustment)
            hyp_copy["cultural_priors_applied"] = applied_priors
            hyp_copy["prior_adjustment"] = adjustment
            
            modified.append(hyp_copy)
        
        # Renormalize probabilities
        total = sum(h.get("probability", 0) for h in modified)
        if total > 0:
            for h in modified:
                h["probability"] = h["probability"] / total
        
        return modified
    
    def _prior_relevant_to_hypothesis(self, prior: CulturalPrior,
                                       hypothesis: Dict[str, Any]) -> bool:
        """Check if a prior is relevant to a hypothesis.
        
        Uses applicability_conditions (e.g., signal_trend) to determine
        specific relevance rather than broad domain keyword matching.
        This prevents all hypotheses from getting equal adjustments
        (which would cancel out after renormalization).
        """
        hyp_desc = hypothesis.get("description", "").lower()
        
        # Check applicability_conditions for specific relevance
        # Only string conditions provide specific relevance (e.g., "increasing" in description)
        # Numeric conditions (e.g., phase=0) don't help match hypothesis descriptions
        for key, value in prior.applicability_conditions.items():
            if key == "always_apply":
                continue
            if isinstance(value, str):
                # Check if the condition value appears in hypothesis description
                if value in hyp_desc:
                    return True
        
        # No match from string conditions means prior is not relevant to this hypothesis
        # This prevents all hypotheses from getting equal adjustments
        return False
    
    def record_application(self, prior_id: str, hypothesis_correct: bool):
        """Record how a prior application went."""
        if prior_id not in self._priors:
            return
        
        self._priors[prior_id].record_application(hypothesis_correct)
        
        # Record in history
        self._application_history.append(PriorApplication(
            application_id=str(uuid.uuid4()),
            prior_id=prior_id,
            observation_hash=str(hash(str(time.time()))),
            was_correct=hypothesis_correct,
        ))
    
    def record_contradiction(self, prior_id: str):
        """Record a contradiction event."""
        if prior_id not in self._priors:
            return
        
        self._priors[prior_id].record_contradiction()
    
    def decay_priors(self, current_time: float):
        """Decay priors that haven't been used recently."""
        for prior in self._priors.values():
            time_since_use = current_time - prior.last_validated
            if time_since_use > 100:  # Arbitrary threshold
                prior.confidence *= (1 - self._decay_rate)
    
    def get_domain_priors(self, domain: str) -> List[CulturalPrior]:
        """Get all priors for a domain."""
        if domain not in self._domain_index:
            return []
        
        return [self._priors[pid] for pid in self._domain_index[domain]
                if pid in self._priors]
    
    def get_prior_effectiveness(self) -> Dict[str, Any]:
        """Get effectiveness statistics for all priors."""
        if not self._priors:
            return {"total": 0}
        
        total_applied = sum(p.times_applied for p in self._priors.values())
        total_helped = sum(p.times_helped for p in self._priors.values())
        total_hurt = sum(p.times_hurt for p in self._priors.values())
        
        effective_priors = sum(1 for p in self._priors.values() 
                              if p.get_effective_weight() > 0.5)
        
        return {
            "total_priors": len(self._priors),
            "effective_priors": effective_priors,
            "total_applications": total_applied,
            "total_helped": total_helped,
            "total_hurt": total_hurt,
            "help_rate": total_helped / total_applied if total_applied > 0 else 0,
            "domains": len(self._domain_index),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_priors": len(self._priors),
            "domains": len(self._domain_index),
            "effectiveness": self.get_prior_effectiveness(),
        }