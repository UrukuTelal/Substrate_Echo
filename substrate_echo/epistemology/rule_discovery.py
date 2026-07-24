"""Rule Discovery Engine — Moving from pattern detection to learned rules.

The rule discovery engine detects recurring patterns in observations
and generates candidate rules that explain those patterns. Rules
are then tested through prediction and validation.

Architecture:
    Observation History
         |
    Pattern Detection
         |
    Rule Candidate
         |
    Prediction Generation
         |
    Validation
         |
    Rule Memory
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import time
from collections import defaultdict


class RuleStatus(Enum):
    """Lifecycle status of a rule."""
    CANDIDATE = "candidate"       # Newly detected pattern
    TESTING = "testing"           # Being validated
    VALIDATED = "validated"       # Passed validation
    REJECTED = "rejected"         # Failed validation
    CONSOLIDATED = "consolidated" # Strong enough to be a learned rule


class RuleType(Enum):
    """Types of rules that can be discovered."""
    CONDITIONAL = "conditional"   # IF condition THEN outcome
    SEQUENTIAL = "sequential"     # A follows B
    CORRELATIONAL = "correlational" # A and B co-occur
    CAUSAL = "causal"            # A causes B
    BEHAVIORAL = "behavioral"    # Entity tends to do X


@dataclass
class Pattern:
    """A detected pattern in observations."""
    id: str
    description: str
    pattern_type: RuleType
    
    # Pattern components
    conditions: Dict[str, Any] = field(default_factory=dict)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    occurrence_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    
    # Confidence
    confidence: float = 0.0
    support: float = 0.0          # How often pattern occurs
    lift: float = 1.0             # How much more likely given conditions
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rule:
    """A learned rule from validated patterns."""
    id: str
    description: str
    rule_type: RuleType
    
    # Rule structure
    conditions: Dict[str, Any] = field(default_factory=dict)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    validation_attempts: int = 0
    validation_successes: int = 0
    confidence: float = 0.0
    
    # Source
    source_pattern: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    
    # Usage
    application_count: int = 0
    success_count: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def apply(self, conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply rule given conditions.
        
        Returns predicted outcomes if conditions match, None otherwise.
        """
        # Check if conditions match
        match = True
        for key, value in self.conditions.items():
            if key not in conditions:
                match = False
                break
            
            actual = conditions[key]
            if isinstance(value, (int, float)) and isinstance(actual, (int, float)):
                if abs(value - actual) / max(abs(value), 1e-6) > 0.2:
                    match = False
                    break
            elif value != actual:
                match = False
                break
        
        if not match:
            return None
        
        self.application_count += 1
        return self.outcomes
    
    def record_outcome(self, success: bool):
        """Record outcome of rule application."""
        self.validation_attempts += 1
        if success:
            self.validation_successes += 1
            self.success_count += 1
        
        # Update confidence
        self.confidence = self.validation_successes / self.validation_attempts
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "type": self.rule_type.value,
            "confidence": self.confidence,
            "status": "validated" if self.confidence > 0.7 else "testing",
            "applications": self.application_count,
            "successes": self.success_count,
        }


class PatternDetector:
    """Detects patterns in observation history.
    
    Looks for recurring sequences, correlations, and conditional
    relationships in feature data.
    """
    
    def __init__(self, min_support: float = 0.3, min_confidence: float = 0.5):
        self._min_support = min_support
        self._min_confidence = min_confidence
        self._pattern_count = 0
        self._observed_sequences: List[Tuple[str, ...]] = []
        self._feature_cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        self._total_observations = 0
    
    def record_observation(self, features: 'FeatureSet'):
        """Record observation for pattern detection."""
        self._total_observations += 1
        
        # Extract significant features
        significant = [
            name for name, f in features.features.items()
            if f.confidence > 0.5 and isinstance(f.value, (int, float))
        ]
        
        # Update co-occurrence
        for i, feat_a in enumerate(significant):
            for feat_b in significant[i+1:]:
                pair = tuple(sorted([feat_a, feat_b]))
                self._feature_cooccurrence[pair] += 1
        
        # Track sequence
        self._observed_sequences.append(tuple(significant))
        if len(self._observed_sequences) > 1000:
            self._observed_sequences.pop(0)
    
    def detect_patterns(self) -> List[Pattern]:
        """Detect patterns from observed data."""
        patterns = []
        
        # Detect co-occurrence patterns
        for (feat_a, feat_b), count in self._feature_cooccurrence.items():
            support = count / max(self._total_observations, 1)
            
            if support >= self._min_support:
                pattern = Pattern(
                    id=f"PAT{self._pattern_count:04d}",
                    description=f"{feat_a} and {feat_b} co-occur",
                    pattern_type=RuleType.CORRELATIONAL,
                    conditions={feat_a: "present", feat_b: "present"},
                    outcomes={"correlation": True},
                    occurrence_count=count,
                    confidence=support,
                    support=support,
                )
                patterns.append(pattern)
                self._pattern_count += 1
        
        # Detect sequential patterns
        if len(self._observed_sequences) >= 10:
            patterns.extend(self._detect_sequential_patterns())
        
        return patterns
    
    def _detect_sequential_patterns(self) -> List[Pattern]:
        """Detect sequential patterns."""
        patterns = []
        
        # Look for A followed by B
        for i in range(len(self._observed_sequences) - 1):
            seq_a = self._observed_sequences[i]
            seq_b = self._observed_sequences[i + 1]
            
            if seq_a and seq_b:
                # Check if features in A predict features in B
                for feat_a in seq_a:
                    for feat_b in seq_b:
                        if feat_a != feat_b:
                            # This is a simplified pattern detection
                            pass
        
        return patterns
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pattern detection statistics."""
        return {
            "total_observations": self._total_observations,
            "unique_patterns": self._pattern_count,
            "co_occurrence_pairs": len(self._feature_cooccurrence),
        }


class RuleDiscoveryEngine:
    """Discovers rules from detected patterns.
    
    Takes patterns and generates candidate rules,
    then validates them through prediction.
    """
    
    def __init__(self):
        self._rules: Dict[str, Rule] = {}
        self._rule_count = 0
        self._pattern_detector = PatternDetector()
    
    def observe(self, features: 'FeatureSet'):
        """Record observation for rule discovery."""
        self._pattern_detector.record_observation(features)
    
    def discover_rules(self) -> List[Rule]:
        """Discover rules from observed patterns."""
        patterns = self._pattern_detector.detect_patterns()
        new_rules = []
        
        for pattern in patterns:
            # Convert pattern to rule candidate
            rule = Rule(
                id=f"RULE{self._rule_count:04d}",
                description=pattern.description,
                rule_type=pattern.pattern_type,
                conditions=pattern.conditions,
                outcomes=pattern.outcomes,
                source_pattern=pattern.id,
                created_at=time.time(),
                updated_at=time.time(),
                confidence=pattern.confidence,
            )
            
            self._rules[rule.id] = rule
            new_rules.append(rule)
            self._rule_count += 1
        
        return new_rules
    
    def get_rules(self, min_confidence: float = 0.0) -> List[Rule]:
        """Get rules above confidence threshold."""
        return [
            r for r in self._rules.values()
            if r.confidence >= min_confidence
        ]
    
    def get_validated_rules(self) -> List[Rule]:
        """Get validated rules."""
        return [
            r for r in self._rules.values()
            if r.confidence >= 0.7
        ]
    
    def apply_rules(self, conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply all rules to given conditions."""
        results = []
        
        for rule in self._rules.values():
            outcomes = rule.apply(conditions)
            if outcomes is not None:
                results.append({
                    "rule": rule.id,
                    "description": rule.description,
                    "outcomes": outcomes,
                    "confidence": rule.confidence,
                })
        
        return results
    
    def consolidate_rules(self):
        """Consolidate high-confidence rules."""
        for rule in self._rules.values():
            if rule.confidence >= 0.8 and rule.application_count >= 5:
                rule.metadata["consolidated"] = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_rules": len(self._rules),
            "validated": len(self.get_validated_rules()),
            "rules": [r.to_dict() for r in self._rules.values()],
            "patterns": self._pattern_detector.get_statistics(),
        }