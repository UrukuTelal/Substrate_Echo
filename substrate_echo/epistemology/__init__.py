"""Epistemology Layer — How the agent knows what it knows.

This layer provides epistemic continuity: the ability to track
what was observed, what was inferred, what was predicted,
and whether reality confirmed or contradicted those predictions.

Architecture:
    Raw Observation
         |
    Feature Extraction (observation.py)
         |
    Hypothesis Generation (hypothesis.py)
         |
    Prediction (prediction.py)
         |
    Outcome Verification
         |
    Belief Update
         |
    Rule Discovery (rule_discovery.py)
         |
    Development Record (development_record.py)

The epistemology layer is a cognitive capability running on the
substrate kernel, not part of the kernel itself.
"""
from .observation import (
    RawObservation,
    Feature,
    FeatureSet,
    FeatureExtractor,
    FeatureType,
    ObservationMemory,
)
from .hypothesis import (
    Hypothesis,
    HypothesisSpace,
    HypothesisStatus,
    Evidence,
)
from .prediction import (
    PredictionRecord,
    PredictionEngine,
    PredictionMemory,
    PredictionStatus,
    PredictionType,
)
from .rule_discovery import (
    Rule,
    RuleDiscoveryEngine,
    Pattern,
    PatternDetector,
    RuleType,
    RuleStatus,
)
from .development_record import (
    DevelopmentRecord,
    DevelopmentEvent,
    BeliefSnapshot,
    EventType,
)
from .perturbation import (
    Perturbation,
    PerturbationEngine,
    PerturbationType,
    BaselineState,
    SystemResponse,
    CausalInference,
)


__all__ = [
    # Observation
    "RawObservation",
    "Feature",
    "FeatureSet",
    "FeatureExtractor",
    "FeatureType",
    "ObservationMemory",
    
    # Hypothesis
    "Hypothesis",
    "HypothesisSpace",
    "HypothesisStatus",
    "Evidence",
    
    # Prediction
    "PredictionRecord",
    "PredictionEngine",
    "PredictionMemory",
    "PredictionStatus",
    "PredictionType",
    
    # Rule Discovery
    "Rule",
    "RuleDiscoveryEngine",
    "Pattern",
    "PatternDetector",
    "RuleType",
    "RuleStatus",
    
    # Development Record
    "DevelopmentRecord",
    "DevelopmentEvent",
    "BeliefSnapshot",
    "EventType",
    
    # Perturbation
    "Perturbation",
    "PerturbationEngine",
    "PerturbationType",
    "BaselineState",
    "SystemResponse",
    "CausalInference",
]