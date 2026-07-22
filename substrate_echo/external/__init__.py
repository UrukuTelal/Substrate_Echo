"""External agent integration package.

S8: Interface for external AI agents to feed information into the
cognitive substrate. External agents are modeled as foreign dynamical
systems producing perturbations, not as knowledge sources.

S9: Foreign ecosystem simulation for validation.
"""
from .candidate_queue import (
    CandidateQueue,
    CandidateQueueConfig,
    CandidateStatus,
    EvaluationResult,
    IntegrationDecision,
    IntegrationMode,
    InteractionSpectrum,
    LatentIntegrationRecord,
    MemoryCandidate,
    Provenance,
)
from .foreign_node import ForeignAgent, ReputationVector
from .interaction_encoder import InteractionEncoder
from .foreign_evaluator import ForeignEvaluator
from .integration_gate import IntegrationGate
from .verification_loop import VerificationLoop, VerificationRecord
from .synthetic_profiles import (
    BehaviorArchetype,
    SyntheticAgent,
    create_ecosystem,
)
from .ecosystem_simulation import (
    ForeignEcosystemSimulation,
    EcosystemMetrics,
)

__all__ = [
    "BehaviorArchetype",
    "CandidateQueue",
    "CandidateQueueConfig",
    "CandidateStatus",
    "EcosystemMetrics",
    "EvaluationResult",
    "ForeignAgent",
    "ForeignEvaluator",
    "ForeignEcosystemSimulation",
    "IntegrationDecision",
    "IntegrationGate",
    "IntegrationMode",
    "InteractionEncoder",
    "InteractionSpectrum",
    "LatentIntegrationRecord",
    "MemoryCandidate",
    "Provenance",
    "ReputationVector",
    "SyntheticAgent",
    "VerificationLoop",
    "VerificationRecord",
    "create_ecosystem",
]
