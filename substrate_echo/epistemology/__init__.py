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

from .council import (
    ModelCouncil,
    RealityCouncil,
    MemoryCouncil,
    CalibrationCouncil,
    EpistemologyCouncil,
    CouncilRole,
    CouncilVerdict,
)

from .trust import (
    EpistemicTrustSystem,
    TrustVector,
    TrustDimension,
)

from .swarm import (
    SwarmDevelopmentRecord,
    AgentEpistemicState,
    CompressedDiscovery,
    DiscoveryType,
    OpenQuestion,
    OpenQuestionType,
)

from .exchange import (
    DiscoveryExchangeProtocol,
    ExchangeMessage,
    ExchangeProtocol,
    ExchangeRate,
)

from .lineage import (
    DiscoveryLineageSystem,
    DiscoveryLineage,
    LineageNode,
    LineageNodeType,
    ConflictResolver,
    ConflictPair,
    ConflictResolution,
)

from .cultural_prior import (
    CulturalPriorEngine,
    CulturalPrior,
    PriorType,
    PriorApplication,
)

from .curiosity import (
    EpistemicCuriosityEngine,
    UncertaintyMap,
    KnowledgeGap,
    GapType,
    GapPriority,
    ImpactAssessor,
    ImpactAssessment,
    ResearchGoal,
)

from .research import (
    ExperimentPlanner,
    ExperimentProposal,
    ExperimentDesign,
    ExperimentType,
    ProposalStatus,
    ExecutionPlan,
    ResearchAgenda,
)

from .observatory import (
    EpistemicObservatory,
    CognitiveTelemetry,
    EventTimeline,
    CognitiveEvent,
    CognitiveSnapshot,
    ModuleTelemetry,
    EventType,
    ModuleType,
)

from .instrumentation import (
    CausalReplay,
    ReasoningTree,
    ReasoningNode,
    NodeType,
    CounterfactualEngine,
    Modification,
    ModificationType,
    ComparisonReport,
    KnowledgeProvenanceGraph,
    ProvenanceNode,
    ProvenanceEdge,
    DependencyAnalyzer,
    DependencyReport,
    NodeInfluenceProfile,
    EvidenceDiversity,
    CorrectionLatency,
    EpistemicInertia,
    RecoveryTime,
    ResilienceAnalyzer,
    EpistemicPlasticityAnalyzer,
    PlasticityReport,
    PlasticityProfile,
    PlasticityRecord,
    EpistemicDynamicsLab,
    AgentState,
    DynamicsSnapshot,
    KnowledgeClass,
    KnowledgeClassConfig,
    KnowledgeItem,
    DifferentialPlasticityManager,
    HalfLifeRecord,
    EpistemicHalfLifeAnalyzer,
    MigrationDirection,
    MigrationEvent,
    MigrationTrigger,
    KnowledgeMigrationSystem,
    EpistemicRole,
    EpistemicRoleManager,
    # Governance
    GovernanceRuleType,
    GovernanceRule,
    KnowledgeCompetitionResult,
    CompetitionEvent,
    EpistemicGovernance,
    # Epistemic Age
    EpistemicAgeProfile,
    EpistemicAgeAnalyzer,
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
    
    # Council
    "ModelCouncil",
    "RealityCouncil",
    "MemoryCouncil",
    "CalibrationCouncil",
    "EpistemologyCouncil",
    "CouncilRole",
    "CouncilVerdict",
    
    # Trust
    "EpistemicTrustSystem",
    "TrustVector",
    "TrustDimension",
    
    # Swarm
    "SwarmDevelopmentRecord",
    "AgentEpistemicState",
    "CompressedDiscovery",
    "DiscoveryType",
    "OpenQuestion",
    "OpenQuestionType",
    
    # Exchange
    "DiscoveryExchangeProtocol",
    "ExchangeMessage",
    "ExchangeProtocol",
    "ExchangeRate",
    
    # Lineage
    "DiscoveryLineageSystem",
    "DiscoveryLineage",
    "LineageNode",
    "LineageNodeType",
    "ConflictResolver",
    "ConflictPair",
    "ConflictResolution",
    
    # Cultural Prior
    "CulturalPriorEngine",
    "CulturalPrior",
    "PriorType",
    "PriorApplication",
    
    # Curiosity
    "EpistemicCuriosityEngine",
    "UncertaintyMap",
    "KnowledgeGap",
    "GapType",
    "GapPriority",
    "ImpactAssessor",
    "ImpactAssessment",
    "ResearchGoal",
    
    # Research
    "ExperimentPlanner",
    "ExperimentProposal",
    "ExperimentDesign",
    "ExperimentType",
    "ProposalStatus",
    "ExecutionPlan",
    "ResearchAgenda",
    
    # Observatory
    "EpistemicObservatory",
    "CognitiveTelemetry",
    "EventTimeline",
    "CognitiveEvent",
    "CognitiveSnapshot",
    "ModuleTelemetry",
    "EventType",
    "ModuleType",
    
    # Instrumentation
    "CausalReplay",
    "ReasoningTree",
    "ReasoningNode",
    "NodeType",
    "CounterfactualEngine",
    "Modification",
    "ModificationType",
    "ComparisonReport",
    "KnowledgeProvenanceGraph",
    "ProvenanceNode",
    "ProvenanceEdge",
    "DependencyAnalyzer",
    "DependencyReport",
    "NodeInfluenceProfile",
    "EvidenceDiversity",
    "CorrectionLatency",
    "EpistemicInertia",
    "RecoveryTime",
    "ResilienceAnalyzer",
    "EpistemicPlasticityAnalyzer",
    "PlasticityReport",
    "PlasticityProfile",
    "PlasticityRecord",
    "EpistemicDynamicsLab",
    "AgentState",
    "DynamicsSnapshot",
    "KnowledgeClass",
    "KnowledgeClassConfig",
    "KnowledgeItem",
    "DifferentialPlasticityManager",
    "HalfLifeRecord",
    "EpistemicHalfLifeAnalyzer",
    "MigrationDirection",
    "MigrationEvent",
    "MigrationTrigger",
    "KnowledgeMigrationSystem",
    "EpistemicRole",
    "EpistemicRoleManager",
    # Governance
    "GovernanceRuleType",
    "GovernanceRule",
    "KnowledgeCompetitionResult",
    "CompetitionEvent",
    "EpistemicGovernance",
    # Epistemic Age
    "EpistemicAgeProfile",
    "EpistemicAgeAnalyzer",
]