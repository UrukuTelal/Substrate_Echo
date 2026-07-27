"""Representational Layer — the shared semantic substrate.

Four representations of the same reality, each answering a different question:

    Perceptual    What did I observe?
    Semantic      What does it mean?
    Dynamical     What patterns are emerging?
    Executive     What should I do?

Architecture:
    The Ontology provides static knowledge (concepts, taxonomy, rules).
    The EntityDescriptor is the universal cognitive object — everything
    that can be reasoned about is represented through this structure.
    The StateGraph tracks dynamic state across all entities.
    The SemanticInterpreter translates raw observations into semantic meaning.
    The CausalGraph chains events into consequences.
    The FrameSystem queries the world model through different lenses.
    The NarrativeLayer compresses causality into temporal explanations.

Design Principles:
    1. The ontology lives AROUND the kernel, not inside it.
    2. Every subsystem operates on the same world model while maintaining
       its own representation of that model.
    3. The ontology stays small — only what's needed for reasoning.
    4. Evidence is never overwritten by hypotheses.
    5. Uncertainty is always explicit.
"""
from substrate_echo.representational.ontology import (
    Concept, ConceptCategory, TaxonomyNode, Rule, Constraint,
    PropertySchema, Ontology,
)
from substrate_echo.representational.entity_descriptor import (
    EntityDescriptor, EntityIdentity, EntityEmbodiment, EntityClassification,
    EntityComposition, EntityCapabilities, EntityAffordances,
    EntityRelationships, EntityState, EntityObservation, EntityHypothesis,
    EntityEvidence, EntityCausality, EntityHistory,
)
from substrate_echo.representational.state_graph import (
    StateGraph, GraphEdge, EntityStateType,
)
from substrate_echo.representational.interpreter import SemanticInterpreter
from substrate_echo.representational.causal_graph import (
    CausalGraph, CausalEvent, Consequence, EventChain,
    EventType, ConsequenceType,
)
from substrate_echo.representational.frames import (
    FrameSystem, FrameResult, Perspective, FrameType,
    PERSPECTIVE_EARLY_GAME, PERSPECTIVE_MID_GAME,
    PERSPECTIVE_UNDER_ATTACK, PERSPECTIVE_ATTACKING,
)
from substrate_echo.representational.narrative import (
    NarrativeLayer, Narrative, NarrativeEvent, NarrativeType, NarrativeArc,
)
