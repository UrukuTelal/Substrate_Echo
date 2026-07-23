"""Field dynamics — evolution, attractor formation, diffusion, state transitions, conservation, metric, topology."""

from .field_evolution import FieldEvolver, EvolutionConfig, FieldConfig, SolverType
from .diffusion import DiffusionTensor, DiffusionConfig
from .attractor_dynamics import AttractorDynamics, DynamicsConfig
from .state_transitions import (
    StateTransitionManager, StateTransition, TransitionCause,
    TransitionStatus, TransitionConstraint, TransitionCallback,
)
from .conservation import ConservationHooks, ConservationFramework, ConservationResult
from .metric_interface import MetricInterface, MetricTensor
from .topology_events import TopologyEventQueue, TopologyEvent, TopologyEventType
from .pillar_coupling import PillarCoupling, PillarCouplingConfig, PILLAR_NAMES
from .vortex_dynamics import VortexDynamics, VortexConfig, Vortex
from .topology_transitions import TopologyManager, TransitionConfig, TopologyTransition, TransitionType
from .basin_topology import BasinTopology, BasinMetrics, BasinEvent, AttractorState
from .abstraction import AbstractionEngine, MetaAttractor, AttractorCorrelation, CognitiveBudget

__all__ = [
    "FieldEvolver", "EvolutionConfig", "SolverType",
    "DiffusionTensor", "DiffusionConfig",
    "AttractorDynamics", "DynamicsConfig",
    "StateTransitionManager", "StateTransition", "TransitionCause",
    "TransitionStatus", "TransitionConstraint", "TransitionCallback",
    "ConservationHooks", "ConservationResult",
    "MetricInterface", "MetricTensor",
    "TopologyEventQueue", "TopologyEvent", "TopologyEventType",
    "PillarCoupling", "PillarCouplingConfig", "PILLAR_NAMES",
    "VortexDynamics", "VortexConfig", "Vortex",
    "TopologyManager", "TransitionConfig", "TopologyTransition", "TransitionType",
    "BasinTopology", "BasinMetrics", "BasinEvent", "AttractorState",
    "AbstractionEngine", "MetaAttractor", "AttractorCorrelation", "CognitiveBudget",
]
