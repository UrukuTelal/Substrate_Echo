"""Core ontological architecture — spatial world, field dynamics, memory, agents."""

from .spatial_world import SpatialWorldModel, WorldObject
from .ontological_field import OntologicalField, Attractor, Repulsor
from .attractor_memory import AttractorMemory, Experience, MemoryTrace
from .cognitive_agents import AgentEcology, CognitiveAgent
from .embodiment_bridge import EmbodimentBridge
from .cognitive_loop import CognitiveLoop, CognitiveLoopConfig
from .multi_agent_dynamics import SocialField, SocialConfig, AgentProfile
