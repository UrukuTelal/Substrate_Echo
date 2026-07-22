"""Social Persona Ecology Package.

Six cognitive pressures inside a social ecosystem:
  Cartographer — expands search space
  Engineer — constrains it
  Archivist — preserves it
  Philosopher — questions it
  Trickster — mutates it
  Diplomat — stabilizes it

Personas are attractor seeds, not prompt wrappers.
The social environment determines which attractors survive.
"""
from .persona_genome import AgentGenome, PersonalityVector, COMMUNICATION_STYLES
from .social_memory import SocialEpisode, SocialMemory
from .relationship_memory import RelationshipRecord, RelationshipMemory
from .persona_dynamics import PersonaDynamics

__all__ = [
    "AgentGenome",
    "PersonalityVector",
    "COMMUNICATION_STYLES",
    "SocialEpisode",
    "SocialMemory",
    "RelationshipRecord",
    "RelationshipMemory",
    "PersonaDynamics",
]
