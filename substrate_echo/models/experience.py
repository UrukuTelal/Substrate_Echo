"""Experience — a unit of lived interaction with the world.

Experiences are the raw material that gets transformed into attractor memories.
Each experience captures what happened, the context, and the ontological state at the time.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time


class ExperienceType(Enum):
    """Types of experiences an agent can have."""
    PERCEPTION = auto()      # saw something in the world
    INTERACTION = auto()     # acted on an object
    SOCIAL = auto()          # interacted with another agent
    REFLECTION = auto()      # internal thought/analysis
    SURPRISE = auto()        # expectation violated
    GOAL_ACHIEVED = auto()   # completed a planned action
    GOAL_FAILED = auto()     # failed to complete action
    LEARNING = auto()        # acquired new knowledge


@dataclass
class Experience:
    """A single experience event.
    
    Contains:
    - What happened (event description)
    - The context (where, when, what was happening)
    - The ontological state (16D PSV at time of experience)
    - Sensory snapshot (what was perceived)
    - Action taken (what the agent did)
    - Result (what happened as a consequence)
    """
    experience_id: str
    experience_type: ExperienceType
    timestamp: float = field(default_factory=time.time)
    
    # Content
    description: str = ""
    object_ids: list[str] = field(default_factory=list)  # involved objects
    agent_ids: list[str] = field(default_factory=list)   # involved agents
    
    # Ontological state at time of experience
    psv_snapshot: Optional[list[float]] = None  # 16D state (BlochPSV values)
    context_psv: Optional[list[float]] = None   # broader context state
    
    # Sensory data
    sensory_data: Optional[dict] = None  # raw sensor snapshot
    
    # Action and result
    action_taken: Optional[str] = None
    action_result: Optional[str] = None
    result_valence: float = 0.0  # -1 to 1: negative=bad, positive=good
    
    # Metadata
    importance: float = 0.5  # how important was this experience
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.experience_id,
            "type": self.experience_type.name,
            "time": self.timestamp,
            "description": self.description,
            "objects": self.object_ids,
            "action": self.action_taken,
            "result": self.action_result,
            "valence": self.result_valence,
            "importance": self.importance,
        }

    @classmethod
    def perception(cls, exp_id: str, description: str, object_ids: list[str],
                   psv_snapshot: Optional[list[float]] = None) -> Experience:
        """Create a perception experience."""
        return cls(
            experience_id=exp_id,
            experience_type=ExperienceType.PERCEPTION,
            description=description,
            object_ids=object_ids,
            psv_snapshot=psv_snapshot,
        )

    @classmethod
    def interaction(cls, exp_id: str, description: str, object_id: str,
                    action: str, result: str, valence: float = 0.0,
                    psv_snapshot: Optional[list[float]] = None) -> Experience:
        """Create an interaction experience."""
        return cls(
            experience_id=exp_id,
            experience_type=ExperienceType.INTERACTION,
            description=description,
            object_ids=[object_id],
            action_taken=action,
            action_result=result,
            result_valence=valence,
            psv_snapshot=psv_snapshot,
        )
