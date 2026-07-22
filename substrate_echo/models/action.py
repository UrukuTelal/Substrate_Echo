"""Action — decisions and motor commands.

Actions flow from cognitive decisions through to physical motor commands.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time


class ActionType(Enum):
    """Types of actions an agent can take."""
    MOVE_TO = auto()           # move to a location
    GRASP = auto()             # pick up an object
    RELEASE = auto()           # put down an object
    OBSERVE = auto()           # focus attention on something
    COMMUNICATE = auto()       # send message/signal
    THINK = auto()             # internal reflection
    PLAN = auto()              # generate action plan
    CREATE = auto()            # create something new
    MODIFY = auto()            # change something
    AVOID = auto()             # move away from something
    WAIT = auto()              # do nothing for a duration


@dataclass
class Action:
    """A cognitive decision to perform an action."""
    action_id: str
    action_type: ActionType
    timestamp: float = field(default_factory=time.time)
    
    # Target
    target_object_id: Optional[str] = None
    target_position: Optional[tuple[float, float, float]] = None
    target_agent_id: Optional[str] = None
    
    # Parameters
    parameters: dict = field(default_factory=dict)
    duration: float = 0.0  # seconds, 0 = instantaneous
    
    # Provenance
    source_agent: str = ""  # which cognitive agent proposed this
    confidence: float = 0.5  # 0-1: how confident in this action
    reasoning: str = ""     # why this action was chosen
    
    # Outcome (filled after execution)
    executed: bool = False
    result: Optional[str] = None
    success: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "id": self.action_id,
            "type": self.action_type.name,
            "target": self.target_object_id,
            "agent": self.source_agent,
            "confidence": round(self.confidence, 3),
            "executed": self.executed,
            "success": self.success,
        }


@dataclass
class MotorCommand:
    """Physical motor command for embodiment.
    
    Translated from Action by the EmbodimentBridge.
    Contains joint angles, velocities, etc. specific to the robot platform.
    """
    command_id: str
    timestamp: float = field(default_factory=time.time)
    
    # Joint-space commands
    joint_positions: list[float] = field(default_factory=list)  # radians
    joint_velocities: list[float] = field(default_factory=list)  # rad/s
    
    # End-effector
    gripper_open: float = 0.0  # 0=closed, 1=open
    
    # Base movement (for mobile robots)
    linear_velocity: float = 0.0   # m/s
    angular_velocity: float = 0.0  # rad/s
    
    # Safety
    max_force: float = 10.0  # Newtons
    emergency_stop: bool = False
    
    # Source
    source_action_id: str = ""
    validity_duration: float = 0.1  # seconds (commands expire quickly)

    def is_valid(self) -> bool:
        """Check if command is within safety limits."""
        if self.emergency_stop:
            return False
        for v in self.joint_velocities:
            if abs(v) > 10.0:  # rad/s limit
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "id": self.command_id,
            "joints": self.joint_positions,
            "gripper": self.gripper_open,
            "base_linear": self.linear_velocity,
            "base_angular": self.angular_velocity,
            "valid": self.is_valid(),
        }
