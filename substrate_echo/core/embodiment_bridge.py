"""Embodiment Bridge — connects internal world model to physical reality.

Handles the transition from cognitive decisions to motor commands,
and from raw sensor data to world model updates.

Progression: Virtual → AR → Robotics
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol
import numpy as np
import time

from ..models.action import Action, ActionType, MotorCommand


class SensorInterface(Protocol):
    """Interface for sensor hardware (AR, robot, simulation)."""
    def get_reading(self) -> dict: ...
    def get_position(self) -> tuple[float, float, float]: ...
    def get_orientation(self) -> tuple[float, float, float]: ...


class ActuatorInterface(Protocol):
    """Interface for motor hardware."""
    def send_command(self, command: MotorCommand) -> bool: ...
    def get_status(self) -> dict: ...


@dataclass
class SensorData:
    """Processed sensor data from the environment."""
    timestamp: float = field(default_factory=time.time)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    detected_objects: list[dict] = field(default_factory=list)
    depth_map: Optional[np.ndarray] = None
    rgb_image: Optional[np.ndarray] = None
    lidar_points: Optional[np.ndarray] = None
    imu_data: Optional[dict] = None
    metadata: dict = field(default_factory=dict)


class EmbodimentBridge:
    """Connects internal world model to physical embodiment.
    
    Handles:
    - Sensor data → world model updates
    - Cognitive decisions → motor commands
    - AR environment sync
    - Robotics platform abstraction
    """
    
    def __init__(self):
        self.sensor: Optional[SensorInterface] = None
        self.actuator: Optional[ActuatorInterface] = None
        self.calibration_offset = np.zeros(3)
        self.last_sensor_time = 0.0
        self._action_queue: list[Action] = []
    
    def connect_sensor(self, sensor: SensorInterface) -> None:
        self.sensor = sensor
    
    def connect_actuator(self, actuator: ActuatorInterface) -> None:
        self.actuator = actuator
    
    def process_sensor_input(self, raw: SensorData) -> list[dict]:
        """Convert raw sensor data to world model updates.
        
        Returns a list of object update dicts that can be applied
        to the SpatialWorldModel.
        """
        updates = []
        for obj_data in raw.detected_objects:
            update = {
                "object_id": obj_data.get("id", "unknown"),
                "name": obj_data.get("label", "unknown"),
                "position": tuple(obj_data.get("position", [0, 0, 0])),
                "dimensions": tuple(obj_data.get("dimensions", [0.1, 0.1, 0.1])),
                "confidence": obj_data.get("confidence", 0.5),
            }
            updates.append(update)
        
        self.last_sensor_time = raw.timestamp
        return updates
    
    def generate_action(self, decision: Action) -> MotorCommand:
        """Convert cognitive decision to physical motor commands.
        
        Maps high-level actions to joint-space or base-space commands.
        """
        cmd = MotorCommand(
            command_id=f"cmd_{decision.action_id}",
            source_action_id=decision.action_id,
        )
        
        if decision.action_type == ActionType.MOVE_TO:
            # Simple proportional control toward target
            if decision.target_position is not None:
                target = np.array(decision.target_position)
                current = np.array(self.sensor.get_position()) if self.sensor else np.zeros(3)
                diff = target - current
                distance = np.linalg.norm(diff)
                if distance > 0.01:
                    cmd.linear_velocity = min(0.5, distance * 2.0)
                    cmd.angular_velocity = float(np.arctan2(diff[1], diff[0]))
        
        elif decision.action_type == ActionType.GRASP:
            cmd.gripper_open = 0.0  # close gripper
        
        elif decision.action_type == ActionType.RELEASE:
            cmd.gripper_open = 1.0  # open gripper
        
        elif decision.action_type == ActionType.OBSERVE:
            # Just look — no motor command needed
            pass
        
        elif decision.action_type == ActionType.WAIT:
            pass
        
        return cmd
    
    def execute_command(self, command: MotorCommand) -> bool:
        """Send motor command to actuators."""
        if self.actuator is None:
            return False
        if not command.is_valid():
            return False
        return self.actuator.send_command(command)
    
    def sync_ar_environment(self, ar_state: dict) -> list[dict]:
        """Sync with AR environment mapping.
        
        AR state contains detected planes, objects, and spatial anchors.
        """
        updates = []
        
        # Convert AR anchors to world objects
        for anchor in ar_state.get("anchors", []):
            updates.append({
                "object_id": anchor.get("id", ""),
                "name": anchor.get("name", "anchor"),
                "position": tuple(anchor.get("position", [0, 0, 0])),
                "type": "spatial_anchor",
            })
        
        # Convert AR objects to world objects
        for obj in ar_state.get("objects", []):
            updates.append({
                "object_id": obj.get("id", ""),
                "name": obj.get("label", "unknown"),
                "position": tuple(obj.get("position", [0, 0, 0])),
                "dimensions": tuple(obj.get("size", [0.1, 0.1, 0.1])),
                "type": "ar_object",
            })
        
        return updates
    
    def transfer_to_robotics(self) -> dict:
        """Export configuration for robotic platform.
        
        Returns a config dict that can be loaded by a robotics driver.
        """
        return {
            "sensor_type": type(self.sensor).__name__ if self.sensor else "none",
            "actuator_type": type(self.actuator).__name__ if self.actuator else "none",
            "calibration": self.calibration_offset.tolist(),
            "last_sensor_time": self.last_sensor_time,
            "pending_actions": len(self._action_queue),
        }
    
    def queue_action(self, action: Action) -> None:
        self._action_queue.append(action)
    
    def dequeue_action(self) -> Optional[Action]:
        if self._action_queue:
            return self._action_queue.pop(0)
        return None
