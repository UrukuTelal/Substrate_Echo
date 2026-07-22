"""Tests for Embodiment Bridge."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from substrate_echo.core.embodiment_bridge import EmbodimentBridge, SensorData
from substrate_echo.models.action import Action, ActionType, MotorCommand


def test_bridge_creation():
    bridge = EmbodimentBridge()
    assert bridge.sensor is None
    assert bridge.actuator is None
    print("PASS: test_bridge_creation")


def test_sensor_processing():
    bridge = EmbodimentBridge()
    
    sensor_data = SensorData(
        detected_objects=[
            {"id": "cup_1", "label": "cup", "position": [1.0, 0.0, 0.5], "confidence": 0.9},
            {"id": "table_1", "label": "table", "position": [1.0, 0.0, 0.0], "confidence": 0.95},
        ]
    )
    
    updates = bridge.process_sensor_input(sensor_data)
    assert len(updates) == 2
    assert updates[0]["object_id"] == "cup_1"
    print("PASS: test_sensor_processing")


def test_action_to_motor_command():
    bridge = EmbodimentBridge()
    
    action = Action(
        action_id="act_001",
        action_type=ActionType.GRASP,
        target_object_id="cup_1",
    )
    
    cmd = bridge.generate_action(action)
    assert cmd.source_action_id == "act_001"
    assert cmd.gripper_open == 0.0  # closed for grasp
    print("PASS: test_action_to_motor_command")


def test_motor_command_safety():
    cmd = MotorCommand(command_id="test")
    cmd.joint_velocities = [5.0] * 6  # normal
    assert cmd.is_valid()
    
    cmd.joint_velocities = [100.0] * 6  # too fast
    assert not cmd.is_valid()
    
    cmd2 = MotorCommand(command_id="estop", emergency_stop=True)
    assert not cmd2.is_valid()
    print("PASS: test_motor_command_safety")


def test_ar_sync():
    bridge = EmbodimentBridge()
    
    ar_state = {
        "anchors": [{"id": "anchor_1", "name": "floor", "position": [0, 0, 0]}],
        "objects": [{"id": "obj_1", "label": "chair", "position": [2, 0, 0], "size": [0.5, 0.5, 1.0]}],
    }
    
    updates = bridge.sync_ar_environment(ar_state)
    assert len(updates) == 2
    print("PASS: test_ar_sync")


def test_robotics_transfer():
    bridge = EmbodimentBridge()
    config = bridge.transfer_to_robotics()
    assert "sensor_type" in config
    assert "actuator_type" in config
    print("PASS: test_robotics_transfer")


if __name__ == "__main__":
    test_bridge_creation()
    test_sensor_processing()
    test_action_to_motor_command()
    test_motor_command_safety()
    test_ar_sync()
    test_robotics_transfer()
    print("\nAll embodiment bridge tests passed!")
