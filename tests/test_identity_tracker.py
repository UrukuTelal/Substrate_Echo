"""Tests for Object Identity Tracking."""
import numpy as np
import sys
sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.identity_tracker import (
    IdentityTracker, IdentityTrackerConfig, TrackedEntity,
)


# ── Basic Tracking ───────────────────────────────────────────────

def test_single_entity_appears():
    t = IdentityTracker()
    r = t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}], timestamp=1.0)
    assert len(r["active"]) == 1
    assert len(r["new"]) == 1
    assert r["active"][0].entity_type == "human"


def test_entity_persists_across_frames():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}], timestamp=1.0)
    r = t.update([{"type": "human", "position": [1.1, 0.1, 0.0]}], timestamp=2.0)
    assert len(r["active"]) == 1
    assert len(r["new"]) == 0  # matched existing
    entity = r["active"][0]
    assert entity.frames_tracked == 2


def test_entity_gets_stable_id():
    t = IdentityTracker()
    r1 = t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}])
    r2 = t.update([{"type": "human", "position": [1.1, 0.1, 0.0]}])
    assert r1["active"][0].entity_id == r2["active"][0].entity_id


def test_two_entities_distinct_ids():
    t = IdentityTracker()
    t.update([
        {"type": "human", "position": [0.0, 0.0, 0.0]},
        {"type": "human", "position": [5.0, 5.0, 0.0]},
    ])
    active = t.active_entities
    assert len(active) == 2
    assert active[0].entity_id != active[1].entity_id


def test_different_types_different_ids():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0]},
              {"type": "animal", "position": [1.0, 0.0, 0.0]}])
    active = t.active_entities
    assert len(active) == 2
    types = {e.entity_type for e in active}
    assert "human" in types
    assert "animal" in types


# ── Association Distance ─────────────────────────────────────────

def test_close_match():
    t = IdentityTracker(IdentityTrackerConfig(max_association_distance=2.0))
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}], timestamp=1.0)
    r = t.update([{"type": "human", "position": [1.5, 0.0, 0.0]}], timestamp=2.0)
    assert len(r["new"]) == 0  # matched


def test_far_creates_new():
    t = IdentityTracker(IdentityTrackerConfig(max_association_distance=1.0))
    t.update([{"type": "human", "position": [0.0, 0.0, 0.0]}], timestamp=1.0)
    r = t.update([{"type": "human", "position": [5.0, 0.0, 0.0]}], timestamp=2.0)
    assert len(r["new"]) == 1  # new entity


def test_type_mismatch_creates_new():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}], timestamp=1.0)
    r = t.update([{"type": "animal", "position": [1.0, 0.0, 0.0]}], timestamp=2.0)
    assert len(r["new"]) == 1


# ── Exit Detection ───────────────────────────────────────────────

def test_entity_exits_after_timeout():
    config = IdentityTrackerConfig(exit_timeout=3)
    t = IdentityTracker(config)
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}], timestamp=1.0)
    
    # Entity not seen for 4 frames
    for i in range(4):
        r = t.update([], timestamp=2.0 + i)
    
    assert len(r["exited"]) == 1
    assert len(t.active_entities) == 0


def test_entity_stays_if_seen():
    t = IdentityTracker(IdentityTrackerConfig(exit_timeout=3))
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}], timestamp=1.0)
    
    for i in range(5):
        t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}],
                 timestamp=2.0 + i)
    
    assert len(t.active_entities) == 1


# ── Velocity Tracking ────────────────────────────────────────────

def test_velocity_computed():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [0.0, 0.0, 0.0]}], timestamp=0.0)
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}], timestamp=1.0)
    
    entity = t.active_entities[0]
    assert entity.speed > 0
    assert np.dot(entity.velocity, np.array([1.0, 0.0, 0.0])) > 0


def test_velocity_smoothed():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [0.0, 0.0, 0.0]}], timestamp=0.0)
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}], timestamp=1.0)
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}], timestamp=2.0)
    
    entity = t.active_entities[0]
    # After stopping, velocity should decay
    assert entity.speed < 1.0


# ── Position History ─────────────────────────────────────────────

def test_position_history_grows():
    t = IdentityTracker()
    for i in range(5):
        t.update([{"type": "human", "position": [float(i), 0.0, 0.0]}],
                 timestamp=float(i))
    
    entity = t.active_entities[0]
    assert len(entity.position_history) == 5


def test_position_history_bounded():
    config = IdentityTrackerConfig()
    t = IdentityTracker(config)
    # Create entity first
    t.update([{"type": "human", "position": [0.0, 0.0, 0.0]}])
    entity = t.active_entities[0]
    entity.max_history = 10
    
    for i in range(15):
        t.update([{"type": "human", "position": [float(i), 0.0, 0.0]}])
    
    assert len(entity.position_history) <= 10


def test_mean_position():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [0.0, 0.0, 0.0]}])
    t.update([{"type": "human", "position": [2.0, 0.0, 0.0]}])
    
    entity = t.active_entities[0]
    mean = entity.mean_position
    assert abs(mean[0] - 1.0) < 0.1


# ── Trajectory Length ────────────────────────────────────────────

def test_trajectory_length():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [0.0, 0.0, 0.0]}])
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}])
    t.update([{"type": "human", "position": [1.0, 1.0, 0.0]}])
    
    entity = t.active_entities[0]
    assert abs(entity.trajectory_length - 2.0) < 0.1


# ── Queries ──────────────────────────────────────────────────────

def test_get_entities_by_type():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [0.0, 0.0, 0.0]}])
    t.update([{"type": "animal", "position": [1.0, 0.0, 0.0]}])
    
    humans = t.get_entities_by_type("human")
    animals = t.get_entities_by_type("animal")
    assert len(humans) == 1
    assert len(animals) == 1


def test_get_nearest():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [0.0, 0.0, 0.0]},
              {"type": "human", "position": [5.0, 0.0, 0.0]}])
    
    nearest = t.get_nearest(np.array([0.5, 0.0, 0.0]))
    assert nearest is not None
    assert abs(nearest.position[0]) < 1.0


def test_get_nearest_with_type_filter():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [0.0, 0.0, 0.0]},
              {"type": "animal", "position": [0.5, 0.0, 0.0]}])
    
    nearest_human = t.get_nearest(np.array([0.2, 0.0, 0.0]), entity_type="human")
    assert nearest_human is not None
    assert nearest_human.entity_type == "human"


def test_get_nearest_empty():
    t = IdentityTracker()
    assert t.get_nearest(np.array([0.0, 0.0, 0.0])) is None


# ── Visit Count ──────────────────────────────────────────────────

def test_visit_count():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [0.0, 0.0, 0.0]}])
    t.update([{"type": "human", "position": [0.1, 0.0, 0.0]}])
    t.update([{"type": "animal", "position": [0.0, 0.0, 0.0]}])
    
    count = t.get_visit_count(np.array([0.0, 0.0, 0.0]), radius=1.0)
    assert count == 2  # human + animal


# ── Properties ───────────────────────────────────────────────────

def test_properties_carry_forward():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0],
               "properties": {"mood": "happy"}}])
    t.update([{"type": "human", "position": [1.1, 0.0, 0.0]}])
    
    entity = t.active_entities[0]
    assert entity.properties.get("mood") == "happy"


def test_properties_update():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0],
               "properties": {"mood": "happy"}}])
    t.update([{"type": "human", "position": [1.1, 0.0, 0.0],
               "properties": {"mood": "sad"}}])
    
    entity = t.active_entities[0]
    assert entity.properties.get("mood") == "sad"


# ── Dwell Time ───────────────────────────────────────────────────

def test_dwell_time():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}], timestamp=1.0)
    t.update([{"type": "human", "position": [1.1, 0.0, 0.0]}], timestamp=3.0)
    
    entity = t.active_entities[0]
    assert abs(entity.dwell_time - 2.0) < 0.1


# ── Reset ────────────────────────────────────────────────────────

def test_reset():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}])
    t.reset()
    assert len(t.active_entities) == 0
    assert t._frame_count == 0


# ── Multiple Entities in Frame ───────────────────────────────────

def test_multiple_per_frame():
    t = IdentityTracker()
    r = t.update([
        {"type": "human", "position": [0.0, 0.0, 0.0]},
        {"type": "human", "position": [5.0, 0.0, 0.0]},
        {"type": "animal", "position": [2.0, 2.0, 0.0]},
    ])
    assert len(r["active"]) == 3
    assert len(r["new"]) == 3


def test_partial_match():
    """One entity matches, one is new."""
    t = IdentityTracker(IdentityTrackerConfig(max_association_distance=2.0))
    t.update([
        {"type": "human", "position": [0.0, 0.0, 0.0]},
        {"type": "human", "position": [5.0, 0.0, 0.0]},
    ])
    r = t.update([
        {"type": "human", "position": [0.1, 0.0, 0.0]},  # matches first
        {"type": "human", "position": [10.0, 0.0, 0.0]},  # too far, new
    ])
    assert len(r["active"]) == 2
    assert len(r["new"]) == 1


# ── to_dict ──────────────────────────────────────────────────────

def test_to_dict():
    t = IdentityTracker()
    t.update([{"type": "human", "position": [1.0, 0.0, 0.0]}])
    entity = t.active_entities[0]
    d = entity.to_dict()
    assert d["entity_type"] == "human"
    assert "position" in d
    assert "speed" in d
