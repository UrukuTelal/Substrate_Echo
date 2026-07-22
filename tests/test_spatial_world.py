"""Tests for Spatial World Model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from substrate_echo.models.world_object import (
    WorldObject, PhysicalState, RelationalState, Relationship, SpatialGraph,
    RelationshipType, Affordance,
)
from substrate_echo.core.spatial_world import SpatialWorldModel


def test_world_object_creation():
    obj = WorldObject(
        object_id="cup_1",
        name="Coffee Cup",
        object_type="cup",
        physical=PhysicalState(position=(1.0, 2.0, 0.5)),
    )
    assert obj.object_id == "cup_1"
    assert obj.physical.position == (1.0, 2.0, 0.5)
    assert obj.relational.familiarity == 0.0
    print("PASS: test_world_object_creation")


def test_relational_state_reinforcement():
    rel = RelationalState()
    assert rel.familiarity == 0.0
    rel.reinforce(0.2)
    assert rel.familiarity == 0.2
    assert rel.interaction_count == 1
    rel.reinforce(0.3)
    assert rel.familiarity == 0.5
    print("PASS: test_relational_state_reinforcement")


def test_spatial_world_model_queries():
    model = SpatialWorldModel()
    
    cup = WorldObject("cup_1", "Cup", "cup",
                       physical=PhysicalState(position=(1.0, 0.0, 0.0)))
    table = WorldObject("table_1", "Table", "table",
                         physical=PhysicalState(position=(1.0, 0.0, 0.0)))
    far_obj = WorldObject("book_1", "Book", "book",
                           physical=PhysicalState(position=(10.0, 0.0, 0.0)))
    
    model.add_object(cup)
    model.add_object(table)
    model.add_object(far_obj)
    
    # Query region
    nearby = model.query_region(center=(1.0, 0.0, 0.0), radius=1.0)
    assert len(nearby) == 2  # cup and table
    
    far = model.query_region(center=(10.0, 0.0, 0.0), radius=1.0)
    assert len(far) == 1  # book
    
    # Nearest
    nearest = model.get_nearest((1.1, 0.0, 0.0), k=1)
    assert len(nearest) == 1
    
    print("PASS: test_spatial_world_model_queries")


def test_spatial_graph():
    graph = SpatialGraph()
    
    rel = Relationship("cup_1", "table_1", RelationshipType.SPATIAL_PROXIMITY, 0.8)
    graph.add_relationship(rel)
    
    neighbors = graph.get_neighbors("cup_1")
    assert len(neighbors) == 1
    assert neighbors[0].target_id == "table_1"
    
    print("PASS: test_spatial_graph")


def test_object_distance():
    a = PhysicalState(position=(0, 0, 0))
    b = PhysicalState(position=(3, 4, 0))
    assert a.distance_to(b) == 5.0
    print("PASS: test_object_distance")


if __name__ == "__main__":
    test_world_object_creation()
    test_relational_state_reinforcement()
    test_spatial_world_model_queries()
    test_spatial_graph()
    test_object_distance()
    print("\nAll spatial world tests passed!")
