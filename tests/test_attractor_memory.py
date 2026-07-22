"""Tests for Attractor Memory."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from substrate_echo.core.ontological_field import OntologicalField
from substrate_echo.core.attractor_memory import AttractorMemory
from substrate_echo.models.experience import Experience, ExperienceType


def test_memory_encoding():
    field = OntologicalField()
    memory = AttractorMemory(field)
    
    exp = Experience(
        experience_id="exp_001",
        experience_type=ExperienceType.PERCEPTION,
        description="I see a cup on the table",
        psv_snapshot=[0.7] * 16,
        importance=0.8,
    )
    
    attractor = memory.encode(exp)
    assert attractor is not None
    assert len(memory.traces) == 1
    print("PASS: test_memory_encoding")


def test_memory_recall():
    field = OntologicalField()
    memory = AttractorMemory(field)
    
    # Encode a memory
    exp = Experience(
        experience_id="exp_001",
        experience_type=ExperienceType.INTERACTION,
        description="I picked up the cup",
        psv_snapshot=[0.6] * 16,
        importance=0.9,
        object_ids=["cup_1"],
    )
    memory.encode(exp)
    
    # Recall with similar cue
    cue = np.full(16, 0.6)
    results = memory.recall(cue, k=5)
    
    assert len(results) >= 1
    assert results[0].description == "I picked up the cup"
    print("PASS: test_memory_recall")


def test_memory_by_association():
    field = OntologicalField()
    memory = AttractorMemory(field)
    
    # Encode several memories
    for i in range(5):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            experience_type=ExperienceType.INTERACTION,
            description=f"Interaction {i}",
            psv_snapshot=[0.3 + i * 0.1] * 16,
            object_ids=["cup_1"] if i < 3 else ["book_1"],
        )
        memory.encode(exp)
    
    # Recall by object association
    cup_memories = memory.recall_by_association("cup_1")
    assert len(cup_memories) == 3
    
    book_memories = memory.recall_by_association("book_1")
    assert len(book_memories) == 2
    print("PASS: test_memory_by_association")


def test_memory_stats():
    field = OntologicalField()
    memory = AttractorMemory(field)
    
    for i in range(3):
        exp = Experience(
            experience_id=f"exp_{i:03d}",
            experience_type=ExperienceType.LEARNING,
            description=f"Learning {i}",
            psv_snapshot=[0.5] * 16,
        )
        memory.encode(exp)
    
    stats = memory.memory_stats()
    assert stats["total_memories"] == 3
    assert stats["active_memories"] == 3
    print("PASS: test_memory_stats")


def test_surprising_events_stronger():
    field = OntologicalField()
    memory = AttractorMemory(field)
    
    # Normal event with unique pattern
    normal = Experience(
        experience_id="normal",
        experience_type=ExperienceType.PERCEPTION,
        description="Normal perception",
        psv_snapshot=[0.3] * 16,
        importance=0.5,
    )
    a1 = memory.encode(normal)
    
    # Surprising event with different pattern
    surprise = Experience(
        experience_id="surprise",
        experience_type=ExperienceType.SURPRISE,
        description="Surprising event!",
        psv_snapshot=[0.8] * 16,
        importance=0.5,
    )
    a2 = memory.encode(surprise)
    
    # Both should form distinct attractors
    assert len(field.attractors) == 2
    # Surprise should have higher strength (0.5 * 1.5 = 0.75) vs normal (0.5)
    assert a2.strength > a1.strength
    print("PASS: test_surprising_events_stronger")


if __name__ == "__main__":
    test_memory_encoding()
    test_memory_recall()
    test_memory_by_association()
    test_memory_stats()
    test_surprising_events_stronger()
    print("\nAll attractor memory tests passed!")
