"""Tests for Ontological Field."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from substrate_echo.core.ontological_field import OntologicalField, Attractor, Repulsor


def test_field_creation():
    field = OntologicalField()
    assert field.state.shape == (16,)
    assert np.all(field.state == 0.0)
    print("PASS: test_field_creation")


def test_field_evolution():
    field = OntologicalField()
    field.set_state(np.full(16, 0.5))
    
    # Evolve for several steps
    for _ in range(10):
        field.evolve(0.01)
    
    # State should still be in [0, 1]
    assert np.all(field.state >= 0.0)
    assert np.all(field.state <= 1.0)
    print("PASS: test_field_evolution")


def test_attractor_formation():
    field = OntologicalField()
    pattern = np.full(16, 0.7)
    att = field.form_attractor(pattern, strength=0.9, label="test_memory")
    
    assert att.label == "test_memory"
    assert att.strength == 0.9
    assert len(field.attractors) == 1
    print("PASS: test_attractor_formation")


def test_attractor_pull():
    center = np.full(16, 0.8)
    att = Attractor(center=center, strength=1.0)
    
    point = np.full(16, 0.2)
    pull = att.pull(point, strength=0.1)
    
    # Pull should be toward the center (positive direction)
    assert np.all(pull > 0)
    print("PASS: test_attractor_pull")


def test_repulsor_push():
    center = np.full(16, 0.5)
    rep = Repulsor(center=center, strength=1.0, radius=0.5)
    
    point = np.full(16, 0.4)  # within radius
    push = rep.push(point, strength=0.1)
    
    # Push should be away from center (negative direction since point < center)
    assert np.all(push < 0)
    
    # Outside radius — no push
    far_point = np.full(16, 10.0)
    push_far = rep.push(far_point, strength=0.1)
    assert np.all(push_far == 0)
    print("PASS: test_repulsor_push")


def test_attractor_basin():
    center = np.full(16, 0.5)
    att = Attractor(center=center, basin_width=0.3)
    
    inside = np.full(16, 0.55)  # distance = 0.05 * sqrt(16) ≈ 0.2
    outside = np.full(16, 0.9)  # far away
    
    assert att.is_in_basin(inside)
    assert not att.is_in_basin(outside)
    print("PASS: test_attractor_basin")


def test_nearest_attractors():
    field = OntologicalField()
    
    # Create several attractors
    field.form_attractor(np.full(16, 0.2), label="a1")
    field.form_attractor(np.full(16, 0.5), label="a2")
    field.form_attractor(np.full(16, 0.8), label="a3")
    
    query = np.full(16, 0.25)  # closest to a1
    nearest = field.find_nearest_attractors(query, k=1)
    assert len(nearest) == 1
    assert nearest[0].label == "a1"
    print("PASS: test_nearest_attractors")


def test_coherence():
    field = OntologicalField()
    
    # Low coherence: attractors spread out
    field.form_attractor(np.zeros(16), label="far1")
    field.form_attractor(np.ones(16), label="far2")
    low_coh = field.coherence()
    
    # Higher coherence: attractors close together
    field2 = OntologicalField()
    field2.form_attractor(np.full(16, 0.5), label="close1")
    field2.form_attractor(np.full(16, 0.51), label="close2")
    high_coh = field2.coherence()
    
    assert high_coh > low_coh
    print("PASS: test_coherence")


def test_identity_pattern():
    field = OntologicalField()
    
    # No strong attractors → no identity
    assert field.identity_pattern() is None
    
    # Strong attractors → identity emerges
    field.form_attractor(np.full(16, 0.7), strength=0.8, label="core1")
    field.form_attractor(np.full(16, 0.71), strength=0.9, label="core2")
    
    identity = field.identity_pattern()
    assert identity is not None
    assert identity.shape == (16,)
    print("PASS: test_identity_pattern")


if __name__ == "__main__":
    test_field_creation()
    test_field_evolution()
    test_attractor_formation()
    test_attractor_pull()
    test_repulsor_push()
    test_attractor_basin()
    test_nearest_attractors()
    test_coherence()
    test_identity_pattern()
    print("\nAll ontological field tests passed!")
