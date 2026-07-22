"""Tests for Probabilistic PSV."""
import numpy as np
import sys
sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.probabilistic_psv import (
    ProbabilisticPSV, PILLAR_NAMES, NUM_PILLARS,
)


def test_creation():
    p = ProbabilisticPSV()
    assert p.means.shape == (16,)
    assert p.variances.shape == (16,)


def test_default_values():
    p = ProbabilisticPSV()
    assert all(abs(m - 0.5) < 0.01 for m in p.means)


def test_to_deterministic():
    p = ProbabilisticPSV()
    det = p.to_deterministic()
    assert det.shape == (16,)
    assert np.allclose(det, 0.5)


def test_from_deterministic():
    arr = np.random.rand(16)
    p = ProbabilisticPSV.from_deterministic(arr)
    assert np.allclose(p.means, arr)
    assert all(v == 0.25 for v in p.variances)


def test_array_roundtrip():
    p = ProbabilisticPSV()
    arr = p.to_array()
    assert arr.shape == (32,)
    p2 = ProbabilisticPSV.from_array(arr)
    assert np.allclose(p.means, p2.means)
    assert np.allclose(p.variances, p2.variances)


def test_update_from_observation():
    p = ProbabilisticPSV()
    p.update_from_observation({"Awareness": 0.9, "Force": 0.1})
    assert p.pillars[0].mean > 0.5  # Awareness pulled up
    assert p.pillars[2].mean < 0.5  # Force pulled down


def test_update_from_array():
    p = ProbabilisticPSV()
    obs = np.random.rand(16)
    p.update_from_array(obs, noise=0.1)
    # Means should move toward observations
    for i in range(16):
        assert abs(p.pillars[i].mean - obs[i]) < abs(0.5 - obs[i])


def test_confidence():
    p = ProbabilisticPSV()
    c = p.confidence
    assert 0.0 < c < 1.0


def test_dominant_pillar():
    p = ProbabilisticPSV()
    p.pillars[5].mean = 0.9
    assert p.dominant_pillar == 5


def test_weakest_pillar():
    p = ProbabilisticPSV()
    p.pillars[3].mean = 0.1
    assert p.weakest_pillar == 3


def test_similarity_identical():
    p1 = ProbabilisticPSV()
    p2 = ProbabilisticPSV()
    assert p1.similarity(p2) > 0.99


def test_distance_zero():
    p1 = ProbabilisticPSV()
    p2 = ProbabilisticPSV()
    assert p1.distance(p2) < 0.01


def test_coherence():
    p = ProbabilisticPSV()
    # All same value = high coherence
    for pillar in p.pillars:
        pillar.mean = 0.5
    assert p.coherence() > 0.9


def test_pillar_summary():
    p = ProbabilisticPSV()
    s = p.pillar_summary()
    assert "Awareness" in s
    assert len(s) == 16


def test_uncertain_pillars():
    p = ProbabilisticPSV()
    p.pillars[0].variance = 0.5
    p.pillars[1].variance = 0.1
    uncertain = p.uncertain_pillars(threshold=0.3)
    assert "Awareness" in uncertain
    assert "Willpower" not in uncertain


def test_copy():
    p = ProbabilisticPSV()
    p.pillars[0].mean = 0.9
    p2 = p.copy()
    p2.pillars[0].mean = 0.1
    assert p.pillars[0].mean == 0.9  # original unchanged


def test_update_reduces_variance():
    p = ProbabilisticPSV()
    before = p.pillars[0].variance
    for _ in range(10):
        p.update_from_observation({"Awareness": 0.8})
    after = p.pillars[0].variance
    assert after < before
