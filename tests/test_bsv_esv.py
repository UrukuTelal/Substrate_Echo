"""Tests for BSV and ESV standardization."""
import numpy as np
import sys
sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.bsv import (
    BiologicalStateVector, BSV_DIMENSIONS, NUM_BSV_DIMS,
)
from substrate_echo.core.esv import (
    EnvironmentalStateVector, ESV_DIMENSIONS, NUM_ESV_DIMS,
)
from substrate_echo.core.human_state import GaussianDim


# ═════════════════════════════════════════════════════════════════
# BSV — Biological State Vector
# ═════════════════════════════════════════════════════════════════

def test_bsv_creation():
    b = BiologicalStateVector()
    assert len(b.dim_names) == 8
    assert b.means.shape == (8,)
    assert b.variances.shape == (8,)


def test_bsv_default_viability():
    b = BiologicalStateVector()
    assert b.is_viable()


def test_bsv_composite_health_range():
    b = BiologicalStateVector()
    h = b.composite_health()
    assert 0.0 <= h <= 1.0


def test_bsv_update_from_observation():
    b = BiologicalStateVector()
    b.update_from_observation({"metabolic_rate": 0.9, "stress_level": 0.1})
    assert b.metabolic_rate.mean > 0.5
    assert b.stress_level.mean < 0.5


def test_bsv_update_unrecognized_key():
    b = BiologicalStateVector()
    old_health = b.health.mean
    b.update_from_observation({"nonexistent_dim": 0.9})
    assert b.health.mean == old_health


def test_bsv_array_roundtrip():
    b = BiologicalStateVector()
    arr = b.to_array()
    assert arr.shape == (16,)
    b2 = BiologicalStateVector.from_array(arr)
    for d in BSV_DIMENSIONS:
        assert abs(getattr(b, d).mean - getattr(b2, d).mean) < 1e-6
        assert abs(getattr(b, d).variance - getattr(b2, d).variance) < 1e-6


def test_bsv_update_from_engine_bio_state():
    b = BiologicalStateVector()
    b.update_from_engine_bio_state(food_level=80.0, has_shelter=True,
                                    offspring_count=2)
    assert b.energy_level.mean > 0.5
    assert b.reproductive_readiness.mean > 0.0


def test_bsv_update_from_engine_projection():
    b = BiologicalStateVector()
    b.update_from_engine_projection(metabolic_rate=0.9,
                                     membrane_integrity=0.7,
                                     stress_level=0.2)
    assert b.metabolic_rate.mean > 0.7
    assert b.membrane_integrity.mean > 0.6


def test_bsv_uncertainty():
    b = BiologicalStateVector()
    u = b.uncertainty
    assert 0.0 < u < 1.0


def test_bsv_not_viable_when_damaged():
    b = BiologicalStateVector()
    b.health = GaussianDim(0.05, 0.01)
    b.energy_level = GaussianDim(0.05, 0.01)
    b.membrane_integrity = GaussianDim(0.05, 0.01)
    assert not b.is_viable(threshold=0.3)


def test_bsv_to_dict():
    b = BiologicalStateVector()
    d = b.to_dict()
    assert "metabolic_rate" in d
    assert "mean" in d["metabolic_rate"]


# ═════════════════════════════════════════════════════════════════
# ESV — Environmental State Vector
# ═════════════════════════════════════════════════════════════════

def test_esv_creation():
    e = EnvironmentalStateVector()
    assert len(e.dim_names) == 8
    assert e.means.shape == (8,)
    assert e.variances.shape == (8,)


def test_esv_default_habitability():
    e = EnvironmentalStateVector()
    h = e.habitability()
    assert 0.0 <= h <= 1.0


def test_esv_danger_score_range():
    e = EnvironmentalStateVector()
    d = e.danger_score()
    assert 0.0 <= d <= 1.0


def test_esv_update_from_observation():
    e = EnvironmentalStateVector()
    e.update_from_observation({"temperature": 0.9, "hazard_level": 0.8})
    assert e.temperature.mean > 0.5
    assert e.hazard_level.mean > 0.1  # pulled up from default 0.1


def test_esv_update_unrecognized_key():
    e = EnvironmentalStateVector()
    old_temp = e.temperature.mean
    e.update_from_observation({"fake_key": 0.9})
    assert e.temperature.mean == old_temp


def test_esv_array_roundtrip():
    e = EnvironmentalStateVector()
    arr = e.to_array()
    assert arr.shape == (16,)
    e2 = EnvironmentalStateVector.from_array(arr)
    for d in ESV_DIMENSIONS:
        assert abs(getattr(e, d).mean - getattr(e2, d).mean) < 1e-6
        assert abs(getattr(e, d).variance - getattr(e2, d).variance) < 1e-6


def test_esv_update_from_engine_cellstate():
    e = EnvironmentalStateVector()
    e.update_from_engine_cellstate(temp=22.0, depth=0.5, flux=0.8)
    assert e.resource_density.mean > 0.5
    assert e.light.mean > 0.5


def test_esv_habitable_environment():
    e = EnvironmentalStateVector()
    e.update_from_observation({
        "temperature": 0.5,
        "light": 0.7,
        "resource_density": 0.8,
        "hazard_level": 0.05,
        "pollution": 0.05,
    })
    assert e.habitability() > 0.5


def test_esv_dangerous_environment():
    e = EnvironmentalStateVector()
    e.hazard_level = GaussianDim(0.9, 0.05)
    e.pollution = GaussianDim(0.8, 0.05)
    e.terrain_complexity = GaussianDim(0.7, 0.05)
    assert e.danger_score() > 0.5


def test_esv_uncertainty():
    e = EnvironmentalStateVector()
    u = e.uncertainty
    assert 0.0 < u < 1.0


def test_esv_to_dict():
    e = EnvironmentalStateVector()
    d = e.to_dict()
    assert "temperature" in d
    assert "mean" in d["temperature"]


# ═════════════════════════════════════════════════════════════════
# Cross-type compatibility
# ═════════════════════════════════════════════════════════════════

def test_bsv_esv_same_interface():
    """BSV and ESV share the same structural interface."""
    b = BiologicalStateVector()
    e = EnvironmentalStateVector()
    assert b.means.shape == e.means.shape
    assert b.to_array().shape == e.to_array().shape
    assert len(b.dim_names) == len(e.dim_names)
