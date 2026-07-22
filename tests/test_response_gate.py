"""Tests for Privacy-Aware Response Gating."""
import sys
sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.response_gate import (
    ResponseGate, ResponseGateConfig, GateDecision,
)


# ── Basic Gating ─────────────────────────────────────────────────

def test_high_confidence_allowed():
    g = ResponseGate()
    d = g.evaluate(intent_confidence=0.9, dwell_frames=5)
    assert d.allowed is True


def test_low_confidence_blocked():
    g = ResponseGate()
    d = g.evaluate(intent_confidence=0.1, dwell_frames=5)
    assert d.allowed is False


def test_threshold_boundary():
    g = ResponseGate(ResponseGateConfig(confidence_threshold=0.5, privacy_level=0.0))
    d = g.evaluate(intent_confidence=0.5, dwell_frames=5)
    assert d.allowed is True


# ── Privacy Level ────────────────────────────────────────────────

def test_high_privacy_raises_threshold():
    g = ResponseGate(ResponseGateConfig(
        confidence_threshold=0.3,
        privacy_level=0.8,
    ))
    d = g.evaluate(intent_confidence=0.5, dwell_frames=5)
    assert d.allowed is False  # threshold raised by privacy


def test_zero_privacy():
    g = ResponseGate(ResponseGateConfig(
        confidence_threshold=0.3,
        privacy_level=0.0,
    ))
    d = g.evaluate(intent_confidence=0.5, dwell_frames=5)
    assert d.allowed is True


# ── Observer Effect ──────────────────────────────────────────────

def test_observers_raise_threshold():
    g = ResponseGate(ResponseGateConfig(
        confidence_threshold=0.3,
        privacy_level=0.3,
    ))
    d_no = g.evaluate(intent_confidence=0.5, observers=0, dwell_frames=5)
    g.reset()
    d_yes = g.evaluate(intent_confidence=0.5, observers=5, dwell_frames=5)
    assert d_no.effective_threshold <= d_yes.effective_threshold


# ── Relationship Discount ────────────────────────────────────────

def test_relationship_lowers_threshold():
    g = ResponseGate(ResponseGateConfig(
        confidence_threshold=0.5,
        privacy_level=0.3,
    ))
    d_stranger = g.evaluate(
        intent_confidence=0.55, relationship_strength=0.0, dwell_frames=5)
    g.reset()
    d_friend = g.evaluate(
        intent_confidence=0.55, relationship_strength=1.0, dwell_frames=5)
    assert d_stranger.allowed is False or d_friend.effective_threshold < d_stranger.effective_threshold


# ── Dwell Time Gate ──────────────────────────────────────────────

def test_new_entity_suppressed():
    g = ResponseGate()
    d = g.evaluate(intent_confidence=0.9, dwell_frames=1)
    assert d.allowed is False
    assert "dwell_frames" in d.reason


def test_entity_enough_dwell():
    g = ResponseGate()
    d = g.evaluate(intent_confidence=0.9, dwell_frames=3)
    assert d.allowed is True


def test_custom_min_dwell():
    g = ResponseGate(ResponseGateConfig(min_dwell_frames=10))
    d = g.evaluate(intent_confidence=0.9, dwell_frames=8)
    assert d.allowed is False


# ── Cooldown ─────────────────────────────────────────────────────

def test_cooldown_after_response():
    g = ResponseGate(ResponseGateConfig(cooldown_frames=3))
    g.evaluate(intent_confidence=0.9, entity_id="a", dwell_frames=5)
    d2 = g.evaluate(intent_confidence=0.9, entity_id="a", dwell_frames=5)
    assert d2.allowed is False
    assert "cooldown" in d2.reason


def test_cooldown_expires():
    g = ResponseGate(ResponseGateConfig(cooldown_frames=2))
    g.evaluate(intent_confidence=0.9, entity_id="a", dwell_frames=5)
    g.tick()
    g.tick()
    g.tick()
    d = g.evaluate(intent_confidence=0.9, entity_id="a", dwell_frames=5)
    assert d.allowed is True


def test_different_entities_independent_cooldowns():
    g = ResponseGate(ResponseGateConfig(cooldown_frames=3))
    g.evaluate(intent_confidence=0.9, entity_id="a", dwell_frames=5)
    d = g.evaluate(intent_confidence=0.9, entity_id="b", dwell_frames=5)
    assert d.allowed is True


# ── Response Cost ────────────────────────────────────────────────

def test_high_cost_raises_threshold():
    g = ResponseGate(ResponseGateConfig(confidence_threshold=0.3))
    d_low = g.evaluate(intent_confidence=0.5, response_cost=0.0, dwell_frames=5)
    g.reset()
    d_high = g.evaluate(intent_confidence=0.5, response_cost=0.9, dwell_frames=5)
    assert d_high.effective_threshold >= d_low.effective_threshold


# ── Social Openness ──────────────────────────────────────────────

def test_high_openness_lowers_threshold():
    g = ResponseGate(ResponseGateConfig(confidence_threshold=0.5))
    d_closed = g.evaluate(intent_confidence=0.45, social_openness=0.0, dwell_frames=5)
    g.reset()
    d_open = g.evaluate(intent_confidence=0.45, social_openness=1.0, dwell_frames=5)
    assert d_open.effective_threshold < d_closed.effective_threshold


# ── Suppression Strength ─────────────────────────────────────────

def test_suppressed_has_high_suppression():
    g = ResponseGate()
    d = g.evaluate(intent_confidence=0.1, dwell_frames=5)
    assert d.suppression_strength > 0.5


def test_allowed_has_low_suppression():
    g = ResponseGate()
    d = g.evaluate(intent_confidence=0.9, dwell_frames=5)
    assert d.suppression_strength < 0.5


# ── Stats ────────────────────────────────────────────────────────

def test_stats_tracking():
    g = ResponseGate()
    g.evaluate(intent_confidence=0.9, dwell_frames=5)  # allowed
    g.evaluate(intent_confidence=0.1, dwell_frames=5)  # suppressed
    s = g.stats
    assert s["allowed"] == 1
    assert s["suppressed"] == 1
    assert s["suppression_rate"] == 0.5


# ── Tick ─────────────────────────────────────────────────────────

def test_tick_advances_cooldowns():
    g = ResponseGate(ResponseGateConfig(cooldown_frames=3))
    g.evaluate(intent_confidence=0.9, entity_id="x", dwell_frames=5)
    for _ in range(3):
        g.tick()
    d = g.evaluate(intent_confidence=0.9, entity_id="x", dwell_frames=5)
    assert d.allowed is True


# ── Reset ────────────────────────────────────────────────────────

def test_reset():
    g = ResponseGate()
    g.evaluate(intent_confidence=0.9, entity_id="a", dwell_frames=5)
    g.reset()
    assert g.stats["allowed"] == 0
    assert g.stats["suppressed"] == 0
    d = g.evaluate(intent_confidence=0.9, entity_id="a", dwell_frames=5)
    assert d.allowed is True


# ── to_dict ──────────────────────────────────────────────────────

def test_to_dict():
    g = ResponseGate()
    d = g.evaluate(intent_confidence=0.7, dwell_frames=5)
    dct = d.to_dict()
    assert "allowed" in dct
    assert "effective_threshold" in dct


# ── Effective Threshold Bounds ───────────────────────────────────

def test_threshold_always_in_0_1():
    g = ResponseGate(ResponseGateConfig(
        confidence_threshold=0.9,
        privacy_level=0.9,
    ))
    d = g.evaluate(
        intent_confidence=0.5,
        observers=10,
        relationship_strength=1.0,
        dwell_frames=5,
    )
    assert 0.0 <= d.effective_threshold <= 1.0
