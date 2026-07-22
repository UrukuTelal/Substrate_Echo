"""Tests for Communicative Intent Detection."""
import numpy as np
import sys
sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.communicative_intent import (
    CommunicativeIntent, BehavioralSignals, CommunicativeSignal,
    CommunicativeIntentDetector,
)


AI_POS = np.array([0.0, 0.0, 0.0])


# ── Basic Analysis ───────────────────────────────────────────────

def test_no_signals_returns_none():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals()  # all zeros
    result = d.analyze(sig, entity_position=np.array([2.0, 2.0, 0.0]))
    assert result.intent == CommunicativeIntent.NONE
    assert result.confidence < 0.3


def test_empty_history():
    d = CommunicativeIntentDetector(AI_POS)
    assert len(d.get_history()) == 0


# ── Gaze Evidence ────────────────────────────────────────────────

def test_gaze_at_me_high_evidence():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gaze_direction=np.array([-1.0, 0.0, 0.0]),  # looking toward AI at origin
        gaze_confidence=0.9,
        distance=1.0,
    )
    result = d.analyze(sig, entity_position=np.array([1.0, 0.0, 0.0]))
    assert result.gaze_evidence > 0.7


def test_gaze_away_low_evidence():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gaze_direction=np.array([1.0, 0.0, 0.0]),  # looking away from AI
        gaze_confidence=0.9,
        distance=1.0,
    )
    result = d.analyze(sig, entity_position=np.array([1.0, 0.0, 0.0]))
    assert result.gaze_evidence < 0.1


def test_gaze_low_confidence():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gaze_direction=np.array([-1.0, 0.0, 0.0]),
        gaze_confidence=0.05,
    )
    result = d.analyze(sig)
    assert result.gaze_evidence == 0.0


# ── Speech Evidence ──────────────────────────────────────────────

def test_no_speech():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(speech_level=0.0)
    result = d.analyze(sig)
    assert result.speech_evidence == 0.0


def test_speech_with_facing():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        speech_level=0.8,
        speech_duration=2.0,
        facing_toward_me=True,
    )
    result = d.analyze(sig, entity_position=np.array([1.0, 0.0, 0.0]))
    assert result.speech_evidence > 0.4


def test_longer_speech_higher_evidence():
    d = CommunicativeIntentDetector(AI_POS)
    sig1 = BehavioralSignals(speech_level=0.5, speech_duration=1.0)
    sig2 = BehavioralSignals(speech_level=0.5, speech_duration=5.0)
    r1 = d.analyze(sig1)
    d2 = CommunicativeIntentDetector(AI_POS)
    r2 = d2.analyze(sig2)
    assert r2.speech_evidence >= r1.speech_evidence


# ── Gesture Evidence ─────────────────────────────────────────────

def test_gesture_directed_at_me():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gesture_speed=0.8,
        gesture_direction=np.array([-1.0, 0.0, 0.0]),
    )
    result = d.analyze(sig, entity_position=np.array([1.0, 0.0, 0.0]))
    assert result.gesture_evidence > 0.5


def test_gesture_away_from_me():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gesture_speed=0.8,
        gesture_direction=np.array([1.0, 0.0, 0.0]),  # away from AI
    )
    result = d.analyze(sig, entity_position=np.array([1.0, 0.0, 0.0]))
    assert result.gesture_evidence < 0.5


def test_repeated_gesture_boost():
    d1 = CommunicativeIntentDetector(AI_POS)
    d2 = CommunicativeIntentDetector(AI_POS)
    sig1 = BehavioralSignals(gesture_speed=0.5, gesture_repetition=0)
    sig2 = BehavioralSignals(gesture_speed=0.5, gesture_repetition=3)
    r1 = d1.analyze(sig1)
    r2 = d2.analyze(sig2)
    assert r2.gesture_evidence > r1.gesture_evidence


# ── Directed Detection ───────────────────────────────────────────

def test_directed_gaze_close():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gaze_direction=np.array([-1.0, 0.0, 0.0]),
        gaze_confidence=0.9,
        distance=1.0,
    )
    result = d.analyze(sig, entity_position=np.array([1.0, 0.0, 0.0]))
    assert result.is_directed_at_me


def test_directed_speech_facing():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        speech_level=0.6,
        facing_toward_me=True,
        distance=1.5,
    )
    result = d.analyze(sig, entity_position=np.array([1.0, 0.0, 0.0]))
    assert result.is_directed_at_me


def test_directed_gesture_facing():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gesture_speed=0.5,
        facing_toward_me=True,
        distance=1.0,
    )
    result = d.analyze(sig, entity_position=np.array([1.0, 0.0, 0.0]))
    assert result.is_directed_at_me


def test_not_directed_far_away():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gaze_direction=np.array([0.0, 1.0, 0.0]),
        gaze_confidence=0.5,
        distance=5.0,
        facing_toward_me=False,
    )
    result = d.analyze(sig, entity_position=np.array([5.0, 0.0, 0.0]))
    assert not result.is_directed_at_me


# ── Intent Classification ────────────────────────────────────────

def test_greeting():
    """Brief facing + gaze = greeting."""
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gaze_direction=np.array([-1.0, 0.0, 0.0]),
        gaze_confidence=0.9,
        facing_toward_me=True,
        distance=1.5,
        signal_duration=1.5,
        speech_level=0.2,
    )
    result = d.analyze(sig, entity_position=np.array([1.0, 0.0, 0.0]))
    assert result.intent in (CommunicativeIntent.GREETING, CommunicativeIntent.INFORMATION)
    assert result.is_directed_at_me


def test_request():
    """Gesture + speech + close = request."""
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gesture_speed=0.7,
        gesture_direction=np.array([-1.0, 0.0, 0.0]),
        gesture_repetition=2,
        speech_level=0.6,
        speech_duration=2.0,
        distance=1.0,
        facing_toward_me=True,
    )
    result = d.analyze(sig, entity_position=np.array([1.0, 0.0, 0.0]))
    assert result.intent in (CommunicativeIntent.REQUEST, CommunicativeIntent.DIRECTIVE)
    assert result.is_request or result.intent == CommunicativeIntent.DIRECTIVE


def test_warning():
    """High speech + fast gesture = warning."""
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        speech_level=0.9,
        speech_duration=1.0,
        gesture_speed=0.9,
        gesture_direction=np.array([-1.0, 0.0, 0.0]),
        facing_toward_me=True,
        distance=2.0,
    )
    result = d.analyze(sig, entity_position=np.array([2.0, 0.0, 0.0]))
    assert result.intent == CommunicativeIntent.WARNING
    assert result.is_warning


def test_bid_for_attention():
    """Gesture + approach + facing = bid for attention or invitation."""
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gesture_speed=0.9,
        gesture_direction=np.array([-1.0, 0.0, 0.0]),
        approach_speed=0.7,
        distance=2.0,
        facing_toward_me=True,
        gesture_repetition=3,
    )
    result = d.analyze(sig, entity_position=np.array([2.0, 0.0, 0.0]))
    assert result.intent in (CommunicativeIntent.BID_FOR_ATTENTION,
                             CommunicativeIntent.INVITATION,
                             CommunicativeIntent.REQUEST)
    assert result.is_directed_at_me


# ── Temporal Refinement ──────────────────────────────────────────

def test_repeated_signal_increases_confidence():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gesture_speed=0.5,
        gesture_direction=np.array([-1.0, 0.0, 0.0]),
        facing_toward_me=True,
        distance=1.5,
    )
    entity_pos = np.array([1.5, 0.0, 0.0])
    
    # First observation
    r1 = d.analyze(sig, entity_position=entity_pos)
    conf1 = r1.confidence
    
    # Repeat several times
    for _ in range(5):
        r = d.analyze(sig, entity_position=entity_pos)
    
    assert r.confidence >= conf1  # should not decrease


def test_consistent_gaze_boosts_confidence():
    d = CommunicativeIntentDetector(AI_POS)
    entity_pos = np.array([1.0, 0.0, 0.0])
    
    # Consistent gaze at AI
    for _ in range(5):
        sig = BehavioralSignals(
            gaze_direction=np.array([-1.0, 0.0, 0.0]),
            gaze_confidence=0.8,
            distance=1.0,
        )
        r = d.analyze(sig, entity_position=entity_pos)
    
    assert r.gaze_evidence > 0.5
    assert r.confidence > 0.3


# ── Requires Response ────────────────────────────────────────────

def test_request_requires_response():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gesture_speed=0.8,
        gesture_direction=np.array([-1.0, 0.0, 0.0]),
        gesture_repetition=3,
        speech_level=0.7,
        speech_duration=3.0,
        distance=0.8,
        facing_toward_me=True,
    )
    result = d.analyze(sig, entity_position=np.array([0.8, 0.0, 0.0]))
    # With strong gesture + speech + close proximity, should be REQUEST or DIRECTIVE
    assert result.is_directed_at_me
    assert result.confidence > 0.4
    if result.intent in (CommunicativeIntent.REQUEST, CommunicativeIntent.DIRECTIVE):
        assert result.requires_response


def test_greeting_no_response_needed():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gaze_direction=np.array([-1.0, 0.0, 0.0]),
        gaze_confidence=0.7,
        facing_toward_me=True,
        distance=2.0,
        signal_duration=1.0,
        speech_level=0.1,
    )
    result = d.analyze(sig, entity_position=np.array([2.0, 0.0, 0.0]))
    # Greetings don't require response
    if result.intent == CommunicativeIntent.GREETING:
        assert not result.requires_response


# ── Signal Strength ──────────────────────────────────────────────

def test_high_signal_strength():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gaze_direction=np.array([-1.0, 0.0, 0.0]),
        gaze_confidence=0.9,
        speech_level=0.8,
        speech_duration=2.0,
        gesture_speed=0.7,
        gesture_direction=np.array([-1.0, 0.0, 0.0]),
        facing_toward_me=True,
        distance=1.0,
    )
    result = d.analyze(sig, entity_position=np.array([1.0, 0.0, 0.0]))
    assert result.signal_strength > 0.5


def test_low_signal_strength():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        distance=3.0,
        facing_toward_me=False,
    )
    result = d.analyze(sig, entity_position=np.array([3.0, 0.0, 0.0]))
    assert result.signal_strength < 0.3


# ── Proximity Evidence ───────────────────────────────────────────

def test_close_proximity_higher_evidence():
    d1 = CommunicativeIntentDetector(AI_POS)
    d2 = CommunicativeIntentDetector(AI_POS)
    sig1 = BehavioralSignals(approach_speed=0.3, distance=0.5)
    sig2 = BehavioralSignals(approach_speed=0.3, distance=2.5)
    r1 = d1.analyze(sig1, entity_position=np.array([0.5, 0.0, 0.0]))
    r2 = d2.analyze(sig2, entity_position=np.array([2.5, 0.0, 0.0]))
    assert r1.proximity_evidence > r2.proximity_evidence


# ── History ──────────────────────────────────────────────────────

def test_history_bounded():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(speech_level=0.3, facing_toward_me=True, distance=1.0)
    for _ in range(25):
        d.analyze(sig, entity_position=np.array([1.0, 0.0, 0.0]))
    assert len(d.get_history()) <= 20


# ── Set AI Position ──────────────────────────────────────────────

def test_set_ai_position():
    d = CommunicativeIntentDetector(np.array([5.0, 5.0, 0.0]))
    sig = BehavioralSignals(
        gaze_direction=np.array([-1.0, -1.0, 0.0]),
        gaze_confidence=0.9,
        distance=1.0,
    )
    result = d.analyze(sig, entity_position=np.array([6.0, 6.0, 0.0]))
    assert result.gaze_evidence > 0.5


# ── Edge Cases ───────────────────────────────────────────────────

def test_all_zeros():
    d = CommunicativeIntentDetector(AI_POS)
    result = d.analyze(BehavioralSignals())
    assert result.intent == CommunicativeIntent.NONE
    assert result.confidence <= 0.3
    assert not result.is_directed_at_me


def test_extreme_values():
    d = CommunicativeIntentDetector(AI_POS)
    sig = BehavioralSignals(
        gaze_confidence=1.0,
        speech_level=1.0,
        speech_duration=100.0,
        gesture_speed=1.0,
        gesture_repetition=50,
        distance=0.0,
        approach_speed=1.0,
    )
    result = d.analyze(sig, entity_position=np.array([0.01, 0.0, 0.0]))
    assert result.confidence <= 1.0  # capped
    assert result.signal_strength <= 1.0
