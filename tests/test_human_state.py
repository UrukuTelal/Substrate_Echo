"""Tests for HSV (Human State Vector)."""
import numpy as np
import sys
sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.human_state import (
    GaussianDim, HSVState, HumanObservation, HumanStateEstimator,
    IntentHypothesis, HSV_DIM, HSV_DIM_NAMES, SIGNAL_MAP,
)


# ── GaussianDim ──────────────────────────────────────────────────

def test_gaussian_dim_default():
    d = GaussianDim()
    assert d.mean == 0.5
    assert d.variance == 0.25
    assert d.confidence > 0.0


def test_gaussian_dim_update_reduces_variance():
    d = GaussianDim(mean=0.5, variance=0.25)
    for _ in range(10):
        d.update(0.7, 0.05)
    assert d.variance < 0.1
    assert abs(d.mean - 0.7) < 0.1


def test_gaussian_dim_update_moves_mean():
    d = GaussianDim(mean=0.5, variance=0.25)
    d.update(0.9, 0.05)
    assert d.mean > 0.5


def test_gaussian_dim_update_clamps():
    d = GaussianDim(mean=0.1, variance=0.01)
    d.update(0.0, 0.001)
    assert d.mean >= 0.0
    d.update(1.0, 0.001)
    assert d.mean <= 1.0


def test_gaussian_dim_sample():
    d = GaussianDim(mean=0.5, variance=0.01)
    samples = [d.sample() for _ in range(50)]
    assert all(0.0 <= s <= 1.0 for s in samples)
    assert abs(np.mean(samples) - 0.5) < 0.1


def test_gaussian_dim_confidence():
    d1 = GaussianDim(mean=0.5, variance=0.01)  # low variance = high conf
    d2 = GaussianDim(mean=0.5, variance=0.4)   # high variance = low conf
    assert d1.confidence > d2.confidence


# ── HSVState ─────────────────────────────────────────────────────

def test_hsv_state_default():
    s = HSVState()
    assert s.means.shape == (7,)
    assert s.variances.shape == (7,)
    assert all(m == 0.5 for m in s.means)
    assert s.uncertainty == 0.25


def test_hsv_state_to_from_array():
    s1 = HSVState()
    s1.arousal.mean = 0.8
    s1.arousal.variance = 0.1
    s1.valence.mean = 0.3
    
    arr = s1.to_array()
    assert arr.shape == (14,)
    
    s2 = HSVState.from_array(arr)
    assert abs(s2.arousal.mean - 0.8) < 1e-6
    assert abs(s2.arousal.variance - 0.1) < 1e-6
    assert abs(s2.valence.mean - 0.3) < 1e-6


def test_hsv_state_confidence():
    s = HSVState()
    # All at default 0.25 variance → confidence = 1 - 0.25*2 = 0.5
    assert abs(s.confidence - 0.5) < 1e-6


def test_hsv_state_repr():
    s = HSVState()
    r = repr(s)
    assert "HSV" in r
    assert "±" in r


# ── HumanObservation ─────────────────────────────────────────────

def test_observation_nan_default():
    obs = HumanObservation()
    assert obs.get_signal("speech_level") is None
    assert obs.get_signal("nonexistent") is None


def test_observation_get_signal():
    obs = HumanObservation(speech_level=0.6, facing_toward=0.9)
    assert obs.get_signal("speech_level") == 0.6
    assert obs.get_signal("facing_toward") == 0.9
    assert obs.get_signal("gesture_speed") is None


# ── Signal Map Coverage ──────────────────────────────────────────

def test_signal_map_covers_all_dims():
    for dim_name in HSV_DIM_NAMES:
        assert dim_name in SIGNAL_MAP
        assert len(SIGNAL_MAP[dim_name]) > 0


def test_signal_map_weights_in_range():
    for dim_name, signals in SIGNAL_MAP.items():
        for sig_name, weight in signals:
            assert -1.0 <= weight <= 1.0


# ── HumanStateEstimator ──────────────────────────────────────────

def test_estimator_init():
    e = HumanStateEstimator()
    assert e.observation_count == 0
    assert e.estimate is not None
    assert e.estimate.uncertainty == 0.25


def test_estimator_single_observation():
    e = HumanStateEstimator()
    obs = HumanObservation(speech_level=0.8, gesture_speed=0.7)
    e.observe(obs)
    assert e.observation_count == 1
    # Arousal should increase (speech + gesture = arousal)
    assert e.estimate.arousal.mean > 0.5


def test_estimator_variance_decreases():
    e = HumanStateEstimator()
    for _ in range(10):
        e.observe(HumanObservation(
            speech_level=0.7, gesture_speed=0.6,
            motion_speed=0.5))
    assert e.estimate.arousal.variance < 0.1


def test_estimator_observe_signals():
    e = HumanStateEstimator()
    e.observe_signals(speech_level=0.8, gesture_speed=0.7)
    assert e.observation_count == 1
    assert e.estimate.arousal.mean > 0.5


def test_estimator_no_signal_grows_uncertainty():
    e = HumanStateEstimator()
    initial_var = e.estimate.attention.variance
    # Observe with no attention-related signals
    e.observe(HumanObservation(speech_level=0.5))
    assert e.estimate.attention.variance >= initial_var


def test_estimator_high_fidget_reduces_attention():
    e = HumanStateEstimator()
    for _ in range(5):
        e.observe(HumanObservation(fidget_level=0.9))
    assert e.estimate.attention.mean < 0.4


def test_estimator_social_signals():
    e = HumanStateEstimator()
    for _ in range(5):
        e.observe(HumanObservation(
            facing_toward=0.9, approach_behavior=0.8,
            eye_contact_frequency=0.7))
    assert e.estimate.social_openness.mean > 0.6


def test_estimator_fatigue_signals():
    e = HumanStateEstimator()
    for _ in range(5):
        e.observe(HumanObservation(
            blink_rate=0.9, posture_slump=0.8,
            response_latency=0.7, motion_speed=0.1))
    assert e.estimate.fatigue.mean > 0.6


def test_estimator_valence_signals():
    e = HumanStateEstimator()
    for _ in range(5):
        e.observe(HumanObservation(
            facial_openness=0.9, vocal_tone=0.8,
            posture_openness=0.7))
    assert e.estimate.valence.mean > 0.6


# ── Intent Inference ─────────────────────────────────────────────

def test_infer_intents_returns_list():
    e = HumanStateEstimator()
    e.observe(HumanObservation(speech_level=0.5))
    intents = e.infer_intents()
    assert len(intents) == 5  # 5 hypotheses


def test_infer_intents_normalized():
    e = HumanStateEstimator()
    for _ in range(5):
        e.observe(HumanObservation(
            speech_level=0.7, facing_toward=0.8,
            facial_openness=0.9))
    intents = e.infer_intents()
    total = sum(i.probability for i in intents)
    assert abs(total - 1.0) < 0.01


def test_infer_intents_sorted():
    e = HumanStateEstimator()
    e.observe(HumanObservation(facial_openness=0.9, facing_toward=0.9))
    intents = e.infer_intents()
    for i in range(len(intents) - 1):
        assert intents[i].probability >= intents[i + 1].probability


def test_friendly_when_positive_valence():
    e = HumanStateEstimator()
    for _ in range(10):
        e.observe(HumanObservation(
            facial_openness=0.9, vocal_tone=0.8,
            posture_openness=0.8, facing_toward=0.9,
            approach_behavior=0.7))
    intents = e.infer_intents()
    # Should be friendly or social
    top = intents[0].label
    assert top in ("friendly_approach", "social_engagement")


def test_threatening_when_negative():
    e = HumanStateEstimator()
    for _ in range(10):
        e.observe(HumanObservation(
            speech_level=0.9, gesture_speed=0.9,
            motion_speed=0.8, body_tension=0.9,
            facial_openness=0.1, vocal_tone=0.1))
    intents = e.infer_intents()
    # Threat should be high
    threat = [i for i in intents if i.label == "threatening"]
    assert len(threat) == 1
    assert threat[0].probability > 0.15


def test_fatigued_when_tired():
    e = HumanStateEstimator()
    for _ in range(10):
        e.observe(HumanObservation(
            blink_rate=0.9, posture_slump=0.9,
            response_latency=0.8, motion_speed=0.1))
    intents = e.infer_intents()
    fatigued = [i for i in intents if i.label == "fatigued"]
    assert len(fatigued) == 1
    assert fatigued[0].probability > 0.2


def test_low_confidence_when_no_data():
    e = HumanStateEstimator()
    intents = e.infer_intents()
    # With no observations, confidence is 0.5 → all probabilities scaled down
    total = sum(i.probability for i in intents)
    assert total < 1.0 or all(i.probability < 0.3 for i in intents)


# ── Prediction ───────────────────────────────────────────────────

def test_predict_decays_to_neutral():
    e = HumanStateEstimator()
    e._state.arousal.mean = 0.9
    predicted = e.predict_next(dt=5.0)
    assert predicted.arousal.mean < 0.9
    assert abs(predicted.arousal.mean - 0.5) < 0.5


def test_predict_increases_uncertainty():
    e = HumanStateEstimator()
    e._state.arousal.variance = 0.1
    predicted = e.predict_next(dt=2.0)
    assert predicted.arousal.variance > 0.1


# ── History ──────────────────────────────────────────────────────

def test_history_stored():
    e = HumanStateEstimator()
    for _ in range(5):
        e.observe(HumanObservation(speech_level=0.5))
    assert len(e.get_history()) == 5


def test_history_bounded():
    e = HumanStateEstimator()
    for _ in range(25):
        e.observe(HumanObservation(speech_level=0.5))
    assert len(e.get_history()) <= 20


# ── Reset ────────────────────────────────────────────────────────

def test_reset():
    e = HumanStateEstimator()
    e.observe(HumanObservation(speech_level=0.9))
    e.reset()
    assert e.observation_count == 0
    assert e.estimate.arousal.mean == 0.5
    assert e.estimate.uncertainty == 0.25


# ── Edge Cases ───────────────────────────────────────────────────

def test_all_nan_observation():
    e = HumanStateEstimator()
    e.observe(HumanObservation())  # all NaN
    assert e.observation_count == 1
    # Variances should grow slightly
    assert e.estimate.arousal.variance >= 0.25


def test_mixed_signals():
    """Conflicting signals should produce higher variance."""
    e1 = HumanStateEstimator()
    e2 = HumanStateEstimator()
    
    # Consistent: high arousal signals
    for _ in range(5):
        e1.observe(HumanObservation(
            speech_level=0.8, gesture_speed=0.8,
            motion_speed=0.8))
    
    # Mixed: some high, some low
    for _ in range(5):
        e2.observe(HumanObservation(
            speech_level=0.8, gesture_speed=0.2,
            motion_speed=0.8))
    
    # Mixed should have higher variance (lower confidence)
    assert e2.estimate.arousal.variance >= e1.estimate.arousal.variance


def test_intent_hypothesis_has_pillars():
    e = HumanStateEstimator()
    e.observe(HumanObservation(facial_openness=0.9))
    intents = e.infer_intents()
    for intent in intents:
        assert len(intent.supporting_pillars) > 0
        assert len(intent.pillar_deltas) == len(intent.supporting_pillars)
