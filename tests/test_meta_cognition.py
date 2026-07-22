"""Tests for Meta-Cognition — P8"""
import numpy as np
from substrate_echo.core.meta_cognition import (
    MetaCognition, MetaCognitionConfig, MetaState,
    CalibrationRecord, ConfidenceSource
)


class TestMetaUpdate:
    def test_update_single(self):
        mc = MetaCognition()
        mc.update(0.8, actual_outcome_correct=True, source="dynamics_memory")
        assert len(mc._history) == 1

    def test_update_multiple_sources(self):
        mc = MetaCognition()
        mc.update(0.8, True, source="dynamics_memory")
        mc.update(0.6, True, source="self_model")
        mc.update(0.9, False, source="theory_of_mind")
        assert len(mc._history) == 3

    def test_update_multi(self):
        mc = MetaCognition()
        mc.update_multi({
            "dynamics_memory": 0.8,
            "self_model": 0.7,
        }, actual_outcome_correct=True)
        assert len(mc._history) == 2

    def test_history_eviction(self):
        config = MetaCognitionConfig(history_size=20)
        mc = MetaCognition(config)
        for i in range(30):
            mc.update(0.5, True, source="dynamics_memory")
        assert len(mc._history) == 20


class TestCalibration:
    def test_well_calibrated(self):
        mc = MetaCognition()
        # Predict 0.8 and be correct 80% of the time
        for i in range(50):
            mc.update(0.8, actual_outcome_correct=(i % 5 != 0),
                      source="dynamics_memory")
        state = mc.get_meta_state()
        # Brier score for 0.8/80% is ~0.16 (inherent variance)
        assert state.calibration_error < 0.25
        assert not state.overconfidence  # mean_pred ~= mean_outcome

    def test_overconfident(self):
        mc = MetaCognition()
        # Predict 0.9 but only correct 50% of the time
        for i in range(50):
            mc.update(0.9, actual_outcome_correct=(i % 2 == 0),
                      source="dynamics_memory")
        state = mc.get_meta_state()
        assert state.overconfidence > 0

    def test_underconfident(self):
        mc = MetaCognition()
        # Predict 0.3 but correct 80% of the time
        for i in range(50):
            mc.update(0.3, actual_outcome_correct=(i % 5 != 0),
                      source="dynamics_memory")
        state = mc.get_meta_state()
        assert state.underconfidence > 0


class TestMetaState:
    def test_initial_state(self):
        mc = MetaCognition()
        state = mc.get_meta_state()
        assert isinstance(state, MetaState)
        assert state.calibrated_confidence == 0.5 or state.calibrated_confidence > 0

    def test_state_summary(self):
        mc = MetaCognition()
        state = mc.get_meta_state()
        assert state.summary in ("confident", "uncertain", "cautious")

    def test_should_be_cautious_low_confidence(self):
        mc = MetaCognition()
        for i in range(30):
            mc.update(0.1, actual_outcome_correct=(i % 10 == 0),
                      source="dynamics_memory")
        state = mc.get_meta_state()
        assert state.should_be_cautious

    def test_should_be_cautious_disagreement(self):
        mc = MetaCognition()
        # Sources that disagree wildly
        for i in range(30):
            mc.update(0.9, actual_outcome_correct=True,
                      source="dynamics_memory")
            mc.update(0.1, actual_outcome_correct=True,
                      source="theory_of_mind")
        state = mc.get_meta_state()
        assert state.model_disagreement > 0


class TestTrust:
    def test_should_trust_initial(self):
        mc = MetaCognition()
        # Initially should trust (default 0.5 confidence)
        trust = mc.should_trust_myself()
        assert isinstance(trust, bool)

    def test_trust_decreases_with_overconfidence(self):
        mc = MetaCognition()
        for i in range(50):
            mc.update(0.9, actual_outcome_correct=(i % 2 == 0),
                      source="dynamics_memory")
        trust = mc.should_trust_myself()
        # After consistent overconfidence, should not trust
        assert not trust

    def test_trust_increases_with_calibration(self):
        mc = MetaCognition()
        for i in range(50):
            mc.update(0.7, actual_outcome_correct=(i % 10 < 7),
                      source="dynamics_memory")
        trust = mc.should_trust_myself()
        # Well calibrated should trust
        assert trust


class TestSourceReliability:
    def test_reliability_perfect(self):
        mc = MetaCognition()
        for i in range(20):
            mc.update(0.8, actual_outcome_correct=True,
                      source="dynamics_memory")
        rel = mc.get_source_reliability("dynamics_memory")
        assert rel == 1.0

    def test_reliability_poor(self):
        mc = MetaCognition()
        for i in range(20):
            mc.update(0.8, actual_outcome_correct=False,
                      source="dynamics_memory")
        rel = mc.get_source_reliability("dynamics_memory")
        assert rel == 0.0

    def test_reliability_unknown_source(self):
        mc = MetaCognition()
        rel = mc.get_source_reliability("unknown_source")
        assert rel == 0.5  # default


class TestDisagreement:
    def test_no_disagreement_single_source(self):
        mc = MetaCognition()
        d = mc.check_disagreement({"dynamics_memory": 0.8})
        assert d == 0.0

    def test_disagreement_high(self):
        mc = MetaCognition()
        d = mc.check_disagreement({
            "dynamics_memory": 0.9,
            "theory_of_mind": 0.1,
        })
        assert d > 0.3

    def test_disagreement_low(self):
        mc = MetaCognition()
        d = mc.check_disagreement({
            "dynamics_memory": 0.7,
            "self_model": 0.75,
        })
        assert d < 0.2


class TestConfidence:
    def test_confidence_overall(self):
        mc = MetaCognition()
        for i in range(20):
            mc.update(0.7, True, source="dynamics_memory")
        conf = mc.get_confidence()
        assert 0 <= conf <= 1

    def test_confidence_per_source(self):
        mc = MetaCognition()
        for i in range(20):
            mc.update(0.8, True, source="dynamics_memory")
            mc.update(0.3, True, source="theory_of_mind")
        dm_conf = mc.get_confidence("dynamics_memory")
        tom_conf = mc.get_confidence("theory_of_mind")
        # DM should be higher trust than ToM
        assert dm_conf > tom_conf


class TestSummary:
    def test_summary(self):
        mc = MetaCognition()
        mc.update(0.8, True, source="dynamics_memory")
        s = mc.summary()
        assert s["n_predictions"] == 1
        assert "calibrated_confidence" in s
        assert "should_trust" in s


class TestMetaStateFields:
    def test_state_fields(self):
        state = MetaState(
            calibrated_confidence=0.8,
            source_trust={"dynamics_memory": 0.9},
            calibration_error=0.1,
            model_disagreement=0.2,
        )
        assert state.calibrated_confidence == 0.8
        assert state.model_disagreement == 0.2
        assert not state.should_be_cautious

    def test_cautious_state(self):
        state = MetaState(
            calibrated_confidence=0.2,
            model_disagreement=0.6,
        )
        assert state.should_be_cautious
        assert state.summary == "cautious"
