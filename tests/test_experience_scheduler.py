"""Tests for Experience Adapter and Experience Scheduler"""
import numpy as np
from substrate_echo.core.experience_adapter import (
    ExperienceAdapter, ExperienceAdapterConfig, PerceptionFrame,
    ActionFrame, RewardSignal, ExperienceRecord
)
from substrate_echo.core.experience_scheduler import (
    ExperienceScheduler, ExperienceSchedulerConfig, EnvironmentType,
    EnvironmentProfile, IdlePrediction, WeaknessAnalysis,
    ENVIRONMENT_CATALOG
)
from substrate_echo.core.meta_cognition import MetaCognition
from substrate_echo.core.self_model import SelfModel


class TestExperienceAdapter:
    def test_create_adapter(self):
        adapter = ExperienceAdapter(env_name="test")
        assert adapter.env_name == "test"

    def test_observe_returns_frame(self):
        adapter = ExperienceAdapter()
        frame = adapter.observe({"x": 1.0, "y": 2.0})
        assert isinstance(frame, PerceptionFrame)
        assert frame.tick == 0

    def test_observe_with_entities(self):
        adapter = ExperienceAdapter()
        entities = [{"position": [3, 4], "type": "block"}]
        frame = adapter.observe({"x": 1}, entities=entities)
        assert len(frame.entities) == 1

    def test_observe_increments_tick(self):
        adapter = ExperienceAdapter()
        adapter.observe({})
        adapter.observe({})
        assert adapter._tick == 2

    def test_act_returns_frame(self):
        adapter = ExperienceAdapter()
        result = adapter.act({"type": "move", "direction": [1, 0]})
        assert isinstance(result, ActionFrame)
        assert result.action_type == "move"

    def test_record_outcome(self):
        adapter = ExperienceAdapter()
        adapter.observe({"x": 1})
        record = adapter.record_outcome(
            {"type": "move"}, reward=0.5, success=True)
        assert isinstance(record, ExperienceRecord)
        assert record.reward == 0.5

    def test_export_experience(self):
        adapter = ExperienceAdapter()
        adapter.observe({})
        adapter.record_outcome({"type": "move"}, reward=0.5)
        adapter.record_outcome({"type": "grasp"}, reward=1.0)
        batch = adapter.export_experience(n=2)
        assert len(batch) == 2

    def test_information_gap(self):
        adapter = ExperienceAdapter()
        # No history = high gap
        gaps = adapter.get_information_gap()
        assert gaps["general"] == 1.0

        # With history
        for i in range(10):
            adapter.observe({})
            adapter.record_outcome({"type": "move"}, reward=np.random.random())
        gaps = adapter.get_information_gap()
        assert "move" in gaps

    def test_stats(self):
        adapter = ExperienceAdapter(env_name="test")
        s = adapter.stats()
        assert s["env_name"] == "test"
        assert s["tick"] == 0

    def test_history_eviction(self):
        config = ExperienceAdapterConfig(history_size=20)
        adapter = ExperienceAdapter(config=config)
        for i in range(30):
            adapter.observe({})
            adapter.record_outcome({"type": "move"}, reward=0.5)
        assert len(adapter._history) <= 20

    def test_encode_state_dict(self):
        adapter = ExperienceAdapter()
        state = adapter._encode_state({"x": 1.0, "y": 2.0})
        assert isinstance(state, np.ndarray)
        assert len(state) == 16

    def test_encode_state_entities(self):
        adapter = ExperienceAdapter()
        entities = [{"position": [3, 4], "type": "block"}]
        state = adapter._encode_state(None, entities=entities)
        assert state[0] == 3 / 10.0
        assert state[1] == 4 / 10.0


class TestExperienceScheduler:
    def test_create_scheduler(self):
        mc = MetaCognition()
        sm = SelfModel()
        scheduler = ExperienceScheduler(meta_cognition=mc, self_model=sm)
        assert scheduler is not None

    def test_should_activate_initial(self):
        mc = MetaCognition()
        sm = SelfModel()
        scheduler = ExperienceScheduler(meta_cognition=mc, self_model=sm)
        # Initially, may or may not activate depending on load
        result = scheduler.should_activate()
        assert isinstance(result, bool)

    def test_predict_idle_period(self):
        mc = MetaCognition()
        scheduler = ExperienceScheduler(meta_cognition=mc)
        prediction = scheduler.predict_idle_period()
        assert isinstance(prediction, IdlePrediction)
        assert prediction.predicted_duration >= 0

    def test_analyze_weaknesses(self):
        mc = MetaCognition()
        sm = SelfModel()
        scheduler = ExperienceScheduler(meta_cognition=mc, self_model=sm)
        weakness = scheduler.analyze_weaknesses()
        assert isinstance(weakness, WeaknessAnalysis)
        assert len(weakness.weakest_modules) >= 0

    def test_select_environment(self):
        mc = MetaCognition()
        sm = SelfModel()
        scheduler = ExperienceScheduler(meta_cognition=mc, self_model=sm)
        env = scheduler.select_environment()
        assert isinstance(env, EnvironmentType)

    def test_create_session(self):
        mc = MetaCognition()
        sm = SelfModel()
        scheduler = ExperienceScheduler(meta_cognition=mc, self_model=sm)
        session = scheduler.create_session()
        assert "env_type" in session
        assert "budget" in session
        assert scheduler._session_active

    def test_complete_session(self):
        mc = MetaCognition()
        sm = SelfModel()
        scheduler = ExperienceScheduler(meta_cognition=mc, self_model=sm)
        session = scheduler.create_session()
        result = scheduler.complete_session(session)
        assert result["ticks_run"] == 0
        assert not scheduler._session_active

    def test_get_adapter(self):
        mc = MetaCognition()
        scheduler = ExperienceScheduler(meta_cognition=mc)
        session = scheduler.create_session()
        adapter = scheduler.get_adapter(session)
        assert isinstance(adapter, ExperienceAdapter)

    def test_session_history(self):
        mc = MetaCognition()
        scheduler = ExperienceScheduler(meta_cognition=mc)
        session = scheduler.create_session()
        scheduler.complete_session(session)
        history = scheduler.get_session_history()
        assert len(history) == 1

    def test_stats(self):
        mc = MetaCognition()
        scheduler = ExperienceScheduler(meta_cognition=mc)
        s = scheduler.stats()
        assert "sessions_completed" in s


class TestEnvironmentCatalog:
    def test_catalog_populated(self):
        assert len(ENVIRONMENT_CATALOG) >= 4

    def test_spatial_navigation_profile(self):
        profile = ENVIRONMENT_CATALOG[EnvironmentType.SPATIAL_NAVIGATION]
        assert "dynamics_memory" in profile.trains_modules
        assert profile.spatial_complexity > 0

    def test_social_simulation_profile(self):
        profile = ENVIRONMENT_CATALOG[EnvironmentType.SOCIAL_SIMULATION]
        assert "theory_of_mind" in profile.trains_modules
        assert profile.social_density > 0.5

    def test_adversarial_profile(self):
        profile = ENVIRONMENT_CATALOG[EnvironmentType.ADVERSARIAL_PLANNING]
        assert profile.adversarial
        assert "counterfactual" in profile.trains_modules


class TestWeaknessAnalysis:
    def test_fields(self):
        wa = WeaknessAnalysis(
            module_confidences={"dynamics_memory": 0.9, "theory_of_mind": 0.3},
            weakest_modules=["theory_of_mind"],
            recommendation="Focus on social simulation",
        )
        assert wa.weakest_modules == ["theory_of_mind"]
        assert "social" in wa.recommendation.lower()


class TestIdlePrediction:
    def test_fields(self):
        ip = IdlePrediction(
            predicted_duration=100,
            confidence=0.8,
            current_load=0.2,
        )
        assert ip.predicted_duration == 100
        assert ip.confidence == 0.8
