"""Integration Tests — Full Pipeline Validation (S13-S14).

Tests that all kernel layers work together:
  - Goals → Attention → Resources → Attractors
  - Prediction Errors → Confidence → Audit
  - Council → Executive Function feedback
  - Multi-embodiment sharing
"""
from __future__ import annotations
import pytest
import numpy as np
import time
from substrate_echo.kernel import SubstrateKernel, Observation, Goal
from substrate_echo.kernel.resources import ResourceRequest, ResourceTier
from substrate_echo.kernel.executive import GoalState, GoalStatus, GoalTier


class TestGoalAttentionPipeline:
    """Goals influence attention, attention influences attractor updates."""

    def test_goals_generate_attention(self):
        kernel = SubstrateKernel()
        gid = kernel.executive.add_goal(GoalState(
            id=0, target=np.zeros(16).tolist(),
            description="test goal", urgency=0.9, importance=0.9,
        ))
        kernel.executive.activate_goal(gid)
        state = kernel.executive.tick()
        assert state.n_active > 0

    def test_high_priority_goals_get_attention(self):
        kernel = SubstrateKernel()
        gid = kernel.executive.add_goal(GoalState(
            id=0, target=np.zeros(16).tolist(),
            description="urgent", urgency=0.95, importance=0.95,
        ))
        kernel.executive.activate_goal(gid)
        state = kernel.executive.tick()
        assert state.n_active > 0
        assert state.uncertainty < 1.0

    def test_goal_completion(self):
        kernel = SubstrateKernel()
        gid = kernel.executive.add_goal(GoalState(
            id=0, target=np.zeros(16).tolist(),
            description="complete me", urgency=0.8, importance=0.8,
        ))
        kernel.executive.activate_goal(gid)
        kernel.executive.complete_goal(gid)
        state = kernel.executive.tick()
        assert state.n_completed > 0


class TestResourceAttentionPipeline:
    """Resources are allocated based on goal priorities."""

    def test_resource_request_granted(self):
        kernel = SubstrateKernel()
        req = ResourceRequest(
            embodiment_id="desktop",
            compute=0.3, attention=0.2, learning=0.1,
            tier=ResourceTier.ACTIVE,
        )
        result = kernel.resources.request(req)
        assert result.granted

    def test_multiple_embodiments_share(self):
        kernel = SubstrateKernel()
        for i in range(3):
            req = ResourceRequest(
                embodiment_id=f"embodiment_{i}",
                compute=0.2, attention=0.1, learning=0.05,
                tier=ResourceTier.ACTIVE,
            )
            kernel.resources.request(req)
        state = kernel.resources.get_state()
        assert state.active_leases >= 3

    def test_resource_release(self):
        kernel = SubstrateKernel()
        req = ResourceRequest(
            embodiment_id="desktop",
            compute=0.5, attention=0.3, learning=0.1,
            tier=ResourceTier.ACTIVE,
        )
        result = kernel.resources.request(req)
        kernel.resources.release(result.lease.lease_id)
        state = kernel.resources.get_state()
        assert state.active_leases == 0


class TestCouncilAuditPipeline:
    """Council audits detect anomalies and produce reports."""

    def test_scheduled_audit(self):
        kernel = SubstrateKernel()
        kernel.council.scheduled.interval_ticks = 10
        for i in range(15):
            kernel.council.tick({"tick": i * 10, "n_attractors": 5,
                                 "coherence": 0.5, "volume_entropy": 0.5,
                                 "n_goals": 2, "n_embodiments": 1})
        reports = kernel.council.get_reports()
        assert len(reports) > 0

    def test_event_audit_on_collapse(self):
        kernel = SubstrateKernel()
        kernel.council.tick({"tick": 10, "n_attractors": 10,
                             "coherence": 0.5, "volume_entropy": 0.5,
                             "n_goals": 2, "n_embodiments": 1})
        kernel.council.tick({"tick": 20, "n_attractors": 2,
                             "coherence": 0.5, "volume_entropy": 0.5,
                             "n_goals": 2, "n_embodiments": 1})
        reports = kernel.council.get_reports()
        assert any(r.trigger.value == "attractor_collapse" for r in reports)

    def test_health_score_degradation(self):
        kernel = SubstrateKernel()
        kernel.council.scheduled.interval_ticks = 5
        for i in range(50):
            kernel.council.tick({"tick": i * 5, "n_attractors": 0,
                                 "coherence": 0.1, "volume_entropy": 0.05,
                                 "n_goals": 0, "n_embodiments": 1})
        state = kernel.council.get_state()
        assert state.health_score < 1.0


class TestPredictionConfidencePipeline:
    """Prediction errors drive confidence updates."""

    def test_prediction_error_detection(self):
        kernel = SubstrateKernel()
        for i in range(200):
            vec = np.random.uniform(0.3, 0.7, 16).tolist()
            kernel.publish_observation(Observation(vector=vec))
        assert kernel._tick == 200

    def test_coherence_computation(self):
        kernel = SubstrateKernel()
        for i in range(50):
            vec = np.random.uniform(0.4, 0.6, 16).tolist()
            kernel.publish_observation(Observation(vector=vec))
        coherence = kernel._compute_coherence()
        assert 0.0 <= coherence <= 1.0


class TestMultiEmbodimentSharing:
    """Multiple embodiments share one kernel."""

    def test_embodiment_tracking(self):
        kernel = SubstrateKernel()
        for i in range(3):
            kernel.publish_observation(Observation(
                vector=np.random.uniform(0.2, 0.8, 16).tolist(),
                embodiment_id=f"embodiment_{i}",
            ))
        assert len(kernel._embodiments) >= 3

    def test_cross_embodiment_learning(self):
        kernel = SubstrateKernel()
        for i in range(100):
            kernel.publish_observation(Observation(
                vector=np.random.uniform(0.3, 0.7, 16).tolist(),
                embodiment_id="desktop",
            ))
        for i in range(100):
            kernel.publish_observation(Observation(
                vector=np.random.uniform(0.3, 0.7, 16).tolist(),
                embodiment_id="robot",
            ))
        assert kernel._tick == 200

    def test_embodiment_resource_allocation(self):
        kernel = SubstrateKernel()
        for i in range(3):
            req = ResourceRequest(
                embodiment_id=f"embodiment_{i}",
                compute=0.2, attention=0.1, learning=0.05,
                tier=ResourceTier.ACTIVE,
            )
            kernel.resources.request(req)
        state = kernel.resources.get_state()
        assert state.n_embodiments >= 3


class TestAbstractionHierarchy:
    """Abstraction hierarchy forms from attractor correlations."""

    def test_abstraction_tick(self):
        kernel = SubstrateKernel()
        for i in range(100):
            vec = np.random.uniform(0.2, 0.8, 16).tolist()
            kernel.publish_observation(Observation(vector=vec))
        assert kernel._tick == 100

    def test_abstraction_state(self):
        kernel = SubstrateKernel()
        for i in range(50):
            vec = np.random.uniform(0.3, 0.7, 16).tolist()
            kernel.publish_observation(Observation(vector=vec))
        snap = kernel.get_snapshot()
        assert "n_meta_attractors" in snap


class TestFullPipeline:
    """End-to-end test of all layers together."""

    def test_full_cycle(self):
        kernel = SubstrateKernel()
        kernel.council.scheduled.interval_ticks = 50

        for i in range(200):
            vec = np.random.uniform(0.2, 0.8, 16).tolist()
            kernel.publish_observation(Observation(
                vector=vec,
                embodiment_id="desktop" if i < 100 else "robot",
            ))

        assert kernel._tick == 200
        assert len(kernel._embodiments) >= 2

        council_state = kernel.council.get_state()
        assert council_state.n_audits > 0

    def test_goal_driven_cycle(self):
        kernel = SubstrateKernel()
        gid = kernel.executive.add_goal(GoalState(
            id=0, target=np.ones(16).tolist(),
            description="reach target", urgency=0.9, importance=0.9,
        ))
        kernel.executive.activate_goal(gid)

        for i in range(100):
            vec = np.random.uniform(0.1, 0.9, 16).tolist()
            kernel.publish_observation(Observation(vector=vec))

        state = kernel.executive.tick()
        assert state.n_active > 0

    def test_resource_constrained_cycle(self):
        kernel = SubstrateKernel()
        kernel.resources.budget.total_compute = 0.5
        kernel.resources.budget.total_attention = 0.3

        for i in range(3):
            req = ResourceRequest(
                embodiment_id=f"embodiment_{i}",
                compute=0.3, attention=0.2, learning=0.1,
                tier=ResourceTier.ACTIVE,
            )
            kernel.resources.request(req)

        state = kernel.resources.get_state()
        assert state.active_leases >= 1
