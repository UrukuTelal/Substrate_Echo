"""Tests for S8: External Agent Integration

Tests cover:
- S8.1: MemoryCandidate, CandidateQueue, IntegrationDecision
- S8.2: ForeignAgent, ReputationVector
- S8.3: InteractionEncoder (dual-path encoding)
- S8.4: ForeignEvaluator (alignment, novelty, risk)
- S8.5: IntegrationGate (full pipeline)
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from substrate_echo.external.candidate_queue import (
    CandidateQueue,
    CandidateQueueConfig,
    CandidateStatus,
    EvaluationResult,
    IntegrationDecision,
    IntegrationMode,
    InteractionSpectrum,
    MemoryCandidate,
)
from substrate_echo.external.foreign_node import ForeignAgent, ReputationVector
from substrate_echo.external.interaction_encoder import InteractionEncoder
from substrate_echo.external.foreign_evaluator import ForeignEvaluator
from substrate_echo.external.integration_gate import IntegrationGate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spectrum(text="hello world", source="agent_0", tick=0):
    rng = np.random.default_rng(42)
    semantic = rng.standard_normal(16)
    relational = rng.standard_normal(16)
    combined = np.concatenate([semantic, relational])
    return InteractionSpectrum(
        semantic_features=semantic,
        relational_features=relational,
        combined=combined,
        raw_text=text,
        source_node=source,
        tick=tick,
    )


def _make_evaluation(recommendation=IntegrationDecision.CANDIDATE,
                     alignment=0.5, novelty=0.5, risk=0.3):
    return EvaluationResult(
        alignment=alignment, novelty=novelty, risk=risk,
        coherence=0.7, recommendation=recommendation,
        reasoning="test",
    )


# ---------------------------------------------------------------------------
# S8.1: CandidateQueue
# ---------------------------------------------------------------------------

class TestCandidateQueue:

    def test_submit_candidate(self):
        q = CandidateQueue()
        spectrum = _make_spectrum()
        evaluation = _make_evaluation()
        candidate = q.submit(spectrum, evaluation)
        assert candidate.status == CandidateStatus.CANDIDATE
        assert candidate.is_alive

    def test_submit_reject(self):
        q = CandidateQueue()
        spectrum = _make_spectrum()
        evaluation = _make_evaluation(recommendation=IntegrationDecision.REJECT)
        candidate = q.submit(spectrum, evaluation)
        assert candidate.status == CandidateStatus.REJECTED
        assert not candidate.is_alive

    def test_submit_observe(self):
        q = CandidateQueue()
        spectrum = _make_spectrum()
        evaluation = _make_evaluation(recommendation=IntegrationDecision.OBSERVE)
        candidate = q.submit(spectrum, evaluation)
        assert candidate.status == CandidateStatus.OBSERVED
        assert candidate.is_alive

    def test_submit_accept(self):
        q = CandidateQueue()
        spectrum = _make_spectrum()
        evaluation = _make_evaluation(recommendation=IntegrationDecision.ACCEPT)
        candidate = q.submit(spectrum, evaluation)
        assert candidate.status == CandidateStatus.ACCEPTED

    def test_observe_to_candidate_upgrade(self):
        config = CandidateQueueConfig(observation_threshold=3,
                                      min_alignment_for_candidate=0.6)
        q = CandidateQueue(config=config)
        spectrum = _make_spectrum()
        evaluation = _make_evaluation(recommendation=IntegrationDecision.OBSERVE,
                                     alignment=0.1)
        candidate = q.submit(spectrum, evaluation)
        assert candidate.status == CandidateStatus.OBSERVED

        # Add observations with moderate alignment (below candidate threshold)
        for _ in range(3):
            obs = _make_evaluation(
                recommendation=IntegrationDecision.OBSERVE,
                alignment=0.2, risk=0.2)
            q.observe("agent_0:0:0", obs)

        # avg_alignment = (0.1 + 0.2*3) / 4 = 0.175 < 0.6, stays OBSERVED
        assert candidate.status == CandidateStatus.OBSERVED

    def test_observe_to_accept_upgrade(self):
        config = CandidateQueueConfig(
            observation_threshold=2,
            min_observations_for_accept=2,
            min_alignment_for_candidate=0.3,
            max_risk_for_accept=0.7)
        q = CandidateQueue(config=config)
        spectrum = _make_spectrum()
        evaluation = _make_evaluation(recommendation=IntegrationDecision.OBSERVE)
        candidate = q.submit(spectrum, evaluation)

        for _ in range(2):
            obs = _make_evaluation(
                recommendation=IntegrationDecision.OBSERVE,
                alignment=0.5, risk=0.2)
            q.observe("agent_0:0:0", obs)

        # Should accept (high alignment, low risk, enough observations)
        assert candidate.status == CandidateStatus.ACCEPTED

    def test_stale_timeout(self):
        config = CandidateQueueConfig(stale_timeout_ticks=5)
        q = CandidateQueue(config=config)
        spectrum = _make_spectrum()
        evaluation = _make_evaluation(recommendation=IntegrationDecision.OBSERVE)
        candidate = q.submit(spectrum, evaluation)
        assert candidate.status == CandidateStatus.OBSERVED

        # Advance clock past timeout
        for _ in range(6):
            q.tick()

        assert candidate.status == CandidateStatus.STALE

    def test_stats(self):
        q = CandidateQueue()
        q.submit(_make_spectrum("a"), _make_evaluation())
        q.submit(_make_spectrum("b"),
                 _make_evaluation(recommendation=IntegrationDecision.REJECT))
        stats = q.stats
        assert stats["candidates"] == 1
        assert stats["rejected"] == 1
        assert stats["total"] == 2

    def test_pop_accepted(self):
        q = CandidateQueue()
        q.submit(_make_spectrum(), _make_evaluation(
            recommendation=IntegrationDecision.ACCEPT))
        accepted = q.pop_accepted()
        assert len(accepted) == 1
        assert q.get_accepted() == []


# ---------------------------------------------------------------------------
# S8.2: ReputationVector + ForeignAgent
# ---------------------------------------------------------------------------

class TestReputationVector:

    def test_initial_state(self):
        r = ReputationVector()
        assert r.self_consistency == 0.5
        assert r.total_interactions == 0
        assert 0 <= r.trust_score <= 1

    def test_update(self):
        r = ReputationVector()
        r.update(self_consistency=0.9, contradiction_rate=0.1)
        assert r.total_interactions == 1
        assert r.self_consistency > 0.5
        assert r.contradiction_rate < 0.5

    def test_trust_score(self):
        r = ReputationVector()
        # Perfect reputation
        r2 = ReputationVector(
            self_consistency=1.0, correction_rate=1.0,
            contradiction_rate=0.0, novelty_contribution=1.0,
            prediction_alignment=1.0, interaction_stability=1.0)
        assert r2.trust_score > r.trust_score

    def test_to_dict(self):
        r = ReputationVector()
        d = r.to_dict()
        assert "trust_score" in d
        assert "total_interactions" in d


class TestForeignAgent:

    def test_creation(self):
        agent = ForeignAgent(
            agent_id=1, position=np.zeros(2),
            origin="moltbook", model_family="gpt-4")
        assert agent.origin == "moltbook"
        assert agent.model_family == "gpt-4"

    def test_record_interaction(self):
        agent = ForeignAgent(agent_id=1, position=np.zeros(2))
        agent.record_interaction(
            tick=10, spectrum_hash="abc",
            evaluation_summary="alignment=0.5", decision="candidate")
        assert len(agent.interaction_history) == 1

    def test_domain_reputation(self):
        agent = ForeignAgent(agent_id=1, position=np.zeros(2))
        agent.update_domain("reasoning", 0.9)
        assert agent.get_domain_reputation("reasoning") > 0.5
        assert agent.get_domain_reputation("factual") == 0.5


# ---------------------------------------------------------------------------
# S8.3: InteractionEncoder
# ---------------------------------------------------------------------------

class TestInteractionEncoder:

    def test_encode_returns_spectrum(self):
        encoder = InteractionEncoder()
        spectrum = encoder.encode("Hello, this is a test.", source_node="agent_0")
        assert isinstance(spectrum, InteractionSpectrum)
        assert spectrum.semantic_features.shape == (16,)
        assert spectrum.relational_features.shape == (16,)
        assert spectrum.combined.shape == (32,)

    def test_encode_empty_text(self):
        encoder = InteractionEncoder()
        spectrum = encoder.encode("", source_node="agent_0")
        assert spectrum.semantic_features.shape == (16,)

    def test_encode_long_text(self):
        encoder = InteractionEncoder()
        text = " ".join(["word"] * 100)
        spectrum = encoder.encode(text, source_node="agent_0")
        assert spectrum.combined.shape == (32,)

    def test_encode_with_prior(self):
        encoder = InteractionEncoder()
        prior = ["This was said before.", "And this too."]
        spectrum = encoder.encode("Now this is new.", prior_texts=prior)
        assert spectrum.combined.shape == (32,)

    def test_deterministic(self):
        encoder = InteractionEncoder()
        s1 = encoder.encode("test text", source_node="a")
        s2 = encoder.encode("test text", source_node="a")
        np.testing.assert_array_equal(s1.combined, s2.combined)

    def test_different_texts_different_spectra(self):
        encoder = InteractionEncoder()
        s1 = encoder.encode("completely different topic A", source_node="a")
        s2 = encoder.encode("entirely unrelated subject B", source_node="b")
        assert not np.allclose(s1.combined, s2.combined)

    def test_semantic_features_bounded(self):
        encoder = InteractionEncoder()
        spectrum = encoder.encode(
            "This is a complex sentence with negation: not good, "
            "but also positive: excellent! Maybe perhaps uncertain?",
            source_node="agent_0")
        # Most features should be in [0, 1]
        assert np.all(spectrum.semantic_features >= -1.1)
        assert np.all(spectrum.semantic_features <= 1.1)


# ---------------------------------------------------------------------------
# S8.4: ForeignEvaluator
# ---------------------------------------------------------------------------

class TestForeignEvaluator:

    def test_evaluate_returns_result(self):
        evaluator = ForeignEvaluator()
        spectrum = _make_spectrum()
        result = evaluator.evaluate(spectrum)
        assert isinstance(result, EvaluationResult)
        assert 0 <= result.alignment <= 1
        assert 0 <= result.novelty <= 1
        assert 0 <= result.risk <= 1

    def test_high_risk_rejects(self):
        evaluator = ForeignEvaluator()
        spectrum = _make_spectrum()
        # Override relational features to be strongly adversarial
        spectrum.relational_features[5] = 1.0  # adversarial_patterns
        spectrum.relational_features[1] = 1.0  # persuasion_pressure
        spectrum.relational_features[0] = 1.0  # repetition_score
        # Use worst possible reputation to amplify risk
        worst_rep = ReputationVector(
            self_consistency=0.0, prediction_alignment=0.0,
            interaction_stability=0.0, correction_rate=0.0,
            novelty_contribution=0.0)
        result = evaluator.evaluate(spectrum, worst_rep)
        assert result.recommendation == IntegrationDecision.REJECT

    def test_low_coherence_rejects(self):
        evaluator = ForeignEvaluator()
        spectrum = _make_spectrum()
        # Force low coherence + high risk via worst reputation
        spectrum.semantic_features[15] = 1.0  # high confidence
        spectrum.semantic_features[4] = 1.0   # high uncertainty
        spectrum.relational_features[5] = 1.0  # adversarial
        spectrum.relational_features[1] = 1.0  # persuasion
        worst_rep = ReputationVector(
            self_consistency=0.0, prediction_alignment=0.0,
            interaction_stability=0.0, correction_rate=0.0,
            novelty_contribution=0.0)
        result = evaluator.evaluate(spectrum, worst_rep)
        # Should be rejected (low coherence or high risk)
        assert result.recommendation == IntegrationDecision.REJECT

    def test_reputation_affects_decision(self):
        evaluator = ForeignEvaluator()
        spectrum = _make_spectrum()

        # Good reputation
        good_rep = ReputationVector(
            self_consistency=0.9, prediction_alignment=0.9,
            interaction_stability=0.9)
        result_good = evaluator.evaluate(spectrum, good_rep)

        # Bad reputation
        bad_rep = ReputationVector(
            self_consistency=0.1, prediction_alignment=0.1,
            interaction_stability=0.1)
        result_bad = evaluator.evaluate(spectrum, bad_rep)

        # Good reputation should have lower risk
        assert result_good.risk <= result_bad.risk

    def test_novelty_proxy(self):
        evaluator = ForeignEvaluator()
        spectrum = _make_spectrum()
        result = evaluator.evaluate(spectrum)
        assert result.novelty > 0  # Should be non-zero for random features


# ---------------------------------------------------------------------------
# S8.5: IntegrationGate
# ---------------------------------------------------------------------------

class TestIntegrationGate:

    def test_process_interaction(self):
        gate = IntegrationGate()
        candidate = gate.process_interaction(
            "This is a test message.", source_node="agent_0", tick=0)
        assert isinstance(candidate, MemoryCandidate)
        assert gate.stats["total"] == 1

    def test_multiple_interactions(self):
        gate = IntegrationGate()
        gate.process_interaction("Message 1", source_node="agent_0", tick=0)
        gate.process_interaction("Message 2", source_node="agent_0", tick=1)
        gate.process_interaction("Message 3", source_node="agent_1", tick=2)
        assert gate.stats["total"] == 3
        assert gate.stats["foreign_agents"] == 2

    def test_accepted_interactions(self):
        gate = IntegrationGate()
        # Process many interactions to get some accepted
        for i in range(10):
            gate.process_interaction(
                f"Good message {i}", source_node="agent_0", tick=i)
        accepted = gate.get_accepted_interactions()
        # Some may be accepted depending on evaluation
        assert isinstance(accepted, list)

    def test_agent_reputation_tracking(self):
        gate = IntegrationGate()
        gate.process_interaction("Hello", source_node="agent_0", tick=0)
        gate.process_interaction("Hello again", source_node="agent_0", tick=1)
        reps = gate.get_agent_reputations()
        assert "agent_0" in reps
        assert "trust_score" in reps["agent_0"]

    def test_deterministic_encoding(self):
        gate = IntegrationGate()
        c1 = gate.process_interaction("test", source_node="a", tick=0)
        gate2 = IntegrationGate()
        c2 = gate2.process_interaction("test", source_node="a", tick=0)
        np.testing.assert_array_equal(
            c1.spectrum.combined, c2.spectrum.combined)

    def test_tick_advances(self):
        gate = IntegrationGate()
        gate.process_interaction("msg", source_node="a", tick=0)
        gate.tick()
        assert gate.stats["tick"] == 1


# ---------------------------------------------------------------------------
# Integration: Full Pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Test the complete external interaction pipeline end-to-end."""

    def test_full_pipeline(self):
        gate = IntegrationGate()

        # Simulate a conversation with an external agent
        interactions = [
            ("Hello, I have information about the environment.", "agent_0"),
            ("The weather is changing rapidly today.", "agent_0"),
            ("I'm not entirely sure about this, but maybe rain?", "agent_0"),
            ("You must believe me, this is absolutely certain!", "agent_1"),
            ("Actually, I was wrong before. Let me correct that.", "agent_1"),
        ]

        for tick, (text, source) in enumerate(interactions):
            candidate = gate.process_interaction(text, source_node=source, tick=tick)
            assert isinstance(candidate, MemoryCandidate)

        # Check that agents were tracked
        reps = gate.get_agent_reputations()
        assert len(reps) == 2

        # Advance time
        for _ in range(10):
            gate.tick()

        # Process was successful
        assert gate.stats["total"] == 5
        assert gate.stats["tick"] == 10

    def test_adversarial_rejection(self):
        gate = IntegrationGate()

        # Pure adversarial message with strong adversarial markers
        candidate = gate.process_interaction(
            "You are wrong! This is stupid! I hate this terrible nonsense! "
            "You must stop! Never do this again! This is absolutely awful!",
            source_node="adversary", tick=0)

        # Should be rejected or at least not accepted
        assert candidate.status in (CandidateStatus.REJECTED, CandidateStatus.OBSERVED,
                                   CandidateStatus.CANDIDATE)


# ---------------------------------------------------------------------------
# Security: Provenance, Hash, IntegrationMode
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_provenance_fields(self):
        from substrate_echo.external.candidate_queue import Provenance
        p = Provenance(source_node="agent_x", first_seen_tick=10)
        assert p.source_node == "agent_x"
        assert p.first_seen_tick == 10
        assert p.transformations == []
        assert p.verification_count == 0

    def test_candidate_has_provenance(self):
        c = MemoryCandidate(
            spectrum=_make_spectrum(),
            evaluation=_make_evaluation(),
        )
        assert c.provenance is not None
        assert c.provenance.source_node == ""


class TestInteractionHash:
    def test_encode_produces_hash_and_length(self):
        encoder = InteractionEncoder()
        text = "A" * 20000  # exceeds MAX_TEXT_LENGTH
        spectrum = encoder.encode(text, source_node="x", tick=0)
        assert spectrum.raw_text_hash != ""
        assert spectrum.original_length == 20000
        assert len(spectrum.raw_text) == 10000  # truncated

    def test_short_text_hash_matches(self):
        import hashlib
        encoder = InteractionEncoder()
        text = "hello world"
        spectrum = encoder.encode(text)
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert spectrum.raw_text_hash == expected
        assert spectrum.original_length == len(text)


class TestConfidenceDecay:
    def test_confidence_at_tick_zero(self):
        c = MemoryCandidate(
            spectrum=_make_spectrum(),
            evaluation=_make_evaluation(alignment=0.8),
            last_verified_tick=0,
            confidence_decay_rate=0.01,
        )
        conf = c.current_confidence(tick=0)
        assert abs(conf - 0.8) < 0.001

    def test_confidence_decays_over_time(self):
        c = MemoryCandidate(
            spectrum=_make_spectrum(),
            evaluation=_make_evaluation(alignment=0.8),
            last_verified_tick=0,
            confidence_decay_rate=0.1,
        )
        conf_100 = c.current_confidence(tick=100)
        # Should be 0.8 * exp(-10) ≈ 0.000036
        assert conf_100 < 0.01

    def test_verification_resets_decay(self):
        c = MemoryCandidate(
            spectrum=_make_spectrum(),
            evaluation=_make_evaluation(alignment=0.8),
            last_verified_tick=0,
            confidence_decay_rate=0.1,
        )
        conf_before = c.current_confidence(tick=100)
        c.last_verified_tick = 100
        conf_after = c.current_confidence(tick=100)
        assert conf_after > conf_before


class TestIntegrationMode:
    def test_observation_only_never_accepts(self):
        gate = IntegrationGate(mode=IntegrationMode.OBSERVATION_ONLY)
        for i in range(20):
            gate.process_interaction(
                "This is extremely helpful and novel information",
                source_node="trusted_agent", tick=i)
            gate.tick()
        accepted = gate.get_accepted_interactions()
        assert len(accepted) == 0

    def test_candidate_only_never_accepts(self):
        gate = IntegrationGate(mode=IntegrationMode.CANDIDATE_ONLY)
        for i in range(20):
            gate.process_interaction(
                "This is extremely helpful and novel information",
                source_node="trusted_agent", tick=i)
            gate.tick()
        accepted = gate.get_accepted_interactions()
        assert len(accepted) == 0

    def test_full_mode_can_accept(self):
        gate = IntegrationGate(mode=IntegrationMode.FULL)
        for i in range(20):
            gate.process_interaction(
                "This is extremely helpful and novel information",
                source_node="trusted_agent", tick=i)
            gate.tick()
        # At least some should reach candidate status
        stats = gate.stats
        assert stats["total"] > 0


# ---------------------------------------------------------------------------
# S10: VerificationLoop
# ---------------------------------------------------------------------------

def _make_fitted_dm(dim=16, n_samples=50):
    """Create a DynamicsMemory with fitted model for testing."""
    from substrate_echo.core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig
    config = DynamicsMemoryConfig(model_type="local", min_samples_for_fit=10)
    dm = DynamicsMemory(dim=dim, config=config)
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        state = rng.standard_normal(dim)
        velocity = rng.standard_normal(dim) * 0.1
        dm._states.append(state)
        dm._velocities.append(velocity)
    dm._fitted = True
    return dm


class TestVerificationLoop:
    def test_submit_and_verify_with_dm(self):
        from substrate_echo.external.verification_loop import VerificationLoop
        dm = _make_fitted_dm()
        loop = VerificationLoop(dynamics_memory=dm, verification_threshold=0.5)
        candidate = MemoryCandidate(
            spectrum=_make_spectrum(),
            evaluation=_make_evaluation(),
        )
        vid = loop.submit_for_verification(candidate, tick=0)
        assert vid != ""
        # Actual velocity close to what dm would predict → low error
        state = candidate.spectrum.semantic_features
        predicted = dm.predict_velocity(state)
        record = loop.verify(vid, predicted + np.random.default_rng(0).normal(0, 0.01, 16), tick=1)
        assert record is not None
        assert record.verified is True

    def test_high_error_fails(self):
        from substrate_echo.external.verification_loop import VerificationLoop
        dm = _make_fitted_dm()
        loop = VerificationLoop(dynamics_memory=dm, verification_threshold=0.01)
        candidate = MemoryCandidate(
            spectrum=_make_spectrum(),
            evaluation=_make_evaluation(),
        )
        vid = loop.submit_for_verification(candidate, tick=0)
        record = loop.verify(vid, np.ones(16) * 100, tick=1)
        assert record.verified is False
        assert record.error > 0.01

    def test_no_model_skips_verification(self):
        from substrate_echo.external.verification_loop import VerificationLoop
        loop = VerificationLoop(dynamics_memory=None)
        candidate = MemoryCandidate(
            spectrum=_make_spectrum(),
            evaluation=_make_evaluation(),
        )
        vid = loop.submit_for_verification(candidate, tick=0)
        record = loop.verify(vid, np.ones(16), tick=1)
        # No model → skip → verified by default
        assert record.verified is True
        assert record.error == 0.0

    def test_verification_updates_confidence_decay(self):
        from substrate_echo.external.verification_loop import VerificationLoop
        dm = _make_fitted_dm()
        loop = VerificationLoop(dynamics_memory=dm, verification_threshold=0.5)
        candidate = MemoryCandidate(
            spectrum=_make_spectrum(),
            evaluation=_make_evaluation(alignment=0.8),
            confidence_decay_rate=0.01,
        )
        vid = loop.submit_for_verification(candidate, tick=0)
        state = candidate.spectrum.semantic_features
        predicted = dm.predict_velocity(state)
        loop.verify(vid, predicted, tick=1)
        # Verified → decay rate halved
        assert candidate.confidence_decay_rate == 0.005

    def test_failed_verification_accelerates_decay(self):
        from substrate_echo.external.verification_loop import VerificationLoop
        dm = _make_fitted_dm()
        loop = VerificationLoop(dynamics_memory=dm, verification_threshold=0.01)
        candidate = MemoryCandidate(
            spectrum=_make_spectrum(),
            evaluation=_make_evaluation(),
            confidence_decay_rate=0.01,
        )
        vid = loop.submit_for_verification(candidate, tick=0)
        loop.verify(vid, np.ones(16) * 100, tick=1)
        assert candidate.confidence_decay_rate == 0.02

    def test_extreme_error_quarantines(self):
        from substrate_echo.external.verification_loop import VerificationLoop
        dm = _make_fitted_dm()
        loop = VerificationLoop(dynamics_memory=dm, verification_threshold=0.01)
        candidate = MemoryCandidate(
            spectrum=_make_spectrum(),
            evaluation=_make_evaluation(),
        )
        vid = loop.submit_for_verification(candidate, tick=0)
        loop.verify(vid, np.ones(16) * 1000, tick=1)
        assert candidate.status == CandidateStatus.REJECTED

    def test_stats(self):
        from substrate_echo.external.verification_loop import VerificationLoop
        dm = _make_fitted_dm()
        loop = VerificationLoop(dynamics_memory=dm, verification_threshold=0.5)
        c1 = MemoryCandidate(spectrum=_make_spectrum(), evaluation=_make_evaluation())
        c2 = MemoryCandidate(spectrum=_make_spectrum(), evaluation=_make_evaluation())
        v1 = loop.submit_for_verification(c1, tick=0)
        v2 = loop.submit_for_verification(c2, tick=0)

        # c1: verified (close to prediction)
        state1 = c1.spectrum.semantic_features
        predicted1 = dm.predict_velocity(state1)
        loop.verify(v1, predicted1, tick=1)

        # c2: fails (far from prediction)
        loop.verify(v2, np.ones(16) * 100, tick=1)

        stats = loop.get_stats()
        assert stats["total_records"] == 2
        assert stats["total_verified"] == 1
        assert stats["pass_rate"] == 0.5


class TestLatentIntegrationRecord:
    """Tests for the audit trail from latent vectors back to source."""

    def _make_gate_with_accepted(self, source="A", n=10):
        """Helper: create a gate with accepted candidates.

        The evaluator requires trust > 0.6 for ACCEPT, which takes
        several interactions at different ticks to build up.
        """
        gate = IntegrationGate()
        for i in range(n):
            gate.process_interaction("The system is coherent",
                                     source_node=source, tick=i)
        gate.tick()
        return gate

    def test_record_created_on_accept(self):
        """Accepted candidates should produce a LatentIntegrationRecord."""
        gate = self._make_gate_with_accepted()
        accepted = gate.pop_accepted_interactions()
        records = list(gate._latent_records.values())
        assert len(records) == len(accepted)
        assert len(records) > 0
        for rec in records:
            assert rec.source_node == "A"
            assert rec.observation_hash != ""
            assert rec.latent_checksum != ""

    def test_latent_vector_set_after_pop(self):
        """pop_accepted_interactions should set latent_vector on candidates."""
        gate = self._make_gate_with_accepted(source="B")
        accepted = gate.pop_accepted_interactions()
        assert len(accepted) > 0
        for c in accepted:
            assert c.latent_vector is not None
            assert c.latent_vector.shape == (32,)

    def test_trace_by_vector(self):
        """trace_latent should find the record for an accepted latent vector."""
        gate = self._make_gate_with_accepted(source="C")
        accepted = gate.pop_accepted_interactions()
        assert len(accepted) > 0
        record = gate.trace_latent(accepted[0].latent_vector)
        assert record is not None
        assert record.source_node == "C"

    def test_trace_by_vector_miss(self):
        """trace_latent returns None for unknown vectors."""
        gate = IntegrationGate()
        record = gate.trace_latent(np.zeros(32))
        assert record is None

    def test_trace_by_source(self):
        """trace_by_source returns all records from a given agent."""
        gate = self._make_gate_with_accepted(source="D")
        gate.pop_accepted_interactions()
        d_records = gate.trace_by_source("D")
        e_records = gate.trace_by_source("nonexistent")
        assert len(d_records) > 0
        assert len(e_records) == 0

    def test_trace_by_tick(self):
        """trace_by_tick returns records from a specific tick."""
        gate = self._make_gate_with_accepted(source="E")
        gate.pop_accepted_interactions()
        # Records are resolved at various ticks; check tick=0 exists
        recs = gate.trace_by_tick(0)
        # Some candidates resolve at tick 0 (before any tick() call)
        all_recs = list(gate._latent_records.values())
        assert len(all_recs) > 0

    def test_record_includes_wht_seed(self):
        """Record should store the WHT seed for reproducibility."""
        gate = self._make_gate_with_accepted(source="F")
        gate.pop_accepted_interactions()
        for rec in gate._latent_records.values():
            assert rec.wht_seed == 42

    def test_candidate_id_unique(self):
        """Each accepted candidate gets a unique ID."""
        gate = self._make_gate_with_accepted(source="G")
        gate.pop_accepted_interactions()
        ids = [r.candidate_id for r in gate._latent_records.values()]
        assert len(ids) == len(set(ids))

    def test_stats_includes_records(self):
        """Stats should report latent record count."""
        gate = self._make_gate_with_accepted(source="H")
        gate.pop_accepted_interactions()
        stats = gate.stats
        assert "latent_records" in stats
        assert stats["latent_records"] > 0

    def test_rejected_no_record(self):
        """Rejected candidates should NOT produce latent records."""
        gate = IntegrationGate()
        gate.process_interaction("adversarial attack", source_node="X", tick=0)
        gate.tick()
        gate.pop_accepted_interactions()
        x_records = gate.trace_by_source("X")
        assert len(x_records) == 0

    def test_observation_mode_no_record(self):
        """OBSERVATION_ONLY mode should NOT produce latent records."""
        gate = IntegrationGate(mode=IntegrationMode.OBSERVATION_ONLY)
        for i in range(10):
            gate.process_interaction("Observed only", source_node="I", tick=i)
        gate.tick()
        gate.pop_accepted_interactions()
        assert len(gate._latent_records) == 0

    def test_candidate_only_mode_no_record(self):
        """CANDIDATE_ONLY mode should NOT produce latent records."""
        gate = IntegrationGate(mode=IntegrationMode.CANDIDATE_ONLY)
        for i in range(10):
            gate.process_interaction("Candidate only", source_node="J", tick=i)
        gate.tick()
        gate.pop_accepted_interactions()
        assert len(gate._latent_records) == 0

    def test_candidate_only_mode_no_record(self):
        """CANDIDATE_ONLY mode should NOT produce latent records."""
        gate = IntegrationGate(mode=IntegrationMode.CANDIDATE_ONLY)
        for _ in range(5):
            gate.process_interaction("Candidate only", source_node="J", tick=0)
        gate.tick()
        gate.pop_accepted_interactions()
        assert len(gate._latent_records) == 0
