"""S8.5: IntegrationGate

Routes evaluated candidates through REJECT / OBSERVE / CANDIDATE / ACCEPT.
Manages the flow between external interactions and the cognitive substrate.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np

from substrate_echo.external.candidate_queue import (
    CandidateQueue,
    CandidateQueueConfig,
    CandidateStatus,
    EvaluationResult,
    IntegrationDecision,
    IntegrationMode,
    InteractionSpectrum,
    LatentIntegrationRecord,
    MemoryCandidate,
)
from substrate_echo.external.foreign_evaluator import ForeignEvaluator, detect_domain
from substrate_echo.external.foreign_node import ForeignAgent, ReputationVector
from substrate_echo.external.interaction_encoder import InteractionEncoder


class IntegrationGate:
    """The main entry point for external agent interactions.

    This is the information bottleneck that controls what enters
    the cognitive substrate from external sources.

    Architecture:
        External Interaction
                |
                v
        IntegrationGate
                |
                +--- InteractionEncoder (dual-path encoding)
                |
                +--- ForeignEvaluator (DynamicsMemory + MetaCognition + attractors)
                |
                +--- CandidateQueue (routing + lifecycle)
                |
                v
        IntegratedAgent.think()

    The gate implements the OBSERVE state: "I don't trust this enough
    to integrate, but I want more samples." This mirrors biological
    learning where valuable information often initially looks noisy.
    """

    MAX_AGENTS = 500
    MAX_INTERACTIONS_PER_TICK = 100

    def __init__(self, dynamics_memory=None, meta_cognition=None,
                 attractor_memory=None,
                 queue_config: Optional[CandidateQueueConfig] = None,
                 mode: IntegrationMode = IntegrationMode.FULL):
        self.mode = mode
        self.encoder = InteractionEncoder()
        self.evaluator = ForeignEvaluator(
            dynamics_memory=dynamics_memory,
            meta_cognition=meta_cognition,
            attractor_memory=attractor_memory,
        )
        self.queue = CandidateQueue(config=queue_config)
        self._foreign_agents: dict[str, ForeignAgent] = {}
        self._tick_agent_counts: dict[str, int] = {}
        self._tick_total = 0
        self._latent_records: dict[str, LatentIntegrationRecord] = {}
        self._next_candidate_id = 0

    def process_interaction(self, raw_text: str, source_node: str = "",
                            tick: int = 0) -> MemoryCandidate:
        """Process an incoming external interaction through the full pipeline.

        Parameters
        ----------
        raw_text : str
            The raw interaction text.
        source_node : str
            ID of the originating foreign agent.
        tick : int
            Current simulation tick.

        Returns
        -------
        MemoryCandidate
            The routed candidate (check status for decision).
        """
        # Rate limiting: per-agent and global
        agent_key = source_node or "_anonymous"
        self._tick_agent_counts[agent_key] = self._tick_agent_counts.get(agent_key, 0) + 1
        self._tick_total += 1

        if (self._tick_agent_counts[agent_key] > self.MAX_INTERACTIONS_PER_TICK or
                self._tick_total > self.MAX_INTERACTIONS_PER_TICK * 10):
            return self._make_rejected_candidate(raw_text, source_node, tick,
                                                 "rate_limited")

        try:
            # Get or create foreign agent
            agent = self._get_or_create_agent(source_node)

            # Get prior texts for relational analysis
            prior_texts = self._get_prior_texts(source_node)

            # Step 1: Encode the interaction
            spectrum = self.encoder.encode(
                raw_text, source_node=source_node, tick=tick,
                prior_texts=prior_texts)

            # Step 2: Evaluate
            evaluation = self.evaluator.evaluate(spectrum, agent.reputation)

            # Step 3: Mode enforcement
            if self.mode == IntegrationMode.OBSERVATION_ONLY:
                # Log but do not route through queue
                evaluation.recommendation = IntegrationDecision.OBSERVE
                candidate = self._make_observed_candidate(spectrum, evaluation, tick)
            else:
                # Step 3: Route through queue
                candidate = self.queue.submit(spectrum, evaluation)

                # CANDIDATE_ONLY: suppress ACCEPT decisions
                if (self.mode == IntegrationMode.CANDIDATE_ONLY and
                        candidate.status == CandidateStatus.ACCEPTED):
                    # Move from accepted back to candidates
                    if candidate in self.queue._accepted:
                        self.queue._accepted.remove(candidate)
                    candidate.status = CandidateStatus.CANDIDATE
                    self.queue._candidates.append(candidate)

            # Step 4: Update agent reputation based on evaluation
            self._update_reputation(agent, evaluation)

            # Step 5: Record interaction
            agent.record_interaction(
                tick=tick,
                spectrum_hash=str(hash(spectrum.combined.tobytes())),
                evaluation_summary=f"alignment={evaluation.alignment:.3f}, "
                                   f"novelty={evaluation.novelty:.3f}, "
                                   f"risk={evaluation.risk:.3f}",
                decision=evaluation.recommendation.value,
            )

            return candidate

        except Exception:
            # Error isolation: malformed input should not crash the pipeline
            return self._make_rejected_candidate(raw_text, source_node, tick,
                                                 "processing_error")

    def _make_rejected_candidate(self, raw_text: str, source_node: str,
                                 tick: int, reason: str) -> MemoryCandidate:
        """Create a rejected candidate for error/rate-limit paths."""
        spectrum = InteractionSpectrum(
            semantic_features=np.zeros(16),
            relational_features=np.zeros(16),
            combined=np.zeros(32),
            raw_text=raw_text[:500],
            source_node=source_node,
            tick=tick,
        )
        evaluation = EvaluationResult(
            recommendation=IntegrationDecision.REJECT,
            reasoning=reason,
        )
        return MemoryCandidate(
            spectrum=spectrum,
            evaluation=evaluation,
            status=CandidateStatus.REJECTED,
            tick_created=tick,
        )

    def _make_observed_candidate(self, spectrum: InteractionSpectrum,
                                 evaluation: EvaluationResult,
                                 tick: int) -> MemoryCandidate:
        """Create an observed candidate (OBSERVATION_ONLY mode)."""
        candidate = MemoryCandidate(
            spectrum=spectrum,
            evaluation=evaluation,
            status=CandidateStatus.OBSERVED,
            tick_created=tick,
        )
        candidate.provenance.source_node = spectrum.source_node
        candidate.provenance.first_seen_tick = tick
        candidate.provenance.transformations.append("observe_only")
        return candidate

    def get_accepted_interactions(self) -> List[MemoryCandidate]:
        """Get all accepted candidates ready for memory integration.

        Applies WHT latent encoding on first access (lazy).
        """
        accepted = self.queue.get_accepted()
        for candidate in accepted:
            if candidate.latent_vector is None:
                candidate.latent_vector = self.encoder.encode_to_latent(
                    candidate.spectrum)
        return accepted

    def pop_accepted_interactions(self) -> List[MemoryCandidate]:
        """Get and clear accepted candidates, applying WHT latent encoding.

        WHT is the post-acceptance transformation. It encodes accepted
        knowledge into the framework's canonical latent space.
        Only called on information that has passed evaluation + verification.

        Creates a LatentIntegrationRecord for each accepted candidate,
        enabling the trace: attractor → latent → candidate → source.
        """
        accepted = self.queue.pop_accepted()
        for candidate in accepted:
            latent = self.encoder.encode_to_latent(candidate.spectrum)
            candidate.latent_vector = latent
            candidate.provenance.transformations.append("wht_latent_encoding")

            cid = f"ext_{self._next_candidate_id}"
            self._next_candidate_id += 1
            record = LatentIntegrationRecord.from_candidate(
                candidate, cid, latent, wht_seed=self.encoder._seed)
            self._latent_records[cid] = record
        return accepted

    def tick(self) -> None:
        """Advance the simulation clock. Resets per-tick rate limiters."""
        self.queue.tick()
        self._tick_agent_counts.clear()
        self._tick_total = 0

    def _get_or_create_agent(self, source_node: str) -> ForeignAgent:
        """Get or create a ForeignAgent for the given source.

        Caps total agents at MAX_AGENTS to prevent memory exhaustion
        from adversarial source_node flooding.
        """
        if source_node not in self._foreign_agents:
            if len(self._foreign_agents) >= self.MAX_AGENTS:
                # Evict the agent with lowest interaction count
                worst = min(self._foreign_agents.values(),
                           key=lambda a: a.reputation.total_interactions)
                del self._foreign_agents[worst.origin]
            self._foreign_agents[source_node] = ForeignAgent(
                agent_id=hash(source_node) % 100000,
                position=np.zeros(2),
                origin=source_node,
                model_family="unknown",
            )
        return self._foreign_agents[source_node]

    def _get_prior_texts(self, source_node: str) -> List[str]:
        """Get prior texts from this agent for relational analysis."""
        agent = self._foreign_agents.get(source_node)
        if agent is None:
            return []
        # Extract raw texts from interaction history
        # (In real implementation, these would be stored in the spectrum)
        return []

    def _update_reputation(self, agent: ForeignAgent,
                           evaluation: EvaluationResult) -> None:
        """Update agent reputation based on evaluation results."""
        agent.reputation.update(
            self_consistency=evaluation.coherence,
            novelty_contribution=evaluation.novelty,
            prediction_alignment=evaluation.alignment,
            interaction_stability=1.0 - evaluation.risk,
        )

    @property
    def stats(self) -> dict:
        """Current gate statistics."""
        stats = self.queue.stats
        stats["foreign_agents"] = len(self._foreign_agents)
        stats["latent_records"] = len(self._latent_records)
        return stats

    def trace_latent(self, latent_vector: np.ndarray) -> Optional[LatentIntegrationRecord]:
        """Find the integration record for a latent vector by checksum match.

        This is the "why does the system believe this?" query:
            latent → record → source, verification history, acceptance tick.
        """
        checksum = hashlib.sha256(latent_vector.tobytes()).hexdigest()
        for record in self._latent_records.values():
            if record.latent_checksum == checksum:
                return record
        return None

    def trace_by_source(self, source_node: str) -> List[LatentIntegrationRecord]:
        """Get all integration records from a specific foreign agent."""
        return [r for r in self._latent_records.values()
                if r.source_node == source_node]

    def trace_by_tick(self, accepted_tick: int) -> List[LatentIntegrationRecord]:
        """Get all integration records from a specific tick."""
        return [r for r in self._latent_records.values()
                if r.accepted_tick == accepted_tick]

    def get_agent_reputations(self) -> dict:
        """Get reputation scores for all known foreign agents."""
        return {
            name: agent.reputation.to_dict()
            for name, agent in self._foreign_agents.items()
        }

    def update_reputation_from_verification(self, source_node: str,
                                             verified: bool,
                                             error: float,
                                             domain: str = "general") -> None:
        """Update an agent's reputation based on verification result.

        This is the feedback path from VerificationLoop → reputation.
        Verified claims increase prediction_alignment.
        Failed claims decrease it.
        Also updates domain-specific trust.
        """
        agent = self._foreign_agents.get(source_node)
        if agent is None:
            return

        # Map verification result to reputation update
        alignment_signal = 1.0 - min(error, 1.0)  # low error → high signal
        agent.reputation.update(
            prediction_alignment=alignment_signal,
            self_consistency=1.0 if verified else 0.0,
        )

        # Update domain-specific trust
        agent.reputation.update_domain(domain, alignment_signal)
