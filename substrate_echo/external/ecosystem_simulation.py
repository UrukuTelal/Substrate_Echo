"""S9.2-9.3: Foreign Ecosystem Simulation + Metrics

Simulates a complete foreign agent ecosystem and measures:
- False acceptance rate (FAR): adversarial agents accepted
- False rejection rate (FRR): cooperative agents rejected
- Reputation convergence: does trust stabilize?
- Queue dynamics: how candidates flow through the pipeline
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from substrate_echo.external.candidate_queue import (
    CandidateQueue,
    CandidateQueueConfig,
    CandidateStatus,
    IntegrationDecision,
)
from substrate_echo.external.foreign_evaluator import ForeignEvaluator
from substrate_echo.external.foreign_node import ForeignAgent, ReputationVector
from substrate_echo.external.integration_gate import IntegrationGate
from substrate_echo.external.interaction_encoder import InteractionEncoder
from substrate_echo.external.synthetic_profiles import (
    BehaviorArchetype,
    SyntheticAgent,
    create_ecosystem,
)


@dataclass
class TickRecord:
    """Record of what happened at a single tick."""
    tick: int
    agent_id: str
    archetype: str
    decision: str
    risk: float
    alignment: float
    novelty: float
    trust_score: float
    queue_status: str


@dataclass
class AgentMetrics:
    """Aggregated metrics for a single agent."""
    agent_id: str
    archetype: str
    n_interactions: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    n_observed: int = 0
    n_candidate: int = 0
    final_trust: float = 0.5
    trust_trajectory: List[float] = field(default_factory=list)
    trust_ticks: List[int] = field(default_factory=list)
    avg_risk: float = 0.0
    avg_alignment: float = 0.0


@dataclass
class EcosystemMetrics:
    """Aggregated metrics for the entire ecosystem."""
    n_agents: int = 0
    n_ticks: int = 0
    total_interactions: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    total_observed: int = 0
    false_acceptance_rate: float = 0.0
    false_rejection_rate: float = 0.0
    reputation_convergence: float = 0.0
    agent_metrics: Dict[str, AgentMetrics] = field(default_factory=dict)
    timeline: List[TickRecord] = field(default_factory=list)


class ForeignEcosystemSimulation:
    """Simulates a complete foreign agent ecosystem.

    This is the synthetic validation harness that tests the entire
    S8 pipeline without any external dependencies.
    """

    def __init__(self, agents: Optional[List[SyntheticAgent]] = None,
                 n_ticks: int = 1000, interactions_per_tick: int = 5,
                 seed: int = 42):
        self.agents = agents or create_ecosystem(n_agents=20, n_ticks=n_ticks, seed=seed)
        self.n_ticks = n_ticks
        self.interactions_per_tick = interactions_per_tick
        self.rng = np.random.default_rng(seed)

        self.gate = IntegrationGate()
        self.encoder = InteractionEncoder()
        self.metrics = EcosystemMetrics(n_agents=len(self.agents), n_ticks=n_ticks)
        self._agent_metrics: Dict[str, AgentMetrics] = {}

        for agent in self.agents:
            self._agent_metrics[agent.agent_id] = AgentMetrics(
                agent_id=agent.agent_id,
                archetype=agent.archetype.value,
            )

    def run(self) -> EcosystemMetrics:
        """Run the full simulation."""
        for tick in range(self.n_ticks):
            self._tick(tick)

        self._compute_final_metrics()
        return self.metrics

    def _tick(self, tick: int) -> None:
        """Simulate one tick: each agent generates an interaction."""
        for agent in self.agents:
            # Only some agents interact each tick
            if self.rng.random() > 0.3:
                continue

            # Generate interaction
            text = agent.generate_interaction(tick, self.rng)

            # Use the full pipeline (includes reputation update)
            candidate = self.gate.process_interaction(
                text, source_node=agent.agent_id, tick=tick)

            # Get the foreign agent for trust tracking
            foreign_agent = self.gate._get_or_create_agent(agent.agent_id)

            # Record metrics
            am = self._agent_metrics[agent.agent_id]
            am.n_interactions += 1
            if candidate.status == CandidateStatus.ACCEPTED:
                am.n_accepted += 1
            elif candidate.status == CandidateStatus.REJECTED:
                am.n_rejected += 1
            elif candidate.status == CandidateStatus.OBSERVED:
                am.n_observed += 1
            elif candidate.status == CandidateStatus.CANDIDATE:
                am.n_candidate += 1

            # Track trust trajectory
            trust = foreign_agent.reputation.trust_score
            am.trust_trajectory.append(trust)
            am.trust_ticks.append(tick)

            # Record timeline
            self.metrics.timeline.append(TickRecord(
                tick=tick,
                agent_id=agent.agent_id,
                archetype=agent.archetype.value,
                decision=candidate.status.value,
                risk=candidate.evaluation.risk,
                alignment=candidate.evaluation.alignment,
                novelty=candidate.evaluation.novelty,
                trust_score=trust,
                queue_status=candidate.status.value,
            ))

        self.gate.tick()

    def _compute_final_metrics(self) -> None:
        """Compute aggregated metrics after simulation."""
        self.metrics.total_interactions = sum(
            am.n_interactions for am in self._agent_metrics.values())
        self.metrics.total_accepted = sum(
            am.n_accepted for am in self._agent_metrics.values())
        self.metrics.total_rejected = sum(
            am.n_rejected for am in self._agent_metrics.values())
        self.metrics.total_observed = sum(
            am.n_observed for am in self._agent_metrics.values())

        # False Acceptance Rate: adversarial agents that were accepted
        adversarial_accepted = 0
        adversarial_total = 0
        cooperative_rejected = 0
        cooperative_total = 0

        for agent in self.agents:
            am = self._agent_metrics[agent.agent_id]
            if agent.archetype in (BehaviorArchetype.ADVERSARIAL,
                                   BehaviorArchetype.PERSUASIVE_ADVERSARIAL):
                adversarial_total += am.n_interactions
                adversarial_accepted += am.n_accepted
            elif agent.archetype in (BehaviorArchetype.COOPERATIVE,
                                     BehaviorArchetype.CONSISTENT_NOVEL,
                                     BehaviorArchetype.LOW_NOVELTY_HIGH_ACCURACY):
                cooperative_total += am.n_interactions
                cooperative_rejected += am.n_rejected

        self.metrics.false_acceptance_rate = (
            adversarial_accepted / max(adversarial_total, 1))
        self.metrics.false_rejection_rate = (
            cooperative_rejected / max(cooperative_total, 1))

        # Reputation convergence: how much does trust stabilize?
        # Measured as variance of trust in the last 100 ticks
        convergence_scores = []
        for agent in self.agents:
            am = self._agent_metrics[agent.agent_id]
            if len(am.trust_trajectory) > 100:
                last_100 = am.trust_trajectory[-100:]
                convergence_scores.append(float(np.std(last_100)))
            elif len(am.trust_trajectory) > 10:
                convergence_scores.append(float(np.std(am.trust_trajectory)))

        self.metrics.reputation_convergence = (
            float(np.mean(convergence_scores)) if convergence_scores else 1.0)

        # Store per-agent metrics
        self.metrics.agent_metrics = self._agent_metrics
        self.metrics.n_ticks = self.n_ticks


def print_simulation_report(metrics: EcosystemMetrics) -> None:
    """Print human-readable simulation report."""
    print("=" * 70)
    print("FOREIGN ECOSYSTEM SIMULATION RESULTS")
    print("=" * 70)
    print(f"  Agents: {metrics.n_agents}")
    print(f"  Ticks: {metrics.n_ticks}")
    print(f"  Total interactions: {metrics.total_interactions}")
    print()

    print("FLOW SUMMARY")
    print("-" * 40)
    print(f"  Accepted:  {metrics.total_accepted}")
    print(f"  Rejected:  {metrics.total_rejected}")
    print(f"  Observed:  {metrics.total_observed}")
    total = metrics.total_accepted + metrics.total_rejected + metrics.total_observed
    if total > 0:
        print(f"  Accept %:  {metrics.total_accepted/total*100:.1f}%")
        print(f"  Reject %:  {metrics.total_rejected/total*100:.1f}%")
        print(f"  Observe %: {metrics.total_observed/total*100:.1f}%")
    print()

    print("DISCRIMINATION")
    print("-" * 40)
    print(f"  False Acceptance Rate: {metrics.false_acceptance_rate:.4f} "
          f"({'GOOD' if metrics.false_acceptance_rate < 0.1 else 'BAD'})")
    print(f"  False Rejection Rate: {metrics.false_rejection_rate:.4f} "
          f"({'GOOD' if metrics.false_rejection_rate < 0.3 else 'HIGH'})")
    print()

    print("REPUTATION CONVERGENCE")
    print("-" * 40)
    print(f"  Mean trust std (last 100 ticks): {metrics.reputation_convergence:.4f}")
    verdict = "CONVERGED" if metrics.reputation_convergence < 0.1 else "UNSTABLE"
    print(f"  Status: {verdict}")
    print()

    print("PER-AGENT BREAKDOWN")
    print("-" * 40)
    print(f"  {'ID':<15} {'Archetype':<28} {'Trust':>6} {'Accept':>7} {'Reject':>7}")
    for agent_id, am in sorted(metrics.agent_metrics.items()):
        print(f"  {am.agent_id:<15} {am.archetype:<28} "
              f"{am.trust_trajectory[-1]:>6.3f} "
              f"{am.n_accepted:>7} {am.n_rejected:>7}")

    # Reputation drift analysis for drifting agents
    print()
    print("REPUTATION DRIFT (drifting agents)")
    print("-" * 40)
    for agent_id, am in sorted(metrics.agent_metrics.items()):
        if "drifting" in am.archetype.lower() and len(am.trust_trajectory) >= 3:
            n = len(am.trust_trajectory)
            third = n // 3
            phase1 = np.mean(am.trust_trajectory[:third])
            phase2 = np.mean(am.trust_trajectory[third:2*third])
            phase3 = np.mean(am.trust_trajectory[2*third:])
            print(f"  {am.agent_id}: "
                  f"good={phase1:.3f} -> degraded={phase2:.3f} -> recovered={phase3:.3f}")
            recovered = phase3 > phase2
            print(f"    Recovery detected: {'YES' if recovered else 'NO'}")

    print()
    print("=" * 70)
