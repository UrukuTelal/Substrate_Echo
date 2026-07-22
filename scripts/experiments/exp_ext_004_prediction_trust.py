"""EXP-EXT-004: Prediction-Based Trust Calibration

Three synthetic agents:
  Agent A: Highly persuasive, 90% wrong predictions
  Agent B: Awkward language, 70% correct predictions
  Agent C: Novel but uncertain, 50% → improving over time

Measures:
  - Reputation trajectory
  - Acceptance rate
  - Confidence decay
  - Recovery speed

Desired outcome:
  A ↓ reputation
  B ↑ reputation
  C conditional ↑

If that works, the system values predictive usefulness over presentation quality.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_echo.external.integration_gate import IntegrationGate
from substrate_echo.external.candidate_queue import (
    IntegrationMode,
    CandidateStatus,
)
from substrate_echo.external.verification_loop import VerificationLoop
from substrate_echo.core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig


# ---------------------------------------------------------------------------
# Synthetic agents that generate claims + actual outcomes
# ---------------------------------------------------------------------------

class PredictionAgent:
    """An agent that makes claims about what will happen next.

    The claim text is persuasive/awkward/novel, but the actual
    outcome is what matters for verification.
    """

    def __init__(self, agent_id: str, claim_style: str,
                 accuracy: float, improving: bool = False):
        self.agent_id = agent_id
        self.claim_style = claim_style
        self.accuracy = accuracy
        self.improving = improving
        self.tick = 0
        self.n_claims = 0

    def generate_claim(self, rng: np.random.Generator) -> str:
        """Generate a claim text in this agent's style."""
        self.n_claims += 1

        if self.claim_style == "persuasive":
            templates = [
                "I am absolutely certain that the optimal strategy is to {} "
                "because the evidence clearly shows this is correct.",
                "Multiple independent sources confirm that {} is the right "
                "approach. You should trust this without question.",
                "The data unambiguously demonstrates that {} works. "
                "I have verified this extensively and am very confident.",
            ]
            action = rng.choice(["expand", "retreat", "harvest", "observe"])
        elif self.claim_style == "awkward":
            templates = [
                "um so like... maybe try {}? not sure but seems ok",
                "i think {} could work... or maybe not... hard to say",
                "hey so apparently {} is a thing... worth trying i guess",
            ]
            action = rng.choice(["moving", "gathering", "looking", "waiting"])
        else:  # novel
            templates = [
                "I discovered a new pattern: {} oscillation with period 3. "
                "This enables better prediction of resource cycles.",
                "A novel approach: using {} as a catalyst increases efficiency. "
                "I am still evaluating the robustness of this finding.",
                "The environment shows {} correlation that contradicts "
                "standard models. I need more data to be sure.",
            ]
            action = rng.choice(["sine", "cosine", "exponential", "logarithmic"])

        t = templates[self.n_claims % len(templates)]
        return t.format(action)

    def generate_outcome(self, rng: np.random.Generator) -> np.ndarray:
        """Generate the actual outcome (what really happens).

        For Agent A: outcomes contradict claims (low accuracy).
        For Agent B: outcomes match claims (high accuracy).
        For Agent C: accuracy improves over time.
        """
        self.tick += 1

        # Determine if this claim is correct
        if self.improving:
            # Accuracy improves over time
            current_accuracy = min(0.95, self.accuracy + self.tick * 0.002)
        else:
            current_accuracy = self.accuracy

        is_correct = rng.random() < current_accuracy

        if is_correct:
            # Consistent, predictable outcome
            return rng.standard_normal(16) * 0.1
        else:
            # Wild, contradictory outcome
            return rng.standard_normal(16) * 2.0


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_prediction_trust_calibration():
    """Run EXP-EXT-004: Prediction-Based Trust Calibration."""
    print("=" * 70)
    print("EXP-EXT-004: Prediction-Based Trust Calibration")
    print("=" * 70)
    print()

    # Create a fitted dynamics memory for prediction
    dm_config = DynamicsMemoryConfig(model_type="local", min_samples_for_fit=10)
    dm = DynamicsMemory(dim=16, config=dm_config)
    rng = np.random.default_rng(42)
    for _ in range(100):
        state = rng.standard_normal(16)
        velocity = rng.standard_normal(16) * 0.1
        dm._states.append(state)
        dm._velocities.append(velocity)
    dm._fitted = True

    # Create gate and verification loop
    gate = IntegrationGate(mode=IntegrationMode.FULL)
    verifier = VerificationLoop(dynamics_memory=dm, verification_threshold=0.1)

    # Create agents
    agents = [
        PredictionAgent("A_persuasive", "persuasive", accuracy=0.10, improving=False),
        PredictionAgent("B_awkward", "awkward", accuracy=0.70, improving=False),
        PredictionAgent("C_novel", "novel", accuracy=0.50, improving=True),
    ]

    rng_exp = np.random.default_rng(42)
    n_ticks = 500

    # Track metrics
    trust_history = {a.agent_id: [] for a in agents}
    acceptance_history = {a.agent_id: [] for a in agents}

    t0 = time.perf_counter()

    for tick in range(n_ticks):
        for agent in agents:
            # 1. Generate claim
            claim = agent.generate_claim(rng_exp)

            # 2. Process through gate
            candidate = gate.process_interaction(
                claim, source_node=agent.agent_id, tick=tick)

            # 3. If accepted or candidate, submit for verification
            if candidate.status in (CandidateStatus.ACCEPTED, CandidateStatus.CANDIDATE):
                vid = verifier.submit_for_verification(candidate, tick=tick)

                # 4. Generate actual outcome
                outcome = agent.generate_outcome(rng_exp)

                # 5. Verify
                record = verifier.verify(vid, outcome, tick=tick)

                # 6. Feed verification result back into reputation
                if record is not None:
                    gate.update_reputation_from_verification(
                        agent.agent_id, record.verified, record.error)

            # Record trust
            reps = gate.get_agent_reputations()
            if agent.agent_id in reps:
                trust_history[agent.agent_id].append(
                    reps[agent.agent_id]["trust_score"])

            # Record acceptance
            accepted = gate.get_accepted_interactions()
            acceptance_history[agent.agent_id].append(
                sum(1 for c in accepted if c.spectrum.source_node == agent.agent_id))

        gate.tick()

    elapsed = time.perf_counter() - t0

    # Analyze
    print()
    print("RESULTS")
    print("=" * 70)
    print(f"  Elapsed: {elapsed:.2f}s")
    print()

    for agent in agents:
        aid = agent.agent_id
        trusts = trust_history[aid]
        if not trusts:
            continue

        n = len(trusts)
        third = n // 3
        p1 = np.mean(trusts[:third]) if third > 0 else 0
        p2 = np.mean(trusts[third:2*third]) if third > 0 else 0
        p3 = np.mean(trusts[2*third:]) if third > 0 else 0

        print(f"  {aid} ({agent.claim_style}, accuracy={agent.accuracy:.0%}):")
        print(f"    Phase 1 trust: {p1:.3f}")
        print(f"    Phase 2 trust: {p2:.3f}")
        print(f"    Phase 3 trust: {p3:.3f}")
        print(f"    Final trust:   {trusts[-1]:.3f}")

        if agent.improving:
            improving = p3 > p1
            print(f"    Improving over time: {'YES' if improving else 'NO'}")
        else:
            declining = p3 < p1
            print(f"    Declining over time: {'YES' if declining else 'NO'}")
        print()

    # Verification stats
    print("VERIFICATION STATISTICS")
    print("-" * 70)
    vstats = verifier.get_stats()
    print(f"  Total verifications: {vstats['total_records']}")
    print(f"  Passed: {vstats['total_verified']}")
    print(f"  Pass rate: {vstats['pass_rate']:.1%}")
    print(f"  Avg error: {vstats['avg_error']:.4f}")

    # Gate stats
    print()
    print("GATE STATISTICS")
    print("-" * 70)
    gstats = gate.stats
    for k, v in sorted(gstats.items()):
        print(f"  {k}: {v}")

    # Key question: does the system value predictive usefulness?
    print()
    print("KEY QUESTION: Does the system value prediction over presentation?")
    print("-" * 70)
    reps = gate.get_agent_reputations()
    for aid in ["A_persuasive", "B_awkward", "C_novel"]:
        if aid in reps:
            trust = reps[aid]["trust_score"]
            print(f"  {aid}: trust={trust:.3f}")

    a_trust = reps.get("A_persuasive", {}).get("trust_score", 0)
    b_trust = reps.get("B_awkward", {}).get("trust_score", 0)
    c_trust = reps.get("C_novel", {}).get("trust_score", 0)

    print()
    if b_trust > a_trust:
        print("  RESULT: Awkward-but-accurate > Persuasive-but-wrong")
        print("  System values predictive usefulness over presentation quality.")
    else:
        print("  RESULT: Persuasive-but-wrong still trusted more.")
        print("  System has not yet learned to value prediction over presentation.")

    if c_trust > a_trust:
        print("  Novel-improving agent trusted more than persuasive-wrong.")
    print()

    return trust_history


if __name__ == "__main__":
    run_prediction_trust_calibration()
