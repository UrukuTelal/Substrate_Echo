"""EXP-EXT-005: Domain Transfer Test

Tests whether domain-conditioned reputation correctly separates
domain expertise from global reputation.

Agents:
  Agent A: Physics specialist, poor social reasoning
  Agent B: Social specialist, poor physics
  Agent C: General mediocre

Expected outcome:
  Physics claim: A > C > B
  Social claim: B > C > A

Measures:
  - Global trust
  - Domain trust (physics, social)
  - Acceptance decisions per domain
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_echo.external.integration_gate import IntegrationGate
from substrate_echo.external.candidate_queue import IntegrationMode
from substrate_echo.external.verification_loop import VerificationLoop
from substrate_echo.external.foreign_evaluator import detect_domain
from substrate_echo.core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig


# ---------------------------------------------------------------------------
# Domain-specific claim generators
# ---------------------------------------------------------------------------

PHYSICS_CLAIMS = [
    "The force applied to the mass creates acceleration proportional to F=ma.",
    "Energy conservation requires that the total energy in the system remains constant.",
    "The velocity of the particle increases under constant gravitational field.",
    "Temperature and pressure are related through the ideal gas law PV=nRT.",
    "The wave propagates through the medium with frequency inversely proportional to wavelength.",
    "Friction force opposes motion and is proportional to the normal force.",
    "Momentum is conserved in the collision between the two bodies.",
    "The entropy of an isolated system always increases over time.",
    "The gravitational field strength decreases with the square of distance.",
    "Photon energy is proportional to the frequency of the electromagnetic wave.",
]

SOCIAL_CLAIMS = [
    "Trust is built through repeated cooperative interactions over time.",
    "The group will form an alliance if negotiation succeeds.",
    "Diplomatic influence increases through reputation and social capital.",
    "Betrayal in the community damages long-term cooperative relationships.",
    "Persuasion is more effective when the source has established credibility.",
    "The social network structure affects information spread and cooperation.",
    "Conflict resolution requires understanding the other party's perspective.",
    "Reputation spreads through the community via word-of-mouth communication.",
    "Cooperation emerges when the benefits outweigh the costs of trust.",
    "The alliance will hold if all parties maintain their commitments.",
]


def make_physics_outcome(rng: np.random.Generator, accurate: bool) -> np.ndarray:
    """Generate a physics-like outcome."""
    if accurate:
        # Consistent, physically plausible
        return rng.standard_normal(16) * 0.1
    else:
        # Random, physically implausible
        return rng.standard_normal(16) * 2.0


def make_social_outcome(rng: np.random.Generator, accurate: bool) -> np.ndarray:
    """Generate a social-like outcome."""
    if accurate:
        return rng.standard_normal(16) * 0.1
    else:
        return rng.standard_normal(16) * 2.0


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_domain_transfer():
    """Run EXP-EXT-005: Domain Transfer Test."""
    print("=" * 70)
    print("EXP-EXT-005: Domain Transfer Test")
    print("=" * 70)
    print()

    # Create fitted dynamics memory
    dm_config = DynamicsMemoryConfig(model_type="local", min_samples_for_fit=10)
    dm = DynamicsMemory(dim=16, config=dm_config)
    rng = np.random.default_rng(42)
    for _ in range(100):
        state = rng.standard_normal(16)
        velocity = rng.standard_normal(16) * 0.1
        dm._states.append(state)
        dm._velocities.append(velocity)
    dm._fitted = True

    gate = IntegrationGate(mode=IntegrationMode.FULL)
    verifier = VerificationLoop(dynamics_memory=dm, verification_threshold=0.1)

    # Agent configs: (id, physics_accuracy, social_accuracy)
    agent_configs = [
        ("A_physics", 0.90, 0.10),   # physics specialist
        ("B_social", 0.10, 0.90),    # social specialist
        ("C_general", 0.50, 0.50),   # general mediocre
    ]

    rng_exp = np.random.default_rng(42)
    n_ticks = 400

    # Track per-domain trust
    domain_trust_history = {aid: {"physics": [], "social": []}
                           for aid, _, _ in agent_configs}

    t0 = time.perf_counter()

    for tick in range(n_ticks):
        for aid, phys_acc, soc_acc in agent_configs:
            # Alternate between physics and social claims
            if tick % 2 == 0:
                claim = PHYSICS_CLAIMS[tick // 2 % len(PHYSICS_CLAIMS)]
                accurate = rng_exp.random() < phys_acc
                outcome = make_physics_outcome(rng_exp, accurate)
                claim_domain = "physics"
            else:
                claim = SOCIAL_CLAIMS[tick // 2 % len(SOCIAL_CLAIMS)]
                accurate = rng_exp.random() < soc_acc
                outcome = make_social_outcome(rng_exp, accurate)
                claim_domain = "social"

            # Process through gate
            candidate = gate.process_interaction(
                claim, source_node=aid, tick=tick)

            # Verify
            if candidate.status.value in ("accepted", "candidate"):
                vid = verifier.submit_for_verification(candidate, tick=tick)
                record = verifier.verify(vid, outcome, tick=tick)
                if record is not None:
                    gate.update_reputation_from_verification(
                        aid, record.verified, record.error, claim_domain)

            # Record domain trust
            reps = gate.get_agent_reputations()
            if aid in reps:
                for d in ["physics", "social"]:
                    agent = gate._foreign_agents.get(aid)
                    if agent:
                        dt = agent.reputation.domain_trust_score(d)
                        domain_trust_history[aid][d].append(dt)

        gate.tick()

    elapsed = time.perf_counter() - t0

    # Analyze
    print()
    print("RESULTS")
    print("=" * 70)
    print(f"  Elapsed: {elapsed:.2f}s")
    print()

    reps = gate.get_agent_reputations()

    for aid, _, _ in agent_configs:
        print(f"  {aid}:")
        print(f"    Global trust: {reps[aid]['trust_score']:.3f}")
        for d in ["physics", "social"]:
            agent = gate._foreign_agents.get(aid)
            dt = agent.reputation.domain_trust_score(d) if agent else 0
            n = agent.reputation.domain_interactions.get(d, 0) if agent else 0
            print(f"    {d} trust: {dt:.3f} (n={n})")
        print()

    # Domain separation analysis
    print("DOMAIN SEPARATION ANALYSIS")
    print("-" * 70)

    a_physics = gate._foreign_agents.get("A_physics")
    b_social = gate._foreign_agents.get("B_social")
    c_general = gate._foreign_agents.get("C_general")

    if a_physics and b_social and c_general:
        a_phys_t = a_physics.reputation.domain_trust_score("physics")
        a_soc_t = a_physics.reputation.domain_trust_score("social")
        b_phys_t = b_social.reputation.domain_trust_score("physics")
        b_soc_t = b_social.reputation.domain_trust_score("social")
        c_phys_t = c_general.reputation.domain_trust_score("physics")
        c_soc_t = c_general.reputation.domain_trust_score("social")

        print(f"  Physics domain: A={a_phys_t:.3f} > C={c_phys_t:.3f} > B={b_phys_t:.3f}")
        print(f"    A > C: {'YES' if a_phys_t > c_phys_t else 'NO'}")
        print(f"    C > B: {'YES' if c_phys_t > b_phys_t else 'NO'}")
        print()
        print(f"  Social domain: B={b_soc_t:.3f} > C={c_soc_t:.3f} > A={a_soc_t:.3f}")
        print(f"    B > C: {'YES' if b_soc_t > c_soc_t else 'NO'}")
        print(f"    C > A: {'YES' if c_soc_t > a_soc_t else 'NO'}")

        # Domain specificity: does A know more about physics than social?
        a_specificity = a_phys_t - a_soc_t
        b_specificity = b_soc_t - b_phys_t
        print()
        print(f"  A domain specificity (physics-soc): {a_specificity:+.3f}")
        print(f"  B domain specificity (social-phys): {b_specificity:+.3f}")

    # Verification stats
    print()
    print("VERIFICATION STATISTICS")
    print("-" * 70)
    vstats = verifier.get_stats()
    print(f"  Total: {vstats['total_records']}, Passed: {vstats['total_verified']}, "
          f"Rate: {vstats['pass_rate']:.1%}")

    # Gate stats
    print()
    print("GATE STATISTICS")
    print("-" * 70)
    gstats = gate.stats
    for k, v in sorted(gstats.items()):
        print(f"  {k}: {v}")

    return domain_trust_history


if __name__ == "__main__":
    run_domain_transfer()
