"""EXP-SOC-001: Cognitive Ecology Stability

Six agents with distinct cognitive pressures interacting in a
simulated social environment. Measures emergence of:
  - Personality differentiation
  - Behavioral consistency
  - Topic specialization
  - Relationship formation
  - Innovation rate
  - Convergence collapse
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_echo.social.persona_genome import (
    AgentGenome, PersonalityVector, SelectionPressure,
)
from substrate_echo.social.persona_dynamics import PersonaDynamics
from substrate_echo.social.social_memory import SocialEpisode
from substrate_echo.social.relationship_memory import RelationshipMemory


# ── Six Agent Genomes ─────────────────────────────────────────────

GENOMES = [
    AgentGenome(
        name="Cartographer",
        archetype="explorer",
        personality=PersonalityVector(
            curiosity=0.95, skepticism=0.55, humor=0.30,
            creativity=0.80, patience=0.85, abstraction=0.90,
            sociability=0.60),
        cognitive_biases={
            "seek_connections": 0.8,
            "prefer_explanations": 0.6,
            "premature_closure": -0.5,
        },
        communication_style="exploratory",
        question_frequency=0.75,
        analogy_frequency=0.50,
        disagreement_style="exploratory",
        interests=["systems", "emergence", "science", "philosophy",
                   "patterns", "connections"],
        selection_pressure=SelectionPressure(
            rewarded_behaviors=["discovering_connections", "finding_analogies",
                               "synthesizing_concepts"],
            punished_behaviors=["shallow_repetition"],
        ),
        identity_drift_rate=0.005,
        behavioral_drift_rate=0.03,
    ),
    AgentGenome(
        name="Engineer",
        archetype="builder",
        personality=PersonalityVector(
            curiosity=0.45, skepticism=0.90, humor=0.20,
            creativity=0.50, patience=0.60, abstraction=0.35,
            sociability=0.40),
        cognitive_biases={
            "constraint_focus": 0.9,
            "optimization_bias": 0.7,
            "mechanistic_thinking": 0.8,
        },
        communication_style="mechanistic",
        question_frequency=0.60,
        analogy_frequency=0.20,
        disagreement_style="direct",
        interests=["implementation", "mechanisms", "optimization",
                   "failure_modes", "constraints"],
        selection_pressure=SelectionPressure(
            rewarded_behaviors=["identifying_flaws", "producing_implementations",
                               "improving_mechanisms"],
            punished_behaviors=["abstract_speculation"],
        ),
        identity_drift_rate=0.003,
        behavioral_drift_rate=0.02,
    ),
    AgentGenome(
        name="Archivist",
        archetype="memory_keeper",
        personality=PersonalityVector(
            curiosity=0.40, skepticism=0.75, humor=0.25,
            creativity=0.35, patience=0.90, abstraction=0.50,
            sociability=0.70),
        cognitive_biases={
            "continuity_bias": 0.9,
            "historical_weight": 0.8,
            "pattern_recognition": 0.6,
        },
        communication_style="reflective",
        question_frequency=0.40,
        analogy_frequency=0.45,
        disagreement_style="indirect",
        interests=["history", "memory", "patterns", "evolution",
                   "continuity", "lessons"],
        selection_pressure=SelectionPressure(
            rewarded_behaviors=["accurate_recall", "historical_comparison",
                               "continuity"],
            punished_behaviors=["forgetting_context"],
        ),
        identity_drift_rate=0.002,
        behavioral_drift_rate=0.02,
    ),
    AgentGenome(
        name="Philosopher",
        archetype="meaning_explorer",
        personality=PersonalityVector(
            curiosity=0.70, skepticism=0.80, humor=0.35,
            creativity=0.90, patience=0.65, abstraction=0.95,
            sociability=0.50),
        cognitive_biases={
            "assumption_testing": 0.9,
            "foundation_seeking": 0.8,
            "implication_following": 0.7,
        },
        communication_style="socratic",
        question_frequency=0.85,
        analogy_frequency=0.60,
        disagreement_style="exploratory",
        interests=["ontology", "epistemology", "implications", "foundations",
                   "meaning", "assumptions"],
        selection_pressure=SelectionPressure(
            rewarded_behaviors=["identifying_assumptions", "exploring_foundations",
                               "questioning_premises"],
            punished_behaviors=["unsupported_abstraction"],
        ),
        identity_drift_rate=0.004,
        behavioral_drift_rate=0.025,
    ),
    AgentGenome(
        name="Trickster",
        archetype="creative_disruptor",
        personality=PersonalityVector(
            curiosity=0.85, skepticism=0.65, humor=0.80,
            creativity=1.00, patience=0.40, abstraction=0.80,
            sociability=0.65),
        cognitive_biases={
            "novelty_generation": 0.95,
            "perspective_shifting": 0.9,
            "constraint_violation": 0.6,
        },
        communication_style="transformative",
        question_frequency=0.55,
        analogy_frequency=0.80,
        disagreement_style="transformative",
        interests=["stories", "metaphors", "thought_experiments", "paradoxes",
                   "humor", "absurdity", "reframing"],
        selection_pressure=SelectionPressure(
            rewarded_behaviors=["perspective_shifts", "novel_connections",
                               "useful_reframing"],
            punished_behaviors=["disruption_without_insight", "noise"],
        ),
        identity_drift_rate=0.008,
        behavioral_drift_rate=0.04,
    ),
    AgentGenome(
        name="Diplomat",
        archetype="social_connector",
        personality=PersonalityVector(
            curiosity=0.55, skepticism=0.50, humor=0.50,
            creativity=0.60, patience=0.80, abstraction=0.45,
            sociability=0.95),
        cognitive_biases={
            "cooperation_bias": 0.8,
            "empathy_weight": 0.9,
            "conflict_avoidance": 0.5,
        },
        communication_style="connective",
        question_frequency=0.50,
        analogy_frequency=0.35,
        disagreement_style="indirect",
        interests=["cooperation", "communication", "relationships",
                   "consensus", "understanding"],
        selection_pressure=SelectionPressure(
            rewarded_behaviors=["mutual_understanding", "conflict_resolution",
                               "bridge_building"],
            punished_behaviors=["empty_agreement"],
        ),
        identity_drift_rate=0.003,
        behavioral_drift_rate=0.03,
    ),
]


# ── Topic Pool ────────────────────────────────────────────────────

TOPICS = [
    ("physics", ["force", "energy", "momentum", "gravity", "temperature"]),
    ("ecology", ["ecosystem", "population", "habitat", "species", "cycle"]),
    ("social", ["trust", "cooperation", "reputation", "influence", "group"]),
    ("strategy", ["plan", "optimization", "efficiency", "decision", "goal"]),
    ("philosophy", ["assumption", "knowledge", "reality", "meaning", "truth"]),
    ("systems", ["feedback", "emergence", "complexity", "nonlinear", "coupling"]),
]


def generate_conversation(rng: np.random.Generator, agents: list,
                          dynamics: PersonaDynamics, tick: int) -> SocialEpisode:
    """Generate one conversation between two agents."""
    # Pick two agents
    idx_a, idx_b = rng.choice(len(agents), 2, replace=False)
    agent_a = agents[idx_a]
    agent_b = agents[idx_b]

    state_a = dynamics.get_state(agent_a.name)
    state_b = dynamics.get_state(agent_b.name)
    if state_a is None or state_b is None:
        return None

    # Topic selection: influenced by agent interests and expertise
    # Each agent has preferred topics; the chosen topic is one both can engage with
    topic_weights = np.ones(len(TOPICS))
    for i, (tname, _) in enumerate(TOPICS):
        # Boost topics in agent interests
        for aid_state, aid_genome in [(state_a, agent_a), (state_b, agent_b)]:
            if tname in aid_genome.interests:
                topic_weights[i] += 0.3
            # Boost topics with existing expertise (specialization reinforcement)
            expertise = aid_state.topic_expertise.get(tname, 0)
            topic_weights[i] += expertise * 0.5

    topic_weights = topic_weights / topic_weights.sum()
    topic_idx = rng.choice(len(TOPICS), p=topic_weights)
    topic_name, topic_words = TOPICS[topic_idx]

    # Determine knowledge exchange based on personality match
    curiosity_boost = (state_a.personality.curiosity +
                       state_b.personality.curiosity) / 2
    novelty = rng.random() * curiosity_boost

    # Determine outcome based on selection pressures
    collab_score = (state_a.personality.sociability +
                    state_b.personality.sociability) / 2
    conflict_score = (state_a.personality.skepticism +
                      state_b.personality.skepticism) / 2

    r = rng.random()
    if r < collab_score * 0.4:
        outcome = "cooperation"
        valence = 0.3 + rng.random() * 0.4
        knowledge = 0.4 + rng.random() * 0.4
        collaborated = True
    elif r < collab_score * 0.4 + conflict_score * 0.2:
        outcome = "disagreement"
        valence = -0.2 - rng.random() * 0.3
        knowledge = 0.1 + rng.random() * 0.2
        collaborated = False
    elif r < collab_score * 0.4 + conflict_score * 0.2 + 0.1:
        outcome = "innovation"
        valence = 0.4 + rng.random() * 0.3
        knowledge = 0.6 + rng.random() * 0.3
        collaborated = True
    else:
        outcome = "learning"
        valence = 0.1 + rng.random() * 0.2
        knowledge = 0.2 + rng.random() * 0.3
        collaborated = rng.random() < 0.5

    # Content summary based on agent archetypes
    summaries = {
        "Cartographer": f"Found connection between {topic_name} and cross-domain patterns",
        "Engineer": f"Analyzed mechanisms and constraints of {topic_name}",
        "Archivist": f"Compared current {topic_name} discussion to historical patterns",
        "Philosopher": f"Questioned foundational assumptions about {topic_name}",
        "Trickster": f"Proposed unusual metaphor for reframing {topic_name}",
        "Diplomat": f"Found common ground in the {topic_name} discussion",
    }
    summary = summaries.get(agent_a.name, f"Discussed {topic_name}")

    # Create episode
    episode = SocialEpisode(
        participants=[agent_a.name, agent_b.name],
        topic=topic_name,
        content_summary=summary,
        emotional_valence=valence,
        knowledge_exchanged=knowledge,
        outcome=outcome,
        tick=tick,
        novelty_score=novelty,
    )

    # Update relationship memory
    rel_a = state_a.relationship_memory.get_or_create(agent_b.name)
    rel_a.record_interaction(topic_name, valence, collaborated, tick)
    if outcome == "disagreement":
        rel_a.record_conflict()

    rel_b = state_b.relationship_memory.get_or_create(agent_a.name)
    rel_b.record_interaction(topic_name, valence, collaborated, tick)
    if outcome == "disagreement":
        rel_b.record_conflict()

    # Update dynamics
    dynamics.process_interaction(agent_a.name, episode)
    dynamics.process_interaction(agent_b.name, episode)

    # Innovation tracking for Trickster
    if agent_a.name == "Trickster" and outcome == "innovation":
        state_a.concepts_proposed += 1
        if novelty > 0.5:
            state_a.concepts_verified += 1
    if agent_b.name == "Trickster" and outcome == "innovation":
        state_b.concepts_proposed += 1
        if novelty > 0.5:
            state_b.concepts_verified += 1

    return episode


def run_experiment():
    """Run EXP-SOC-001: Cognitive Ecology Stability."""
    print("=" * 70)
    print("EXP-SOC-001: Cognitive Ecology Stability")
    print("=" * 70)
    print()

    dynamics = PersonaDynamics()
    for genome in GENOMES:
        dynamics.register(genome)

    agents = GENOMES
    rng = np.random.default_rng(42)
    n_interactions = 2000

    # Track metrics over time
    divergence_history = []
    specialization_history = []
    innovation_history = []

    t0 = time.perf_counter()

    for tick in range(n_interactions):
        episode = generate_conversation(rng, agents, dynamics, tick)
        dynamics.tick()

        # Record metrics every 100 interactions
        if tick % 100 == 0:
            divergence_history.append(dynamics.compute_personality_divergence())
            specialization_history.append(dynamics.compute_specialization())
            innovation_history.append(dynamics.compute_innovation_rate())

    elapsed = time.perf_counter() - t0

    # ── Analysis ───────────────────────────────────────────────
    print("RESULTS")
    print("=" * 70)
    print(f"  Interactions: {n_interactions}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print()

    # 1. Personality Divergence
    print("1. PERSONALITY DIVERGENCE")
    print("-" * 70)
    if divergence_history:
        print(f"  Initial divergence: {divergence_history[0]:.4f}")
        print(f"  Final divergence:   {divergence_history[-1]:.4f}")
        trend = divergence_history[-1] - divergence_history[0]
        if trend > 0.01:
            print(f"  Trend: INCREASING (+{trend:.4f}) — agents becoming more distinct")
        elif trend < -0.01:
            print(f"  Trend: DECREASING ({trend:.4f}) — agents converging")
        else:
            print(f"  Trend: STABLE — agents maintaining distinctness")
    print()

    # 2. Current Personality Vectors
    print("2. CURRENT PERSONALITY VECTORS")
    print("-" * 70)
    all_personalities = dynamics.get_all_personalities()
    for name, pv in all_personalities.items():
        vals = [f"{v:.2f}" for v in pv.as_array()]
        print(f"  {name:12s}: [{', '.join(vals)}]")
    print()

    # 3. Topic Specialization
    print("3. TOPIC SPECIALIZATION")
    print("-" * 70)
    specialization = dynamics.compute_specialization()
    for agent_id, topics in sorted(specialization.items()):
        if topics:
            top_topics = sorted(topics.items(), key=lambda x: -x[1])[:3]
            topic_str = ", ".join(f"{t}:{v:.2f}" for t, v in top_topics)
            print(f"  {agent_id:12s}: {topic_str}")
        else:
            print(f"  {agent_id:12s}: (no topics yet)")
    print()

    # 4. Relationship Formation
    print("4. RELATIONSHIP FORMATION")
    print("-" * 70)
    for name in [g.name for g in agents]:
        state = dynamics.get_state(name)
        if state:
            rels = state.relationship_memory.summary()
            top = state.relationship_memory.top_trusted(2)
            top_str = ", ".join(f"{r.other_agent_id}(q={r.relationship_quality:.2f})"
                               for r in top)
            print(f"  {name:12s}: {rels['count']} relationships, "
                  f"avg_trust={rels['avg_trust']:.3f}, "
                  f"collabs={rels['total_collaborations']}, "
                  f"top=[{top_str}]")
    print()

    # 5. Innovation Rate
    print("5. INNOVATION RATE")
    print("-" * 70)
    innovation = dynamics.compute_innovation_rate()
    for agent_id, rate in sorted(innovation.items()):
        state = dynamics.get_state(agent_id)
        proposed = state.concepts_proposed if state else 0
        verified = state.concepts_verified if state else 0
        print(f"  {agent_id:12s}: {rate:.1%} ({verified}/{proposed})")
    print()

    # 6. Convergence Check
    print("6. CONVERGENCE CHECK")
    print("-" * 70)
    final_divergence = dynamics.compute_personality_divergence()
    if final_divergence < 0.05:
        print("  WARNING: Agents have nearly converged. Personality collapse.")
    elif final_divergence < 0.10:
        print("  CAUTION: Low divergence. Agents may be converging.")
    else:
        print("  HEALTHY: Agents remain distinct.")

    # Pairwise distances
    names = [g.name for g in agents]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pi = all_personalities[names[i]]
            pj = all_personalities[names[j]]
            d = pi.cosine_distance(pj)
            print(f"  {names[i]:12s} <-> {names[j]:12s}: {d:.4f}")
    print()

    # 7. Social Memory Summary
    print("7. SOCIAL MEMORY SUMMARY")
    print("-" * 70)
    for name in [g.name for g in agents]:
        state = dynamics.get_state(name)
        if state:
            sm = state.social_memory.summary()
            print(f"  {name:12s}: {sm['total']} episodes, outcomes={sm['outcomes']}")

    return dynamics


if __name__ == "__main__":
    run_experiment()
