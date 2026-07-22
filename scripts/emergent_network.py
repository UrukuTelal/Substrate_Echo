#!/usr/bin/env python3
"""
D. Emergent Network Structure — topology correlation across layers.

Measures and compares:
1. Field topology: correlation structure of 16D field evolution
2. Memory topology: which memories are similar/recalled together
3. Communication topology: who talks to whom
4. Social topology: who influences whom (reputation-weighted)

Question: Do these topologies become correlated over time?
"""
import sys, os, json, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.field_evolution import FieldEvolver, FieldConfig
from substrate_echo.dynamics.pillar_coupling import PillarCoupling
from substrate_echo.core.multi_agent_dynamics import SocialField, SocialConfig
from substrate_echo.core.attractor_memory import AttractorMemory
from substrate_echo.core.ontological_field import OntologicalField
from substrate_echo.models.experience import Experience, ExperienceType


PILLAR_NAMES = [
    "Awareness", "Willpower", "Force", "Influence",
    "Resistance", "Integrity", "Cohesion", "Relation",
    "Presence", "Warmth", "Memory", "Attraction",
    "Harm", "Distortion", "Flux", "Depth",
]


def build_field_topology(field_history, window=50):
    """Build 16x16 correlation matrix from field state history."""
    if len(field_history) < 2:
        return np.eye(16)

    data = np.array(field_history[-window:])
    if data.shape[0] < 2:
        return np.eye(16)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std < 1e-10] = 1.0

    centered = (data - mean) / std
    corr = (centered.T @ centered) / data.shape[0]

    return (corr + 1) / 2


def build_memory_topology(memory, n_agents=16):
    """Build 16x16 correlation matrix from memory patterns.
    
    Computes pairwise similarity between stored attractor centers,
    projected into 16D pillar space.
    """
    recent = memory.get_recent(100)
    if len(recent) < 2:
        return np.eye(16)

    centers = []
    for t in recent:
        if t.attractor_center is not None and len(t.attractor_center) == 16:
            centers.append(np.array(t.attractor_center))

    if len(centers) < 2:
        return np.eye(16)

    # Compute pairwise cosine similarity → 16x16 correlation-like matrix
    data = np.array(centers[-50:])
    n_pat = data.shape[0]

    # Build 16x16 matrix: how often pillar i and pillar j co-vary across memories
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std < 1e-10] = 1.0
    centered = (data - mean) / std
    corr = (centered.T @ centered) / n_pat

    return (corr + 1) / 2


def build_communication_topology_16d(sf):
    """Build 16x16 matrix from communication patterns.
    
    For each message, compute the state delta between sender and receiver,
    then correlate across all messages.
    """
    agents = list(sf.get_all_agents().keys())
    if len(agents) < 2:
        return np.eye(16)

    # Build state-delta matrix from messages
    deltas = []
    for msg in sf._message_log[-100:]:
        sender_id = msg.get("sender")
        receiver_id = msg.get("receiver")
        if sender_id in agents and receiver_id in agents:
            sender = sf.get_agent(sender_id)
            receiver = sf.get_agent(receiver_id)
            delta = sender.state - receiver.state
            deltas.append(delta)

    if len(deltas) < 3:
        return np.eye(16)

    data = np.array(deltas)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std < 1e-10] = 1.0
    centered = (data - mean) / std
    corr = (centered.T @ centered) / data.shape[0]

    return (corr + 1) / 2


def build_social_topology_16d(sf):
    """Build 16x16 matrix from agent influence patterns.
    
    For each pair of agents, compute the influence vector,
    then correlate across all pairs.
    """
    agents = list(sf.get_all_agents().keys())
    if len(agents) < 2:
        return np.eye(16)

    # Compute influence vectors for each agent
    influence_data = []
    for aid in agents:
        influence = sf.compute_social_influence(aid)
        influence_data.append(influence)

    data = np.array(influence_data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std < 1e-10] = 1.0
    centered = (data - mean) / std
    corr = (centered.T @ centered) / data.shape[0]

    return (corr + 1) / 2


def topology_correlation(topo_a, topo_b):
    """Compute correlation between two topology matrices.
    
    Uses the full flattened upper triangle (excluding diagonal).
    Handles different sizes by truncating to the smaller matrix.
    """
    min_size = min(topo_a.shape[0], topo_b.shape[0])
    if min_size < 2:
        return 0.0
    a = topo_a[:min_size, :min_size]
    b = topo_b[:min_size, :min_size]

    # Flatten upper triangles (excluding diagonal)
    mask = np.triu(np.ones_like(a, dtype=bool), k=1)
    a_flat = a[mask]
    b_flat = b[mask]

    if len(a_flat) < 2:
        return 0.0

    # Pearson correlation
    std_a = np.std(a_flat)
    std_b = np.std(b_flat)
    if std_a < 1e-10 or std_b < 1e-10:
        return 0.0

    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def run_experiment(n_agents=10, n_ticks=500):
    print("=" * 60)
    print("EMERGENT NETWORK STRUCTURE")
    print(f"  {n_agents} agents, {n_ticks} ticks")
    print("=" * 60)

    # Setup
    cfg = FieldConfig(dt=0.01, gamma=0.01, lambda_gl=1.0, temperature=0.001)
    evolver = FieldEvolver(cfg)
    coupling = PillarCoupling()

    field = np.full(16, 0.5)

    sf = SocialField(SocialConfig(
        influence_strength=0.05,
        influence_range=1.5,
        max_communication_range=1.0,
    ))
    roles = ['perceiver', 'analyzer', 'synthesizer', 'guardian', 'explorer']
    for i in range(n_agents):
        state = np.random.uniform(0.2, 0.8, 16)
        sf.add_agent(f"a_{i}", roles[i % len(roles)], state)

    # Memory
    ont_field = OntologicalField()
    memory = AttractorMemory(field=ont_field)

    # Track topology evolution
    field_history = []
    correlations_over_time = []

    for tick in range(n_ticks):
        # Field evolution
        rhs = evolver.rhs(field, cfg.dt)
        field = field + cfg.dt * rhs
        field = np.clip(field, 0.0, 1.0)
        pillar = coupling.project_to_pillars(field)

        field_history.append(pillar.copy())

        # Agent dynamics
        for aid in sf.get_all_agents():
            influence = sf.compute_social_influence(aid)
            agent = sf.get_agent(aid)
            agent.state = np.clip(agent.state + influence, 0.0, 1.0)

            # Some agents communicate
            agents = list(sf.get_all_agents().keys())
            partners = [a for a in agents if a != aid][:2]
            for partner in partners:
                sf.send_message(aid, partner, "update", agent.state)

        # Memory encoding
        if tick % 20 == 0 and len(field_history) >= 10:
            try:
                exp = Experience(
                    experience_id=f"exp_{tick}",
                    experience_type=ExperienceType.LEARNING,
                    psv_snapshot=pillar.tolist(),
                    description=f"Tick {tick} field state",
                )
                memory.encode(exp)
            except Exception:
                pass

        # Measure topology correlations every 50 ticks
        if tick % 50 == 0 and tick > 0:
            field_topo = build_field_topology(field_history, window=50)
            comm_topo = build_communication_topology_16d(sf)
            social_topo = build_social_topology_16d(sf)
            mem_topo = build_memory_topology(memory, n_agents)

            # Cross-topology correlations (all 16x16 now)
            fc_corr = topology_correlation(field_topo, comm_topo)
            fs_corr = topology_correlation(field_topo, social_topo)
            fm_corr = topology_correlation(field_topo, mem_topo)
            cs_corr = topology_correlation(comm_topo, social_topo)
            cm_corr = topology_correlation(comm_topo, mem_topo)
            sm_corr = topology_correlation(social_topo, mem_topo)

            correlations_over_time.append({
                'tick': tick,
                'field_comm': fc_corr,
                'field_social': fs_corr,
                'field_memory': fm_corr,
                'comm_social': cs_corr,
                'comm_memory': cm_corr,
                'social_memory': sm_corr,
                'mean': (fc_corr + fs_corr + fm_corr + cs_corr + cm_corr + sm_corr) / 6,
            })

            if tick % 200 == 0:
                print(f"  tick {tick:>4d}: F-C={fc_corr:+.3f} "
                      f"F-S={fs_corr:+.3f} F-M={fm_corr:+.3f} "
                      f"C-S={cs_corr:+.3f} C-M={cm_corr:+.3f} S-M={sm_corr:+.3f}")

    # Final analysis
    if correlations_over_time:
        initial = correlations_over_time[0]
        final = correlations_over_time[-1]

        print(f"\n--- Topology Correlation Evolution ---")
        print(f"  {'Pair':<15s} {'Initial':>8s} {'Final':>8s} {'Delta':>8s}")
        for key, label in [('field_comm', 'Field-Comm'),
                           ('field_social', 'Field-Social'),
                           ('field_memory', 'Field-Memory'),
                           ('comm_social', 'Comm-Social'),
                           ('comm_memory', 'Comm-Memory'),
                           ('social_memory', 'Social-Memory')]:
            if key in final and key in initial:
                d = final[key] - initial[key]
                print(f"  {label:<15s} {initial[key]:+8.3f} {final[key]:+8.3f} "
                      f"{d:+8.3f}")

        mean_init = initial['mean']
        mean_final = final['mean']
        print(f"\n  Mean correlation: {mean_init:+.3f} -> {mean_final:+.3f} "
              f"({mean_final - mean_init:+.3f})")

        if mean_final > 0.3:
            print(f"  Topologies ARE correlated (mean > 0.3)")
        elif mean_final > 0.1:
            print(f"  Topologies show moderate correlation")
        else:
            print(f"  Topologies are largely independent")

    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'network_structure.json')
    with open(out, 'w') as f:
        json.dump({
            'config': {'n_agents': n_agents, 'n_ticks': n_ticks},
            'correlations': correlations_over_time,
            'final_topologies': {
                'field': field_topo.tolist() if 'field_topo' in dir() else None,
            },
        }, f, indent=2)
    print(f"\nResults saved to {out}")

    return correlations_over_time


if __name__ == '__main__':
    agents = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    run_experiment(n_agents=agents, n_ticks=ticks)
