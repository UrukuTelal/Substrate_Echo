#!/usr/bin/env python3
"""
Multi-Agent Behavioral Experiments.

Tests: consensus formation, cluster emergence, role specialization,
reputation dynamics, communication topology evolution.
"""
import sys, os, json, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.core.multi_agent_dynamics import SocialField, SocialConfig


def experiment_consensus(n_agents=8, n_ticks=200):
    """Test: do agents reach consensus through social influence?"""
    print("\n--- Consensus Experiment ---")
    sf = SocialField(SocialConfig(influence_strength=0.05, influence_range=2.0))

    # Initialize agents with diverse states
    roles = ['perceiver', 'analyzer', 'synthesizer', 'guardian'] * (n_agents // 4)
    for i in range(n_agents):
        state = np.random.uniform(0.0, 1.0, 16)
        sf.add_agent(f"a_{i}", roles[i], state)

    # Record coherence over time
    coherence_history = []
    diversity_history = []

    for tick in range(n_ticks):
        for aid in sf.get_all_agents():
            influence = sf.compute_social_influence(aid)
            agent = sf.get_agent(aid)
            agent.state = np.clip(agent.state + influence, 0.0, 1.0)

        coherence_history.append(sf.compute_collective_coherence())
        diversity_history.append(sf.compute_collective_diversity())

    initial_c = coherence_history[0]
    final_c = coherence_history[-1]
    initial_d = diversity_history[0]
    final_d = diversity_history[-1]

    print(f"  Coherence: {initial_c:.3f} -> {final_c:.3f} "
          f"(delta={final_c - initial_c:+.3f})")
    print(f"  Diversity: {initial_d:.3f} -> {final_d:.3f} "
          f"(delta={final_d - initial_d:+.3f})")
    print(f"  Converged: {'yes' if final_c > 0.8 else 'no'} "
          f"(threshold=0.8)")

    return {
        'initial_coherence': initial_c,
        'final_coherence': final_c,
        'initial_diversity': initial_d,
        'final_diversity': final_d,
        'converged': final_c > 0.8,
    }


def experiment_clustering(n_agents=12, n_ticks=300):
    """Test: do agents form clusters based on role similarity?"""
    print("\n--- Clustering Experiment ---")
    sf = SocialField(SocialConfig(influence_strength=0.03, influence_range=1.5))

    # Two groups with distinct initial states
    for i in range(n_agents // 2):
        state = np.full(16, 0.3) + np.random.randn(16) * 0.05
        sf.add_agent(f"left_{i}", "perceiver", np.clip(state, 0, 1))

    for i in range(n_agents // 2):
        state = np.full(16, 0.7) + np.random.randn(16) * 0.05
        sf.add_agent(f"right_{i}", "guardian", np.clip(state, 0, 1))

    states_over_time = []
    for tick in range(n_ticks):
        for aid in sf.get_all_agents():
            influence = sf.compute_social_influence(aid)
            agent = sf.get_agent(aid)
            agent.state = np.clip(agent.state + influence, 0.0, 1.0)

        if tick % 50 == 0:
            states = {aid: a.state.mean() for aid, a in sf.get_all_agents().items()}
            states_over_time.append(states)

    # Measure cluster separation
    left_states = [sf.get_agent(f"left_{i}").state for i in range(n_agents // 2)]
    right_states = [sf.get_agent(f"right_{i}").state for i in range(n_agents // 2)]

    left_center = np.mean(left_states, axis=0)
    right_center = np.mean(right_states, axis=0)
    cluster_dist = np.linalg.norm(left_center - right_center)

    left_spread = np.mean([np.linalg.norm(s - left_center) for s in left_states])
    right_spread = np.mean([np.linalg.norm(s - right_center) for s in right_states])

    print(f"  Cluster distance: {cluster_dist:.3f}")
    print(f"  Left spread: {left_spread:.3f}, Right spread: {right_spread:.3f}")
    print(f"  Separated: {'yes' if cluster_dist > 0.3 else 'no'}")

    return {
        'cluster_distance': cluster_dist,
        'left_spread': left_spread,
        'right_spread': right_spread,
        'separated': cluster_dist > 0.3,
    }


def experiment_reputation(n_agents=6, n_ticks=200):
    """Test: does reputation reward good behavior and penalize bad?"""
    print("\n--- Reputation Experiment ---")
    sf = SocialField(SocialConfig(
        reputation_bonus=0.05,
        reputation_penalty=0.08,
        reputation_decay=0.001,
    ))

    for i in range(n_agents):
        sf.add_agent(f"a_{i}", "analyst", np.random.uniform(0.3, 0.7, 16))

    # Half agents always succeed, half always fail
    rep_history = {aid: [] for aid in sf.get_all_agents()}

    for tick in range(n_ticks):
        for aid in sf.get_all_agents():
            idx = int(aid.split('_')[1])
            success = idx < n_agents // 2
            sf.record_action(aid, success)

        for aid in sf.get_all_agents():
            rep_history[aid].append(sf.get_agent(aid).reputation)

    # Compare final reputations
    good_agents = [f"a_{i}" for i in range(n_agents // 2)]
    bad_agents = [f"a_{i}" for i in range(n_agents // 2, n_agents)]

    avg_good = np.mean([sf.get_agent(a).reputation for a in good_agents])
    avg_bad = np.mean([sf.get_agent(a).reputation for a in bad_agents])

    print(f"  Good agents reputation: {avg_good:.3f}")
    print(f"  Bad agents reputation:  {avg_bad:.3f}")
    print(f"  Separation: {avg_good - avg_bad:.3f}")
    print(f"  Effective: {'yes' if avg_good > avg_bad + 0.1 else 'no'}")

    return {
        'avg_good_reputation': avg_good,
        'avg_bad_reputation': avg_bad,
        'separation': avg_good - avg_bad,
        'effective': avg_good > avg_bad + 0.1,
    }


def experiment_communication(n_agents=8, n_ticks=200):
    """Test: does communication topology emerge?"""
    print("\n--- Communication Topology Experiment ---")
    sf = SocialField(SocialConfig(max_communication_range=1.5))

    for i in range(n_agents):
        state = np.random.uniform(0.2, 0.8, 16)
        sf.add_agent(f"a_{i}", "analyst", state)

    msg_counts = []
    for tick in range(n_ticks):
        # Agents communicate with nearby agents
        agents = list(sf.get_all_agents().keys())
        for i, a_id in enumerate(agents):
            for j, b_id in enumerate(agents):
                if i >= j:
                    continue
                if sf.can_communicate(a_id, b_id):
                    sf.send_message(a_id, b_id, "update", 
                                   sf.get_agent(a_id).state)

        # Apply social influence
        for aid in agents:
            influence = sf.compute_social_influence(aid)
            agent = sf.get_agent(aid)
            agent.state = np.clip(agent.state + influence, 0.0, 1.0)

        msg_counts.append(len(sf._message_log))

    topology = sf.get_communication_topology()
    n_edges = sum(len(v) for v in topology.values())
    n_possible = n_agents * (n_agents - 1)
    density = n_edges / max(n_possible, 1)

    # Connected components
    visited = set()
    components = 0
    for start in topology:
        if start in visited:
            continue
        components += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in topology.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)

    print(f"  Total messages: {msg_counts[-1]}")
    print(f"  Network density: {density:.2f}")
    print(f"  Connected components: {components}")
    print(f"  Fully connected: {'yes' if components == 1 else 'no'}")

    return {
        'total_messages': msg_counts[-1],
        'density': density,
        'components': components,
        'fully_connected': components == 1,
    }


def run_all():
    print("=" * 60)
    print("MULTI-AGENT BEHAVIORAL EXPERIMENTS")
    print("=" * 60)

    results = {}
    results['consensus'] = experiment_consensus()
    results['clustering'] = experiment_clustering()
    results['reputation'] = experiment_reputation()
    results['communication'] = experiment_communication()

    print("\n--- Summary ---")
    all_passed = all([
        results['consensus']['converged'],
        results['clustering']['separated'],
        results['reputation']['effective'],
        results['communication']['fully_connected'],
    ])
    print(f"  All experiments passed: {all_passed}")

    out = os.path.join(os.path.dirname(__file__), '..', 'multi_agent_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {out}")

    return results


if __name__ == '__main__':
    run_all()
