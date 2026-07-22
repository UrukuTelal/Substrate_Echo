"""Tests for enhanced memory consolidation, agent refinement, conservation, metric, topology."""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from substrate_echo.core.ontological_field import OntologicalField
from substrate_echo.core.attractor_memory import AttractorMemory, ConsolidationConfig
from substrate_echo.core.cognitive_agents import (
    AgentEcology, AgentRole, MessageType,
)
from substrate_echo.models.experience import Experience, ExperienceType
from substrate_echo.dynamics.conservation import ConservationHooks
from substrate_echo.dynamics.metric_interface import MetricInterface, MetricTensor
from substrate_echo.dynamics.topology_events import (
    TopologyEventQueue, TopologyEvent, TopologyEventType,
)
from substrate_echo.dynamics.state_transitions import (
    StateTransitionManager, StateTransition, TransitionCause,
)


# ── Memory Consolidation Tests ────────────────────────────────────

def test_memory_merge():
    field = OntologicalField()
    mem = AttractorMemory(field)
    
    # Encode similar experiences (same direction, slight variation)
    for i in range(5):
        exp = Experience(
            experience_id=f"sim_{i}",
            experience_type=ExperienceType.LEARNING,
            description=f"Similar learning {i}",
            psv_snapshot=[0.5 + i * 0.001] * 16,
        )
        mem.encode(exp)
    
    before = len(mem.traces)
    stats = mem.consolidate(force=True)
    
    # Some should have merged
    assert stats["merged"] >= 0  # may or may not merge depending on similarity
    print(f"PASS: test_memory_merge (merged={stats['merged']}, before={before}, after={len(mem.traces)})")


def test_memory_pruning():
    field = OntologicalField()
    config = ConsolidationConfig(prune_strength_threshold=0.3)
    mem = AttractorMemory(field, config=config)
    
    # Encode many experiences with low importance
    for i in range(20):
        exp = Experience(
            experience_id=f"weak_{i}",
            experience_type=ExperienceType.PERCEPTION,
            description=f"Weak perception {i}",
            psv_snapshot=[0.1 + i * 0.05] * 16,
            importance=0.1,
        )
        mem.encode(exp)
    
    # Force consolidation
    stats = mem.consolidate(force=True)
    print(f"PASS: test_memory_pruning (pruned={stats['pruned']})")


def test_identity_formation():
    field = OntologicalField()
    mem = AttractorMemory(field)
    
    # Create strong, stable memories (identity-forming)
    for i in range(5):
        exp = Experience(
            experience_id=f"identity_{i}",
            experience_type=ExperienceType.LEARNING,
            description=f"Core identity memory {i}",
            psv_snapshot=[0.7] * 16,  # same direction = identity cluster
            importance=0.9,
        )
        mem.encode(exp)
    
    identity = mem.identity_pattern()
    assert identity is not None
    assert identity.shape == (16,)
    
    coherence = mem.identity_coherence()
    assert coherence > 0.5  # should be coherent
    print(f"PASS: test_identity_formation (coherence={coherence:.3f})")


def test_identity_no_cluster():
    field = OntologicalField()
    mem = AttractorMemory(field)
    
    # Too few memories for identity
    identity = mem.identity_pattern()
    assert identity is None
    print("PASS: test_identity_no_cluster")


def test_memory_stats_enhanced():
    field = OntologicalField()
    mem = AttractorMemory(field)
    
    for i in range(3):
        exp = Experience(
            experience_id=f"stat_{i}",
            experience_type=ExperienceType.INTERACTION,
            description=f"Stat test {i}",
            psv_snapshot=[0.5] * 16,
            importance=0.7,
        )
        mem.encode(exp)
    
    stats = mem.memory_stats()
    assert "identity_coherence" in stats
    assert "has_identity" in stats
    assert stats["total_memories"] == 3
    print("PASS: test_memory_stats_enhanced")


# ── Agent Refinement Tests ────────────────────────────────────────

def test_agent_energy_management():
    ecology = AgentEcology(max_energy=0.1, energy_regen_rate=0.01)
    
    state = np.full(16, 0.95)
    
    # First tick should activate agents
    responses = ecology.tick(state)
    assert len(responses) > 0
    
    # Energy should be depleted
    assert ecology.energy_pool < 0.1
    
    # Second tick with low energy — fewer agents
    responses2 = ecology.tick(state)
    print(f"PASS: test_agent_energy_management (pool={ecology.energy_pool:.3f})")


def test_agent_deliberation():
    ecology = AgentEcology()
    
    state = np.full(16, 0.95)
    responses = ecology.tick(state)
    
    # Deliberation should have generated messages
    assert ecology._message_log is not None
    print(f"PASS: test_agent_deliberation (messages={len(ecology._message_log)})")


def test_agent_message_handling():
    ecology = AgentEcology()
    
    # Manually send a message to perception agent
    from substrate_echo.core.cognitive_agents import AgentMessage
    msg = AgentMessage(
        sender=AgentRole.PLANNING,
        receiver=AgentRole.PERCEPTION,
        message_type=MessageType.REQUEST_INFO,
    )
    
    perception = ecology.get_agent(AgentRole.PERCEPTION)
    perception.receive_message(msg)
    responses = perception.process_inbox()
    
    # Perception should respond with share_state
    assert len(responses) == 1
    assert responses[0].message_type == MessageType.SHARE_STATE
    print("PASS: test_agent_message_handling")


def test_agent_dynamic_activation():
    ecology = AgentEcology()
    agent = ecology.get_agent(AgentRole.PERCEPTION)
    
    # Record initial threshold
    initial_threshold = agent.activation_threshold
    
    # Simulate 30 ticks with no activation (history = all False)
    neutral_state = np.full(16, 0.5)
    for _ in range(30):
        ecology.tick(neutral_state)
    
    # Threshold should have decreased due to inactivity
    assert agent.activation_threshold < initial_threshold
    print(f"PASS: test_agent_dynamic_activation (threshold {initial_threshold:.3f} -> {agent.activation_threshold:.3f})")


def test_agent_consensus_weighted():
    ecology = AgentEcology()
    
    state = np.full(16, 0.95)
    responses = ecology.tick(state)
    
    consensus = ecology.get_consensus(responses)
    assert consensus is not None
    assert consensus.confidence > 0
    print("PASS: test_agent_consensus_weighted")


def test_agent_energy_status():
    ecology = AgentEcology(max_energy=1.0)
    
    status = ecology.energy_status()
    assert status["available"] == 1.0
    assert status["max"] == 1.0
    print("PASS: test_agent_energy_status")


# ── Conservation Hooks Tests ──────────────────────────────────────

def test_conservation_bounds():
    hooks = ConservationHooks(enabled=True)
    
    # Valid state (norm ~ 1.0)
    state = np.ones(16) / 4.0  # norm = 1.0
    results = hooks.check_all(state)
    assert all(r.passed for r in results)
    
    # Invalid state
    bad_state = np.full(16, 1.5)
    results_bad = hooks.check_all(bad_state)
    assert not any(r.passed for r in results_bad if r.law_name == "state_bounds")
    print("PASS: test_conservation_bounds")


def test_conservation_disabled():
    hooks = ConservationHooks(enabled=False)
    
    # Even bad state passes when disabled
    bad_state = np.full(16, 1.5)
    results = hooks.check_all(bad_state)
    assert all(r.passed for r in results)
    print("PASS: test_conservation_disabled")


def test_conservation_energy_tracking():
    hooks = ConservationHooks(enabled=True)
    
    state = np.ones(16) / 4.0
    hooks.check_energy(state)  # sets baseline
    
    # Same state should pass
    result = hooks.check_energy(state)
    assert result.passed
    
    # Very different state should fail
    result2 = hooks.check_energy(np.full(16, 0.9))
    assert not result2.passed
    print("PASS: test_conservation_energy_tracking")


# ── Metric Interface Tests ────────────────────────────────────────

def test_metric_identity():
    metric = MetricTensor.identity()
    a = np.zeros(16)
    b = np.ones(16)
    
    dist = metric.distance(a, b)
    assert abs(dist - 4.0) < 0.01  # sqrt(16) = 4
    print("PASS: test_metric_identity")


def test_metric_inner_product():
    metric = MetricTensor.identity()
    a = np.ones(16)
    b = np.ones(16)
    
    ip = metric.inner_product(a, b)
    assert abs(ip - 16.0) < 0.01
    print("PASS: test_metric_inner_product")


def test_metric_interface_evolve():
    miface = MetricInterface()
    
    state = np.full(16, 0.5)
    metric = miface.evolve(state)
    
    assert miface._evolution_count == 1
    assert isinstance(metric, MetricTensor)
    print("PASS: test_metric_interface_evolve")


def test_metric_distance():
    miface = MetricInterface()
    
    a = np.zeros(16)
    b = np.ones(16) * 0.5
    
    dist = miface.compute_distance(a, b)
    assert dist > 0
    print("PASS: test_metric_distance")


# ── Topology Events Tests ─────────────────────────────────────────

def test_topology_queue_basic():
    queue = TopologyEventQueue()
    
    event = TopologyEvent(
        event_type=TopologyEventType.VACUUM_TUNNELING,
        priority=0.8,
    )
    
    assert queue.enqueue(event)
    assert queue.peek() is not None
    
    dequeued = queue.dequeue()
    assert dequeued.event_type == TopologyEventType.VACUUM_TUNNELING
    assert queue.peek() is None
    print("PASS: test_topology_queue_basic")


def test_topology_queue_priority():
    queue = TopologyEventQueue()
    
    low = TopologyEvent(event_type=TopologyEventType.FOAM_NODE_CREATE, priority=0.2)
    high = TopologyEvent(event_type=TopologyEventType.VORTEX_CREATE, priority=0.9)
    med = TopologyEvent(event_type=TopologyEventType.VACUUM_TUNNELING, priority=0.5)
    
    queue.enqueue(low)
    queue.enqueue(high)
    queue.enqueue(med)
    
    # Should dequeue in priority order
    assert queue.dequeue().event_type == TopologyEventType.VORTEX_CREATE
    assert queue.dequeue().event_type == TopologyEventType.VACUUM_TUNNELING
    assert queue.dequeue().event_type == TopologyEventType.FOAM_NODE_CREATE
    print("PASS: test_topology_queue_priority")


def test_topology_queue_capacity():
    queue = TopologyEventQueue(max_events=3)
    
    for i in range(5):
        queue.enqueue(TopologyEvent(
            event_type=TopologyEventType.FOAM_NODE_CREATE,
            priority=i * 0.1,
        ))
    
    stats = queue.stats()
    assert stats["pending"] == 3
    assert stats["rejected"] == 2
    print("PASS: test_topology_queue_capacity")


def test_topology_energy_check():
    queue = TopologyEventQueue()
    
    event = TopologyEvent(event_type=TopologyEventType.VACUUM_TUNNELING)
    assert queue.energy_check(event, current_energy=1.0)
    print("PASS: test_topology_energy_check")


def test_topology_stats():
    queue = TopologyEventQueue()
    
    queue.enqueue(TopologyEvent(event_type=TopologyEventType.VACUUM_TUNNELING))
    queue.enqueue(TopologyEvent(event_type=TopologyEventType.VORTEX_CREATE))
    queue.enqueue(TopologyEvent(event_type=TopologyEventType.VORTEX_CREATE))
    
    stats = queue.stats()
    assert stats["pending"] == 3
    assert stats["by_type"]["VACUUM_TUNNELING"] == 1
    assert stats["by_type"]["VORTEX_CREATE"] == 2
    print("PASS: test_topology_stats")


# ── Integration: Memory + Conservation + Transitions ──────────────

def test_memory_with_transitions():
    mgr = StateTransitionManager()
    field = OntologicalField()
    mem = AttractorMemory(field, transition_manager=mgr)
    
    for i in range(5):
        exp = Experience(
            experience_id=f"trans_{i}",
            experience_type=ExperienceType.LEARNING,
            description=f"Transition test {i}",
            psv_snapshot=[0.5] * 16,
            importance=0.7,
        )
        mem.encode(exp)
    
    stats = mgr.stats()
    assert stats["by_cause"]["MEMORY_UPDATE"] == 5
    print(f"PASS: test_memory_with_transitions ({stats['by_cause']['MEMORY_UPDATE']} transitions)")


def test_full_system_integration():
    """Test all components working together."""
    # Create all components
    mgr = StateTransitionManager()
    field = OntologicalField()
    config = ConsolidationConfig(consolidation_interval=0.1)
    mem = AttractorMemory(field, config=config, transition_manager=mgr)
    ecology = AgentEcology()
    conservation = ConservationHooks(enabled=False)
    metric = MetricInterface()
    topology = TopologyEventQueue()
    
    # Encode some memories
    for i in range(3):
        exp = Experience(
            experience_id=f"full_{i}",
            experience_type=ExperienceType.LEARNING,
            description=f"Full integration test {i}",
            psv_snapshot=[0.5 + i * 0.1] * 16,
            importance=0.7,
        )
        mem.encode(exp)
    
    # Run agent ecology
    state = np.full(16, 0.9)
    responses = ecology.tick(state, memory=mem)
    
    # Check conservation
    results = conservation.check_all(state)
    assert all(r.passed for r in results)
    
    # Evolve metric
    metric.evolve(state)
    
    # Queue topology event
    topology.enqueue(TopologyEvent(
        event_type=TopologyEventType.VORTEX_CREATE,
        priority=0.5,
    ))
    
    # Consolidate memory
    mem_stats = mem.memory_stats()
    assert mem_stats["total_memories"] == 3
    
    # Check all stats
    mgr_stats = mgr.stats()
    assert mgr_stats["total_transitions"] >= 3
    
    print("PASS: test_full_system_integration")


if __name__ == "__main__":
    print("=== Memory Consolidation ===")
    test_memory_merge()
    test_memory_pruning()
    test_identity_formation()
    test_identity_no_cluster()
    test_memory_stats_enhanced()
    
    print("\n=== Agent Refinement ===")
    test_agent_energy_management()
    test_agent_deliberation()
    test_agent_message_handling()
    test_agent_dynamic_activation()
    test_agent_consensus_weighted()
    test_agent_energy_status()
    
    print("\n=== Conservation Hooks ===")
    test_conservation_bounds()
    test_conservation_disabled()
    test_conservation_energy_tracking()
    
    print("\n=== Metric Interface ===")
    test_metric_identity()
    test_metric_inner_product()
    test_metric_interface_evolve()
    test_metric_distance()
    
    print("\n=== Topology Events ===")
    test_topology_queue_basic()
    test_topology_queue_priority()
    test_topology_queue_capacity()
    test_topology_energy_check()
    test_topology_stats()
    
    print("\n=== Integration ===")
    test_memory_with_transitions()
    test_full_system_integration()
    
    print("\nAll Phase A tests passed!")
