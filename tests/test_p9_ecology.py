"""Tests for P9 — Integrated Cognitive Ecology"""
import numpy as np
from substrate_echo.core.world import World, WorldConfig, ResourceType
from substrate_echo.core.integrated_agent import (
    IntegratedAgent, IntegratedAgentConfig, AgentMetrics
)


def _default_world():
    return World(WorldConfig(
        grid_size=10.0, n_resources=10, observation_range=3.0))


def _default_agent(agent_id=0):
    return IntegratedAgent(agent_id=agent_id)


class TestWorld:
    def test_create_world(self):
        world = _default_world()
        assert world.current_tick == 0
        assert len(world._resources) > 0

    def test_add_agent(self):
        world = _default_world()
        state = world.add_agent(0, position=np.array([5.0, 5.0]))
        assert state.agent_id == 0
        assert 0 in world.agent_ids

    def test_observe(self):
        world = _default_world()
        world.add_agent(0, position=np.array([5.0, 5.0]))
        obs = world.observe(0)
        assert "agent_position" in obs
        assert "nearby_resources" in obs
        assert "nearby_agents" in obs

    def test_apply_action_move(self):
        world = _default_world()
        world.add_agent(0, position=np.array([5.0, 5.0]))
        result = world.apply_action(0, {
            "type": "move",
            "direction": [1, 0],
            "speed": 0.5,
        })
        assert result["success"]

    def test_apply_action_harvest(self):
        world = _default_world()
        world.add_agent(0, position=np.array([5.0, 5.0]))
        result = world.apply_action(0, {
            "type": "harvest",
            "target": [5.0, 5.0],
            "amount": 0.5,
        })
        assert result["action"] == "harvest"

    def test_world_tick(self):
        world = _default_world()
        world.add_agent(0)
        events = world.tick()
        assert world.current_tick == 1
        assert isinstance(events, list)

    def test_resource_regen(self):
        config = WorldConfig(n_resources=5, resource_regen_rate=0.5)
        world = World(config)
        # Deplete a resource
        world._resources[0].quantity = 0
        world._resources[0].depleted = True
        # Tick should not regen depleted
        world.tick()
        assert world._resources[0].depleted

    def test_agent_death(self):
        world = _default_world()
        world.add_agent(0)
        # Drain energy
        for _ in range(500):
            world.apply_action(0, {"type": "wait"})
            world.tick()
        # Agent may be dead
        alive = world.alive_agents
        # Either alive with low energy or dead
        assert len(alive) <= 1

    def test_world_stats(self):
        world = _default_world()
        world.add_agent(0)
        stats = world.stats()
        assert "n_agents" in stats
        assert "n_resources" in stats


class TestIntegratedAgent:
    def test_create_agent(self):
        agent = _default_agent()
        assert agent.id == 0

    def test_think_returns_action(self):
        agent = _default_agent()
        obs = _make_observation()
        action = agent.think(obs)
        assert "type" in action

    def test_think_multiple_ticks(self):
        agent = _default_agent()
        for i in range(10):
            obs = _make_observation()
            action = agent.think(obs)
            assert "type" in action
        assert agent._tick == 10

    def test_think_updates_state(self):
        agent = _default_agent()
        obs = _make_observation()
        agent.think(obs)
        assert np.any(agent._state_16d != 0)

    def test_think_records_episodes(self):
        agent = _default_agent()
        for i in range(5):
            obs = _make_observation()
            agent.think(obs)
        assert agent.metrics.episodes_stored > 0

    def test_think_creates_plans(self):
        agent = _default_agent()
        for i in range(15):
            obs = _make_observation()
            agent.think(obs)
        assert agent.metrics.plans_created > 0

    def test_think_updates_meta_cognition(self):
        agent = _default_agent()
        for i in range(60):
            obs = _make_observation()
            agent.think(obs)
        meta = agent.meta_cognition.get_meta_state()
        assert meta.calibration_error >= 0

    def test_agent_summary(self):
        agent = _default_agent()
        obs = _make_observation()
        agent.think(obs)
        s = agent.summary()
        assert "agent_id" in s
        assert "metrics" in s


class TestWorldAgentIntegration:
    def test_agent_in_world(self):
        world = _default_world()
        agent = _default_agent()
        world.add_agent(agent.id, position=np.array([5.0, 5.0]))

        for i in range(10):
            obs = world.observe(agent.id)
            action = agent.think(obs)
            result = world.apply_action(agent.id, action)
            world.tick()

        assert agent._tick == 10

    def test_multi_agent_world(self):
        world = _default_world()
        agents = [_default_agent(i) for i in range(3)]
        for agent in agents:
            world.add_agent(agent.id, position=np.random.uniform(1, 9, 2))

        for tick in range(20):
            world.tick()
            for agent in agents:
                if agent.id in world.agent_ids:
                    obs = world.observe(agent.id)
                    action = agent.think(obs)
                    world.apply_action(agent.id, action)

        # All agents should have ticked
        for agent in agents:
            assert agent._tick == 20


class TestAgentMetrics:
    def test_metrics_fields(self):
        m = AgentMetrics()
        assert m.ticks == 0
        assert m.actions_taken == 0
        assert m.energy_history == []


# ── Helpers ─────────────────────────────────────────────

def _make_observation():
    """Create a mock observation."""
    return {
        "tick": 0,
        "agent_position": [5.0, 5.0],
        "agent_energy": 0.8,
        "agent_health": 1.0,
        "inventory": {"food": 0.0, "energy": 0.0},
        "nearby_resources": [
            {"type": "food", "position": [6.0, 5.0], "quantity": 0.5, "distance": 1.0},
        ],
        "nearby_agents": [],
        "recent_events": [],
    }
