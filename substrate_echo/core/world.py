"""World — The environment agents live in.

A spatial world with resources, other agents, and environmental
dynamics. Not a simulation engine — a ecology.

The world provides:
- Spatial grid with resource nodes
- Agent presence and movement
- Environmental events (storms, resource depletion, new discoveries)
- Observation model (what each agent can see)
- Reward signal (resource acquisition, survival)

Usage:
    world = World(config)
    world.tick()

    obs = world.observe(agent_id)
    world.apply_action(agent_id, action)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from collections import defaultdict
from enum import Enum


class ResourceType(Enum):
    FOOD = "food"
    ENERGY = "energy"
    INFORMATION = "information"
    MATERIAL = "material"


class EventType(Enum):
    NONE = "none"
    RESOURCE_SPAWN = "resource_spawn"
    RESOURCE_DEPLETE = "resource_deplete"
    STORM = "storm"
    DISCOVERY = "discovery"
    SOCIAL_ENCOUNTER = "social_encounter"


@dataclass
class Resource:
    """A resource node in the world."""
    resource_id: int
    resource_type: ResourceType
    position: np.ndarray
    quantity: float = 1.0
    max_quantity: float = 1.0
    regen_rate: float = 0.01
    depleted: bool = False

    def tick(self):
        if not self.depleted and self.quantity < self.max_quantity:
            self.quantity = min(self.max_quantity,
                                self.quantity + self.regen_rate)

    def harvest(self, amount: float) -> float:
        actual = min(amount, self.quantity)
        self.quantity -= actual
        if self.quantity <= 0:
            self.depleted = True
        return actual


@dataclass
class AgentState:
    """State of an agent in the world."""
    agent_id: int
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    energy: float = 1.0
    health: float = 1.0
    alive: bool = True
    inventory: dict = field(default_factory=lambda: {
        "food": 0.0, "energy": 0.0, "information": 0.0, "material": 0.0
    })

    @property
    def position_16d(self) -> np.ndarray:
        """Convert 2D position to 16D state vector."""
        state = np.zeros(16)
        state[0] = self.position[0]
        state[1] = self.position[1]
        state[2] = self.velocity[0]
        state[3] = self.velocity[1]
        state[4] = self.energy
        state[5] = self.health
        state[6] = sum(self.inventory.values())
        return state


@dataclass
class WorldEvent:
    """An event that happened in the world."""
    event_type: EventType
    tick: int
    position: Optional[np.ndarray] = None
    agent_id: Optional[int] = None
    message: str = ""


@dataclass
class WorldConfig:
    """Configuration for the world."""
    grid_size: float = 10.0
    n_resources: int = 20
    resource_regen_rate: float = 0.01
    energy_decay_rate: float = 0.002
    observation_range: float = 3.0
    max_agents: int = 100
    storm_interval: int = 500
    storm_duration: int = 10
    storm_energy_drain: float = 0.05
    event_probability: float = 0.01


class World:
    """A spatial world for agents to live in.

    Agents exist in a 2D world with resources, other agents,
    and environmental events. Each tick, agents observe their
    surroundings, take actions, and the world updates.

    Usage:
        world = World(config)
        world.add_agent(agent_id, position)

        for tick in range(10000):
            for agent_id in world.agent_ids:
                obs = world.observe(agent_id)
                action = agent.think(obs)
                world.apply_action(agent_id, action)
            world.tick()
    """

    def __init__(self, config: Optional[WorldConfig] = None):
        self.config = config or WorldConfig()
        self._tick_count = 0

        self._agents: dict[int, AgentState] = {}
        self._resources: list[Resource] = []
        self._events: list[WorldEvent] = []
        self._resource_id_counter = 0

        # Event log for analysis
        self._event_log: list[WorldEvent] = []

        # Initialize resources
        self._spawn_initial_resources()

    def add_agent(self, agent_id: int, position: Optional[np.ndarray] = None) -> AgentState:
        """Add an agent to the world."""
        if position is None:
            position = np.random.uniform(0, self.config.grid_size, 2)

        state = AgentState(agent_id=agent_id, position=np.asarray(position))
        self._agents[agent_id] = state
        return state

    def remove_agent(self, agent_id: int) -> None:
        """Remove an agent from the world."""
        if agent_id in self._agents:
            del self._agents[agent_id]

    def observe(self, agent_id: int) -> dict:
        """Get observation for an agent.

        Returns what the agent can see: nearby resources, agents, events.
        """
        if agent_id not in self._agents:
            return {"error": "agent not found"}

        agent = self._agents[agent_id]
        obs_range = self.config.observation_range

        # Nearby resources
        nearby_resources = []
        for res in self._resources:
            dist = np.linalg.norm(res.position - agent.position)
            if dist <= obs_range and not res.depleted:
                nearby_resources.append({
                    "type": res.resource_type.value,
                    "position": res.position.tolist(),
                    "quantity": res.quantity,
                    "distance": dist,
                })

        # Nearby agents
        nearby_agents = []
        for other_id, other in self._agents.items():
            if other_id == agent_id or not other.alive:
                continue
            dist = np.linalg.norm(other.position - agent.position)
            if dist <= obs_range:
                nearby_agents.append({
                    "agent_id": other_id,
                    "position": other.position.tolist(),
                    "velocity": other.velocity.tolist(),
                    "energy": other.energy,
                    "health": other.health,
                    "distance": dist,
                })

        # Recent events
        recent_events = [
            {"type": e.event_type.value, "tick": e.tick, "message": e.message}
            for e in self._events[-5:]
        ]

        return {
            "tick": self._tick_count,
            "agent_position": agent.position.tolist(),
            "agent_energy": agent.energy,
            "agent_health": agent.health,
            "inventory": dict(agent.inventory),
            "nearby_resources": nearby_resources,
            "nearby_agents": nearby_agents,
            "recent_events": recent_events,
        }

    def apply_action(self, agent_id: int, action: dict) -> dict:
        """Apply an action to an agent.

        Actions: move, harvest, interact, wait
        Returns: result of the action
        """
        if agent_id not in self._agents:
            return {"error": "agent not found"}

        agent = self._agents[agent_id]
        if not agent.alive:
            return {"error": "agent dead"}

        action_type = action.get("type", "wait")
        result = {"success": False, "action": action_type}

        if action_type == "move":
            direction = np.array(action.get("direction", [0, 0]), dtype=np.float64)
            direction = np.clip(direction, -1, 1)
            speed = action.get("speed", 0.5)
            new_pos = agent.position + direction * speed
            new_pos = np.clip(new_pos, 0, self.config.grid_size)
            agent.velocity = direction * speed
            agent.position = new_pos
            result["success"] = True

        elif action_type == "harvest":
            target_pos = np.array(action.get("target", agent.position))
            amount = action.get("amount", 0.5)

            # Find closest resource
            best_res = None
            best_dist = float('inf')
            for res in self._resources:
                dist = np.linalg.norm(res.position - target_pos)
                if dist < best_dist and not res.depleted:
                    best_dist = dist
                    best_res = res

            if best_res and best_dist < 2.0:
                harvested = best_res.harvest(amount)
                rtype = best_res.resource_type.value
                agent.inventory[rtype] = agent.inventory.get(rtype, 0) + harvested
                agent.energy = min(1.0, agent.energy + harvested * 0.5)
                result["success"] = True
                result["harvested"] = harvested
                result["resource_type"] = rtype

        elif action_type == "interact":
            other_id = action.get("target_agent")
            if other_id and other_id in self._agents:
                result["success"] = True
                result["interaction"] = "social"

        elif action_type == "wait":
            result["success"] = True

        # Energy decay only on idle (not on productive actions)
        if action_type == "wait":
            agent.energy = max(0, agent.energy - self.config.energy_decay_rate)

        # Death check
        if agent.energy <= 0:
            agent.alive = False
            self._events.append(WorldEvent(
                event_type=EventType.NONE,
                tick=self._tick_count,
                agent_id=agent_id,
                message=f"Agent {agent_id} died (energy depletion)",
            ))

        return result

    def tick(self) -> list[WorldEvent]:
        """Advance the world by one tick.

        Returns events that happened this tick.
        """
        self._tick_count += 1
        events = []

        # Resource regeneration
        for res in self._resources:
            res.tick()

        # Environmental events
        if np.random.random() < self.config.event_probability:
            event = self._generate_event()
            if event:
                events.append(event)
                self._events.append(event)
                self._event_log.append(event)

        # Storm events
        if (self._tick_count % self.config.storm_interval == 0 and
            self._tick_count > 0):
            storm = WorldEvent(
                event_type=EventType.STORM,
                tick=self._tick_count,
                message="Environmental storm",
            )
            events.append(storm)
            self._events.append(storm)
            self._event_log.append(storm)

            # Storm effects on agents
            for agent in self._agents.values():
                if agent.alive:
                    agent.energy = max(0, agent.energy -
                                       self.config.storm_energy_drain)

        # Maintain event buffer
        if len(self._events) > 100:
            self._events = self._events[-50:]

        return events

    @property
    def agent_ids(self) -> list[int]:
        return [aid for aid, a in self._agents.items() if a.alive]

    @property
    def alive_agents(self) -> list[AgentState]:
        return [a for a in self._agents.values() if a.alive]

    @property
    def current_tick(self) -> int:
        return self._tick_count

    def stats(self) -> dict:
        """World statistics."""
        agents = list(self._agents.values())
        alive = [a for a in agents if a.alive]
        resources = [r for r in self._resources if not r.depleted]
        return {
            "tick": self._tick_count,
            "n_agents": len(agents),
            "n_alive": len(alive),
            "n_resources": len(resources),
            "total_energy": sum(a.energy for a in alive),
            "total_inventory": sum(
                sum(a.inventory.values()) for a in alive),
            "avg_energy": np.mean([a.energy for a in alive]) if alive else 0,
            "n_events": len(self._event_log),
        }

    def _spawn_initial_resources(self) -> None:
        """Spawn initial resources."""
        for _ in range(self.config.n_resources):
            self._spawn_resource()

    def _spawn_resource(self, position: Optional[np.ndarray] = None) -> Resource:
        """Spawn a new resource."""
        if position is None:
            position = np.random.uniform(0, self.config.grid_size, 2)

        rtype = np.random.choice(list(ResourceType))
        res = Resource(
            resource_id=self._resource_id_counter,
            resource_type=rtype,
            position=np.asarray(position),
            quantity=0.5 + np.random.random() * 0.5,
            regen_rate=self.config.resource_regen_rate,
        )
        self._resource_id_counter += 1
        self._resources.append(res)
        return res

    def _generate_event(self) -> Optional[WorldEvent]:
        """Generate a random environmental event."""
        etype = np.random.choice([
            EventType.RESOURCE_SPAWN,
            EventType.DISCOVERY,
        ])

        if etype == EventType.RESOURCE_SPAWN:
            res = self._spawn_resource()
            return WorldEvent(
                event_type=etype,
                tick=self._tick_count,
                position=res.position,
                message=f"New {res.resource_type.value} discovered",
            )
        elif etype == EventType.DISCOVERY:
            return WorldEvent(
                event_type=etype,
                tick=self._tick_count,
                position=np.random.uniform(0, self.config.grid_size, 2),
                message="Interesting discovery",
            )

        return None
