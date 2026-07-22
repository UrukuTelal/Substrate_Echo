"""Cognitive Agent Ecology — specialized agents with deliberation and communication.

Instead of a single monolithic AI, Substrate_Echo uses specialized cognitive
agents that dynamically activate, negotiate through deliberation, and
communicate via message passing. Each agent has pillar affinity and
specializes in different aspects of cognition.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable
import numpy as np
import time


class AgentRole(Enum):
    """Roles for cognitive agents."""
    PERCEPTION = auto()
    MEMORY = auto()
    PLANNING = auto()
    CREATIVITY = auto()
    ENVIRONMENT = auto()
    ROOT = auto()  # orchestrator


class MessageType(Enum):
    """Types of inter-agent messages."""
    REQUEST_INFO = auto()      # ask another agent for data
    PROPOSE_ACTION = auto()    # suggest an action to the group
    SUPPORT = auto()           # agree with another agent's proposal
    OBJECT = auto()            # disagree with another agent's proposal
    SHARE_STATE = auto()       # broadcast own state to others


@dataclass
class AgentMessage:
    """A message between cognitive agents."""
    sender: AgentRole
    receiver: Optional[AgentRole]  # None = broadcast
    message_type: MessageType
    content: str = ""
    data: Optional[np.ndarray] = None
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    """Response from a cognitive agent."""
    agent_role: AgentRole
    confidence: float  # 0-1
    proposed_state_change: Optional[np.ndarray] = None  # suggested PSV update
    proposed_action: Optional[str] = None
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)
    messages_sent: list[AgentMessage] = field(default_factory=list)


class CognitiveAgent:
    """Base class for specialized cognitive processes.
    
    Each agent:
    - Has affinity for specific pillars
    - Activates when context matches its specialty
    - Produces responses that feed into the root agent's decision
    - Can send and receive messages from other agents
    - Consumes energy from a shared resource pool
    """
    
    def __init__(self, role: AgentRole, pillar_affinity: list[int],
                 activation_threshold: float = 0.3,
                 energy_cost_per_tick: float = 0.01):
        self.role = role
        self.pillar_affinity = pillar_affinity
        self.activation_threshold = activation_threshold
        self.energy_cost_per_tick = energy_cost_per_tick
        self.is_active = False
        self.state = np.zeros(16, dtype=np.float64)
        self.last_activation = 0.0
        self.activation_count = 0
        
        # Communication
        self.inbox: list[AgentMessage] = []
        self.outbox: list[AgentMessage] = []
        
        # Adaptation
        self._activation_history: list[bool] = []
        self._adaptation_window: int = 20
    
    def should_activate(self, context_state: np.ndarray,
                        energy_available: float = 1.0) -> bool:
        """Determine if this agent should activate.
        
        Dynamic threshold: adapts based on recent activation pattern.
        If an agent has been inactive for a while, lower its threshold.
        """
        affinity_values = context_state[self.pillar_affinity]
        deviation = np.mean(np.abs(affinity_values - 0.5))
        
        # Dynamic threshold adjustment
        if len(self._activation_history) >= self._adaptation_window:
            recent_activation_rate = sum(self._activation_history[-self._adaptation_window:]) / self._adaptation_window
            if recent_activation_rate < 0.1:
                self.activation_threshold *= 0.8
            elif recent_activation_rate > 0.8:
                self.activation_threshold *= 1.2
        
        # Energy gate
        if energy_available < self.energy_cost_per_tick:
            return False
        
        return deviation > self.activation_threshold
    
    def activate(self) -> None:
        self.is_active = True
        self.last_activation = time.time()
        self.activation_count += 1
        self._activation_history.append(True)
        if len(self._activation_history) > self._adaptation_window:
            self._activation_history.pop(0)
    
    def deactivate(self) -> None:
        self.is_active = False
        self._activation_history.append(False)
        if len(self._activation_history) > self._adaptation_window:
            self._activation_history.pop(0)
    
    def send_message(self, receiver: Optional[AgentRole], msg_type: MessageType,
                     content: str = "", data: Optional[np.ndarray] = None,
                     confidence: float = 0.0) -> AgentMessage:
        """Create and queue a message for another agent or broadcast."""
        msg = AgentMessage(
            sender=self.role,
            receiver=receiver,
            message_type=msg_type,
            content=content,
            data=data,
            confidence=confidence,
        )
        self.outbox.append(msg)
        return msg
    
    def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent."""
        self.inbox.append(message)
    
    def process_inbox(self) -> list[AgentMessage]:
        """Process received messages and generate responses."""
        responses = []
        for msg in self.inbox:
            resp = self._handle_message(msg)
            if resp is not None:
                responses.append(resp)
        self.inbox.clear()
        return responses
    
    def _handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle an incoming message. Override in subclasses."""
        return None
    
    def evaluate(self, context_state: np.ndarray,
                 world_model=None, memory=None) -> AgentResponse:
        """Evaluate context and produce response."""
        raise NotImplementedError
    
    def energy_cost(self) -> float:
        """Energy consumed this tick if active."""
        return self.energy_cost_per_tick if self.is_active else 0.0
    
    def _pillar_summary(self, state: np.ndarray) -> dict:
        names = ["Awareness", "Willpower", "Force", "Influence",
                 "Resistance", "Integrity", "Cohesion", "Relation",
                 "Presence", "Warmth", "Memory", "Attraction",
                 "Harm", "Distortion", "Flux", "Depth"]
        return {names[i]: round(float(state[i]), 3) for i in self.pillar_affinity}


class PerceptionAgent(CognitiveAgent):
    """Processes raw sensor data into world model updates.
    
    Affinity: Awareness(0), Presence(8)
    Role: Transform raw perception into structured world objects.
    """
    
    def __init__(self):
        super().__init__(
            role=AgentRole.PERCEPTION,
            pillar_affinity=[0, 8],
            activation_threshold=0.2,
            energy_cost_per_tick=0.005,
        )
    
    def evaluate(self, context_state, world_model=None, memory=None) -> AgentResponse:
        if world_model is None:
            return AgentResponse(self.role, confidence=0.0, reasoning="No world model")
        
        awareness = float(context_state[0])
        presence = float(context_state[8])
        
        return AgentResponse(
            agent_role=self.role,
            confidence=min(awareness, presence),
            reasoning=f"Perception active: Awareness={awareness:.2f}, Presence={presence:.2f}",
            metadata={"objects_in_world": len(world_model.objects) if world_model else 0},
        )
    
    def _handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.REQUEST_INFO:
            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.SHARE_STATE,
                content="perception_state",
                confidence=0.8,
            )
        return None


class MemoryAgent(CognitiveAgent):
    """Manages attractor memory encoding and recall.
    
    Affinity: Memory(10), Depth(15)
    Role: Encode experiences, recall memories, consolidate.
    """
    
    def __init__(self):
        super().__init__(
            role=AgentRole.MEMORY,
            pillar_affinity=[10, 15],
            activation_threshold=0.2,
            energy_cost_per_tick=0.008,
        )
    
    def evaluate(self, context_state, world_model=None, memory=None) -> AgentResponse:
        if memory is None:
            return AgentResponse(self.role, confidence=0.0, reasoning="No memory system")
        
        memory_pillar = float(context_state[10])
        depth_pillar = float(context_state[15])
        
        stats = memory.memory_stats()
        
        return AgentResponse(
            agent_role=self.role,
            confidence=min(memory_pillar, depth_pillar),
            reasoning=f"Memory active: {stats['active_memories']} memories, "
                     f"avg strength={stats['avg_strength']:.2f}",
            metadata=stats,
        )
    
    def _handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.REQUEST_INFO:
            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.SHARE_STATE,
                content="memory_state",
                confidence=0.7,
            )
        return None


class PlanningAgent(CognitiveAgent):
    """Generates action plans from motivation drives.
    
    Affinity: Willpower(1), Force(2), Influence(3)
    Role: Translate goals into actionable steps.
    """
    
    def __init__(self):
        super().__init__(
            role=AgentRole.PLANNING,
            pillar_affinity=[1, 2, 3],
            activation_threshold=0.25,
            energy_cost_per_tick=0.012,
        )
    
    def evaluate(self, context_state, world_model=None, memory=None) -> AgentResponse:
        willpower = float(context_state[1])
        force = float(context_state[2])
        influence = float(context_state[3])
        
        drive = (willpower + force) / 2.0
        
        return AgentResponse(
            agent_role=self.role,
            confidence=drive,
            reasoning=f"Planning active: Willpower={willpower:.2f}, "
                     f"Force={force:.2f}, Influence={influence:.2f}",
            proposed_action="plan_step",
        )
    
    def _handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.PROPOSE_ACTION:
            # Planning agent supports or objects to proposed actions
            if message.confidence > 0.5:
                return self.send_message(
                    receiver=message.sender,
                    msg_type=MessageType.SUPPORT,
                    content=message.content,
                    confidence=message.confidence,
                )
        return None


class CreativityAgent(CognitiveAgent):
    """Generates novel patterns by combining attractors.
    
    Affinity: Distortion(13), Flux(14), Depth(15)
    Role: Combine existing memories in novel ways.
    """
    
    def __init__(self):
        super().__init__(
            role=AgentRole.CREATIVITY,
            pillar_affinity=[13, 14, 15],
            activation_threshold=0.3,
            energy_cost_per_tick=0.015,
        )
    
    def evaluate(self, context_state, world_model=None, memory=None) -> AgentResponse:
        distortion = float(context_state[13])
        flux = float(context_state[14])
        depth = float(context_state[15])
        
        creativity_level = (distortion + flux + depth) / 3.0
        
        return AgentResponse(
            agent_role=self.role,
            confidence=creativity_level,
            reasoning=f"Creativity active: Distortion={distortion:.2f}, "
                     f"Flux={flux:.2f}, Depth={depth:.2f}",
        )


class EnvironmentAgent(CognitiveAgent):
    """Monitors external world state and detects changes.
    
    Affinity: Relation(7), Resistance(4), Awareness(0)
    Role: Track world changes, maintain spatial awareness.
    """
    
    def __init__(self):
        super().__init__(
            role=AgentRole.ENVIRONMENT,
            pillar_affinity=[7, 4, 0],
            activation_threshold=0.2,
            energy_cost_per_tick=0.006,
        )
    
    def evaluate(self, context_state, world_model=None, memory=None) -> AgentResponse:
        relation = float(context_state[7])
        resistance = float(context_state[4])
        awareness = float(context_state[0])
        
        vigilance = (relation + resistance + awareness) / 3.0
        
        return AgentResponse(
            agent_role=self.role,
            confidence=vigilance,
            reasoning=f"Environment active: Relation={relation:.2f}, "
                     f"Resistance={resistance:.2f}, Awareness={awareness:.2f}",
            metadata={"world_objects": len(world_model.objects) if world_model else 0},
        )


class AgentEcology:
    """Manages the multi-agent cognitive system with deliberation.
    
    Orchestrates agent activation, inter-agent communication,
    deliberation (negotiation), energy management, and consensus.
    """
    
    def __init__(self, max_energy: float = 1.0, energy_regen_rate: float = 0.1):
        self.agents: dict[AgentRole, CognitiveAgent] = {
            AgentRole.PERCEPTION: PerceptionAgent(),
            AgentRole.MEMORY: MemoryAgent(),
            AgentRole.PLANNING: PlanningAgent(),
            AgentRole.CREATIVITY: CreativityAgent(),
            AgentRole.ENVIRONMENT: EnvironmentAgent(),
        }
        self.active_agents: list[AgentRole] = []
        self.response_history: list[AgentResponse] = []
        
        # Energy management
        self.energy_pool = max_energy
        self.max_energy = max_energy
        self.energy_regen_rate = energy_regen_rate
        
        # Deliberation
        self._deliberation_rounds: int = 2
        self._message_log: list[AgentMessage] = []
    
    def tick(self, context_state: np.ndarray,
             world_model=None, memory=None) -> list[AgentResponse]:
        """Run one cognitive cycle with deliberation.
        
        1. Regenerate energy
        2. Check which agents should activate (energy-aware)
        3. Agents evaluate and propose
        4. Deliberation: agents communicate, support/object
        5. Collect final responses
        """
        # Regenerate energy
        self.energy_pool = min(self.max_energy, self.energy_pool + self.energy_regen_rate)
        
        # Deactivate all
        for agent in self.agents.values():
            agent.deactivate()
        self.active_agents.clear()
        
        # Activate relevant agents (energy-aware)
        responses = []
        for role, agent in self.agents.items():
            if agent.should_activate(context_state, self.energy_pool):
                agent.activate()
                self.active_agents.append(role)
                self.energy_pool -= agent.energy_cost_per_tick
                response = agent.evaluate(context_state, world_model, memory)
                responses.append(response)
        
        # Deliberation: agents communicate
        if len(responses) > 1:
            responses = self._deliberate(responses, context_state)
        
        self.response_history.extend(responses)
        return responses
    
    def _deliberate(self, initial_responses: list[AgentResponse],
                    context_state: np.ndarray) -> list[AgentResponse]:
        """Multi-round deliberation where agents communicate.
        
        Round 1: Each active agent shares its proposal with the group.
        Round 2: Agents can support or object to proposals.
        Final: Consensus weighted by confidence and support.
        """
        active_roles = [r.agent_role for r in initial_responses]
        
        # Round 1: Share proposals
        for response in initial_responses:
            agent = self.agents[response.agent_role]
            if response.proposed_action:
                msg = agent.send_message(
                    receiver=None,  # broadcast
                    msg_type=MessageType.PROPOSE_ACTION,
                    content=response.proposed_action,
                    confidence=response.confidence,
                )
                self._message_log.append(msg)
        
        # Deliver messages
        for role, agent in self.agents.items():
            if role in active_roles:
                for other_role, other_agent in self.agents.items():
                    if other_role in active_roles and other_role != role:
                        for msg in other_agent.outbox:
                            if msg.receiver is None or msg.receiver == role:
                                agent.receive_message(msg)
        
        # Round 2: Process messages, generate support/object
        for role, agent in self.agents.items():
            if role in active_roles:
                agent.process_inbox()
        
        # Collect all messages for consensus weighting
        all_messages = []
        for agent in self.agents.values():
            all_messages.extend(agent.outbox)
            agent.outbox.clear()
        
        return initial_responses
    
    def get_consensus(self, responses: list[AgentResponse]) -> Optional[AgentResponse]:
        """Weighted consensus: confidence × support from other agents."""
        if not responses:
            return None
        
        # Simple weighted consensus: highest confidence with support bonus
        scored = []
        for r in responses:
            support_count = sum(
                1 for msg in self._message_log[-20:]
                if msg.message_type == MessageType.SUPPORT
                and msg.content == r.proposed_action
            )
            score = r.confidence + support_count * 0.1
            scored.append((score, r))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
    
    def get_active_roles(self) -> list[AgentRole]:
        return self.active_agents.copy()
    
    def get_agent(self, role: AgentRole) -> CognitiveAgent:
        return self.agents[role]
    
    def energy_status(self) -> dict:
        return {
            "available": round(self.energy_pool, 3),
            "max": self.max_energy,
            "utilization": round(1 - self.energy_pool / self.max_energy, 3),
        }
    
    def stats(self) -> dict:
        return {
            "total_agents": len(self.agents),
            "active_agents": len(self.active_agents),
            "total_responses": len(self.response_history),
            "energy": self.energy_status(),
            "deliberation_messages": len(self._message_log),
            "activation_counts": {
                role.name: agent.activation_count
                for role, agent in self.agents.items()
            },
        }
