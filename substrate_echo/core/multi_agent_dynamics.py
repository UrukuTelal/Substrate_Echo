"""S10: Multi-Agent Dynamics — social field, reputation, communication topology.

Models the social dynamics between multiple cognitive agents:
- Social field: agents influence each other through 16D state space
- Reputation: track agent contributions and trust over time
- Communication topology: who talks to whom, information flow
- Collective intelligence: emergent group behavior

References:
- PLAN.md Phase S10: Multi-Agent Dynamics
- Cognitive agent ecology from cognitive_agents.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import math
import time


@dataclass
class SocialConfig:
    """Configuration for multi-agent dynamics."""
    # Social influence
    influence_strength: float = 0.1   # how much agents affect each other
    influence_range: float = 0.5      # max distance for influence
    
    # Reputation
    initial_reputation: float = 0.5
    reputation_decay: float = 0.001   # per tick
    reputation_bonus: float = 0.02    # for successful actions
    reputation_penalty: float = 0.05  # for failed actions
    
    # Communication
    max_communication_range: float = 1.0
    message_decay: float = 0.1        # message relevance decay
    
    # Collective
    consensus_threshold: float = 0.7  # for group decisions
    diversity_bonus: float = 0.1     # reward for diverse perspectives


@dataclass
class AgentProfile:
    """Profile for an agent in the social field."""
    agent_id: str
    role: str
    state: np.ndarray                 # 16D pillar state
    reputation: float = 0.5
    trust_scores: dict[str, float] = field(default_factory=dict)
    communication_partners: list[str] = field(default_factory=list)
    successful_actions: int = 0
    failed_actions: int = 0
    created_at: float = field(default_factory=time.time)
    
    @property
    def success_rate(self) -> float:
        total = self.successful_actions + self.failed_actions
        if total == 0:
            return 0.5
        return self.successful_actions / total
    
    def update_reputation(self, success: bool, config: SocialConfig) -> None:
        """Update reputation based on action outcome."""
        if success:
            self.successful_actions += 1
            self.reputation = min(1.0, self.reputation + config.reputation_bonus)
        else:
            self.failed_actions += 1
            self.reputation = max(0.0, self.reputation - config.reputation_penalty)


class SocialField:
    """S10 Multi-Agent Dynamics.
    
    Models social interactions between cognitive agents as a
    field theory in 16D pillar space.
    
    Key concepts:
    - Social field: the combined influence of all agents on each other
    - Reputation: trust metric that affects influence strength
    - Communication topology: network of agent interactions
    - Collective intelligence: emergent group behavior
    """
    
    def __init__(self, config: Optional[SocialConfig] = None):
        self.config = config or SocialConfig()
        
        # Agent profiles
        self._agents: dict[str, AgentProfile] = {}
        
        # Social field state (16D)
        self._social_field = np.zeros(16, dtype=np.float64)
        
        # Communication history
        self._message_log: list[dict] = []
        
        # Collective metrics
        self._collective_coherence: float = 0.0
        self._collective_diversity: float = 0.0
    
    # ── Agent Management ──────────────────────────────────────────
    
    def add_agent(self, agent_id: str, role: str,
                  initial_state: Optional[np.ndarray] = None) -> AgentProfile:
        """Add an agent to the social field."""
        if initial_state is None:
            initial_state = np.full(16, 0.5)
        
        profile = AgentProfile(
            agent_id=agent_id,
            role=role,
            state=initial_state.copy(),
            reputation=self.config.initial_reputation,
        )
        
        self._agents[agent_id] = profile
        return profile
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the social field."""
        self._agents.pop(agent_id, None)
    
    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """Get an agent's profile."""
        return self._agents.get(agent_id)
    
    def get_all_agents(self) -> dict[str, AgentProfile]:
        """Get all agent profiles."""
        return dict(self._agents)
    
    # ── Social Influence ──────────────────────────────────────────
    
    def compute_social_influence(self, agent_id: str) -> np.ndarray:
        """Compute the social influence on an agent from all others.
        
        F_social = Σ_j w_j * (ψ_j - ψ_i)
        
        Where w_j = reputation_j * exp(-distance/range)
        """
        if agent_id not in self._agents:
            return np.zeros(16)
        
        target = self._agents[agent_id]
        influence = np.zeros(16)
        
        for other_id, other in self._agents.items():
            if other_id == agent_id:
                continue
            
            # Distance in state space
            dist = np.linalg.norm(target.state - other.state)
            
            if dist > self.config.influence_range:
                continue
            
            # Influence weight: reputation * distance decay
            weight = other.reputation * math.exp(-dist / self.config.influence_range)
            
            # Social force: pull toward other's state
            force = weight * (other.state - target.state)
            influence += force
        
        return self.config.influence_strength * influence
    
    def compute_collective_field(self) -> np.ndarray:
        """Compute the collective social field (weighted average of all agents)."""
        if not self._agents:
            return np.zeros(16)
        
        total_weight = 0.0
        weighted_sum = np.zeros(16)
        
        for agent in self._agents.values():
            weight = agent.reputation
            weighted_sum += weight * agent.state
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        
        return np.zeros(16)
    
    # ── Reputation System ─────────────────────────────────────────
    
    def record_action(self, agent_id: str, success: bool) -> None:
        """Record an action outcome for reputation update."""
        if agent_id in self._agents:
            self._agents[agent_id].update_reputation(success, self.config)
    
    def get_trust_matrix(self) -> dict[str, dict[str, float]]:
        """Compute pairwise trust between agents."""
        trust = {}
        
        for a_id, a in self._agents.items():
            trust[a_id] = {}
            for b_id, b in self._agents.items():
                if a_id == b_id:
                    trust[a_id][b_id] = 1.0
                else:
                    # Trust based on reputation and state similarity
                    state_sim = 1.0 - np.linalg.norm(a.state - b.state) / math.sqrt(16)
                    trust[a_id][b_id] = (
                        0.5 * a.reputation +
                        0.5 * state_sim
                    )
        
        return trust
    
    # ── Communication Topology ────────────────────────────────────
    
    def can_communicate(self, agent_a: str, agent_b: str) -> bool:
        """Check if two agents can communicate."""
        a = self._agents.get(agent_a)
        b = self._agents.get(agent_b)
        
        if a is None or b is None:
            return False
        
        dist = np.linalg.norm(a.state - b.state)
        return dist <= self.config.max_communication_range
    
    def send_message(self, sender_id: str, receiver_id: str,
                     content: str, data: Optional[np.ndarray] = None) -> bool:
        """Send a message between agents."""
        if not self.can_communicate(sender_id, receiver_id):
            return False
        
        message = {
            "sender": sender_id,
            "receiver": receiver_id,
            "content": content,
            "data": data,
            "timestamp": time.time(),
        }
        
        self._message_log.append(message)
        
        # Update communication partners
        sender = self._agents.get(sender_id)
        if sender and receiver_id not in sender.communication_partners:
            sender.communication_partners.append(receiver_id)
        
        return True
    
    def get_communication_topology(self) -> dict[str, list[str]]:
        """Get the communication network (adjacency list)."""
        topology = {aid: [] for aid in self._agents}
        
        for msg in self._message_log:
            sender = msg["sender"]
            receiver = msg["receiver"]
            if sender in topology and receiver not in topology[sender]:
                topology[sender].append(receiver)
        
        return topology
    
    # ── Collective Intelligence ───────────────────────────────────
    
    def compute_collective_coherence(self) -> float:
        """How aligned are all agents (0-1)."""
        if len(self._agents) < 2:
            return 1.0
        
        states = np.array([a.state for a in self._agents.values()])
        mean_state = np.mean(states, axis=0)
        
        # Average distance from mean
        distances = np.linalg.norm(states - mean_state, axis=1)
        avg_distance = np.mean(distances)
        
        # Normalize: 0 = very spread, 1 = perfectly aligned
        return max(0.0, 1.0 - avg_distance)
    
    def compute_collective_diversity(self) -> float:
        """How diverse are the agents' perspectives (0-1)."""
        if len(self._agents) < 2:
            return 0.0
        
        states = np.array([a.state for a in self._agents.values()])
        
        # Compute pairwise distances
        n = len(states)
        total_dist = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += np.linalg.norm(states[i] - states[j])
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_dist = total_dist / count
        
        # Normalize: 0 = identical, 1 = maximally diverse
        return min(1.0, avg_dist / math.sqrt(16))
    
    def collective_decision(self, proposals: dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Make a collective decision using weighted consensus.
        
        Args:
            proposals: dict mapping agent_id to proposed state
        
        Returns:
            Weighted average proposal (or None if no proposals)
        """
        if not proposals:
            return None
        
        weighted_sum = np.zeros(16)
        total_weight = 0.0
        
        for agent_id, proposal in proposals.items():
            agent = self._agents.get(agent_id)
            if agent is None:
                continue
            
            weight = agent.reputation
            weighted_sum += weight * proposal
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        
        return None
    
    # ── Social Field Evolution ────────────────────────────────────
    
    def tick(self, dt: float = 1.0) -> dict:
        """Evolve the social field for one time step.
        
        1. Compute social influences
        2. Update agent states
        3. Decay reputation
        4. Update collective metrics
        """
        # Compute influences
        influences = {}
        for agent_id in self._agents:
            influences[agent_id] = self.compute_social_influence(agent_id)
        
        # Update states
        for agent_id, agent in self._agents.items():
            influence = influences.get(agent_id, np.zeros(16))
            agent.state = agent.state + dt * influence
            agent.state = np.clip(agent.state, 0.0, 1.0)
        
        # Decay reputation
        for agent in self._agents.values():
            agent.reputation = max(
                0.0,
                agent.reputation - self.config.reputation_decay * dt
            )
        
        # Update collective metrics
        self._collective_coherence = self.compute_collective_coherence()
        self._collective_diversity = self.compute_collective_diversity()
        
        # Update social field
        self._social_field = self.compute_collective_field()
        
        return {
            "coherence": self._collective_coherence,
            "diversity": self._collective_diversity,
            "agent_count": len(self._agents),
            "messages": len(self._message_log),
        }
    
    # ── Statistics ─────────────────────────────────────────────────
    
    def stats(self) -> dict:
        """Get social field statistics."""
        if not self._agents:
            return {
                "agent_count": 0,
                "coherence": 0.0,
                "diversity": 0.0,
            }
        
        reputations = [a.reputation for a in self._agents.values()]
        
        return {
            "agent_count": len(self._agents),
            "avg_reputation": sum(reputations) / len(reputations),
            "min_reputation": min(reputations),
            "max_reputation": max(reputations),
            "coherence": self._collective_coherence,
            "diversity": self._collective_diversity,
            "total_messages": len(self._message_log),
            "avg_success_rate": np.mean([
                a.success_rate for a in self._agents.values()
            ]),
        }
    
    def reset(self) -> None:
        """Reset all state."""
        self._agents.clear()
        self._social_field = np.zeros(16)
        self._message_log.clear()
        self._collective_coherence = 0.0
        self._collective_diversity = 0.0
