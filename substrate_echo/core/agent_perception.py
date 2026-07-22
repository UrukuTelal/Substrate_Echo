"""Agent-to-Agent Perception — P6.13

Full agent state reading from raw observations.

The Engine's perceive_agents() only returns nearby agent IDs.
AgentInfo carries rich data: position, velocity, 16D PSV,
shadow_state, active flag. This module reads ALL of it.

For each observed agent, the perception layer produces:
- Position and velocity (raw)
- Distance and direction from observer
- Relative velocity (approaching/retreating)
- Pillar similarity (how similar is their state to mine?)
- Shadow divergence (how much do they differ from their baseline?)
- Threat assessment (based on distance, speed, pillars)
- Social signals (based on velocity direction and pillar state)

Usage:
    perception = AgentPerception(view_distance=50.0)
    
    # Each tick, provide raw agent observations
    results = perception.process(
        observer_position=my_pos,
        observer_pillars=my_pillars,
        raw_agents=[
            {"id": 1, "position": [5,0,0], "velocity": [1,0,0],
             "pillars": [...], "shadow_state": [...], "active": True},
        ],
    )
    
    for agent in results:
        print(f"Agent {agent.agent_id}: dist={agent.distance:.1f}, "
              f"threat={agent.threat_level:.2f}")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class AgentObservation:
    """Raw observation of another agent (from AgentInfo)."""
    agent_id: int
    position: np.ndarray
    velocity: np.ndarray
    pillars: np.ndarray           # 16D PSV
    shadow_state: np.ndarray      # 16D shadow
    active: bool = True
    properties: dict = field(default_factory=dict)


@dataclass
class AgentPerceptionResult:
    """Processed perception of a single agent."""
    agent_id: int
    
    # Raw data
    position: np.ndarray
    velocity: np.ndarray
    pillars: np.ndarray
    shadow_state: np.ndarray
    
    # Derived: spatial
    distance: float               # distance from observer
    direction: np.ndarray         # unit vector from observer to agent
    relative_velocity: float      # positive = approaching, negative = retreating
    
    # Derived: similarity
    pillar_similarity: float      # cosine similarity of PSV (0-1)
    shadow_divergence: float      # how much agent differs from its baseline
    pillar_dominance: int         # which pillar is strongest
    pillar_weakest: int           # which pillar is weakest
    
    # Derived: assessment
    threat_level: float           # 0-1, based on distance, speed, pillars
    social_signal: float          # 0-1, how much this agent seems socially engaged
    
    # Metadata
    is_new: bool = False          # first time seeing this agent
    frames_tracked: int = 0


@dataclass
class AgentPerceptionConfig:
    """Configuration for agent perception."""
    view_distance: float = 50.0
    threat_speed_factor: float = 0.3      # how much speed contributes to threat
    threat_proximity_factor: float = 0.4  # how much proximity contributes
    threat_pillar_factor: float = 0.3     # how much pillar state contributes
    social_warmth_threshold: float = 0.6  # warmth pillar above this = social
    social_relation_threshold: float = 0.5
    similarity_floor: float = 0.0         # minimum pillar similarity


PILLAR_NAMES = [
    "Awareness", "Willpower", "Force", "Influence",
    "Resistance", "Integrity", "Cohesion", "Relation",
    "Presence", "Warmth", "Memory", "Attraction",
    "Harm", "Distortion", "Flux", "Depth",
]


class AgentPerception:
    """Full agent-to-agent perception layer.
    
    Reads all fields from AgentInfo (not just agent count) and
    produces rich per-agent perception results including spatial
    relationships, pillar similarity, threat assessment, and
    social signals.
    
    This replaces the Engine's perceive_agents() which only returns
    nearby agent IDs. The full AgentInfo data exists — this module
    does the translation.
    
    Usage:
        perception = AgentPerception()
        results = perception.process(
            observer_position=np.array([0,0,0]),
            observer_pillars=np.zeros(16),
            raw_agents=agent_info_list,
        )
    """
    
    def __init__(self, config: Optional[AgentPerceptionConfig] = None):
        self.config = config or AgentPerceptionConfig()
        self._seen_agents: dict[int, int] = {}  # agent_id → frame count
    
    def process(self, observer_position: np.ndarray,
                observer_pillars: np.ndarray,
                raw_agents: list[dict]) -> list[AgentPerceptionResult]:
        """Process raw agent observations into rich perception results.
        
        Args:
            observer_position: [x, y, z] of the observing agent
            observer_pillars: 16D PSV of the observing agent
            raw_agents: list of dicts matching AgentInfo fields:
                id, position, velocity, pillars, shadow_state, active
        
        Returns:
            List of AgentPerceptionResult for each active, in-range agent
        """
        observer_pos = np.asarray(observer_position, dtype=np.float64)
        observer_psv = np.asarray(observer_pillars, dtype=np.float64)
        
        results = []
        
        for raw in raw_agents:
            agent_id = raw["id"]
            active = raw.get("active", True)
            if not active:
                continue
            
            pos = np.asarray(raw["position"], dtype=np.float64)
            vel = np.asarray(raw["velocity"], dtype=np.float64)
            psv = np.asarray(raw["pillars"], dtype=np.float64)
            shadow = np.asarray(raw["shadow_state"], dtype=np.float64)
            
            # Distance
            diff = pos - observer_pos
            distance = float(np.linalg.norm(diff))
            
            if distance > self.config.view_distance:
                continue
            
            # Direction
            if distance > 1e-6:
                direction = diff / distance
            else:
                direction = np.zeros(3)
            
            # Relative velocity (positive = approaching)
            if distance > 1e-6:
                rel_vel = float(np.dot(vel, direction))
            else:
                rel_vel = 0.0
            
            # Pillar similarity (cosine)
            psv_norm = np.linalg.norm(psv)
            obs_norm = np.linalg.norm(observer_psv)
            if psv_norm > 1e-6 and obs_norm > 1e-6:
                pillar_sim = float(np.dot(psv, observer_psv) / (psv_norm * obs_norm))
                pillar_sim = max(0.0, (pillar_sim + 1) / 2)  # map [-1,1] → [0,1]
            else:
                pillar_sim = 0.5
            
            # Shadow divergence
            shadow_diff = np.linalg.norm(psv - shadow)
            shadow_div = min(1.0, shadow_diff / np.sqrt(16))  # normalize by dim
            
            # Dominant/weakest pillars
            pillar_dom = int(np.argmax(psv))
            pillar_weak = int(np.argmin(psv))
            
            # Threat assessment
            threat = self._assess_threat(distance, vel, psv)
            
            # Social signal
            social = self._assess_social(psv)
            
            # Track frames
            is_new = agent_id not in self._seen_agents
            self._seen_agents[agent_id] = self._seen_agents.get(agent_id, 0) + 1
            
            results.append(AgentPerceptionResult(
                agent_id=agent_id,
                position=pos,
                velocity=vel,
                pillars=psv,
                shadow_state=shadow,
                distance=distance,
                direction=direction,
                relative_velocity=rel_vel,
                pillar_similarity=pillar_sim,
                shadow_divergence=shadow_div,
                pillar_dominance=pillar_dom,
                pillar_weakest=pillar_weak,
                threat_level=threat,
                social_signal=social,
                is_new=is_new,
                frames_tracked=self._seen_agents[agent_id],
            ))
        
        return results
    
    def _assess_threat(self, distance: float, velocity: np.ndarray,
                       pillars: np.ndarray) -> float:
        """Assess threat level from proximity, speed, and pillar state."""
        cfg = self.config
        
        # Proximity component (closer = more threat)
        if distance < 1.0:
            prox = 1.0
        elif distance > cfg.view_distance * 0.5:
            prox = 0.0
        else:
            prox = 1.0 - (distance / (cfg.view_distance * 0.5))
        
        # Speed component (faster = more threat)
        speed = float(np.linalg.norm(velocity))
        speed_threat = min(1.0, speed * cfg.threat_speed_factor)
        
        # Pillar component: high Force + high Harm = threatening
        if len(pillars) >= 16:
            force_pillar = pillars[2] / 100.0 if pillars[2] > 0 else 0.0
            harm_pillar = pillars[12] / 100.0 if pillars[12] > 0 else 0.0
            pillar_threat = (force_pillar + harm_pillar) / 2.0
        else:
            pillar_threat = 0.0
        
        return (
            cfg.threat_proximity_factor * prox +
            cfg.threat_speed_factor * speed_threat +
            cfg.threat_pillar_factor * pillar_threat
        )
    
    def _assess_social(self, pillars: np.ndarray) -> float:
        """Assess social engagement from pillar state."""
        if len(pillars) < 16:
            return 0.0
        
        warmth = pillars[9] / 100.0 if pillars[9] > 0 else 0.0
        relation = pillars[7] / 100.0 if pillars[7] > 0 else 0.0
        presence = pillars[8] / 100.0 if pillars[8] > 0 else 0.0
        
        return min(1.0, (warmth + relation + presence) / 3.0)
    
    def get_agent_summary(self, agent_id: int) -> Optional[dict]:
        """Get tracking summary for a specific agent."""
        if agent_id not in self._seen_agents:
            return None
        return {
            "agent_id": agent_id,
            "frames_tracked": self._seen_agents[agent_id],
        }
    
    def reset(self) -> None:
        self._seen_agents.clear()
