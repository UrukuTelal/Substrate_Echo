"""Multi-Agent Goal Inference — P6.12

Infer cooperative goals and interactions between tracked agents.

GoalManager tracks individual agents. This module looks at
PAIRS of agents and infers their joint dynamics:

1. Convergence: agents moving toward each other → meeting
2. Following: one agent tracking another's path → following
3. Co-presence: agents at same location → social interaction
4. Joint goal: agents heading to same attractor → cooperation
5. Avoidance: agents diverging when nearby → avoidance

The inference uses simple geometric tests on velocity vectors
and positions — no learned model, just trajectory geometry.

Usage:
    from substrate_echo.core.goal_tracker import GoalManager
    
    gm = GoalManager()
    # ... update agents ...
    
    inference = MultiAgentGoalInference()
    interactions = inference.infer(gm)
    
    for pair in interactions:
        print(f"{pair.entity_a} ↔ {pair.entity_b}: {pair.interaction_type}")
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np

from substrate_echo.core.goal_tracker import GoalManager, GoalPhase


class InteractionType(Enum):
    """Types of pairwise interactions."""
    CONVERGING = auto()      # moving toward each other
    DIVERGING = auto()       # moving away from each other
    FOLLOWING = auto()       # one tracking the other
    CO_PRESENT = auto()      # at same location, not moving much
    JOINT_GOAL = auto()      # heading to same attractor
    AVOIDING = auto()        # actively avoiding each other
    NONE = auto()            # no interaction detected


@dataclass
class PairInteraction:
    """Inferred interaction between two agents."""
    entity_a: int
    entity_b: int
    interaction_type: InteractionType
    confidence: float               # 0-1
    convergence_rate: float         # negative = approaching, positive = departing
    distance: float                 # current distance
    shared_attractor: bool          # heading to same location
    
    def to_dict(self) -> dict:
        return {
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "interaction_type": self.interaction_type.name,
            "confidence": self.confidence,
            "convergence_rate": self.convergence_rate,
            "distance": self.distance,
            "shared_attractor": self.shared_attractor,
        }


@dataclass
class MultiAgentGoalInferenceConfig:
    """Configuration for multi-agent inference."""
    convergence_threshold: float = -0.05   # velocity toward each other (negative = approaching)
    divergence_threshold: float = 0.1      # velocity away from each other
    co_presence_distance: float = 1.5      # distance to consider "co-present"
    following_cos_threshold: float = 0.7   # cosine similarity for "following"
    joint_attractor_distance: float = 3.0  # how close attractors must be
    min_confidence: float = 0.3
    following_min_duration: int = 3        # frames before "following" is confirmed
    # Motion-based group detection
    group_proximity: float = 5.0           # max distance for "same group"
    group_velocity_cos: float = 0.5        # min cosine similarity of velocity
    group_speed_ratio: float = 0.5         # min speed ratio (slower/faster)


class MultiAgentGoalInference:
    """Infers pairwise interactions between tracked agents.
    
    Takes the current state of all agents from GoalManager and
    looks at each pair to determine what kind of interaction
    (if any) is happening.
    
    This is the Stage 4.5 component — it sits between individual
    goal tracking (Stage 4) and response policy (Stage 5).
    
    Usage:
        inference = MultiAgentGoalInference()
        
        # Each tick, after GoalManager.update():
        interactions = inference.infer(goal_manager)
        for pair in interactions:
            if pair.interaction_type == InteractionType.CONVERGING:
                print(f"Agents {pair.entity_a} and {pair.entity_b} are meeting")
    """
    
    def __init__(self, config: Optional[MultiAgentGoalInferenceConfig] = None):
        self.config = config or MultiAgentGoalInferenceConfig()
    
    def infer(self, goal_manager: GoalManager) -> list[PairInteraction]:
        """Infer interactions between all pairs of tracked agents.
        
        Args:
            goal_manager: GoalManager with current agent states
        
        Returns:
            List of PairInteraction for each interacting pair
        """
        states = goal_manager.get_all_states()
        agent_ids = list(states.keys())
        interactions = []
        
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                id_a, id_b = agent_ids[i], agent_ids[j]
                state_a, state_b = states[id_a], states[id_b]
                
                if state_a.position is None or state_b.position is None:
                    continue
                
                pair = self._infer_pair(id_a, state_a, id_b, state_b)
                if pair.interaction_type != InteractionType.NONE:
                    interactions.append(pair)
        
        return interactions
    
    def _infer_pair(self, id_a: int, state_a, id_b: int, state_b) -> PairInteraction:
        """Infer interaction type for a specific pair."""
        pos_a = state_a.position
        pos_b = state_b.position
        diff = pos_b - pos_a
        distance = float(np.linalg.norm(diff))
        
        # ── Compute convergence rate ──
        # Positive = moving apart, negative = moving together
        vel_a = state_a.velocity.direction * state_a.velocity.speed
        vel_b = state_b.velocity.direction * state_b.velocity.speed
        
        # Relative velocity: how fast is B moving relative to A?
        rel_vel = vel_b - vel_a
        
        # Convergence: projection of relative velocity onto direction A→B
        if distance > 1e-6:
            direction_ab = diff / distance
            convergence_rate = float(np.dot(rel_vel, direction_ab))
        else:
            convergence_rate = 0.0
        
        # ── Check shared attractor ──
        shared_attractor = False
        if (state_a.attractor is not None and state_b.attractor is not None
                and state_a.attractor_confidence > 0.3
                and state_b.attractor_confidence > 0.3):
            attractor_dist = float(np.linalg.norm(state_a.attractor - state_b.attractor))
            shared_attractor = attractor_dist < self.config.joint_attractor_distance
        
        # ── Classify interaction ──
        interaction_type, confidence = self._classify(
            distance, convergence_rate, shared_attractor,
            state_a, state_b, id_a, id_b)
        
        return PairInteraction(
            entity_a=id_a,
            entity_b=id_b,
            interaction_type=interaction_type,
            confidence=confidence,
            convergence_rate=convergence_rate,
            distance=distance,
            shared_attractor=shared_attractor,
        )
    
    def _classify(self, distance: float, convergence_rate: float,
                  shared_attractor: bool, state_a, state_b,
                  id_a: int, id_b: int) -> tuple[InteractionType, float]:
        """Classify interaction type from geometric features."""
        cfg = self.config

        # ── Same group check (motion-based) ──
        is_same_group = self._same_group(distance, state_a, state_b)

        # ── Co-present (stationary) ──
        if distance < cfg.co_presence_distance:
            speed_a = state_a.velocity.speed
            speed_b = state_b.velocity.speed
            if speed_a < 0.05 and speed_b < 0.05:
                return InteractionType.CO_PRESENT, 0.8

        # ── Following ──
        following = self._check_following(state_a, state_b, id_a, id_b)
        if following > 0:
            return InteractionType.FOLLOWING, following

        following_rev = self._check_following(state_b, state_a, id_b, id_a)
        if following_rev > 0:
            return InteractionType.FOLLOWING, following_rev

        # ── Joint goal ──
        if shared_attractor:
            conf = min(state_a.attractor_confidence, state_b.attractor_confidence)
            return InteractionType.JOINT_GOAL, conf * 0.9

        # ── Converging (moving toward each other, same group context) ──
        if convergence_rate < cfg.convergence_threshold and distance > 0.5:
            conf = min(1.0, abs(convergence_rate) * 5)
            if is_same_group:
                conf = min(1.0, conf + 0.3)  # bonus for same-group convergence
            return InteractionType.CONVERGING, conf

        # ── Diverging / Avoiding ──
        if convergence_rate > cfg.divergence_threshold and distance <= 5.0:
            speed_a = state_a.velocity.speed
            speed_b = state_b.velocity.speed
            if speed_a > 0.1 and speed_b > 0.1:
                return InteractionType.AVOIDING, 0.6
            return InteractionType.DIVERGING, 0.5

        # ── Same group but no strong signal ──
        if is_same_group and distance < cfg.group_proximity:
            return InteractionType.CO_PRESENT, 0.6

        return InteractionType.NONE, 0.0

    def _same_group(self, distance: float, state_a, state_b) -> bool:
        """Check if two agents are in the same group based on motion.

        A group is defined as agents that are:
        1. Within proximity threshold
        2. Moving in similar directions (cosine > threshold)
        3. Moving at similar speeds (ratio > threshold)
        """
        cfg = self.config

        if distance > cfg.group_proximity:
            return False

        speed_a = state_a.velocity.speed
        speed_b = state_b.velocity.speed

        # Both stationary = same group if close enough
        if speed_a < 0.05 and speed_b < 0.05:
            return distance < cfg.co_presence_distance * 2

        # Need direction data
        dir_a = state_a.velocity.direction
        dir_b = state_b.velocity.direction
        norm_a = np.linalg.norm(dir_a)
        norm_b = np.linalg.norm(dir_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return False

        # Velocity cosine similarity
        cos_sim = float(np.dot(dir_a / norm_a, dir_b / norm_b))

        # Speed ratio (slower / faster)
        speed_ratio = min(speed_a, speed_b) / max(speed_a, speed_b)

        return (cos_sim > cfg.group_velocity_cos
                and speed_ratio > cfg.group_speed_ratio)
    
    def _check_following(self, follower, target, follower_id: int,
                         target_id: int) -> float:
        """Check if one agent is following another.
        
        Following = moving in the same direction with a time lag,
        and the follower's velocity points toward the target.
        """
        if follower.position is None or target.position is None:
            return 0.0
        
        if follower.velocity.speed < 0.05 or target.velocity.speed < 0.05:
            return 0.0
        
        # Need enough history
        if len(follower.recent_positions) < self.config.following_min_duration:
            return 0.0
        
        # Check if follower's velocity points toward target
        to_target = target.position - follower.position
        dist = float(np.linalg.norm(to_target))
        if dist < 0.1:
            return 0.0
        
        dir_to_target = to_target / dist
        cos_alignment = float(np.dot(follower.velocity.direction, dir_to_target))
        
        if cos_alignment < self.config.following_cos_threshold:
            return 0.0
        
        # Check direction similarity (both moving same way)
        cos_dir = float(np.dot(follower.velocity.direction, target.velocity.direction))
        
        if cos_dir < 0.5:
            return 0.0
        
        # Check consistency: look at recent velocity directions
        n = min(len(follower.recent_positions), len(target.recent_positions))
        if n >= 3:
            consistent_count = 0
            for k in range(max(2, n - 3), n):
                v_f = follower.recent_positions[k] - follower.recent_positions[k - 1]
                v_t = target.recent_positions[k] - target.recent_positions[k - 1]
                nf = np.linalg.norm(v_f)
                nt = np.linalg.norm(v_t)
                if nf > 1e-6 and nt > 1e-6:
                    cos_k = float(np.dot(v_f / nf, v_t / nt))
                    if cos_k > 0.5:
                        consistent_count += 1
            frames_consistent = consistent_count
        else:
            frames_consistent = 1
        
        if frames_consistent < self.config.following_min_duration:
            return 0.0
        
        # Confidence from alignment + consistency
        conf = cos_alignment * 0.5 + cos_dir * 0.3
        duration_bonus = min(0.2, frames_consistent * 0.02)
        return min(0.9, conf + duration_bonus)
    
    def reset(self) -> None:
        pass
