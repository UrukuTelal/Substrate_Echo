"""Goal State Machine — per-agent goal tracking with trajectory prediction.

Stage 4 of the developmental architecture: "What is each agent
trying to accomplish?"

This module tracks observed agent trajectories and infers their
current goal state via a finite state machine. Each agent is
independently tracked with velocity estimation, goal-phase
classification, and future trajectory prediction.

The GoalManager provides world-model updates that the Planner
and Evaluator consume — if I know where a human is heading,
I can plan my own trajectory to meet them, avoid them, or
assist them.

State transitions:
    IDLE → EXPLORING (motion begins)
    EXPLORING → APPROACHING (directed toward attractor)
    APPROACHING → RETREATING (reverses direction)
    APPROACHING → IDLE (velocity → 0)
    RETREATING → EXPLORING (new direction detected)
    Any → COMMUNICATING (social intent detected)
    COMMUNICATING → IDLE (social signal lost)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np


class GoalPhase(Enum):
    """Phases of agent behavior."""
    IDLE = auto()
    EXPLORING = auto()
    APPROACHING = auto()
    RETREATING = auto()
    COMMUNICATING = auto()
    WORKING = auto()


class GoalTransition(Enum):
    """Types of state transitions."""
    MOTION_START = auto()
    MOTION_STOP = auto()
    DIRECTION_REVERSE = auto()
    ATTRACTOR_DETECTED = auto()
    SOCIAL_SIGNAL = auto()
    SOCIAL_SIGNAL_LOST = auto()
    TASK_DETECTED = auto()
    TASK_COMPLETE = auto()


@dataclass
class VelocityEstimate:
    """Smoothed velocity estimate for an agent."""
    direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    speed: float = 0.0
    acceleration: float = 0.0
    confidence: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.concatenate([self.direction, [self.speed, self.acceleration]])


@dataclass
class TrajectoryPrediction:
    """Predicted future trajectory of an agent."""
    positions: list[np.ndarray] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    confidence: float = 0.0
    predicted_phase: GoalPhase = GoalPhase.IDLE


@dataclass
class AgentGoalState:
    """Current goal state of a tracked agent."""
    entity_id: int
    phase: GoalPhase = GoalPhase.IDLE
    phase_duration: float = 0.0           # time in current phase
    
    # Velocity tracking
    velocity: VelocityEstimate = field(default_factory=VelocityEstimate)
    recent_positions: list[np.ndarray] = field(default_factory=list)
    recent_timestamps: list[float] = field(default_factory=list)
    
    # Goal inference
    attractor: Optional[np.ndarray] = None  # estimated target position
    attractor_confidence: float = 0.0
    estimated_goal_desc: str = ""
    
    # Social
    social_intent: float = 0.0
    
    # Prediction
    prediction: Optional[TrajectoryPrediction] = None
    
    # Transition history
    transition_count: int = 0
    last_transition: Optional[GoalTransition] = None
    
    @property
    def position(self) -> Optional[np.ndarray]:
        """Most recent observed position."""
        if self.recent_positions:
            return self.recent_positions[-1]
        return None
    
    @property
    def time_in_phase(self) -> float:
        if len(self.recent_timestamps) < 2:
            return 0.0
        return self.recent_timestamps[-1] - self.phase_duration


@dataclass
class GoalManagerConfig:
    """Configuration for goal tracking."""
    max_history: int = 30
    velocity_smoothing: float = 0.3       # EMA alpha for velocity
    attractor_threshold: float = 0.3      # distance to consider "at attractor"
    motion_threshold: float = 0.01        # speed below this = idle
    phase_min_duration: float = 0.5       # min time before transition
    prediction_horizon: float = 3.0       # seconds ahead to predict
    prediction_dt: float = 0.5            # time step for predictions


class GoalManager:
    """Tracks goal states for all observed agents.
    
    The GoalManager is the Stage 4 component of the developmental
    architecture. It answers: "What is each agent trying to do?"
    
    Usage:
        gm = GoalManager()
        # Each tick:
        gm.update(entity_id=1, position=[0.5, 0.3], timestamp=1.0,
                  social_intent=0.2)
        state = gm.get_state(1)
        if state.phase == GoalPhase.APPROACHING:
            print(f"Agent approaching {state.attractor}")
    """
    
    def __init__(self, config: Optional[GoalManagerConfig] = None):
        self.config = config or GoalManagerConfig()
        self._agents: dict[int, AgentGoalState] = {}
    
    def update(self, entity_id: int, position: np.ndarray,
               timestamp: float, social_intent: float = 0.0,
               properties: Optional[dict] = None) -> AgentGoalState:
        """Update tracking for an observed agent.
        
        Call this each tick for every visible agent.
        """
        position = np.asarray(position, dtype=np.float64)
        
        if entity_id not in self._agents:
            self._agents[entity_id] = AgentGoalState(entity_id=entity_id)
        
        agent = self._agents[entity_id]
        
        # Store position history
        agent.recent_positions.append(position.copy())
        agent.recent_timestamps.append(timestamp)
        
        # Bound history
        if len(agent.recent_positions) > self.config.max_history:
            agent.recent_positions = agent.recent_positions[-self.config.max_history:]
            agent.recent_timestamps = agent.recent_timestamps[-self.config.max_history:]
        
        # Update social intent
        agent.social_intent = social_intent
        
        # Update velocity estimate
        self._update_velocity(agent)
        
        # Run state machine
        self._transition(agent, timestamp)
        
        # Update attractor estimate
        self._estimate_attractor(agent)
        
        # Generate prediction
        agent.prediction = self._predict_trajectory(agent)
        
        # Update phase duration
        if len(agent.recent_timestamps) >= 2:
            dt = agent.recent_timestamps[-1] - agent.recent_timestamps[-2]
            agent.phase_duration += dt
        
        return agent
    
    def get_state(self, entity_id: int) -> Optional[AgentGoalState]:
        return self._agents.get(entity_id)
    
    def get_all_states(self) -> dict[int, AgentGoalState]:
        return dict(self._agents)
    
    def remove(self, entity_id: int):
        self._agents.pop(entity_id, None)
    
    def _update_velocity(self, agent: AgentGoalState):
        """Smoothed velocity estimate using exponential moving average."""
        if len(agent.recent_positions) < 2:
            return
        
        dt = agent.recent_timestamps[-1] - agent.recent_timestamps[-2]
        if dt <= 0:
            return
        
        raw_vel = (agent.recent_positions[-1] - agent.recent_positions[-2]) / dt
        raw_speed = np.linalg.norm(raw_vel)
        
        alpha = self.config.velocity_smoothing
        
        if agent.velocity.speed < 1e-6:
            # First velocity estimate
            agent.velocity.direction = raw_vel / (raw_speed + 1e-10)
            agent.velocity.speed = raw_speed
        else:
            # EMA update
            new_speed = alpha * raw_speed + (1 - alpha) * agent.velocity.speed
            if raw_speed > 1e-6:
                new_dir = alpha * (raw_vel / raw_speed) + (1 - alpha) * agent.velocity.direction
                norm = np.linalg.norm(new_dir)
                if norm > 1e-6:
                    agent.velocity.direction = new_dir / norm
            agent.velocity.speed = new_speed
        
        # Acceleration
        agent.velocity.acceleration = raw_speed - agent.velocity.speed
        
        # Confidence from consistency
        if len(agent.recent_positions) >= 3:
            v_prev = (agent.recent_positions[-2] - agent.recent_positions[-3]) / dt
            if raw_speed > 1e-6 and np.linalg.norm(v_prev) > 1e-6:
                cos = np.dot(raw_vel / raw_speed, v_prev / np.linalg.norm(v_prev))
                agent.velocity.confidence = max(0.0, min(1.0, (cos + 1) / 2))
            else:
                agent.velocity.confidence = 0.3
        else:
            agent.velocity.confidence = 0.5
    
    def _transition(self, agent: AgentGoalState, timestamp: float):
        """Run the goal state machine."""
        speed = agent.velocity.speed
        old_phase = agent.phase
        
        # Social signal → COMMUNICATING
        if agent.social_intent > 0.6:
            if old_phase != GoalPhase.COMMUNICATING:
                agent.phase = GoalPhase.COMMUNICATING
                agent.phase_duration = timestamp
                agent.transition_count += 1
                agent.last_transition = GoalTransition.SOCIAL_SIGNAL
                agent.estimated_goal_desc = "communicating with observer"
            return
        
        # Social signal lost
        if old_phase == GoalPhase.COMMUNICATING and agent.social_intent < 0.3:
            agent.phase = GoalPhase.IDLE
            agent.phase_duration = timestamp
            agent.transition_count += 1
            agent.last_transition = GoalTransition.SOCIAL_SIGNAL_LOST
            return
        
        # Skip transitions during social phase
        if old_phase == GoalPhase.COMMUNICATING:
            return
        
        # Motion-based transitions
        if speed < self.config.motion_threshold:
            if old_phase != GoalPhase.IDLE:
                agent.phase = GoalPhase.IDLE
                agent.phase_duration = timestamp
                agent.transition_count += 1
                agent.last_transition = GoalTransition.MOTION_STOP
                agent.estimated_goal_desc = "stationary"
        elif speed >= self.config.motion_threshold:
            if old_phase == GoalPhase.IDLE:
                agent.phase = GoalPhase.EXPLORING
                agent.phase_duration = timestamp
                agent.transition_count += 1
                agent.last_transition = GoalTransition.MOTION_START
                agent.estimated_goal_desc = "exploring"
            elif old_phase == GoalPhase.EXPLORING:
                # Check if approaching an attractor
                if agent.attractor is not None and agent.attractor_confidence > 0.5:
                    dist_to_attractor = np.linalg.norm(agent.position - agent.attractor)
                    if dist_to_attractor < self.config.attractor_threshold * 5:
                        agent.phase = GoalPhase.APPROACHING
                        agent.phase_duration = timestamp
                        agent.transition_count += 1
                        agent.last_transition = GoalTransition.ATTRACTOR_DETECTED
                        agent.estimated_goal_desc = f"approaching target"
            elif old_phase == GoalPhase.APPROACHING:
                # Check for reversal
                if agent.attractor is not None and agent.position is not None:
                    to_attractor = agent.attractor - agent.position
                    to_attractor_norm = np.linalg.norm(to_attractor)
                    if to_attractor_norm > 1e-6:
                        cos = np.dot(agent.velocity.direction, to_attractor / to_attractor_norm)
                        if cos < -0.3:  # moving away from attractor
                            agent.phase = GoalPhase.RETREATING
                            agent.phase_duration = timestamp
                            agent.transition_count += 1
                            agent.last_transition = GoalTransition.DIRECTION_REVERSE
                            agent.estimated_goal_desc = "retreating from target"
    
    def _estimate_attractor(self, agent: AgentGoalState):
        """Estimate where the agent is heading."""
        if len(agent.recent_positions) < 3:
            return
        
        # Simple extrapolation: where would they be at the current speed for 2s?
        if agent.velocity.speed > self.config.motion_threshold:
            predicted = agent.position + agent.velocity.direction * agent.velocity.speed * 2.0
            if agent.attractor is None:
                agent.attractor = predicted
                agent.attractor_confidence = agent.velocity.confidence * 0.5
            else:
                # Smooth toward predicted
                alpha = 0.2
                agent.attractor = alpha * predicted + (1 - alpha) * agent.attractor
                agent.attractor_confidence = min(1.0,
                    agent.attractor_confidence + 0.05 * agent.velocity.confidence)
        else:
            agent.attractor_confidence *= 0.95  # decay when stationary
    
    def _predict_trajectory(self, agent: AgentGoalState) -> TrajectoryPrediction:
        """Predict future positions based on current velocity and phase."""
        if agent.position is None or agent.velocity.speed < 1e-6:
            return TrajectoryPrediction(
                positions=[agent.position] if agent.position is not None else [],
                confidence=0.1,
                predicted_phase=agent.phase,
            )
        
        positions = []
        timestamps = []
        pos = agent.position.copy()
        vel = agent.velocity.direction * agent.velocity.speed
        
        t_start = agent.recent_timestamps[-1] if agent.recent_timestamps else 0.0
        
        for i in range(int(self.config.prediction_horizon / self.config.prediction_dt)):
            t = t_start + (i + 1) * self.config.prediction_dt
            
            # Phase-dependent velocity modification
            if agent.phase == GoalPhase.APPROACHING and agent.attractor is not None:
                to_target = agent.attractor - pos
                dist = np.linalg.norm(to_target)
                if dist > 0.01:
                    approach_vel = to_target / dist * agent.velocity.speed
                    vel = 0.7 * vel + 0.3 * approach_vel  # blend toward attractor
            
            elif agent.phase == GoalPhase.RETREATING and agent.attractor is not None:
                from_target = pos - agent.attractor
                dist = np.linalg.norm(from_target)
                if dist > 0.01:
                    retreat_vel = from_target / dist * agent.velocity.speed
                    vel = 0.5 * vel + 0.5 * retreat_vel
            
            elif agent.phase == GoalPhase.IDLE:
                vel *= 0.9  # gradual slowdown
            
            pos = pos + vel * self.config.prediction_dt
            positions.append(pos.copy())
            timestamps.append(t)
        
        # Confidence decreases with prediction horizon
        conf = agent.velocity.confidence * 0.8
        
        return TrajectoryPrediction(
            positions=positions,
            timestamps=timestamps,
            confidence=conf,
            predicted_phase=agent.phase,
        )
