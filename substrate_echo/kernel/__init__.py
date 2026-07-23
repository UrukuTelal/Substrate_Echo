"""Substrate Kernel — The cognitive backend.

Two planes:

  Control Plane (REST):
    - Create/load/save kernel
    - Query statistics
    - Configure parameters
    - Health checks
    - Checkpoint persistence

  Cognitive Plane (Streaming):
    - Observations (sensor state)
    - Goals (desired states)
    - Actions (suggested trajectories)
    - Predictions (expected next states)
    - Rewards (reinforcement signals)
    - EmbodimentState (client status)

Clients never manipulate cognition directly.
They publish state. The kernel decides how experience
changes the cognitive landscape.

Architecture:
  Embodiment (sensors/I/O)
       |
  Cognitive Plane (streaming)
       |
  Substrate Kernel (cognition)
       |
  Control Plane (admin/persistence)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np
import time

from ..core.dynamics_memory import DynamicsMemory, DynamicsMemoryConfig
from ..dynamics.basin_topology import BasinTopology
from ..dynamics.abstraction import AbstractionEngine
from .executive import ExecutiveFunction, GoalState, GoalStatus, GoalTier, ExecutiveState
from .resources import ResourceManager, ResourceRequest, ResourceAllocation, ResourceState
from .council import Council, AuditReport, CouncilState


# ── Cognitive Plane: State Types ──────────────────────────────────

@dataclass
class Observation:
    """Sensor state from an embodiment.

    Not a message. A snapshot of the world.
    """
    vector: List[float]
    modality: str = "generic"       # "audio", "visual", "text", "proprio"
    embodiment_id: str = "default"
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        return np.array(self.vector, dtype=np.float64)


@dataclass
class Goal:
    """Desired state. The kernel decides how to reach it."""
    target: List[float]
    priority: float = 0.5
    description: str = ""
    embodiment_id: str = "default"
    created_at: float = 0.0


@dataclass
class Reward:
    """Reinforcement signal from the environment.

    The kernel uses this to adjust attractor strengths
    and plasticity, not to directly modify weights.
    """
    value: float                     # [-1, 1] negative=punish, positive=reward
    target_attractor: Optional[int] = None  # which attractor this refers to
    embodiment_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Suggested trajectory from the kernel.

    Not a command. The embodiment decides whether to follow it.
    """
    vector: List[float]
    confidence: float = 0.0
    source: str = ""                 # what produced this action
    embodiment_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """What the kernel expects to happen next.

    The embodiment can compare this to actual outcome
    to compute surprise/novelty.
    """
    expected_next: List[float]
    confidence: float = 0.0
    source: str = ""


@dataclass
class EmbodimentState:
    """Status report from a client.

    The kernel uses this to know what bodies are alive,
    what sensors they have, what their capabilities are.
    """
    embodiment_id: str
    embodiment_type: str = "generic"  # "desktop", "robot", "vr", "simulation"
    available_modalities: List[str] = field(default_factory=lambda: ["generic"])
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveState:
    """Full kernel state returned on each cognitive tick.

    This is what the client receives. It contains everything
    the client needs to know about the kernel's current state.
    """
    tick: int = 0
    action: Optional[Action] = None
    prediction: Optional[Prediction] = None
    n_attractors: int = 0
    n_meta_attractors: int = 0
    coherence: float = 0.0
    basin_balance: float = 0.0
    mean_depth: float = 0.0
    volume_entropy: float = 0.0
    active_goals: int = 0
    active_embodiments: int = 0
    cognitive_energy: float = 0.0
    timestamp: float = 0.0
    executive: Optional[ExecutiveState] = None


# ── Kernel Config ─────────────────────────────────────────────────

@dataclass
class KernelConfig:
    dim: int = 16
    model_type: str = "global"
    min_samples_for_fit: int = 50
    max_samples: int = 4000
    attractor_radius: float = 0.12
    attractor_min_cluster: int = 6
    convergence_window: int = 40
    correlation_threshold: float = 0.25
    meta_sigma: float = 0.5
    coupling_strength: float = 0.3
    total_energy: float = 10.0
    topology_interval: int = 200


# ── Substrate Kernel ──────────────────────────────────────────────

class SubstrateKernel:
    """The cognitive backend.

    Owns all persistent cognitive state.
    Receives state publications, returns cognitive state.
    Never called by the client — the client publishes,
    the kernel responds.
    """

    def __init__(self, config: Optional[KernelConfig] = None):
        self.config = config or KernelConfig()

        # Core cognitive components
        self.dm = DynamicsMemory(dim=self.config.dim, config=DynamicsMemoryConfig(
            model_type=self.config.model_type,
            min_samples_for_fit=self.config.min_samples_for_fit,
            max_samples=self.config.max_samples,
            attractor_samples=300,
            attractor_integration_steps=200))

        self.topology = BasinTopology(sigma=0.3)
        self.abstraction = AbstractionEngine(
            correlation_threshold=self.config.correlation_threshold,
            min_cluster_size=2,
            meta_sigma=self.config.meta_sigma)
        self.executive = ExecutiveFunction()
        self.resources = ResourceManager()
        self.council = Council()

        # Cognitive state
        self._tick = 0
        self._prev_state: Optional[np.ndarray] = None
        self._base_attractors: Dict[int, tuple] = {}
        self._convergence_endpoints: List[np.ndarray] = []
        self._goals: List[Goal] = []
        self._rewards: List[Reward] = []
        self._embodiments: Dict[str, EmbodimentState] = {}
        self._cognitive_energy = self.config.total_energy

    # ── Cognitive Plane ──────────────────────────────────────────

    def publish_observation(self, obs: Observation) -> CognitiveState:
        """Main cognitive entry point. One call = one tick."""
        state = obs.to_array()

        # Track embodiment
        if obs.embodiment_id not in self._embodiments:
            self._embodiments[obs.embodiment_id] = EmbodimentState(
                embodiment_id=obs.embodiment_id)
        self._embodiments[obs.embodiment_id].is_active = True

        # Learn dynamics
        if self._prev_state is not None:
            velocity = state - self._prev_state
            self.dm._states.append(self._prev_state.copy())
            self.dm._velocities.append(velocity.copy())
            if len(self.dm._states) >= self.dm.config.min_samples_for_fit:
                if len(self.dm._states) % 50 == 0:
                    self.dm._fit_dynamics()

        # Convergence detection
        self._convergence_endpoints.append(state.copy())
        if len(self._convergence_endpoints) > self.config.convergence_window:
            self._convergence_endpoints.pop(0)
        if self._tick % 100 == 0 and self._tick > 200:
            self._detect_convergence()

        # Abstraction
        if self._tick % 10 == 0 and self._base_attractors:
            self.abstraction.update(self._tick, state, self._base_attractors)
        if self._tick % 500 == 0 and self._tick > 500 and self._base_attractors:
            self.abstraction.check_abstraction(self._tick, self._base_attractors)

        # Topology
        if self._tick % self.config.topology_interval == 0 and self._tick > 0:
            self.topology.record_snapshot(self._tick)

        # Apply pending rewards
        self._apply_rewards()

        # Executive function: manage goals, attention, priorities
        exec_state = self.executive.tick(obs, current_tick=self._tick)

        # Generate action + prediction (influenced by executive attention)
        action = self._generate_action(state)
        prediction = self._generate_prediction(state)

        # Build response
        topo = self.topology.compute_metrics()
        abs_summary = self.abstraction.summary()

        # Council: periodic audits and health checks
        substrate_state = {
            "tick": self._tick,
            "n_attractors": len(self._base_attractors),
            "n_meta_attractors": abs_summary["n_meta"],
            "coherence": self._compute_coherence(),
            "volume_entropy": topo.volume_entropy,
            "basin_balance": topo.basin_balance,
            "n_goals": exec_state.n_active,
            "n_embodiments": sum(1 for e in self._embodiments.values() if e.is_active),
        }
        new_reports = self.council.tick(substrate_state)

        cognitive_state = CognitiveState(
            tick=self._tick,
            action=action,
            prediction=prediction,
            n_attractors=len(self._base_attractors),
            n_meta_attractors=abs_summary["n_meta"],
            coherence=self._compute_coherence(),
            basin_balance=topo.basin_balance,
            mean_depth=topo.mean_depth,
            volume_entropy=topo.volume_entropy,
            active_goals=exec_state.n_active,
            active_embodiments=sum(1 for e in self._embodiments.values() if e.is_active),
            cognitive_energy=self._cognitive_energy,
            executive=exec_state,
            timestamp=time.time(),
        )

        self._prev_state = state.copy()
        self._tick += 1
        return cognitive_state

    def publish_goal(self, goal: Goal):
        self._goals.append(goal)

    def publish_reward(self, reward: Reward):
        self._rewards.append(reward)

    def publish_embodiment_state(self, emb: EmbodimentState):
        self._embodiments[emb.embodiment_id] = emb

    def set_goals(self, goals: List[Goal]):
        self._goals = goals

    # ── Control Plane ────────────────────────────────────────────

    def get_snapshot(self) -> Dict[str, Any]:
        """Full serializable state for persistence/checkpointing."""
        topo = self.topology.compute_metrics()
        abs_summary = self.abstraction.summary()
        return {
            "tick": self._tick,
            "n_attractors": len(self._base_attractors),
            "n_meta_attractors": abs_summary["n_meta"],
            "basin_balance": topo.basin_balance,
            "mean_depth": topo.mean_depth,
            "volume_entropy": topo.volume_entropy,
            "coherence": self._compute_coherence(),
            "cognitive_energy": self._cognitive_energy,
            "n_goals": len(self._goals),
            "n_executive_goals": self.executive._next_id,
            "n_rewards": len(self._rewards),
            "n_embodiments": len(self._embodiments),
            "active_embodiments": sum(
                1 for e in self._embodiments.values() if e.is_active),
            "abstraction_events": abs_summary["n_abstraction_events"],
            "topology_events": len(self.topology.events),
            "resources": self.resources.get_state().budget,
        }

    def get_topology_history(self) -> List[Dict]:
        return [
            {"tick": t, "attractors": m.n_attractors, "depth": m.mean_depth,
             "entropy": m.volume_entropy, "balance": m.basin_balance}
            for t, m in self.topology.history
        ]

    def get_abstraction_events(self) -> List[Dict]:
        return self.abstraction.abstraction_events

    def get_embodiments(self) -> Dict[str, Dict]:
        return {
            eid: {
                "type": e.embodiment_type,
                "active": e.is_active,
                "modalities": e.available_modalities,
            }
            for eid, e in self._embodiments.items()
        }

    # ── Internal ─────────────────────────────────────────────────

    def _detect_convergence(self):
        if len(self._convergence_endpoints) < self.config.attractor_min_cluster:
            return
        arr = np.array(self._convergence_endpoints)
        assigned = np.zeros(len(arr), dtype=int) - 1
        clusters = []
        for i, pt in enumerate(arr):
            if assigned[i] >= 0:
                continue
            dists = np.linalg.norm(arr - pt, axis=1)
            neighbors = np.where(dists < self.config.attractor_radius)[0]
            if len(neighbors) >= self.config.attractor_min_cluster:
                center = arr[neighbors].mean(axis=0)
                clusters.append((center, len(neighbors)))
                assigned[neighbors] = len(clusters) - 1

        for center, size in clusters:
            is_new = True
            for aid, (existing, _) in self._base_attractors.items():
                if np.linalg.norm(center - existing) < 0.2:
                    is_new = False
                    break
            if is_new:
                strength = min(1.0, size / 20.0)
                aid = len(self._base_attractors)
                self._base_attractors[aid] = (center.copy(), strength)
                self.topology.add_attractor(center, self._tick, strength)

    def _generate_action(self, state: np.ndarray) -> Action:
        if len(self.dm._states) < self.dm.config.min_samples_for_fit:
            return Action(vector=state.tolist(), confidence=0.0, source="default")

        v_learned = self.dm.predict_velocity(state)

        v_attractor = np.zeros_like(state)
        for aid, (center, strength) in self._base_attractors.items():
            diff = state - center
            dist2 = np.sum(diff ** 2)
            v_attractor += strength * diff / (0.3 ** 2) * np.exp(-dist2 / (2 * 0.3 ** 2))

        v_meta = self.abstraction.gradient_contribution(state)

        v_goal = np.zeros_like(state)
        for goal in self._goals:
            target = np.array(goal.target)
            diff = target - state
            v_goal += goal.priority * diff

        v_total = (v_learned
                   + self.config.coupling_strength * (-v_attractor - 0.5 * v_meta)
                   + 0.2 * v_goal)

        new_state = np.clip(state + 0.02 * v_total, 0.0, 1.0)
        confidence = min(1.0, len(self.dm._states) / 500.0)

        return Action(
            vector=new_state.tolist(),
            confidence=confidence,
            source="dynamics+attractors+goals",
            metadata={"v_norm": float(np.linalg.norm(v_total))},
        )

    def _generate_prediction(self, state: np.ndarray) -> Prediction:
        if len(self.dm._states) < self.dm.config.min_samples_for_fit:
            return Prediction(expected_next=state.tolist(), confidence=0.0)
        v = self.dm.predict_velocity(state)
        predicted = np.clip(state + 0.02 * v, 0.0, 1.0)
        confidence = min(1.0, len(self.dm._states) / 500.0)
        return Prediction(
            expected_next=predicted.tolist(),
            confidence=confidence,
            source="dynamics",
        )

    def _apply_rewards(self):
        """Adjust attractor strengths based on rewards."""
        if not self._rewards:
            return
        for reward in self._rewards.pop(0):
            pass  # TODO: apply to specific attractor
        # Simple global reinforcement
        for reward in self._rewards:
            if reward.target_attractor is not None:
                aid = reward.target_attractor
                if aid in self._base_attractors:
                    center, strength = self._base_attractors[aid]
                    new_strength = max(0.05, min(1.0, strength + reward.value * 0.1))
                    self._base_attractors[aid] = (center, new_strength)
        self._rewards.clear()

    def _compute_coherence(self) -> float:
        if not self._base_attractors:
            return 0.0
        centers = [c for c, _ in self._base_attractors.values()]
        if len(centers) < 2:
            return 0.5
        dists = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dists.append(np.linalg.norm(centers[i] - centers[j]))
        return min(1.0, np.mean(dists) / 0.5)
