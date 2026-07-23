"""Substrate Client — Thin embodiment wrapper.

The client handles sensors, I/O, and real-time requirements.
It publishes state to the kernel and receives cognitive state back.

No cognition lives here. Just perception and action.
"""
from __future__ import annotations
from typing import Optional, Dict, List, Callable
import numpy as np
import threading
import queue

from ..kernel import (
    SubstrateKernel, KernelConfig,
    Observation, Goal, Reward, Action, Prediction,
    EmbodimentState, CognitiveState,
)


class SubstrateClient:
    """In-process client. Direct call to kernel.

    For out-of-process, use WebSocketClient.
    """

    def __init__(self, kernel: Optional[SubstrateKernel] = None,
                 config: Optional[KernelConfig] = None,
                 embodiment_id: str = "default"):
        self.kernel = kernel or SubstrateKernel(config)
        self.embodiment_id = embodiment_id
        self._tick = 0
        self._history: List[CognitiveState] = []
        self._on_tick: Optional[Callable[[CognitiveState], None]] = None

    def observe(self, vector: List[float], modality: str = "generic",
                **metadata) -> CognitiveState:
        """Publish observation, get cognitive state."""
        obs = Observation(
            vector=vector,
            modality=modality,
            embodiment_id=self.embodiment_id,
            metadata=metadata,
        )
        state = self.kernel.publish_observation(obs)
        self._history.append(state)
        self._tick += 1
        if self._on_tick:
            self._on_tick(state)
        return state

    def set_goal(self, target: List[float], priority: float = 0.5,
                 description: str = ""):
        goal = Goal(
            target=target,
            priority=priority,
            description=description,
            embodiment_id=self.embodiment_id,
        )
        self.kernel.publish_goal(goal)

    def send_reward(self, value: float, target_attractor: Optional[int] = None):
        reward = Reward(
            value=value,
            target_attractor=target_attractor,
            embodiment_id=self.embodiment_id,
        )
        self.kernel.publish_reward(reward)

    def set_goals(self, goals: List[Goal]):
        self.kernel.set_goals(goals)

    def on_tick(self, callback: Callable[[CognitiveState], None]):
        self._on_tick = callback

    def get_snapshot(self) -> Dict:
        return self.kernel.get_snapshot()

    @property
    def tick(self) -> int:
        return self._tick


class StreamingClient:
    """Client that runs a continuous loop at a fixed tick rate.

    For real-time embodiments (robot, VR).
    """

    def __init__(self, kernel: Optional[SubstrateKernel] = None,
                 config: Optional[KernelConfig] = None,
                 embodiment_id: str = "streaming",
                 tick_rate: float = 10.0):
        self.client = SubstrateClient(kernel, config, embodiment_id)
        self.tick_rate = tick_rate
        self._running = False
        self._obs_queue: queue.Queue = queue.Queue()
        self._state_queue: queue.Queue = queue.Queue()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def feed(self, vector: List[float], modality: str = "generic"):
        self._obs_queue.put((vector, modality))

    def read(self) -> Optional[CognitiveState]:
        try:
            return self._state_queue.get_nowait()
        except queue.Empty:
            return None

    def _loop(self):
        interval = 1.0 / self.tick_rate
        while self._running:
            try:
                vector, modality = self._obs_queue.get(timeout=interval)
            except queue.Empty:
                continue
            state = self.client.observe(vector, modality)
            self._state_queue.put(state)
