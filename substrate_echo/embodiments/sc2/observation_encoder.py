"""SC2 Observation Encoder — Translates game state to kernel vectors.

Encodes SC2 observations into 16-dimensional vectors that the
Substrate Kernel can process.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


# ── Observation Categories ───────────────────────────────────────

@dataclass
class EconomyState:
    """Economic observation components."""
    minerals: float = 0.0
    vespene: float = 0.0
    supply_used: float = 0.0
    supply_cap: float = 0.0
    workers: float = 0.0
    bases: float = 0.0
    production_buildings: float = 0.0
    army_value: float = 0.0

    def to_vector(self) -> List[float]:
        """Normalize to [0, 1] range."""
        return [
            min(1.0, self.minerals / 5000),
            min(1.0, self.vespene / 5000),
            min(1.0, self.supply_used / 200) if self.supply_cap > 0 else 0.0,
            min(1.0, self.supply_cap / 200),
            min(1.0, self.workers / 80),
            min(1.0, self.bases / 5),
            min(1.0, self.production_buildings / 20),
            min(1.0, self.army_value / 5000),
        ]


@dataclass
class MilitaryState:
    """Military observation components."""
    army_count: float = 0.0
    army_value: float = 0.0
    air_count: float = 0.0
    ground_count: float = 0.0
    upgrades: float = 0.0
    position_advantage: float = 0.5  # 0=retreating, 0.5=neutral, 1=advancing
    threat_level: float = 0.0  # 0=none, 1=immediate danger
    map_control: float = 0.5  # 0=none, 1=full

    def to_vector(self) -> List[float]:
        return [
            min(1.0, self.army_count / 100),
            min(1.0, self.army_value / 5000),
            min(1.0, self.air_count / 50),
            min(1.0, self.ground_count / 100),
            min(1.0, self.upgrades / 10),
            self.position_advantage,
            self.threat_level,
            self.map_control,
        ]


@dataclass
class InformationState:
    """Information/uncertainty observation components."""
    scout_count: float = 0.0
    enemy_known_ratio: float = 0.0
    map_revealed: float = 0.0
    enemy_army_known: float = 0.0
    enemy_tech_known: float = 0.0
    enemy_bases_known: float = 0.0
    last_scout_time: float = 0.0  # seconds since last scout
    uncertainty: float = 0.5  # 0=fully known, 1=unknown

    def to_vector(self) -> List[float]:
        return [
            min(1.0, self.scout_count / 10),
            self.enemy_known_ratio,
            self.map_revealed,
            self.enemy_army_known,
            self.enemy_tech_known,
            min(1.0, self.enemy_bases_known / 5),
            min(1.0, self.last_scout_time / 300),
            self.uncertainty,
        ]


# ── Encoder ──────────────────────────────────────────────────────

class SC2ObservationEncoder:
    """Encodes SC2 game state into kernel observation vectors.

    Produces a 16-dimensional vector from game state:
      - 8 dimensions: Economy (minerals, gas, supply, workers, bases, etc.)
      - 5 dimensions: Military (army, position, threat, control)
      - 3 dimensions: Information (scouting, uncertainty)
    """

    DIMENSION = 16

    def __init__(self):
        self.economy = EconomyState()
        self.military = MilitaryState()
        self.information = InformationState()
        self._history: List[np.ndarray] = []

    def encode(self, game_observation: Any = None) -> np.ndarray:
        """Encode current game state into 16D vector.

        If game_observation is None, returns current cached state.
        """
        if game_observation is not None:
            self._update_from_observation(game_observation)

        vec = np.array(
            self.economy.to_vector()[:8] +
            self.military.to_vector()[:5] +
            self.information.to_vector()[:3],
            dtype=np.float64
        )

        self._history.append(vec.copy())
        if len(self._history) > 100:
            self._history.pop(0)

        return vec

    def _update_from_observation(self, obs: Any):
        """Extract state from raw SC2 observation."""
        try:
            # Economy
            self.economy.minerals = obs.observation.player_common.minerals
            self.economy.vespene = obs.observation.player_common.vespene
            self.economy.supply_used = obs.observation.player_common.food_used
            self.economy.supply_cap = obs.observation.player_common.food_cap
            self.economy.workers = len(obs.observation.units)
            self.economy.bases = sum(1 for u in obs.observation.units
                                     if u.unit_type in [86, 130, 131])  # Command Centers
            self.economy.production_buildings = sum(1 for u in obs.observation.units
                                                    if u.unit_type in [21, 22, 23, 39, 62])  # Production
            self.economy.army_value = sum(u.health + u.shield for u in obs.observation.units
                                          if u.unit_type not in [86, 45, 130, 131])  # Non-worker/base

            # Military
            self.military.army_count = sum(1 for u in obs.observation.units
                                           if u.unit_type not in [86, 45, 130, 131, 104])
            self.military.threat_level = min(1.0, self.military.army_count / 50)

            # Information
            self.information.uncertainty = max(0.0, 1.0 - self.information.enemy_known_ratio)

        except (AttributeError, TypeError):
            pass

    def get_velocity(self) -> float:
        """Get rate of change of observation."""
        if len(self._history) < 2:
            return 0.0
        diff = self._history[-1] - self._history[-2]
        return float(np.linalg.norm(diff))

    def get_trend(self, window: int = 10) -> float:
        """Get trend direction over recent window."""
        if len(self._history) < window:
            return 0.0
        recent = self._history[-window:]
        return float(np.mean([np.linalg.norm(recent[i] - recent[i-1])
                             for i in range(1, len(recent))]))
