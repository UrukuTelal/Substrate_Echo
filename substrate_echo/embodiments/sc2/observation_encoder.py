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

    Two encoding paths:
      - encode(game_observation): raw SC2 GameState (fog-of-war limited)
      - encode_from_botai(bot): BotAI persistent knowledge (correct counts)

    State tracking for layered telemetry:
      - encode() records raw_game → parsed → normalized at each tick
      - get_state_trace() returns the full trace for debugging
    """

    DIMENSION = 16

    # SC2 Terran type IDs
    SCV = 45
    COMMAND_CENTER = 18
    ORBITAL_COMMAND = 132
    PLANETARY_FORTRESS = 130
    SUPPLY_DEPOT = 19
    BARRACKS = 21
    FACTORY = 27
    STARPORT = 28
    MARINE = 48
    MULE = 268

    BASE_TYPES = {COMMAND_CENTER, ORBITAL_COMMAND, PLANETARY_FORTRESS}
    PRODUCTION_TYPES = {BARRACKS, FACTORY, STARPORT}
    WORKER_TYPES = {SCV}
    ARMY_EXCLUDE = {SCV, MULE, COMMAND_CENTER, ORBITAL_COMMAND, PLANETARY_FORTRESS}

    def __init__(self):
        self.economy = EconomyState()
        self.military = MilitaryState()
        self.information = InformationState()
        self._history: List[np.ndarray] = []
        self._state_trace: List[Dict[str, Any]] = []

    def encode(self, game_observation: Any = None) -> np.ndarray:
        """Encode current game state into 16D vector.

        Uses raw SC2 GameState. Fog-of-war limited — only visible units.
        For persistent unit knowledge, use encode_from_botai().
        """
        if game_observation is not None:
            self._update_from_observation(game_observation)

        vec = self._build_vector()
        self._history.append(vec.copy())
        if len(self._history) > 100:
            self._history.pop(0)
        return vec

    def encode_from_botai(self, bot: Any) -> np.ndarray:
        """Encode using BotAI persistent knowledge (fog-of-war independent).

        This is the preferred encoding path for own-unit counts.
        BotAI.units maintains all known units regardless of vision.
        """
        self.economy.minerals = bot.minerals
        self.economy.vespene = bot.vespene
        self.economy.supply_used = bot.supply_used
        self.economy.supply_cap = bot.supply_cap

        own_units = bot.units
        self.economy.workers = len(own_units.of_type(UnitTypeId.SCV))
        self.economy.bases = len(own_units.of_type(UnitTypeId.COMMANDCENTER))
        self.economy.production_buildings = (
            len(own_units.of_type(UnitTypeId.BARRACKS))
            + len(own_units.of_type(UnitTypeId.FACTORY))
            + len(own_units.of_type(UnitTypeId.STARPORT))
        )
        self.economy.army_value = sum(
            u.health + u.shield for u in own_units
            if u.type_id not in self.ARMY_EXCLUDE
        )

        self.military.army_count = len(
            own_units.exclude_type(UnitTypeId.SCV)
        )
        self.military.threat_level = min(1.0, self.military.army_count / 50)

        # Information from game state
        self.information.uncertainty = max(
            0.0, 1.0 - self.information.enemy_known_ratio
        )

        vec = self._build_vector()
        self._history.append(vec.copy())
        if len(self._history) > 100:
            self._history.pop(0)
        return vec

    def record_state_trace(self, tick: int, source: str = "unknown"):
        """Record current encoder state for layered telemetry."""
        self._state_trace.append({
            "tick": tick,
            "source": source,
            "economy": {
                "minerals": self.economy.minerals,
                "vespene": self.economy.vespene,
                "supply_used": self.economy.supply_used,
                "supply_cap": self.economy.supply_cap,
                "workers": self.economy.workers,
                "bases": self.economy.bases,
                "production_buildings": self.economy.production_buildings,
                "army_value": self.economy.army_value,
            },
            "military": {
                "army_count": self.military.army_count,
                "threat_level": self.military.threat_level,
                "map_control": self.military.map_control,
            },
        })

    def get_state_trace(self) -> List[Dict[str, Any]]:
        return list(self._state_trace)

    def _build_vector(self) -> np.ndarray:
        return np.array(
            self.economy.to_vector()[:8] +
            self.military.to_vector()[:5] +
            self.information.to_vector()[:3],
            dtype=np.float64
        )

    def _update_from_observation(self, obs: Any):
        """Extract state from raw SC2 GameState (fog-of-war limited)."""
        try:
            self.economy.minerals = obs.observation.player_common.minerals
            self.economy.vespene = obs.observation.player_common.vespene
            self.economy.supply_used = obs.observation.player_common.food_used
            self.economy.supply_cap = obs.observation.player_common.food_cap
            self.economy.workers = sum(
                1 for u in obs.observation.units
                if u.unit_type == self.SCV
            )
            self.economy.bases = sum(
                1 for u in obs.observation.units
                if u.unit_type in self.BASE_TYPES
            )
            self.economy.production_buildings = sum(
                1 for u in obs.observation.units
                if u.unit_type in self.PRODUCTION_TYPES
            )
            self.economy.army_value = sum(
                u.health + u.shield for u in obs.observation.units
                if u.unit_type not in (self.ARMY_EXCLUDE | self.BASE_TYPES)
            )

            self.military.army_count = sum(
                1 for u in obs.observation.units
                if u.unit_type not in (self.ARMY_EXCLUDE | self.BASE_TYPES)
            )
            self.military.threat_level = min(1.0, self.military.army_count / 50)

            self.information.uncertainty = max(
                0.0, 1.0 - self.information.enemy_known_ratio
            )

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
