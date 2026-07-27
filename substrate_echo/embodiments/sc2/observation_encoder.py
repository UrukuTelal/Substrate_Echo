"""SC2 Observation Encoder — Translates game state to kernel vectors.

Encodes SC2 observations into 16-dimensional vectors that the
Substrate Kernel can process.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sc2.ids.unit_typeid import UnitTypeId

from substrate_echo.embodiments.sc2.unit_classifier import (
    UnitClassifier, Role, Movement, AttackCapability,
)
from substrate_echo.embodiments.sc2.building_upgrade_classifier import (
    BuildingClassifier, BuildingCategory,
)


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
    """Information/uncertainty observation components.

    Encoded dims (first3 of to_vector):
      [0] map_revealed:       fraction of walkable map currently visible
      [1] terrain_complexity: fraction of pathing grid that is blocked
      [2] cliff_density:      fraction of cells adjacent to height discontinuity

    Additional fields stored as metadata (not in 16D vector):
      enemy_known_ratio, visibility_advantage, enemy_bases_known,
      last_scout_time, uncertainty
    """
    scout_count: float = 0.0
    enemy_known_ratio: float = 0.0
    map_revealed: float = 0.0
    terrain_complexity: float = 0.0
    cliff_density: float = 0.0
    visibility_advantage: float = 0.5
    enemy_bases_known: float = 0.0
    last_scout_time: float = 0.0
    uncertainty: float = 0.5

    def to_vector(self) -> List[float]:
        return [
            self.map_revealed,
            self.terrain_complexity,
            self.cliff_density,
            self.enemy_known_ratio,
            self.visibility_advantage,
            min(1.0, self.enemy_bases_known / 5),
            min(1.0, self.scout_count / 10),
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
        self._unit_classifier = UnitClassifier()
        self._building_classifier = BuildingClassifier()

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
        """Encode using BotAI persistent knowledge + terrain awareness.

        Race-agnostic: uses UnitClassifier + BuildingClassifier to count
        workers, bases, production buildings, and air/ground composition
        regardless of which race the bot is playing.

        Terrain: reads state.visibility, state.pathing_grid, state.terrain_height
        to compute map coverage, terrain complexity, and cliff density.
        """
        self.economy.minerals = bot.minerals
        self.economy.vespene = bot.vespene
        self.economy.supply_used = bot.supply_used
        self.economy.supply_cap = bot.supply_cap

        own_units = bot.units
        own_structures = bot.units.structure

        # ── Workers (race-agnostic) ──
        worker_count = len(self._unit_classifier.filter_by_role(own_units, Role.ECONOMY))
        self.economy.workers = worker_count

        # ── Bases (race-agnostic via BuildingClassifier) ──
        base_count = 0
        for s in own_structures:
            info = self._building_classifier.classify(s)
            if info and info.category == BuildingCategory.PRODUCTION and info.name in (
                "Hatchery", "Lair", "Hive",
                "CommandCenter", "OrbitalCommand", "PlanetaryFortress",
                "Nexus",
            ):
                base_count += 1
        self.economy.bases = base_count

        # ── Production buildings (race-agnostic) ──
        prod_count = 0
        for s in own_structures:
            info = self._building_classifier.classify(s)
            if info and info.category == BuildingCategory.PRODUCTION and info.name not in (
                "Hatchery", "Lair", "Hive",
                "CommandCenter", "OrbitalCommand", "PlanetaryFortress",
                "Nexus",
            ):
                prod_count += 1
        self.economy.production_buildings = prod_count

        # ── Army value (exclude workers, supply units, bases) ──
        army_value = 0.0
        for u in own_units:
            info = self._unit_classifier.classify(u)
            if info and Role.ARMY in info.roles:
                army_value += u.health + u.shield
            elif info and Role.SUPPORT in info.roles and Role.ECONOMY not in info.roles:
                army_value += u.health + u.shield
        self.economy.army_value = army_value

        # ── Military composition (race-agnostic) ──
        army_units = [u for u in own_units
                      if not u.is_structure
                      and u.can_attack]
        # Exclude workers and supply units from army count
        combat_units = []
        for u in army_units:
            info = self._unit_classifier.classify(u)
            if info and (Role.ARMY in info.roles or Role.SCOUT in info.roles):
                combat_units.append(u)

        self.military.army_count = len(combat_units)
        self.military.threat_level = min(1.0, len(combat_units) / 50)

        # ── Air / Ground / Anti-air breakdown ──
        air_units = self._unit_classifier.filter_by_movement(combat_units, Movement.AIR)
        ground_units = self._unit_classifier.filter_by_movement(combat_units, Movement.GROUND)
        self.military.air_count = len(air_units)
        self.military.ground_count = len(ground_units)

        # ── Map control: our visible tiles / total walkable tiles ──
        total_walkable = 1
        our_visible = 0
        try:
            pathing = bot.state.pathing_grid
            visibility = bot.state.visibility
            if pathing is not None and visibility is not None:
                pathing_arr = np.array(pathing, dtype=np.float32)
                visibility_arr = np.array(visibility, dtype=np.float32)
                total_walkable = max(1, int(np.sum(pathing_arr == 0)))
                # Cells where we have visibility AND are walkable
                our_visible = int(np.sum(
                    (visibility_arr > 0) & (pathing_arr == 0)))
                self.military.map_control = min(
                    1.0, our_visible / total_walkable)
        except (AttributeError, TypeError, ValueError):
            pass

        # ── Terrain awareness from SC2 API ──
        self._compute_terrain_metrics(bot)

        # ── Information from game state ──
        enemy_count = len([u for u in bot.known_enemy_units
                          if not u.is_structure])
        self.information.enemy_known_ratio = min(
            1.0, enemy_count / max(1, 20))
        self.information.enemy_bases_known = len(bot.known_enemy_structures)
        self.information.uncertainty = max(
            0.0, 1.0 - self.information.enemy_known_ratio)

        vec = self._build_vector()
        self._history.append(vec.copy())
        if len(self._history) > 100:
            self._history.pop(0)
        return vec

    def _compute_terrain_metrics(self, bot: Any):
        """Compute terrain complexity, cliff density, and visibility advantage.

        Uses bot.state.pathing_grid, bot.state.terrain_height, and
        bot.state.visibility — all available via BotAI.
        """
        try:
            pathing = bot.state.pathing_grid
            height = bot.state.terrain_height
            visibility = bot.state.visibility

            if pathing is None:
                return

            pathing_arr = np.array(pathing, dtype=np.float32)

            # Terrain complexity: fraction of walkable map that is blocked
            total_cells = pathing_arr.size
            if total_cells > 0:
                blocked = int(np.sum(pathing_arr != 0))
                self.information.terrain_complexity = blocked / total_cells

            # Cliff density: cells where height changes sharply
            if height is not None:
                height_arr = np.array(height, dtype=np.float32)
                # Height difference between adjacent cells
                dh_row = np.abs(np.diff(height_arr, axis=0))
                dh_col = np.abs(np.diff(height_arr, axis=1))
                # Cliff threshold: SC2 uses ~2.0 height units per cliff level
                cliff_threshold = 1.5
                cliff_cells = (
                    int(np.sum(dh_row > cliff_threshold))
                    + int(np.sum(dh_col > cliff_threshold))
                )
                # Normalize by total adjacent-cell pairs
                max_pairs = (height_arr.shape[0] - 1) * height_arr.shape[1] + \
                            height_arr.shape[0] * (height_arr.shape[1] - 1)
                if max_pairs > 0:
                    self.information.cliff_density = min(
                        1.0, cliff_cells / max_pairs)

            # Map revealed: fraction of walkable cells currently visible
            if visibility is not None:
                vis_arr = np.array(visibility, dtype=np.float32)
                walkable = pathing_arr == 0
                walkable_count = max(1, int(np.sum(walkable)))
                visible_walkable = int(np.sum((vis_arr > 0) & walkable))
                self.information.map_revealed = min(
                    1.0, visible_walkable / walkable_count)

                # Visibility advantage: our visible / (our + enemy estimated)
                # Estimate enemy visibility from known enemy unit positions
                enemy_vis_cells = 0
                for eu in bot.known_enemy_units:
                    try:
                        ex, ey = int(eu.position.x), int(eu.position.y)
                        if 0 <= ex < vis_arr.shape[0] and 0 <= ey < vis_arr.shape[1]:
                            # Approximate enemy sight radius (~10 cells)
                            r = 10
                            x0, x1 = max(0, ex-r), min(vis_arr.shape[0], ex+r)
                            y0, y1 = max(0, ey-r), min(vis_arr.shape[1], ey+r)
                            enemy_vis_cells += int(np.sum(
                                walkable[x0:x1, y0:y1]))
                    except (AttributeError, IndexError):
                        pass
                total_vis = visible_walkable + max(1, enemy_vis_cells)
                self.information.visibility_advantage = visible_walkable / total_vis

        except (AttributeError, TypeError, ValueError):
            pass

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
