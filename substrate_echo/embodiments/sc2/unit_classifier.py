"""SC2 Unit Classifier — Full unit taxonomy for role-based decision making.

Classifies units across six dimensions:
  1. Movement:   ground, air
  2. Type:       infantry, vehicle, structure
  3. Combat:     melee, ranged, spell, none
  4. Attack:     ground_vs_ground, ground_vs_air, air_vs_ground, air_vs_air
  5. Behavior:   attack, defend, harvest, build, transport, supply
  6. Role:       army, scout, support, economy (multi-label)

Usage:
    classifier = UnitClassifier()
    info = classifier.classify(unit)
    # UnitInfo(movement='ground', combat='ranged', roles={'army','scout'}, ...)

    army = classifier.filter_by_role(units, 'army')
    anti_air = classifier.filter_by_attack(units, 'ground_vs_air')
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Dict, Optional, Any, List
from enum import Enum


class Movement(Enum):
    GROUND = "ground"
    AIR = "air"


class TerrainTraversal(Enum):
    """How a ground unit interacts with cliffs and terrain obstacles.

    CLIFF_NONE:   Cannot cross cliffs (most ground units)
    CLIFF_JUMP:   Can jump up/down cliffs (Reaper, Zealot with charge)
    CLIFF_WALK:   Walks along cliff edges, not blocked by terrain (Colossus)
    AIR:          Ignores all terrain (flying units)
    BURROW:       Can move underground, ignoring surface terrain (Zerg burrow)
    """
    NONE = "none"
    CLIFF_JUMP = "cliff_jump"
    CLIFF_WALK = "cliff_walk"
    AIR = "air"
    BURROW = "burrow"


class UnitType(Enum):
    INFANTRY = "infantry"
    VEHICLE = "vehicle"
    STRUCTURE = "structure"


class CombatClass(Enum):
    MELEE = "melee"
    RANGED = "ranged"
    SPELL = "spell"
    NONE = "none"


class AttackCapability(Enum):
    GVG = "ground_vs_ground"
    GVA = "ground_vs_air"
    AVG = "air_vs_ground"
    AVA = "air_vs_air"


class Behavior(Enum):
    ATTACK = "attack"
    DEFEND = "defend"
    HARVEST = "harvest"
    BUILD = "build"
    TRANSPORT = "transport"
    SUPPLY = "supply"
    CAST = "cast"
    SCOUT = "scout"


class Role(Enum):
    ARMY = "army"
    SCOUT = "scout"
    SUPPORT = "support"
    ECONOMY = "economy"


@dataclass
class UnitInfo:
    """Full classification of a unit type."""
    name: str
    movement: Movement
    unit_type: UnitType
    combat: CombatClass
    attack_caps: Set[AttackCapability] = field(default_factory=set)
    behaviors: Set[Behavior] = field(default_factory=set)
    roles: Set[Role] = field(default_factory=set)
    cost_minerals: int = 0
    cost_gas: int = 0
    supply: int = 0
    tags: Set[str] = field(default_factory=set)  # misc keywords
    terrain_traversal: TerrainTraversal = TerrainTraversal.NONE


# ── Full SC2 Unit Database ──────────────────────────────────────

# Zerg
_ZERG: Dict[str, UnitInfo] = {
    "DRONE":       UnitInfo("Drone",       Movement.GROUND, UnitType.INFANTRY, CombatClass.NONE,
                            behaviors={Behavior.HARVEST, Behavior.BUILD}, roles={Role.ECONOMY},
                            cost_minerals=50, supply=1),
    "ZERGLING":    UnitInfo("Zergling",    Movement.GROUND, UnitType.INFANTRY, CombatClass.MELEE,
                            attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK},
                            roles={Role.ARMY}, cost_minerals=50, supply=1, tags={"light"}),
    "BANELING":    UnitInfo("Baneling",    Movement.GROUND, UnitType.INFANTRY, CombatClass.MELEE,
                            attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK},
                            roles={Role.ARMY}, cost_minerals=50, cost_gas=25, supply=1,
                            tags={"light", "splash"}),
    "ROACH":       UnitInfo("Roach",       Movement.GROUND, UnitType.INFANTRY, CombatClass.RANGED,
                            attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK, Behavior.DEFEND},
                            roles={Role.ARMY}, cost_minerals=75, cost_gas=25, supply=2,
                            tags={"armored"}),
    "RAVAGER":     UnitInfo("Ravager",     Movement.GROUND, UnitType.INFANTRY, CombatClass.SPELL,
                            attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK, Behavior.CAST},
                            roles={Role.ARMY}, cost_minerals=100, cost_gas=100, supply=3),
    "HYDRALISK":   UnitInfo("Hydralisk",   Movement.GROUND, UnitType.INFANTRY, CombatClass.RANGED,
                            attack_caps={AttackCapability.GVG, AttackCapability.GVA},
                            behaviors={Behavior.ATTACK}, roles={Role.ARMY},
                            cost_minerals=100, cost_gas=50, supply=2),
    "LURKER":      UnitInfo("Lurker",      Movement.GROUND, UnitType.INFANTRY, CombatClass.RANGED,
                            attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK, Behavior.DEFEND},
                            roles={Role.ARMY}, cost_minerals=150, cost_gas=150, supply=3),
    "Mutalisk":    UnitInfo("Mutalisk",    Movement.AIR,    UnitType.INFANTRY, CombatClass.RANGED,
                            attack_caps={AttackCapability.AVG, AttackCapability.AVA},
                            behaviors={Behavior.ATTACK}, roles={Role.ARMY, Role.SCOUT},
                            cost_minerals=100, cost_gas=100, supply=2,
                            terrain_traversal=TerrainTraversal.AIR),
    "CORRUPTOR":   UnitInfo("Corruptor",   Movement.AIR,    UnitType.INFANTRY, CombatClass.RANGED,
                            attack_caps={AttackCapability.AVA, AttackCapability.AVG},
                            behaviors={Behavior.ATTACK}, roles={Role.ARMY},
                            cost_minerals=150, cost_gas=100, supply=2, tags={"armored"},
                            terrain_traversal=TerrainTraversal.AIR),
    "ULTRALISK":   UnitInfo("Ultralisk",   Movement.GROUND, UnitType.VEHICLE, CombatClass.MELEE,
                            attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK},
                            roles={Role.ARMY}, cost_minerals=300, cost_gas=200, supply=6,
                            tags={"massive", "armored"}),
    "VIPER":       UnitInfo("Viper",       Movement.AIR,    UnitType.INFANTRY, CombatClass.SPELL,
                            behaviors={Behavior.CAST}, roles={Role.SUPPORT},
                            cost_minerals=100, cost_gas=200, supply=2,
                            terrain_traversal=TerrainTraversal.AIR),
    "OVERLORD":    UnitInfo("Overlord",    Movement.AIR,    UnitType.VEHICLE, CombatClass.NONE,
                            behaviors={Behavior.SUPPLY}, roles={Role.SUPPORT},
                            cost_minerals=100, supply=0, tags={"supply"},
                            terrain_traversal=TerrainTraversal.AIR),
    "OVERSEER":    UnitInfo("Overseer",    Movement.AIR,    UnitType.VEHICLE, CombatClass.NONE,
                            behaviors={Behavior.SUPPLY, Behavior.CAST}, roles={Role.SCOUT, Role.SUPPORT},
                            cost_minerals=50, cost_gas=50, supply=0,
                            terrain_traversal=TerrainTraversal.AIR),
    "QUEEN":       UnitInfo("Queen",       Movement.GROUND, UnitType.INFANTRY, CombatClass.RANGED,
                            attack_caps={AttackCapability.GVG, AttackCapability.GVA},
                            behaviors={Behavior.DEFEND, Behavior.CAST}, roles={Role.SUPPORT, Role.ARMY},
                            cost_minerals=150, supply=2),
    "LOCUST":      UnitInfo("Locust",      Movement.GROUND, UnitType.INFANTRY, CombatClass.RANGED,
                            attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK},
                            roles={Role.ARMY}, tags={"spawned"}),
    "BROODLING":   UnitInfo("Broodling",   Movement.GROUND, UnitType.INFANTRY, CombatClass.MELEE,
                            attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK},
                            roles={Role.ARMY}, tags={"spawned"}),
}

# Terran
_TERRAN: Dict[str, UnitInfo] = {
    "SCV":              UnitInfo("SCV",              Movement.GROUND, UnitType.INFANTRY, CombatClass.NONE,
                                 behaviors={Behavior.HARVEST, Behavior.BUILD}, roles={Role.ECONOMY},
                                 cost_minerals=50, supply=1),
    "MARINE":           UnitInfo("Marine",           Movement.GROUND, UnitType.INFANTRY, CombatClass.RANGED,
                                 attack_caps={AttackCapability.GVG, AttackCapability.GVA},
                                 behaviors={Behavior.ATTACK}, roles={Role.ARMY},
                                 cost_minerals=50, supply=1, tags={"light"}),
    "MARAUDER":         UnitInfo("Marauder",         Movement.GROUND, UnitType.INFANTRY, CombatClass.RANGED,
                                 attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK},
                                 roles={Role.ARMY}, cost_minerals=125, cost_gas=50, supply=2,
                                 tags={"armored", "light"}),
    "REAPER":           UnitInfo("Reaper",           Movement.GROUND, UnitType.INFANTRY, CombatClass.RANGED,
                                 attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK, Behavior.SCOUT},
                                 roles={Role.ARMY, Role.SCOUT}, cost_minerals=50, cost_gas=50, supply=1,
                                 terrain_traversal=TerrainTraversal.CLIFF_JUMP),
    "HELLION":          UnitInfo("Hellion",          Movement.GROUND, UnitType.VEHICLE, CombatClass.RANGED,
                                 attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK},
                                 roles={Role.ARMY}, cost_minerals=100, supply=2, tags={"light", "splash"}),
    "WIDOWMINE":        UnitInfo("Widowmine",        Movement.GROUND, UnitType.VEHICLE, CombatClass.RANGED,
                                 attack_caps={AttackCapability.GVG, AttackCapability.GVA},
                                 behaviors={Behavior.ATTACK, Behavior.DEFEND}, roles={Role.ARMY},
                                 cost_minerals=75, cost_gas=25, supply=2),
    "Cyclone":          UnitInfo("Cyclone",          Movement.GROUND, UnitType.VEHICLE, CombatClass.RANGED,
                                 attack_caps={AttackCapability.GVG, AttackCapability.GVA},
                                 behaviors={Behavior.ATTACK}, roles={Role.ARMY},
                                 cost_minerals=150, cost_gas=100, supply=3),
    "TANK":             UnitInfo("SiegeTank",        Movement.GROUND, UnitType.VEHICLE, CombatClass.RANGED,
                                 attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK, Behavior.DEFEND},
                                 roles={Role.ARMY}, cost_minerals=150, cost_gas=125, supply=3,
                                 tags={"armored"}),
    "THOR":             UnitInfo("Thor",             Movement.GROUND, UnitType.VEHICLE, CombatClass.RANGED,
                                 attack_caps={AttackCapability.GVG, AttackCapability.GVA},
                                 behaviors={Behavior.ATTACK}, roles={Role.ARMY},
                                 cost_minerals=300, cost_gas=200, supply=6, tags={"massive", "armored"}),
    "VIKING_F":         UnitInfo("Viking (air)",     Movement.AIR,    UnitType.VEHICLE, CombatClass.RANGED,
                                 attack_caps={AttackCapability.AVA, AttackCapability.AVG},
                                 behaviors={Behavior.ATTACK}, roles={Role.ARMY},
                                 cost_minerals=150, cost_gas=75, supply=2,
                                 terrain_traversal=TerrainTraversal.AIR),
    "VIKING_G":         UnitInfo("Viking (ground)",  Movement.GROUND, UnitType.VEHICLE, CombatClass.RANGED,
                                 attack_caps={AttackCapability.GVG},
                                 behaviors={Behavior.ATTACK}, roles={Role.ARMY},
                                 cost_minerals=150, cost_gas=75, supply=2),
    "MEDIVAC":          UnitInfo("Medivac",          Movement.AIR,    UnitType.VEHICLE, CombatClass.NONE,
                                 behaviors={Behavior.TRANSPORT}, roles={Role.SUPPORT},
                                 cost_minerals=100, cost_gas=100, supply=2,
                                 terrain_traversal=TerrainTraversal.AIR),
    "LIBERATOR":        UnitInfo("Liberator",        Movement.AIR,    UnitType.VEHICLE, CombatClass.RANGED,
                                 attack_caps={AttackCapability.AVG, AttackCapability.AVA},
                                 behaviors={Behavior.ATTACK}, roles={Role.ARMY},
                                 cost_minerals=150, cost_gas=150, supply=3,
                                 terrain_traversal=TerrainTraversal.AIR),
    "BATTLECRUISER":    UnitInfo("Battlecruiser",    Movement.AIR,    UnitType.VEHICLE, CombatClass.RANGED,
                                 attack_caps={AttackCapability.AVG, AttackCapability.AVA, AttackCapability.GVG},
                                 behaviors={Behavior.ATTACK, Behavior.CAST}, roles={Role.ARMY},
                                 cost_minerals=400, cost_gas=300, supply=6, tags={"massive", "armored"},
                                 terrain_traversal=TerrainTraversal.AIR),
    "GHOST":            UnitInfo("Ghost",            Movement.GROUND, UnitType.INFANTRY, CombatClass.SPELL,
                                 attack_caps={AttackCapability.GVG, AttackCapability.GVA},
                                 behaviors={Behavior.ATTACK, Behavior.CAST}, roles={Role.ARMY, Role.SUPPORT},
                                 cost_minerals=150, cost_gas=150, supply=2),
    "SIEGE_TANK":       UnitInfo("SiegeTank",        Movement.GROUND, UnitType.VEHICLE, CombatClass.RANGED,
                                 attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK, Behavior.DEFEND},
                                 roles={Role.ARMY}, cost_minerals=150, cost_gas=125, supply=3),
}

# Protoss
_PROTOSS: Dict[str, UnitInfo] = {
    "PROBE":        UnitInfo("Probe",        Movement.GROUND, UnitType.VEHICLE, CombatClass.NONE,
                             behaviors={Behavior.HARVEST, Behavior.BUILD}, roles={Role.ECONOMY},
                             cost_minerals=50, supply=1),
    "ZEALOT":       UnitInfo("Zealot",       Movement.GROUND, UnitType.INFANTRY, CombatClass.MELEE,
                             attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK},
                             roles={Role.ARMY}, cost_minerals=100, supply=2, tags={"light"}),
    "STALKER":      UnitInfo("Stalker",      Movement.GROUND, UnitType.VEHICLE, CombatClass.RANGED,
                             attack_caps={AttackCapability.GVG, AttackCapability.GVA},
                             behaviors={Behavior.ATTACK}, roles={Role.ARMY},
                             cost_minerals=125, cost_gas=50, supply=2, tags={"armored"},
                             terrain_traversal=TerrainTraversal.CLIFF_JUMP),
    "SENTRY":       UnitInfo("Sentry",       Movement.GROUND, UnitType.VEHICLE, CombatClass.SPELL,
                             attack_caps={AttackCapability.GVG}, behaviors={Behavior.CAST},
                             roles={Role.SUPPORT}, cost_minerals=50, cost_gas=100, supply=2),
    "Adept":        UnitInfo("Adept",        Movement.GROUND, UnitType.INFANTRY, CombatClass.RANGED,
                             attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK},
                             roles={Role.ARMY}, cost_minerals=100, cost_gas=25, supply=2),
    "OBSERVER":     UnitInfo("Observer",     Movement.AIR,    UnitType.VEHICLE, CombatClass.NONE,
                             behaviors={Behavior.CAST}, roles={Role.SCOUT, Role.SUPPORT},
                             cost_minerals=25, cost_gas=75, supply=1,
                             terrain_traversal=TerrainTraversal.AIR),
    "IMMORTAL":     UnitInfo("Immortal",     Movement.GROUND, UnitType.VEHICLE, CombatClass.RANGED,
                             attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK},
                             roles={Role.ARMY}, cost_minerals=250, cost_gas=100, supply=4,
                             tags={"armored"}),
    "COLOSSUS":     UnitInfo("Colossus",     Movement.GROUND, UnitType.VEHICLE, CombatClass.RANGED,
                             attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK},
                             roles={Role.ARMY}, cost_minerals=300, cost_gas=200, supply=6,
                             tags={"massive", "armored", "splash"},
                             terrain_traversal=TerrainTraversal.CLIFF_WALK),
    "DISRUPTOR":    UnitInfo("Disruptor",    Movement.GROUND, UnitType.VEHICLE, CombatClass.SPELL,
                             attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK, Behavior.CAST},
                             roles={Role.ARMY}, cost_minerals=150, cost_gas=150, supply=3),
    "PHOENIX":      UnitInfo("Phoenix",      Movement.AIR,    UnitType.VEHICLE, CombatClass.RANGED,
                             attack_caps={AttackCapability.AVA, AttackCapability.AVG},
                             behaviors={Behavior.ATTACK}, roles={Role.ARMY, Role.SCOUT},
                             cost_minerals=150, cost_gas=100, supply=2,
                             terrain_traversal=TerrainTraversal.AIR),
    "VOID_RAY":     UnitInfo("VoidRay",      Movement.AIR,    UnitType.VEHICLE, CombatClass.RANGED,
                             attack_caps={AttackCapability.AVA, AttackCapability.AVG},
                             behaviors={Behavior.ATTACK}, roles={Role.ARMY},
                             cost_minerals=250, cost_gas=150, supply=4, tags={"armored"},
                             terrain_traversal=TerrainTraversal.AIR),
    "ORACLE":       UnitInfo("Oracle",       Movement.AIR,    UnitType.VEHICLE, CombatClass.SPELL,
                             attack_caps={AttackCapability.AVG}, behaviors={Behavior.CAST, Behavior.ATTACK},
                             roles={Role.SUPPORT, Role.SCOUT}, cost_minerals=150, cost_gas=150, supply=3,
                             terrain_traversal=TerrainTraversal.AIR),
    "TEMPEST":      UnitInfo("Tempest",      Movement.AIR,    UnitType.VEHICLE, CombatClass.RANGED,
                             attack_caps={AttackCapability.AVA, AttackCapability.AVG},
                             behaviors={Behavior.ATTACK}, roles={Role.ARMY},
                             cost_minerals=300, cost_gas=250, supply=5, tags={"massive", "armored"},
                             terrain_traversal=TerrainTraversal.AIR),
    "CARRIER":      UnitInfo("Carrier",      Movement.AIR,    UnitType.VEHICLE, CombatClass.RANGED,
                             attack_caps={AttackCapability.AVA, AttackCapability.AVG},
                             behaviors={Behavior.ATTACK}, roles={Role.ARMY},
                             cost_minerals=350, cost_gas=250, supply=6, tags={"massive", "armored"},
                             terrain_traversal=TerrainTraversal.AIR),
    "HIGH_TEMPLAR": UnitInfo("HighTemplar",  Movement.GROUND, UnitType.INFANTRY, CombatClass.SPELL,
                             behaviors={Behavior.CAST}, roles={Role.SUPPORT},
                             cost_minerals=50, cost_gas=150, supply=2),
    "DARK_TEMPLAR": UnitInfo("DarkTemplar",  Movement.GROUND, UnitType.INFANTRY, CombatClass.MELEE,
                             attack_caps={AttackCapability.GVG}, behaviors={Behavior.ATTACK},
                             roles={Role.ARMY}, cost_minerals=125, cost_gas=125, supply=2),
    "ARCHON":       UnitInfo("Archon",       Movement.GROUND, UnitType.INFANTRY, CombatClass.RANGED,
                             attack_caps={AttackCapability.GVG, AttackCapability.GVA},
                             behaviors={Behavior.ATTACK}, roles={Role.ARMY},
                             cost_minerals=0, cost_gas=0, supply=4, tags={"psionic"}),
}

# Merge all
_ALL_UNITS: Dict[str, UnitInfo] = {}
_ALL_UNITS.update(_ZERG)
_ALL_UNITS.update(_TERRAN)
_ALL_UNITS.update(_PROTOSS)


class UnitClassifier:
    """Classifies SC2 units by movement, combat, attack, behavior, role.

    Uses a hybrid approach:
      1. Known-unit lookup table (fast, covers ~80 unit types)
      2. Fallback heuristics from SC2 unit properties (covers custom/new units)
    """

    def __init__(self):
        self._database = dict(_ALL_UNITS)
        # Build uppercase index for case-insensitive lookup
        self._upper_index = {k.upper(): v for k, v in self._database.items()}

    def classify(self, unit: Any) -> Optional[UnitInfo]:
        """Classify a unit by its SC2 API object.

        Returns UnitInfo if known, or constructs one from properties.
        """
        # Try exact name match first
        name = getattr(unit, 'name', '').upper().replace(" ", "").replace("'", "")
        if name in self._upper_index:
            return self._upper_index[name]

        # Try type_id name
        type_name = ""
        try:
            type_name = unit.type_id.name.upper().replace(" ", "")
        except (AttributeError, TypeError):
            pass
        if type_name in self._upper_index:
            return self._upper_index[type_name]

        # Fallback: build from SC2 unit properties
        return self._classify_from_properties(unit)

    def _classify_from_properties(self, unit: Any) -> UnitInfo:
        """Build UnitInfo from SC2 unit properties (unknown unit type)."""
        is_flying = getattr(unit, 'is_flying', False)
        is_structure = getattr(unit, 'is_structure', False)
        can_attack = getattr(unit, 'can_attack', False)
        ground_range = getattr(unit, 'ground_range', 0)
        air_range = getattr(unit, 'air_range', 0)
        name = getattr(unit, 'name', 'unknown')

        movement = Movement.AIR if is_flying else Movement.GROUND
        unit_type = UnitType.STRUCTURE if is_structure else UnitType.INFANTRY

        if not can_attack:
            combat = CombatClass.NONE
        elif ground_range <= 1.5 and air_range <= 1.5:
            combat = CombatClass.MELEE
        else:
            combat = CombatClass.RANGED

        attack_caps = set()
        if can_attack:
            if ground_range > 0:
                attack_caps.add(AttackCapability.GVG)
            if air_range > 0:
                attack_caps.add(AttackCapability.GVA)

        roles = {Role.ARMY} if can_attack else set()
        if is_structure:
            roles = set()

        return UnitInfo(
            name=name, movement=movement, unit_type=unit_type,
            combat=combat, attack_caps=attack_caps, roles=roles,
            terrain_traversal=TerrainTraversal.AIR if is_flying else TerrainTraversal.NONE,
        )

    def filter_by_role(self, units: Any, role: Role) -> List[Any]:
        """Filter units that have a given role."""
        result = []
        for unit in units:
            info = self.classify(unit)
            if info and role in info.roles:
                result.append(unit)
        return result

    def filter_by_attack(self, units: Any, cap: AttackCapability) -> List[Any]:
        """Filter units with a specific attack capability."""
        result = []
        for unit in units:
            info = self.classify(unit)
            if info and cap in info.attack_caps:
                result.append(unit)
        return result

    def filter_by_movement(self, units: Any, movement: Movement) -> List[Any]:
        """Filter units by movement type."""
        result = []
        for unit in units:
            info = self.classify(unit)
            if info and info.movement == movement:
                result.append(unit)
        return result

    def filter_by_combat(self, units: Any, combat: CombatClass) -> List[Any]:
        """Filter units by combat class."""
        result = []
        for unit in units:
            info = self.classify(unit)
            if info and info.combat == combat:
                result.append(unit)
        return result

    def count_by_role(self, units: Any) -> Dict[Role, int]:
        """Count units per role."""
        counts = {r: 0 for r in Role}
        for unit in units:
            info = self.classify(unit)
            if info:
                for role in info.roles:
                    counts[role] += 1
        return counts

    def count_by_movement(self, units: Any) -> Dict[Movement, int]:
        """Count ground vs air units."""
        counts = {Movement.GROUND: 0, Movement.AIR: 0}
        for unit in units:
            info = self.classify(unit)
            if info and info.movement in counts:
                counts[info.movement] += 1
        return counts

    def count_by_combat(self, units: Any) -> Dict[CombatClass, int]:
        """Count melee vs ranged vs spell units."""
        counts = {c: 0 for c in CombatClass}
        for unit in units:
            info = self.classify(unit)
            if info and info.combat in counts:
                counts[info.combat] += 1
        return counts

    def army_composition(self, units: Any) -> Dict[str, int]:
        """Get army composition by unit name."""
        comp = {}
        for unit in units:
            info = self.classify(unit)
            if info and Role.ARMY in info.roles:
                comp[info.name] = comp.get(info.name, 0) + 1
        return comp

    # ── Dual / Combined Attack Helpers ───────────────────────────

    @staticmethod
    def has_combined_attack(info: UnitInfo) -> bool:
        """True if unit can attack BOTH air and ground targets.

        Ground unit with GVG+GVA (e.g. Hydralisk, Marine, Stalker)
        or air unit with AVG+AVA (e.g. Mutalisk, Viking air, Phoenix).
        """
        caps = info.attack_caps
        return (
            (AttackCapability.GVG in caps and AttackCapability.GVA in caps)
            or (AttackCapability.AVG in caps and AttackCapability.AVA in caps)
        )

    def filter_dual_attack(self, units: Any) -> List[Any]:
        """Filter units that can attack both air and ground."""
        result = []
        for unit in units:
            info = self.classify(unit)
            if info and self.has_combined_attack(info):
                result.append(unit)
        return result

    def count_dual_attack(self, units: Any) -> int:
        """Count units with combined air+ground attack."""
        return len(self.filter_dual_attack(units))

    def count_anti_air(self, units: Any) -> int:
        """Count units that can attack air (GVA or AVA capability)."""
        return len(self.filter_by_attack(units, AttackCapability.GVA)) + \
               len([u for u in units
                    if self.classify(u)
                    and AttackCapability.AVA in self.classify(u).attack_caps
                    and AttackCapability.GVA not in self.classify(u).attack_caps])

    def count_by_attack_mode(self, units: Any) -> Dict[str, int]:
        """Count units by attack mode category.

        Returns:
            {
                "ground_only":  units that only hit ground,
                "air_only":     units that only hit air,
                "dual":         units that hit both air and ground,
                "no_attack":    units with no attack,
            }
        """
        counts = {"ground_only": 0, "air_only": 0, "dual": 0, "no_attack": 0}
        for unit in units:
            info = self.classify(unit)
            if not info:
                counts["no_attack"] += 1
                continue
            caps = info.attack_caps
            has_gvg = AttackCapability.GVG in caps
            has_gva = AttackCapability.GVA in caps
            has_avg = AttackCapability.AVG in caps
            has_ava = AttackCapability.AVA in caps

            can_hit_air = has_gva or has_ava
            can_hit_ground = has_gvg or has_avg

            if can_hit_air and can_hit_ground:
                counts["dual"] += 1
            elif can_hit_air:
                counts["air_only"] += 1
            elif can_hit_ground:
                counts["ground_only"] += 1
            else:
                counts["no_attack"] += 1
        return counts

    # ── Terrain Traversal Helpers ──────────────────────────────────

    def filter_by_traversal(self, units: Any,
                            traversal: TerrainTraversal) -> List[Any]:
        """Filter units with a specific terrain traversal capability."""
        result = []
        for unit in units:
            info = self.classify(unit)
            if info and info.terrain_traversal == traversal:
                result.append(unit)
        return result

    def has_cliff_traversal(self, info: UnitInfo) -> bool:
        """True if unit can traverse cliffs (jump or walk)."""
        return info.terrain_traversal in (
            TerrainTraversal.CLIFF_JUMP, TerrainTraversal.CLIFF_WALK)

    def filter_cliff_traversable(self, units: Any) -> List[Any]:
        """Filter ground units that can cross cliffs."""
        result = []
        for unit in units:
            info = self.classify(unit)
            if info and self.has_cliff_traversal(info):
                result.append(unit)
        return result

    def count_by_traversal(self, units: Any) -> Dict[str, int]:
        """Count units by terrain traversal category."""
        counts = {"none": 0, "cliff_jump": 0, "cliff_walk": 0,
                  "air": 0, "burrow": 0}
        for unit in units:
            info = self.classify(unit)
            if info:
                key = info.terrain_traversal.value
                counts[key] = counts.get(key, 0) + 1
            else:
                counts["none"] += 1
        return counts
