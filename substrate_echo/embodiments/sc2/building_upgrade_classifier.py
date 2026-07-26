"""SC2 Building & Upgrade Classifier — Full structure/tech taxonomy.

Classifies buildings across:
  1. Race:       zerg, terran, protoss
  2. Category:   production, tech, economy, defense, supply, static_defense
  3. Tier:       basic, advanced, late
  4. Produces:   unit types, upgrades, other buildings
  5. Requires:   prerequisite buildings

Classifies upgrades across:
  1. Race:       zerg, terran, protoss
  2. Category:   attack, armor, speed, energy, ability, spell
  3. Tier:       tier1, tier2, tier3
  4. Applies_to: unit roles, unit combat classes
  5. Requires:   building prerequisites

Usage:
    bc = BuildingClassifier()
    info = bc.classify(building)
    # BuildingInfo(category='production', tier='basic', produces=['ZERGLING','BANELING'], ...)

    uc = UpgradeClassifier()
    info = uc.classify(upgrade)
    # UpgradeInfo(category='attack', applies_to={CombatClass.RANGED}, tier='tier1', ...)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, Dict, Optional, Any, List
from enum import Enum


# ── Building Enums ──────────────────────────────────────────────

class BuildingCategory(Enum):
    PRODUCTION = "production"       # trains units
    TECH = "tech"                   # unlocks tech tree / upgrades
    ECONOMY = "economy"             # resource gathering / income
    DEFENSE = "defense"             # walls, bunkers
    STATIC_DEFENSE = "static_defense"  # cannons, turrets, spines
    SUPPLY = "supply"               # increases supply cap
    SPECIAL = "special"             # misc (nydus, fleet beacon, etc)


class BuildingTier(Enum):
    BASIC = "basic"         # available from game start or after 1 structure
    ADVANCED = "advanced"   # requires 1+ tech structures
    LATE = "late"           # requires Lair/Hive, lair-tech, or hive-tech


# ── Upgrade Enums ───────────────────────────────────────────────

class UpgradeCategory(Enum):
    ATTACK = "attack"       # weapon damage
    ARMOR = "armor"         # unit armor / plating
    SPEED = "speed"         # movement speed
    ENERGY = "energy"       # caster energy / shield
    ABILITY = "ability"     # unlocks or enhances abilities
    SPELL = "spell"         # spell-specific upgrades
    SIEGE = "siege"         # siege mode / transform
    TRANSPORT = "transport" # drop / elevator


class UpgradeTier(Enum):
    TIER1 = "tier1"     # basic (spawning pool, engineering bay)
    TIER2 = "tier2"     # mid (lair, armory, twilight council)
    TIER3 = "tier3"     # late (hive, fusion core, fleet beacon)


# ── Building Data ───────────────────────────────────────────────

@dataclass
class BuildingInfo:
    """Full classification of a building type."""
    name: str
    race: str                    # "zerg", "terran", "protoss"
    category: BuildingCategory
    tier: BuildingTier
    produces: Set[str] = field(default_factory=set)   # unit type names it trains
    enables: Set[str] = field(default_factory=set)     # buildings/upgrades it unlocks
    requires: Set[str] = field(default_factory=set)     # prerequisite building names
    tags: Set[str] = field(default_factory=set)
    cost_minerals: int = 0
    cost_gas: int = 0
    build_time: float = 0.0


# ── Zerg Buildings ──────────────────────────────────────────────

_ZERG_BUILDINGS: Dict[str, BuildingInfo] = {
    "HATCHERY": BuildingInfo(
        "Hatchery", "zerg", BuildingCategory.PRODUCTION, BuildingTier.BASIC,
        produces={"DRONE", "OVERLORD", "QUEEN", "ZERGLING", "BANELING",
                  "ROACH", "RAVAGER", "HYDRALISK", "LURKER", "MUTALISK",
                  "CORRUPTOR", "ULTRALISK", "VIPER"},
        enables={"EXTRACTOR", "SPAWNINGPOOL", "EVOLUTIONCHAMBER"},
        cost_minerals=300, build_time=71),
    "LAIR": BuildingInfo(
        "Lair", "zerg", BuildingCategory.TECH, BuildingTier.ADVANCED,
        requires={"HATCHERY"},
        enables={"HYDRALISKDEN", "SPIRE", "INFESTATIONPIT", "NYDUSNETWORK"},
        cost_minerals=150, cost_gas=100, build_time=43),
    "HIVE": BuildingInfo(
        "Hive", "zerg", BuildingCategory.TECH, BuildingTier.LATE,
        requires={"LAIR"},
        enables={"GREATERSPIRE", "ULTRALISKCAVERN"},
        cost_minerals=200, cost_gas=150, build_time=71),
    "EXTRACTOR": BuildingInfo(
        "Extractor", "zerg", BuildingCategory.ECONOMY, BuildingTier.BASIC,
        cost_minerals=25, build_time=21),
    "SPAWNINGPOOL": BuildingInfo(
        "SpawningPool", "zerg", BuildingCategory.TECH, BuildingTier.BASIC,
        produces={"ZERGLING", "QUEEN"},
        enables={"BANELINGNEST", "ROACHWARREN", "EVOLUTIONCHAMBER"},
        tags={"melee_upgrades", "zergling_speed"},
        cost_minerals=200, build_time=46),
    "ROACHWARREN": BuildingInfo(
        "RoachWarren", "zerg", BuildingCategory.TECH, BuildingTier.BASIC,
        produces={"ROACH", "RAVAGER"},
        cost_minerals=150, cost_gas=0, build_time=39),
    "BANELINGNEST": BuildingInfo(
        "BanelingNest", "zerg", BuildingCategory.TECH, BuildingTier.BASIC,
        produces={"BANELING"},
        cost_minerals=100, cost_gas=50, build_time=43),
    "EVOLUTIONCHAMBER": BuildingInfo(
        "EvolutionChamber", "zerg", BuildingCategory.TECH, BuildingTier.BASIC,
        enables={"ZERGMELEEWEAPONS", "ZERGMISSILEWEAPONS", "ZERGGROUNDARMORS"},
        tags={"melee_upgrade", "ranged_upgrade", "armor_upgrade"},
        cost_minerals=75, build_time=32),
    "HYDRALISKDEN": BuildingInfo(
        "HydraliskDen", "zerg", BuildingCategory.TECH, BuildingTier.ADVANCED,
        requires={"LAIR"},
        produces={"HYDRALISK", "LURKER"},
        enables={"RESEARCH_HYDRALISKRANGE", "RESEARCH_HYDRALISKSPEED"},
        cost_minerals=100, cost_gas=100, build_time=29),
    "SPIRE": BuildingInfo(
        "Spire", "zerg", BuildingCategory.TECH, BuildingTier.ADVANCED,
        requires={"LAIR"},
        produces={"MUTALISK", "CORRUPTOR"},
        enables={"RESEARCH_ZERGFLYERWEAPONS", "RESEARCH_ZERGFLYERARMORS"},
        cost_minerals=200, cost_gas=200, build_time=65),
    "GREATERSPIRE": BuildingInfo(
        "GreaterSpire", "zerg", BuildingCategory.TECH, BuildingTier.LATE,
        requires={"SPIRE", "HIVE"},
        enables={"BROODLORD"},
        cost_minerals=100, cost_gas=150, build_time=65),
    "INFESTATIONPIT": BuildingInfo(
        "InfestationPit", "zerg", BuildingCategory.TECH, BuildingTier.ADVANCED,
        requires={"LAIR"},
        produces={"VIPER"},
        cost_minerals=100, cost_gas=100, build_time=32),
    "ULTRALISKCAVERN": BuildingInfo(
        "UltraliskCavern", "zerg", BuildingCategory.TECH, BuildingTier.LATE,
        requires={"HIVE"},
        produces={"ULTRALISK"},
        cost_minerals=150, cost_gas=200, build_time=46),
    "NYDUSNETWORK": BuildingInfo(
        "NydusNetwork", "zerg", BuildingCategory.SPECIAL, BuildingTier.ADVANCED,
        requires={"LAIR"},
        cost_minerals=150, cost_gas=200, build_time=29),
    "SPINECRAWLER": BuildingInfo(
        "SpineCrawler", "zerg", BuildingCategory.STATIC_DEFENSE, BuildingTier.BASIC,
        cost_minerals=100, build_time=36),
    "SPORECOLLAR": BuildingInfo(
        "SporeCrawler", "zerg", BuildingCategory.STATIC_DEFENSE, BuildingTier.BASIC,
        tags={"anti_air"}, cost_minerals=75, build_time=21),
    "OVERLORD": BuildingInfo(
        "Overlord", "zerg", BuildingCategory.SUPPLY, BuildingTier.BASIC,
        cost_minerals=100, build_time=18),
}

# ── Terran Buildings ────────────────────────────────────────────

_TERRAN_BUILDINGS: Dict[str, BuildingInfo] = {
    "COMMANDCENTER": BuildingInfo(
        "CommandCenter", "terran", BuildingCategory.PRODUCTION, BuildingTier.BASIC,
        produces={"SCV"},
        enables={"SUPPLYDEPOT", "REFINERY", "BARRACKS"},
        cost_minerals=400, build_time=71),
    "ORBITALCOMMAND": BuildingInfo(
        "OrbitalCommand", "terran", BuildingCategory.TECH, BuildingTier.BASIC,
        requires={"COMMANDCENTER"},
        enables={"MULE", "SCANNINGSWEEP"},
        cost_minerals=150, build_time=25),
    "PLANETARYFORTRESS": BuildingInfo(
        "PlanetaryFortress", "terran", BuildingCategory.DEFENSE, BuildingTier.ADVANCED,
        requires={"COMMANDCENTER"},
        tags={"defensive_base"},
        cost_minerals=150, cost_gas=150, build_time=50),
    "REFINERY": BuildingInfo(
        "Refinery", "terran", BuildingCategory.ECONOMY, BuildingTier.BASIC,
        cost_minerals=75, build_time=21),
    "SUPPLYDEPOT": BuildingInfo(
        "SupplyDepot", "terran", BuildingCategory.SUPPLY, BuildingTier.BASIC,
        cost_minerals=100, build_time=21),
    "BARRACKS": BuildingInfo(
        "Barracks", "terran", BuildingCategory.PRODUCTION, BuildingTier.BASIC,
        produces={"MARINE", "MARAUDER", "REAPER", "GHOST"},
        enables={"FACTORY", "ENGINEERINGBAY"},
        cost_minerals=150, build_time=46),
    "FACTORY": BuildingInfo(
        "Factory", "terran", BuildingCategory.PRODUCTION, BuildingTier.ADVANCED,
        requires={"BARRACKS"},
        produces={"HELLION", "WIDOWMINE", "Cyclone", "TANK", "THOR"},
        enables={"STARPORT", "ARMORY"},
        cost_minerals=150, cost_gas=100, build_time=43),
    "STARPORT": BuildingInfo(
        "Starport", "terran", BuildingCategory.PRODUCTION, BuildingTier.ADVANCED,
        requires={"FACTORY"},
        produces={"VIKING_F", "MEDIVAC", "LIBERATOR", "BATTLECRUISER"},
        enables={"FUSIONCORE"},
        cost_minerals=150, cost_gas=100, build_time=50),
    "ENGINEERINGBAY": BuildingInfo(
        "EngineeringBay", "terran", BuildingCategory.TECH, BuildingTier.BASIC,
        requires={"BARRACKS"},
        enables={"TERRANINFANTRYWEAPONS", "TERRANINFANTRYARMORS", "MISSILETURRET"},
        tags={"infantry_upgrade"},
        cost_minerals=125, build_time=35),
    "ARMORY": BuildingInfo(
        "Armory", "terran", BuildingCategory.TECH, BuildingTier.ADVANCED,
        requires={"FACTORY"},
        enables={"TERRANVEHICLEWEAPONS", "TERRANVEHICLEARMORS", "TERRANAIRWEAPONS", "TERRANAIRARMORS"},
        tags={"vehicle_upgrade", "air_upgrade"},
        cost_minerals=100, cost_gas=100, build_time=65),
    "FUSIONCORE": BuildingInfo(
        "FusionCore", "terran", BuildingCategory.TECH, BuildingTier.LATE,
        requires={"STARPORT", "ARMORY"},
        enables={"BATTLECRUISER", "BATTLECRUISERWEAPONSYSTEMS"},
        cost_minerals=150, cost_gas=150, build_time=50),
    "GHOSTACADEMY": BuildingInfo(
        "GhostAcademy", "terran", BuildingCategory.TECH, BuildingTier.ADVANCED,
        requires={"BARRACKS"},
        produces={"GHOST"},
        enables={"NUKE"},
        cost_minerals=150, cost_gas=50, build_time=29),
    "BUNKER": BuildingInfo(
        "Bunker", "terran", BuildingCategory.DEFENSE, BuildingTier.BASIC,
        cost_minerals=100, build_time=36),
    "MISSILETURRET": BuildingInfo(
        "MissileTurret", "terran", BuildingCategory.STATIC_DEFENSE, BuildingTier.BASIC,
        tags={"anti_air"}, cost_minerals=100, build_time=25),
    "SENSORTOWER": BuildingInfo(
        "SensorTower", "terran", BuildingCategory.SPECIAL, BuildingTier.ADVANCED,
        tags={"detection", "vision"}, cost_minerals=125, cost_gas=100, build_time=18),
}

# ── Protoss Buildings ───────────────────────────────────────────

_PROTOSS_BUILDINGS: Dict[str, BuildingInfo] = {
    "NEXUS": BuildingInfo(
        "Nexus", "protoss", BuildingCategory.PRODUCTION, BuildingTier.BASIC,
        produces={"PROBE"},
        enables={"PYLON", "ASSIMILATOR", "GATEWAY"},
        cost_minerals=400, build_time=71),
    "PYLON": BuildingInfo(
        "Pylon", "protoss", BuildingCategory.SUPPLY, BuildingTier.BASIC,
        enables={"power_field"},
        cost_minerals=100, build_time=18),
    "ASSIMILATOR": BuildingInfo(
        "Assimilator", "protoss", BuildingCategory.ECONOMY, BuildingTier.BASIC,
        cost_minerals=75, build_time=21),
    "GATEWAY": BuildingInfo(
        "Gateway", "protoss", BuildingCategory.PRODUCTION, BuildingTier.BASIC,
        produces={"ZEALOT", "STALKER", "SENTRY", "Adept", "DARK_TEMPLAR"},
        enables={"CYBERNETICSCORE"},
        cost_minerals=150, build_time=46),
    "WARPGATE": BuildingInfo(
        "WarpGate", "protoss", BuildingCategory.PRODUCTION, BuildingTier.BASIC,
        requires={"GATEWAY", "RESEARCH_WARPGATE"},
        produces={"ZEALOT", "STALKER", "SENTRY", "Adept", "DARK_TEMPLAR"},
        tags={"warp_in"},
        cost_minerals=0, build_time=0),
    "CYBERNETICSCORE": BuildingInfo(
        "CyberneticsCore", "protoss", BuildingCategory.TECH, BuildingTier.BASIC,
        requires={"GATEWAY"},
        enables={"TWILIGHTCOUNCIL", "STARGATE", "ROBOTICSFACILITY",
                 "RESEARCH_WARPGATE", "RESEARCH_AIRWEAPONS", "RESEARCH_AIRARMORS"},
        tags={"air_upgrade"},
        cost_minerals=150, cost_gas=0, build_time=50),
    "ROBOTICSFACILITY": BuildingInfo(
        "RoboticsFacility", "protoss", BuildingCategory.PRODUCTION, BuildingTier.ADVANCED,
        requires={"CYBERNETICSCORE"},
        produces={"OBSERVER", "IMMORTAL", "COLOSSUS", "DISRUPTOR"},
        enables={"ROBOTICSBAY"},
        cost_minerals=200, cost_gas=100, build_time=65),
    "ROBOTICSBAY": BuildingInfo(
        "RoboticsBay", "protoss", BuildingCategory.TECH, BuildingTier.ADVANCED,
        requires={"ROBOTICSFACILITY"},
        enables={"COLOSSUS", "RESEARCH_EXTENDEDTHERMALLANCE"},
        cost_minerals=150, cost_gas=100, build_time=50),
    "STARGATE": BuildingInfo(
        "Stargate", "protoss", BuildingCategory.PRODUCTION, BuildingTier.ADVANCED,
        requires={"CYBERNETICSCORE"},
        produces={"PHOENIX", "VOID_RAY", "ORACLE", "TEMPEST", "CARRIER"},
        enables={"FLEETBEACON"},
        cost_minerals=150, cost_gas=150, build_time=50),
    "FLEETBEACON": BuildingInfo(
        "FleetBeacon", "protoss", BuildingCategory.TECH, BuildingTier.LATE,
        requires={"STARGATE"},
        enables={"TEMPEST", "CARRIER", "RESEARCH_PHOENIXRANGE"},
        cost_minerals=200, cost_gas=150, build_time=50),
    "TWILIGHTCOUNCIL": BuildingInfo(
        "TwilightCouncil", "protoss", BuildingCategory.TECH, BuildingTier.ADVANCED,
        requires={"CYBERNETICSCORE"},
        enables={"TEMPLARARCHIVE", "DARKSHRINE",
                 "RESEARCH_CHARGE", "RESEARCH_BLINK", "RESEARCH_ADEPTPSIONICS"},
        tags={"gateway_upgrade"},
        cost_minerals=150, cost_gas=100, build_time=50),
    "TEMPLARARCHIVE": BuildingInfo(
        "TemplarArchive", "protoss", BuildingCategory.TECH, BuildingTier.LATE,
        requires={"TWILIGHTCOUNCIL"},
        produces={"HIGH_TEMPLAR", "ARCHON"},
        enables={"RESEARCH_PSISTORM"},
        cost_minerals=150, cost_gas=200, build_time=50),
    "DARKSHRINE": BuildingInfo(
        "DarkShrine", "protoss", BuildingCategory.TECH, BuildingTier.LATE,
        requires={"TWILIGHTCOUNCIL"},
        produces={"DARK_TEMPLAR", "ARCHON"},
        cost_minerals=150, cost_gas=150, build_time=71),
    "PHOTONCANNON": BuildingInfo(
        "PhotonCannon", "protoss", BuildingCategory.STATIC_DEFENSE, BuildingTier.BASIC,
        tags={"detection"},         cost_minerals=150, build_time=29),
    "SHIELDBattery": BuildingInfo(
        "ShieldBattery", "protoss", BuildingCategory.DEFENSE, BuildingTier.BASIC,
        tags={"shield_regen"}, cost_minerals=100, build_time=21),
}

# Merge all
_ALL_BUILDINGS: Dict[str, BuildingInfo] = {}
_ALL_BUILDINGS.update(_ZERG_BUILDINGS)
_ALL_BUILDINGS.update(_TERRAN_BUILDINGS)
_ALL_BUILDINGS.update(_PROTOSS_BUILDINGS)


# ── Upgrade Database ────────────────────────────────────────────

@dataclass
class UpgradeInfo:
    """Full classification of an upgrade."""
    name: str
    race: str
    category: UpgradeCategory
    tier: UpgradeTier
    applies_to: Set[str] = field(default_factory=set)  # unit names or roles
    requires: Set[str] = field(default_factory=set)      # building names
    description: str = ""
    cost_minerals: int = 0
    cost_gas: int = 0
    research_time: float = 0.0
    tags: Set[str] = field(default_factory=set)


_ALL_UPGRADES: Dict[str, UpgradeInfo] = {
    # ── Zerg Upgrades ───────────────────────────────────────────
    "ZERGLINGMETABOLICBOOST": UpgradeInfo(
        "ZerglingSpeed", "zerg", UpgradeCategory.SPEED, UpgradeTier.TIER1,
        applies_to={"ZERGLING"}, requires={"SPAWNINGPOOL"},
        cost_minerals=100, cost_gas=100, research_time=79),
    "ZERGLINGATTACKSPEED": UpgradeInfo(
        "ZerglingAttackSpeed", "zerg", UpgradeCategory.ABILITY, UpgradeTier.TIER1,
        applies_to={"ZERGLING"}, requires={"SPAWNINGPOOL", "LAIR"},
        cost_minerals=150, cost_gas=150, research_time=93),
    "ZERGMELEEWEAPONS1": UpgradeInfo(
        "MeleeAttack1", "zerg", UpgradeCategory.ATTACK, UpgradeTier.TIER1,
        applies_to={"ZERGLING", "BANELING", "ULTRALISK"}, requires={"EVOLUTIONCHAMBER"},
        cost_minerals=100, cost_gas=100, research_time=114),
    "ZERGMELEEWEAPONS2": UpgradeInfo(
        "MeleeAttack2", "zerg", UpgradeCategory.ATTACK, UpgradeTier.TIER2,
        applies_to={"ZERGLING", "BANELING", "ULTRALISK"},
        requires={"EVOLUTIONCHAMBER", "LAIR"},
        cost_minerals=150, cost_gas=150, research_time=136),
    "ZERGMELEEWEAPONS3": UpgradeInfo(
        "MeleeAttack3", "zerg", UpgradeCategory.ATTACK, UpgradeTier.TIER3,
        applies_to={"ZERGLING", "BANELING", "ULTRALISK"},
        requires={"EVOLUTIONCHAMBER", "HIVE"},
        cost_minerals=200, cost_gas=200, research_time=157),
    "ZERGMISSILEWEAPONS1": UpgradeInfo(
        "RangedAttack1", "zerg", UpgradeCategory.ATTACK, UpgradeTier.TIER1,
        applies_to={"ROACH", "HYDRALISK", "CORRUPTOR", "VIPER"},
        requires={"EVOLUTIONCHAMBER"},
        cost_minerals=100, cost_gas=100, research_time=114),
    "ZERGMISSILEWEAPONS2": UpgradeInfo(
        "RangedAttack2", "zerg", UpgradeCategory.ATTACK, UpgradeTier.TIER2,
        applies_to={"ROACH", "HYDRALISK", "CORRUPTOR", "VIPER"},
        requires={"EVOLUTIONCHAMBER", "LAIR"},
        cost_minerals=150, cost_gas=150, research_time=136),
    "ZERGMISSILEWEAPONS3": UpgradeInfo(
        "RangedAttack3", "zerg", UpgradeCategory.ATTACK, UpgradeTier.TIER3,
        applies_to={"ROACH", "HYDRALISK", "CORRUPTOR", "VIPER"},
        requires={"EVOLUTIONCHAMBER", "HIVE"},
        cost_minerals=200, cost_gas=200, research_time=157),
    "ZERGGROUNDARMORS1": UpgradeInfo(
        "GroundArmor1", "zerg", UpgradeCategory.ARMOR, UpgradeTier.TIER1,
        applies_to={"ZERGLING", "ROACH", "HYDRALISK", "ULTRALISK"},
        requires={"EVOLUTIONCHAMBER"},
        cost_minerals=100, cost_gas=100, research_time=114),
    "ZERGGROUNDARMORS2": UpgradeInfo(
        "GroundArmor2", "zerg", UpgradeCategory.ARMOR, UpgradeTier.TIER2,
        applies_to={"ZERGLING", "ROACH", "HYDRALISK", "ULTRALISK"},
        requires={"EVOLUTIONCHAMBER", "LAIR"},
        cost_minerals=150, cost_gas=150, research_time=136),
    "ZERGGROUNDARMORS3": UpgradeInfo(
        "GroundArmor3", "zerg", UpgradeCategory.ARMOR, UpgradeTier.TIER3,
        applies_to={"ZERGLING", "ROACH", "HYDRALISK", "ULTRALISK"},
        requires={"EVOLUTIONCHAMBER", "HIVE"},
        cost_minerals=200, cost_gas=200, research_time=157),
    "ZERGFLYERWEAPONS1": UpgradeInfo(
        "FlyerAttack1", "zerg", UpgradeCategory.ATTACK, UpgradeTier.TIER1,
        applies_to={"MUTALISK", "CORRUPTOR"}, requires={"SPIRE"},
        cost_minerals=100, cost_gas=100, research_time=114),
    "ZERGFLYERWEAPONS2": UpgradeInfo(
        "FlyerAttack2", "zerg", UpgradeCategory.ATTACK, UpgradeTier.TIER2,
        applies_to={"MUTALISK", "CORRUPTOR"},
        requires={"SPIRE", "LAIR"},
        cost_minerals=150, cost_gas=150, research_time=136),
    "ZERGFLYERWEAPONS3": UpgradeInfo(
        "FlyerAttack3", "zerg", UpgradeCategory.ATTACK, UpgradeTier.TIER3,
        applies_to={"MUTALISK", "CORRUPTOR"},
        requires={"SPIRE", "HIVE"},
        cost_minerals=200, cost_gas=200, research_time=157),
    "ZERGFLYERARMORS1": UpgradeInfo(
        "FlyerArmor1", "zerg", UpgradeCategory.ARMOR, UpgradeTier.TIER1,
        applies_to={"MUTALISK", "CORRUPTOR"}, requires={"SPIRE"},
        cost_minerals=100, cost_gas=100, research_time=114),
    "ZERGFLYERARMORS2": UpgradeInfo(
        "FlyerArmor2", "zerg", UpgradeCategory.ARMOR, UpgradeTier.TIER2,
        applies_to={"MUTALISK", "CORRUPTOR"},
        requires={"SPIRE", "LAIR"},
        cost_minerals=150, cost_gas=150, research_time=136),
    "ZERGFLYERARMORS3": UpgradeInfo(
        "FlyerArmor3", "zerg", UpgradeCategory.ARMOR, UpgradeTier.TIER3,
        applies_to={"MUTALISK", "CORRUPTOR"},
        requires={"SPIRE", "HIVE"},
        cost_minerals=200, cost_gas=200, research_time=157),
    "HYDRALISKRANGE": UpgradeInfo(
        "HydraRange", "zerg", UpgradeCategory.ABILITY, UpgradeTier.TIER1,
        applies_to={"HYDRALISK"}, requires={"HYDRALISKDEN"},
        cost_minerals=100, cost_gas=100, research_time=71),
    "HYDRALISKSPEED": UpgradeInfo(
        "HydraSpeed", "zerg", UpgradeCategory.SPEED, UpgradeTier.TIER1,
        applies_to={"HYDRALISK"}, requires={"HYDRALISKDEN", "LAIR"},
        cost_minerals=100, cost_gas=100, research_time=71),

    # ── Terran Upgrades ─────────────────────────────────────────
    "TERRANINFANTRYWEAPONS1": UpgradeInfo(
        "InfantryAttack1", "terran", UpgradeCategory.ATTACK, UpgradeTier.TIER1,
        applies_to={"MARINE", "MARAUDER", "REAPER", "GHOST"},
        requires={"ENGINEERINGBAY"},
        cost_minerals=100, cost_gas=100, research_time=140),
    "TERRANINFANTRYWEAPONS2": UpgradeInfo(
        "InfantryAttack2", "terran", UpgradeCategory.ATTACK, UpgradeTier.TIER2,
        applies_to={"MARINE", "MARAUDER", "REAPER", "GHOST"},
        requires={"ENGINEERINGBAY", "ARMORY"},
        cost_minerals=175, cost_gas=175, research_time=170),
    "TERRANINFANTRYWEAPONS3": UpgradeInfo(
        "InfantryAttack3", "terran", UpgradeCategory.ATTACK, UpgradeTier.TIER3,
        applies_to={"MARINE", "MARAUDER", "REAPER", "GHOST"},
        requires={"ENGINEERINGBAY", "ARMORY", "ORBITALCOMMAND"},
        cost_minerals=250, cost_gas=250, research_time=200),
    "TERRANINFANTRYARMORS1": UpgradeInfo(
        "InfantryArmor1", "terran", UpgradeCategory.ARMOR, UpgradeTier.TIER1,
        applies_to={"MARINE", "MARAUDER", "REAPER", "GHOST"},
        requires={"ENGINEERINGBAY"},
        cost_minerals=100, cost_gas=100, research_time=140),
    "TERRANINFANTRYARMORS2": UpgradeInfo(
        "InfantryArmor2", "terran", UpgradeCategory.ARMOR, UpgradeTier.TIER2,
        applies_to={"MARINE", "MARAUDER", "REAPER", "GHOST"},
        requires={"ENGINEERINGBAY", "ARMORY"},
        cost_minerals=175, cost_gas=175, research_time=170),
    "TERRANINFANTRYARMORS3": UpgradeInfo(
        "InfantryArmor3", "terran", UpgradeCategory.ARMOR, UpgradeTier.TIER3,
        applies_to={"MARINE", "MARAUDER", "REAPER", "GHOST"},
        requires={"ENGINEERINGBAY", "ARMORY", "ORBITALCOMMAND"},
        cost_minerals=250, cost_gas=250, research_time=200),
    "TERRANVEHICLEWEAPONS1": UpgradeInfo(
        "VehicleAttack1", "terran", UpgradeCategory.ATTACK, UpgradeTier.TIER1,
        applies_to={"HELLION", "TANK", "Cyclone", "THOR", "WIDOWMINE"},
        requires={"ARMORY"},
        cost_minerals=100, cost_gas=100, research_time=140),
    "TERRANVEHICLEWEAPONS2": UpgradeInfo(
        "VehicleAttack2", "terran", UpgradeCategory.ATTACK, UpgradeTier.TIER2,
        applies_to={"HELLION", "TANK", "Cyclone", "THOR", "WIDOWMINE"},
        requires={"ARMORY", "FACTORY"},
        cost_minerals=175, cost_gas=175, research_time=170),
    "TERRANVEHICLEARMORS1": UpgradeInfo(
        "VehicleArmor1", "terran", UpgradeCategory.ARMOR, UpgradeTier.TIER1,
        applies_to={"HELLION", "TANK", "Cyclone", "THOR", "WIDOWMINE"},
        requires={"ARMORY"},
        cost_minerals=100, cost_gas=100, research_time=140),
    "TERRANVEHICLEARMORS2": UpgradeInfo(
        "VehicleArmor2", "terran", UpgradeCategory.ARMOR, UpgradeTier.TIER2,
        applies_to={"HELLION", "TANK", "Cyclone", "THOR", "WIDOWMINE"},
        requires={"ARMORY", "FACTORY"},
        cost_minerals=175, cost_gas=175, research_time=170),
    "TERRANAIRWEAPONS1": UpgradeInfo(
        "AirAttack1", "terran", UpgradeCategory.ATTACK, UpgradeTier.TIER1,
        applies_to={"VIKING_F", "LIBERATOR", "BATTLECRUISER"},
        requires={"ARMORY"},
        cost_minerals=100, cost_gas=100, research_time=140),
    "TERRANAIRWEAPONS2": UpgradeInfo(
        "AirAttack2", "terran", UpgradeCategory.ATTACK, UpgradeTier.TIER2,
        applies_to={"VIKING_F", "LIBERATOR", "BATTLECRUISER"},
        requires={"ARMORY", "STARPORT"},
        cost_minerals=175, cost_gas=175, research_time=170),
    "TERRANAIRARMORS1": UpgradeInfo(
        "AirArmor1", "terran", UpgradeCategory.ARMOR, UpgradeTier.TIER1,
        applies_to={"VIKING_F", "LIBERATOR", "BATTLECRUISER"},
        requires={"ARMORY"},
        cost_minerals=100, cost_gas=100, research_time=140),
    "TERRANAIRARMORS2": UpgradeInfo(
        "AirArmor2", "terran", UpgradeCategory.ARMOR, UpgradeTier.TIER2,
        applies_to={"VIKING_F", "LIBERATOR", "BATTLECRUISER"},
        requires={"ARMORY", "STARPORT"},
        cost_minerals=175, cost_gas=175, research_time=170),
    "STIMPACK": UpgradeInfo(
        "Stimpack", "terran", UpgradeCategory.ABILITY, UpgradeTier.TIER1,
        applies_to={"MARINE", "MARAUDER"}, requires={"BARRACKS", "TECHLAB"},
        cost_minerals=100, cost_gas=100, research_time=100),
    "COMBATSHIELD": UpgradeInfo(
        "CombatShield", "terran", UpgradeCategory.ABILITY, UpgradeTier.TIER1,
        applies_to={"MARINE"}, requires={"BARRACKS", "TECHLAB"},
        cost_minerals=100, cost_gas=100, research_time=110),
    "CONCUSSIVE": UpgradeInfo(
        "ConcussiveShells", "terran", UpgradeCategory.ABILITY, UpgradeTier.TIER1,
        applies_to={"MARAUDER"}, requires={"BARRACKS", "TECHLAB"},
        cost_minerals=50, cost_gas=50, research_time=60),
    "SIEGETECH": UpgradeInfo(
        "SiegeTech", "terran", UpgradeCategory.SIEGE, UpgradeTier.TIER1,
        applies_to={"TANK"}, requires={"FACTORY", "TECHLAB"},
        cost_minerals=150, cost_gas=100, research_time=110),
    "HIBERNATION": UpgradeInfo(
        "Hibernation", "terran", UpgradeCategory.TRANSPORT, UpgradeTier.TIER1,
        applies_to={"MEDIVAC"}, requires={"STARPORT"},
        cost_minerals=100, cost_gas=100, research_time=110),

    # ── Protoss Upgrades ────────────────────────────────────────
    "WARPGATE": UpgradeInfo(
        "WarpGate", "protoss", UpgradeCategory.ABILITY, UpgradeTier.TIER1,
        applies_to={"GATEWAY"}, requires={"CYBERNETICSCORE"},
        cost_minerals=50, cost_gas=50, research_time=160),
    "CHARGE": UpgradeInfo(
        "Charge", "protoss", UpgradeCategory.SPEED, UpgradeTier.TIER1,
        applies_to={"ZEALOT"}, requires={"TWILIGHTCOUNCIL"},
        cost_minerals=100, cost_gas=100, research_time=100),
    "BLINK": UpgradeInfo(
        "Blink", "protoss", UpgradeCategory.ABILITY, UpgradeTier.TIER1,
        applies_to={"STALKER"}, requires={"TWILIGHTCOUNCIL"},
        cost_minerals=150, cost_gas=150, research_time=120),
    "ADEPTPSIONICS": UpgradeInfo(
        "AdeptPsionicTransfer", "protoss", UpgradeCategory.ABILITY, UpgradeTier.TIER1,
        applies_to={"Adept"}, requires={"TWILIGHTCOUNCIL"},
        cost_minerals=100, cost_gas=100, research_time=100),
    "EXTENDEDTHERMALLANCE": UpgradeInfo(
        "ExtendedThermalLance", "protoss", UpgradeCategory.ABILITY, UpgradeTier.TIER1,
        applies_to={"COLOSSUS"}, requires={"ROBOTICSBAY"},
        cost_minerals=200, cost_gas=200, research_time=140),
    "PSISTORM": UpgradeInfo(
        "PsionicStorm", "protoss", UpgradeCategory.SPELL, UpgradeTier.TIER1,
        applies_to={"HIGH_TEMPLAR"}, requires={"TEMPLARARCHIVE"},
        cost_minerals=200, cost_gas=200, research_time=114),
    "AIRWEAPONS1": UpgradeInfo(
        "AirAttack1", "protoss", UpgradeCategory.ATTACK, UpgradeTier.TIER1,
        applies_to={"PHOENIX", "VOID_RAY", "ORACLE", "TEMPEST", "CARRIER"},
        requires={"CYBERNETICSCORE"},
        cost_minerals=100, cost_gas=100, research_time=129),
    "AIRWEAPONS2": UpgradeInfo(
        "AirAttack2", "protoss", UpgradeCategory.ATTACK, UpgradeTier.TIER2,
        applies_to={"PHOENIX", "VOID_RAY", "ORACLE", "TEMPEST", "CARRIER"},
        requires={"CYBERNETICSCORE", "STARGATE"},
        cost_minerals=150, cost_gas=150, research_time=154),
    "AIRARMORS1": UpgradeInfo(
        "AirArmor1", "protoss", UpgradeCategory.ARMOR, UpgradeTier.TIER1,
        applies_to={"PHOENIX", "VOID_RAY", "ORACLE", "TEMPEST", "CARRIER"},
        requires={"CYBERNETICSCORE"},
        cost_minerals=100, cost_gas=100, research_time=129),
    "AIRARMORS2": UpgradeInfo(
        "AirArmor2", "protoss", UpgradeCategory.ARMOR, UpgradeTier.TIER2,
        applies_to={"PHOENIX", "VOID_RAY", "ORACLE", "TEMPEST", "CARRIER"},
        requires={"CYBERNETICSCORE", "STARGATE"},
        cost_minerals=150, cost_gas=150, research_time=154),
    "SHIELDS1": UpgradeInfo(
        "Shields1", "protoss", UpgradeCategory.ARMOR, UpgradeTier.TIER1,
        applies_to={"all"}, requires={"CYBERNETICSCORE"},
        cost_minerals=150, cost_gas=150, research_time=140),
    "PHOENIXRANGE": UpgradeInfo(
        "PhoenixRange", "protoss", UpgradeCategory.ABILITY, UpgradeTier.TIER1,
        applies_to={"PHOENIX"}, requires={"FLEETBEACON"},
        cost_minerals=150, cost_gas=150, research_time=90),
}


class BuildingClassifier:
    """Classifies SC2 buildings by category, tier, and tech tree position."""

    def __init__(self):
        self._database = dict(_ALL_BUILDINGS)

    def classify(self, building: Any) -> Optional[BuildingInfo]:
        """Classify a building by its SC2 API object."""
        name = getattr(building, 'name', '').upper().replace(" ", "").replace("'", "")
        if name in self._database:
            return self._database[name]

        type_name = ""
        try:
            type_name = building.type_id.name.upper().replace(" ", "")
        except (AttributeError, TypeError):
            pass
        if type_name in self._database:
            return self._database[type_name]

        return None

    def filter_by_category(self, buildings: Any, category: BuildingCategory) -> List[Any]:
        result = []
        for b in buildings:
            info = self.classify(b)
            if info and info.category == category:
                result.append(b)
        return result

    def filter_by_tier(self, buildings: Any, tier: BuildingTier) -> List[Any]:
        result = []
        for b in buildings:
            info = self.classify(b)
            if info and info.tier == tier:
                result.append(b)
        return result

    def get_build_chain(self, target_building: str) -> List[str]:
        """Get the prerequisite chain to build a specific building."""
        chain = []
        visited = set()
        current = target_building.upper()
        while current and current not in visited:
            visited.add(current)
            info = self._database.get(current)
            if info and info.requires:
                for req in info.requires:
                    if req not in visited:
                        chain.append(req)
                        current = req
                        break
                else:
                    break
            else:
                break
        return chain

    def what_can_build(self, unit_name: str) -> List[str]:
        """What buildings can a worker build? (by worker type)"""
        return [
            name for name, info in self._database.items()
            if unit_name.upper() in info.tags or not info.requires
        ]


class UpgradeClassifier:
    """Classifies SC2 upgrades by category, tier, and unit applicability."""

    def __init__(self):
        self._database = dict(_ALL_UPGRADES)

    def classify(self, upgrade: Any) -> Optional[UpgradeInfo]:
        """Classify an upgrade by its SC2 API object."""
        name = getattr(upgrade, 'name', '').upper().replace(" ", "").replace("'", "")
        if name in self._database:
            return self._database[name]

        type_name = ""
        try:
            type_name = upgrade.type_id.name.upper().replace(" ", "")
        except (AttributeError, TypeError):
            pass
        if type_name in self._database:
            return self._database[type_name]

        return None

    def get_upgrades_for_unit(self, unit_name: str) -> List[UpgradeInfo]:
        """Get all upgrades that apply to a specific unit."""
        unit_upper = unit_name.upper()
        return [
            info for info in self._database.values()
            if unit_upper in info.applies_to or "all" in info.applies_to
        ]

    def get_available_upgrades(self, buildings: Any, race: str) -> List[UpgradeInfo]:
        """Get upgrades researchable with current buildings."""
        owned = set()
        for b in buildings:
            info = BuildingClassifier().classify(b)
            if info:
                owned.add(info.name.upper())

        race_upper = race.upper()
        available = []
        for info in self._database.values():
            if info.race.upper() != race_upper:
                continue
            meets_prereqs = all(
                req.upper() in owned or req.upper() in {n.upper() for n in owned}
                for req in info.requires
            )
            if meets_prereqs:
                available.append(info)
        return available

    def filter_by_category(self, category: UpgradeCategory, race: str = None) -> List[UpgradeInfo]:
        return [
            info for info in self._database.values()
            if info.category == category and (race is None or info.race == race)
        ]
