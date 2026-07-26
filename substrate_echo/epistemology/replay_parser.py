"""SC2 Replay Parser — Integration with sc2reader for post-game epistemic learning.

Parses SC2 replays and feeds them into the substrate_echo epistemic system
so the agent can learn from complete game trajectories including opponent strategies.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Iterator
from enum import Enum
from pathlib import Path
import json
import time

try:
    import sc2reader
    SC2READER_AVAILABLE = True
except ImportError:
    SC2READER_AVAILABLE = False


class ReplayEventType(Enum):
    """Types of events we extract from replays."""
    UNIT_BORN = "unit_born"
    UNIT_DIED = "unit_died"
    UNIT_DONE = "unit_done"  # construction finished
    UPGRADE_COMPLETE = "upgrade_complete"
    BUILDING_STARTED = "building_started"
    BUILDING_CANCELLED = "building_cancelled"
    UNIT_TYPE_CHANGE = "unit_type_change"  # morph
    PLAYER_STATS = "player_stats"  # resources, supply, etc.
    CAMERA_MOVE = "camera_move"  # what player was looking at
    SELECTION = "selection"  # unit selection
    COMMAND = "command"  # issued commands
    CHAT = "chat"


@dataclass
class ReplayEvent:
    """A single event extracted from a replay."""
    tick: int
    time: float  # game time in seconds
    event_type: ReplayEventType
    player_id: int
    data: Dict[str, Any]
    source_unit: Optional[int] = None  # unit tag if applicable
    target_unit: Optional[int] = None
    location: Optional[Tuple[float, float]] = None


@dataclass
class PlayerTimeline:
    """Aggregated timeline for a single player."""
    player_id: int
    name: str
    race: str
    result: str  # Win/Loss
    apm: float
    eapm: float  # effective APM
    workers_produced: int = 0
    army_value_lost: int = 0
    army_value_killed: int = 0
    resources_collected: Dict[str, int] = field(default_factory=dict)
    supply_used_max: int = 0
    units_produced: Dict[str, int] = field(default_factory=dict)
    buildings_built: Dict[str, int] = field(default_factory=dict)
    upgrades_researched: List[str] = field(default_factory=list)
    key_timings: Dict[str, float] = field(default_factory=dict)  # e.g., "first_zergling": 45.2


@dataclass
class ParsedReplay:
    """Complete parsed replay with both player timelines."""
    replay_path: str
    map_name: str
    game_length: float  # seconds
    game_length_ticks: int
    players: List[PlayerTimeline]
    events: List[ReplayEvent]
    winner_id: int
    date: float  # timestamp
    version: str


class ReplayParser:
    """Parses SC2 replays into structured events for epistemic learning."""

    def __init__(self, load_level: int = 1):
        """Initialize parser.
        
        Args:
            load_level: sc2reader load level (1=basic, 2=details, 3=full)
        """
        if not SC2READER_AVAILABLE:
            raise RuntimeError("sc2reader not installed. Run: pip install sc2reader")
        self.load_level = load_level

    def parse_replay(self, replay_path: str) -> ParsedReplay:
        """Parse a single replay file."""
        # Use load_level=1 to avoid cache_handles issue
        replay = sc2reader.load_replay(replay_path, load_level=1)

        # Parse players
        players = []
        winner_id = 0
        for p in replay.players:
            timeline = self._parse_player(p, replay)
            players.append(timeline)
            if p.result == "Win":
                winner_id = p.pid

        # Parse events
        events = self._parse_events(replay)

        parsed = ParsedReplay(
            replay_path=str(replay.path),
            map_name=replay.map_name,
            game_length=replay.real_length,
            game_length_ticks=replay.game_length,
            players=players,
            events=events,
            winner_id=winner_id,
            date=replay.date.timestamp() if replay.date else time.time(),
            version=replay.version
        )
        return parsed

    def _parse_player(self, player, replay) -> PlayerTimeline:
        """Extract player timeline from replay."""
        # APM and eAPM
        apm = player.avg_apm if hasattr(player, 'avg_apm') else 0
        eapm = player.avg_eapm if hasattr(player, 'avg_eapm') else 0

        # Units and buildings
        units_produced = {}
        buildings_built = {}
        upgrades = []

        # Track key timings
        key_timings = {}

        # Track resources
        workers_produced = 0
        army_value_lost = 0
        army_value_killed = 0
        resources = {"minerals": 0, "vespene": 0}
        supply_max = 0

        # Process events for this player
        for event in replay.events:
            if hasattr(event, 'player') and event.player and hasattr(event.player, 'pid') and event.player.pid == player.pid:
                # Track unit production
                if event.__class__.__name__ in ("UnitBornEvent", "UnitInitEvent"):
                    unit_name = getattr(event, 'unit_type_name', getattr(event, 'unit', None))
                    if unit_name:
                        name = str(unit_name).split('.')[-1]
                        if "Worker" in name or "Drone" in name or "SCV" in name or "Probe" in name:
                            workers_produced += 1
                        else:
                            units_produced[name] = units_produced.get(name, 0) + 1
                            if name not in key_timings:
                                key_timings[f"first_{name.lower()}"] = getattr(event, 'second', 0)

                elif event.__class__.__name__ == "UnitDoneEvent":
                    unit_name = getattr(event, 'unit_type_name', getattr(event, 'unit', None))
                    if unit_name:
                        name = str(unit_name).split('.')[-1]
                        buildings_built[name] = buildings_built.get(name, 0) + 1

                elif event.__class__.__name__ == "UpgradeCompleteEvent":
                    upgrade_name = getattr(event, 'upgrade_type_name', getattr(event, 'upgrade', None))
                    if upgrade_name:
                        name = str(upgrade_name).split('.')[-1]
                        upgrades.append(name)
                        key_timings[f"upgrade_{name.lower()}"] = getattr(event, 'second', 0)

                elif event.__class__.__name__ == "PlayerStatsEvent":
                    resources["minerals"] = getattr(event, 'minerals_current', resources["minerals"])
                    resources["vespene"] = getattr(event, 'vespene_current', resources["vespene"])
                    supply_max = max(supply_max, getattr(event, 'food_made', supply_max))

                elif event.__class__.__name__ == "UnitDiedEvent":
                    unit = getattr(event, 'unit', None)
                    killer = getattr(event, 'killer', None)
                    if unit and hasattr(unit, 'owner') and unit.owner and hasattr(unit.owner, 'pid') and unit.owner.pid == player.pid:
                        army_value_lost += getattr(unit, 'minerals_cost', 0) + getattr(unit, 'vespene_cost', 0) * 2
                    killer = getattr(event, 'killer', None)
                    if killer and hasattr(killer, 'owner') and killer.owner and hasattr(killer.owner, 'pid') and killer.owner.pid == player.pid:
                        if unit:
                            army_value_killed += getattr(unit, 'minerals_cost', 0) + getattr(unit, 'vespene_cost', 0) * 2

        return PlayerTimeline(
            player_id=player.pid,
            name=player.name,
            race=player.pick_race[0].upper() if player.pick_race else "?",
            result=player.result,
            apm=player.avg_apm if hasattr(player, 'avg_apm') else 0,
            eapm=player.avg_eapm if hasattr(player, 'avg_eapm') else 0,
            workers_produced=workers_produced,
            army_value_lost=army_value_lost,
            army_value_killed=army_value_killed,
            resources_collected=resources,
            supply_used_max=supply_max,
            units_produced=units_produced,
            buildings_built=buildings_built,
            upgrades_researched=upgrades,
            key_timings=key_timings
        )

    def _parse_events(self, replay) -> List[ReplayEvent]:
        """Extract all relevant events from replay."""
        events = []
        for event in replay.events:
            try:
                parsed = self._parse_single_event(event)
                if parsed:
                    events.append(parsed)
            except Exception:
                continue
        return events

    def _parse_single_event(self, event) -> Optional[ReplayEvent]:
        """Parse a single replay event."""
        # Map sc2reader event names to our types
        type_map = {
            "UnitBornEvent": ReplayEventType.UNIT_BORN,
            "UnitDiedEvent": ReplayEventType.UNIT_DIED,
            "UnitDoneEvent": ReplayEventType.UNIT_DONE,
            "UpgradeCompleteEvent": ReplayEventType.UPGRADE_COMPLETE,
            "UnitTypeChangeEvent": ReplayEventType.UNIT_TYPE_CHANGE,
            "PlayerStatsEvent": ReplayEventType.PLAYER_STATS,
            "CameraEvent": ReplayEventType.CAMERA_MOVE,
            "SelectionEvent": ReplayEventType.SELECTION,
            "CommandEvent": ReplayEventType.COMMAND,
            "ChatEvent": ReplayEventType.CHAT,
        }

        event_type = type_map.get(event.__class__.__name__)
        if not event_type:
            return None

        player_id = getattr(event, 'player', None)
        if player_id and hasattr(player_id, 'pid'):
            player_id = player_id.pid
        elif player_id and hasattr(player_id, 'pid'):
            player_id = player_id.pid
        else:
            player_id = getattr(event, 'pid', 0)

        data = {}
        for attr in dir(event):
            if not attr.startswith('_'):
                try:
                    val = getattr(event, attr)
                    if not callable(val):
                        data[attr] = val
                except Exception:
                    pass

        location = None
        if hasattr(event, 'location') and event.location:
            location = (event.location.x, event.location.y)
        elif hasattr(event, 'x') and hasattr(event, 'y'):
            location = (event.x, event.y)

        source_unit = None
        if hasattr(event, 'unit') and event.unit and hasattr(event.unit, 'tag'):
            source_unit = event.unit.tag

        target_unit = None
        if hasattr(event, 'target') and event.target and hasattr(event.target, 'tag'):
            target_unit = event.target.tag

        return ReplayEvent(
            tick=int(getattr(event, 'frame', getattr(event, 'second', 0) * 22.4)),
            time=getattr(event, 'second', 0.0),
            event_type=event_type,
            player_id=player_id,
            data=data,
            source_unit=source_unit,
            target_unit=target_unit,
            location=location
        )


class ReplayLearningIntegrator:
    """Integrates parsed replays into the substrate_echo epistemic system."""

    def __init__(self, chain_recorder=None, entity_model=None, affordance_tracer=None):
        self.chain_recorder = chain_recorder
        self.entity_model = entity_model
        self.affordance_tracer = affordance_tracer
        self.learned_patterns: Dict[str, Any] = {}

    def learn_from_replay(self, parsed: ParsedReplay) -> Dict[str, Any]:
        """Learn patterns from a parsed replay."""
        learnings = {
            "opponent_strategies": [],
            "timing_attacks": [],
            "build_orders": [],
            "counter_strategies": [],
            "economic_patterns": []
        }

        for player in parsed.players:
            if player.player_id == parsed.winner_id:
                # Learn from winner
                self._extract_winning_patterns(player, parsed, learnings)
            else:
                # Learn from loser (what to avoid)
                self._extract_losing_patterns(player, parsed, learnings)

        # Store learnings
        self.learned_patterns[str(parsed.replay_path)] = learnings
        return learnings

    def _extract_winning_patterns(self, winner: PlayerTimeline, parsed: ParsedReplay, learnings: Dict):
        """Extract patterns from winning player."""
        # Build order
        build_order = self._reconstruct_build_order(winner, parsed.events)
        if build_order:
            learnings["build_orders"].append({
                "race": winner.race,
                "map": parsed.map_name,
                "build_order": build_order,
                "timings": winner.key_timings,
                "result": "win"
            })

        # Timing attacks
        if winner.army_value_killed > winner.army_value_lost * 2:
            learnings["timing_attacks"].append({
                "race": winner.race,
                "map": parsed.map_name,
                "attack_time": min(winner.key_timings.values()) if winner.key_timings else 0,
                "army_composition": winner.units_produced,
                "effectiveness": winner.army_value_killed / max(1, winner.army_value_lost)
            })

        # Economic efficiency
        total_income = winner.resources_collected.get("minerals", 0) + winner.resources_collected.get("vespene", 0) * 2
        if total_income > 0:
            efficiency = winner.army_value_killed / max(1, total_income / 1000)
            learnings["economic_patterns"].append({
                "race": winner.race,
                "efficiency_ratio": efficiency,
                "workers": winner.workers_produced,
                "max_supply": winner.supply_used_max
            })

    def _extract_losing_patterns(self, loser: PlayerTimeline, parsed: ParsedReplay, learnings: Dict):
        """Extract anti-patterns from losing player."""
        learnings["counter_strategies"].append({
            "race": loser.race,
            "failed_strategy": {
                "build_order": self._reconstruct_build_order(loser, parsed.events),
                "timings": loser.key_timings,
                "composition": loser.units_produced,
                "weakness": "economy" if loser.resources_collected.get("minerals", 0) < 5000 else "army_comp"
            },
            "counter": "early_pressure" if loser.workers_produced > 50 else "tech_switch"
        })

    def _reconstruct_build_order(self, player: PlayerTimeline, events: List[ReplayEvent]) -> List[Dict]:
        """Reconstruct build order from events."""
        build_events = []
        for e in events:
            if e.player_id == player.player_id:
                if e.event_type in (ReplayEventType.UNIT_BORN, ReplayEventType.UNIT_DONE):
                    unit_name = e.data.get('unit_type_name', e.data.get('unit', 'Unknown'))
                    name = str(unit_name).split('.')[-1] if unit_name else "Unknown"
                    build_events.append({
                        "time": e.time,
                        "tick": e.tick,
                        "unit": name,
                        "type": "unit" if e.event_type == ReplayEventType.UNIT_BORN else "building"
                    })
                elif e.event_type == ReplayEventType.UPGRADE_COMPLETE:
                    upgrade = e.data.get('upgrade_type_name', e.data.get('upgrade', 'Unknown'))
                    name = str(upgrade).split('.')[-1]
                    build_events.append({
                        "time": e.time,
                        "tick": e.tick,
                        "upgrade": name,
                        "type": "upgrade"
                    })
        build_events.sort(key=lambda x: x["time"])
        return build_events


def load_and_learn_from_replays(replay_dir: str, integrator: ReplayLearningIntegrator, max_replays: int = 10) -> List[Dict]:
    """Load and learn from multiple replays in a directory."""
    replay_dir = Path(replay_dir)
    if not replay_dir.exists():
        return []

    replays = list(replay_dir.rglob("*.SC2Replay"))
    learnings = []

    for replay_path in replays[:max_replays]:
        try:
            parser = ReplayParser()
            parsed = parser.parse_replay(str(replay_path))
            learnings = integrator.learn_from_replay(parsed)
            learnings["replay_path"] = str(replay_path)
            learnings["map"] = parsed.map_name
            learnings["duration"] = parsed.game_length
            print(f"Learned from: {replay_path.name} ({parsed.game_length:.0f}s)")
        except Exception as e:
            print(f"Failed to parse {replay_path}: {e}")

    return learnings