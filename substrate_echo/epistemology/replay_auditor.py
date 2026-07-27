"""Replay Auditor — Analyzes tick-by-tick data for failure points.

Records every tick's state during a game and produces an audit report
identifying: supply blocks, idle production, economy stalls, army
inaction, composition mismatches, unit losses, and action degeneration.

The auditor also populates an EpistemicLedger with evidence — raw
observations of capabilities used and strategy outcomes.  The ledger
is pure memory; it never recommends or decides.  Other systems
(Strategy Council, Capability Council) read the ledger and reason.

Architecture:
    TickSnapshot         — one frame of game state
    FailurePoint         — a detected issue
    AuditReport          — full game failure analysis
    EvidenceEntry        — a single raw observation
    ConfidenceRecord     — raw evidence counts (wins/losses/draws), not averages
    StrategyContext      — structured context snapshot for hypothesis evaluation
    Capability           — a discovered affordance (what the bot CAN do)
    StrategyHypothesis   — a testable strategy claim with evidence
    EpistemicLedger      — cross-game hierarchical memory
    ReplayAuditor        — records ticks, detects failures, populates ledger

Usage in bot:
    auditor = ReplayAuditor()
    ledger = EpistemicLedger()
    # Each tick:
    auditor.record_tick(tick, bot, action_type)
    # Game end:
    report = auditor.analyze(game_result)
    report.print_summary()
    ledger.end_game(...)
    ledger.save(path)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import json


class FailureCategory(Enum):
    """Categories of failure points detected during gameplay."""
    SUPPLY_BLOCKED = "supply_blocked"
    IDLE_PRODUCTION = "idle_production"
    IDLE_WORKERS = "idle_workers"
    ECONOMY_STALLED = "economy_stalled"
    ARMY_IDLE = "army_idle"
    LATE_EXPANSION = "late_expansion"
    UNIT_LOSS_WAVE = "unit_loss_wave"
    POOR_COMPOSITION = "poor_composition"
    ACTION_DEGENERATE = "action_degenerate"
    TECH_DELAY = "tech_delay"
    DEFENSE_GAP = "defense_gap"
    MINERAL_FLOAT = "mineral_float"


class Severity(Enum):
    """How bad is this failure point."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EvidenceEntry:
    """A single raw evidence observation. Never edited after creation."""
    tick: int
    game_number: int
    observation_type: str  # "produced", "engaged", "terrain_used", "economic", "counter"
    data: Dict[str, Any] = field(default_factory=dict)
    outcome: str = ""  # "won", "lost", "inconclusive", ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "game_number": self.game_number,
            "observation_type": self.observation_type,
            "data": self.data,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceEntry":
        return cls(**d)


@dataclass
class ConfidenceRecord:
    """Tracks confidence for any learned thing. Stores raw evidence, not averages."""
    sample_count: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    confidence: float = 0.0
    variance: float = 0.0
    last_seen_tick: int = 0
    last_seen_game: int = 0
    last_success_tick: int = 0

    def record_outcome(self, outcome: str, tick: int, game_number: int) -> None:
        self.sample_count += 1
        self.last_seen_tick = tick
        self.last_seen_game = game_number
        if outcome == "won":
            self.wins += 1
            self.last_success_tick = tick
        elif outcome == "lost":
            self.losses += 1
        elif outcome == "inconclusive":
            self.draws += 1
        self._recompute()

    def _recompute(self) -> None:
        decided = self.wins + self.losses
        if decided > 0:
            self.confidence = self.wins / decided
            p = self.confidence
            self.variance = p * (1.0 - p) / decided
        else:
            self.confidence = 0.0
            self.variance = 0.0

    def decay(self, factor: float = 0.999) -> None:
        """Apply forgetting decay. Called once per game."""
        self.confidence *= factor
        self.variance *= factor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "confidence": round(self.confidence, 6),
            "variance": round(self.variance, 8),
            "last_seen_tick": self.last_seen_tick,
            "last_seen_game": self.last_seen_game,
            "last_success_tick": self.last_success_tick,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConfidenceRecord":
        return cls(**{k: d.get(k, v) for k, v in {
            "sample_count": 0, "wins": 0, "losses": 0, "draws": 0,
            "confidence": 0.0, "variance": 0.0,
            "last_seen_tick": 0, "last_seen_game": 0, "last_success_tick": 0,
        }.items()})


@dataclass
class StrategyContext:
    """Structured context snapshot when a strategy is evaluated."""
    tick: int = 0
    army_value: float = 0.0
    army_count: int = 0
    worker_count: int = 0
    base_count: int = 0
    minerals: int = 0
    vespene: int = 0
    supply_used: int = 0
    supply_cap: int = 0
    enemy_visible: int = 0
    enemy_army_value: float = 0.0
    terrain_complexity: float = 0.0
    cliff_density: float = 0.0
    visibility_advantage: float = 0.0
    tech_level: int = 0
    pressure_level: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyContext":
        return cls(**{k: d.get(k, v) for k, v in cls().__dict__.items()})


@dataclass
class Capability:
    """A discovered affordance — what the bot CAN do.

    Capabilities are distinct from strategies. A capability is an affordance
    (can cliff jump, can produce ranged units). A strategy is a sequencing
    decision that uses capabilities (attack after enemy moves out).
    """
    name: str
    category: str  # "unit", "structure", "terrain", "tactic", "economic"
    enables: List[str] = field(default_factory=list)
    first_seen_tick: int = 0
    last_seen_tick: int = 0
    games_observed: int = 0
    confidence: Optional[ConfidenceRecord] = None
    evidence: List[EvidenceEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "enables": self.enables,
            "first_seen_tick": self.first_seen_tick,
            "last_seen_tick": self.last_seen_tick,
            "games_observed": self.games_observed,
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "evidence": [e.to_dict() for e in self.evidence[-50:]],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Capability":
        cr = ConfidenceRecord.from_dict(d["confidence"]) if d.get("confidence") else None
        ev = [EvidenceEntry.from_dict(e) for e in d.get("evidence", [])]
        return cls(
            name=d["name"],
            category=d["category"],
            enables=d.get("enables", []),
            first_seen_tick=d.get("first_seen_tick", 0),
            last_seen_tick=d.get("last_seen_tick", 0),
            games_observed=d.get("games_observed", 0),
            confidence=cr,
            evidence=ev,
        )


@dataclass
class StrategyHypothesis:
    """A hypothesis about what strategy might work.

    Not a conclusion. A testable claim with structured evidence
    that other systems (Strategy Council) can reason about.
    """
    name: str
    description: str = ""
    capabilities_used: List[str] = field(default_factory=list)
    context: Optional[StrategyContext] = None
    confidence: Optional[ConfidenceRecord] = None
    evidence: List[EvidenceEntry] = field(default_factory=list)
    counterexamples: List[EvidenceEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "capabilities_used": self.capabilities_used,
            "context": self.context.to_dict() if self.context else None,
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "evidence": [e.to_dict() for e in self.evidence[-50:]],
            "counterexamples": [e.to_dict() for e in self.counterexamples[-20:]],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyHypothesis":
        ctx = StrategyContext.from_dict(d["context"]) if d.get("context") else None
        cr = ConfidenceRecord.from_dict(d["confidence"]) if d.get("confidence") else None
        ev = [EvidenceEntry.from_dict(e) for e in d.get("evidence", [])]
        ce = [EvidenceEntry.from_dict(e) for e in d.get("counterexamples", [])]
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            capabilities_used=d.get("capabilities_used", []),
            context=ctx,
            confidence=cr,
            evidence=ev,
            counterexamples=ce,
        )


class EpistemicLedger:
    """Cross-game epistemic memory. Records, measures, summarizes, indexes.

    Does NOT decide. Does NOT recommend. Other systems read this and reason.

    Hierarchical: Ledger -> Capabilities / Affordances / StrategyHypotheses
    -> Evidence -> Confidence.

    Persistence via JSON. Forgetting via confidence decay per game.
    """

    def __init__(self):
        self.capabilities: Dict[str, Capability] = {}
        self.strategy_hypotheses: Dict[str, StrategyHypothesis] = {}
        self.game_summaries: List[Dict[str, Any]] = []
        self._current_game: int = 0

    def begin_game(self, game_number: int) -> None:
        """Mark the start of a new game. Applies decay to all confidence."""
        self._current_game = game_number
        for cap in self.capabilities.values():
            if cap.confidence:
                cap.confidence.decay()
        for hyp in self.strategy_hypotheses.values():
            if hyp.confidence:
                hyp.confidence.decay()

    def observe_capability(self, name: str, category: str, tick: int,
                           enables: Optional[List[str]] = None,
                           evidence_data: Optional[Dict[str, Any]] = None) -> None:
        """Record that a capability was observed in use."""
        key = f"{category}:{name.upper()}"
        if key not in self.capabilities:
            self.capabilities[key] = Capability(
                name=name.upper(),
                category=category,
                enables=enables or [],
                first_seen_tick=tick,
                last_seen_tick=tick,
                games_observed=1,
                confidence=ConfidenceRecord(),
                evidence=[],
            )
        else:
            cap = self.capabilities[key]
            cap.last_seen_tick = tick
            if enables:
                for e in enables:
                    if e not in cap.enables:
                        cap.enables.append(e)

        ev = EvidenceEntry(
            tick=tick,
            game_number=self._current_game,
            observation_type="produced",
            data=evidence_data or {"capability": name},
        )
        self.capabilities[key].evidence.append(ev)
        self.capabilities[key].confidence.record_outcome("", tick, self._current_game)

    def add_hypothesis(self, name: str, description: str = "",
                       capabilities_used: Optional[List[str]] = None,
                       context: Optional[StrategyContext] = None) -> StrategyHypothesis:
        """Create or retrieve a strategy hypothesis."""
        if name not in self.strategy_hypotheses:
            self.strategy_hypotheses[name] = StrategyHypothesis(
                name=name,
                description=description,
                capabilities_used=capabilities_used or [],
                context=context,
                confidence=ConfidenceRecord(),
                evidence=[],
                counterexamples=[],
            )
        return self.strategy_hypotheses[name]

    def record_hypothesis_outcome(self, name: str, outcome: str, tick: int,
                                  evidence_data: Optional[Dict[str, Any]] = None,
                                  context: Optional[StrategyContext] = None) -> None:
        """Record an outcome for a strategy hypothesis.

        Does not judge. Just stores the evidence.
        """
        if name not in self.strategy_hypotheses:
            self.add_hypothesis(name, context=context)
        hyp = self.strategy_hypotheses[name]
        ev = EvidenceEntry(
            tick=tick,
            game_number=self._current_game,
            observation_type="engaged",
            data=evidence_data or {},
            outcome=outcome,
        )
        hyp.evidence.append(ev)
        hyp.confidence.record_outcome(outcome, tick, self._current_game)
        if context:
            hyp.context = context

        # Counterexamples: losses with strong context
        if outcome == "lost":
            hyp.counterexamples.append(ev)

    def end_game(self, game_number: int, result: str, tick_count: int,
                 capabilities_used: Optional[List[str]] = None) -> None:
        """Finalize a game. Record summary and mark capabilities observed."""
        for cap_name in (capabilities_used or []):
            key_match = [k for k in self.capabilities if k.endswith(f":{cap_name.upper()}")]
            for k in key_match:
                self.capabilities[k].games_observed += 1

        self.game_summaries.append({
            "game_number": game_number,
            "result": result,
            "tick_count": tick_count,
            "capabilities_tracked": len(self.capabilities),
            "hypotheses_tracked": len(self.strategy_hypotheses),
        })

    def get_capability(self, name: str, category: str = "") -> Optional[Capability]:
        key = f"{category}:{name.upper()}" if category else None
        if key and key in self.capabilities:
            return self.capabilities[key]
        for k, v in self.capabilities.items():
            if k.endswith(f":{name.upper()}"):
                return v
        return None

    def get_hypothesis(self, name: str) -> Optional[StrategyHypothesis]:
        return self.strategy_hypotheses.get(name)

    def top_hypotheses(self, n: int = 10) -> List[StrategyHypothesis]:
        ranked = sorted(
            self.strategy_hypotheses.values(),
            key=lambda h: h.confidence.confidence if h.confidence else 0,
            reverse=True,
        )
        return ranked[:n]

    def summary_text(self) -> str:
        lines = ["=== EPISTEMIC LEDGER ==="]
        lines.append(f"  Capabilities tracked: {len(self.capabilities)}")
        cats: Dict[str, int] = {}
        for cap in self.capabilities.values():
            cats[cap.category] = cats.get(cap.category, 0) + 1
        for cat, count in sorted(cats.items()):
            lines.append(f"    {cat}: {count}")

        lines.append(f"  Strategy hypotheses: {len(self.strategy_hypotheses)}")
        decided = [h for h in self.strategy_hypotheses.values()
                   if h.confidence and h.confidence.sample_count >= 3]
        if decided:
            best = sorted(decided, key=lambda h: h.confidence.confidence, reverse=True)[:5]
            lines.append(f"    Top hypotheses (3+ samples):")
            for h in best:
                c = h.confidence
                lines.append(f"      {h.name}: {c.confidence:.1%} "
                             f"({c.wins}W/{c.losses}L/{c.draws}D, n={c.sample_count})")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capabilities": {k: v.to_dict() for k, v in self.capabilities.items()},
            "strategy_hypotheses": {k: v.to_dict() for k, v in self.strategy_hypotheses.items()},
            "game_summaries": self.game_summaries[-100:],
        }

    def load(self, path: str) -> None:
        try:
            with open(path) as f:
                data = json.load(f)
            self.capabilities = {
                k: Capability.from_dict(v)
                for k, v in data.get("capabilities", {}).items()
            }
            self.strategy_hypotheses = {
                k: StrategyHypothesis.from_dict(v)
                for k, v in data.get("strategy_hypotheses", {}).items()
            }
            self.game_summaries = data.get("game_summaries", [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class TickSnapshot:
    """One frame of game state."""
    tick: int
    minerals: int = 0
    vespene: int = 0
    supply_used: int = 0
    supply_cap: int = 0
    worker_count: int = 0
    army_count: int = 0
    base_count: int = 0
    total_units: int = 0
    action_type: str = ""
    own_unit_names: List[str] = field(default_factory=list)
    enemy_unit_names: List[str] = field(default_factory=list)
    own_structure_names: List[str] = field(default_factory=list)
    units_attacking: int = 0
    units_moving: int = 0
    units_idle: int = 0
    army_value: float = 0.0
    enemy_army_value: float = 0.0

    @property
    def supply_blocked(self) -> bool:
        return self.supply_used >= self.supply_cap - 2 and self.supply_cap > 0

    @property
    def supply_ratio(self) -> float:
        if self.supply_cap == 0:
            return 0.0
        return self.supply_used / self.supply_cap

    @property
    def mineral_float(self) -> bool:
        return self.minerals > 500

    @property
    def economy_ratio(self) -> float:
        """Workers relative to army — high means more economy, low means more military."""
        total = self.worker_count + self.army_count
        if total == 0:
            return 0.5
        return self.worker_count / total


@dataclass
class FailurePoint:
    """A detected failure during gameplay."""
    tick: int
    category: FailureCategory
    severity: Severity
    description: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "context": self.context,
        }


@dataclass
class AuditReport:
    """Full game analysis after completion."""
    game_number: int = 0
    result: str = ""
    total_ticks: int = 0
    failure_points: List[FailurePoint] = field(default_factory=list)
    timeline: List[TickSnapshot] = field(default_factory=list)

    # Aggregated metrics
    supply_blocked_ticks: int = 0
    idle_production_ticks: int = 0
    mineral_float_ticks: int = 0
    total_units_lost: int = 0
    peak_army: int = 0
    peak_minerals: int = 0
    action_diversity: float = 0.0

    def severity_counts(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for fp in self.failure_points:
            counts[fp.severity.value] += 1
        return counts

    def category_counts(self) -> Dict[str, int]:
        counts = {c.value: 0 for c in FailureCategory}
        for fp in self.failure_points:
            counts[fp.category.value] += 1
        return counts

    def top_failures(self, n: int = 5) -> List[FailurePoint]:
        severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
        return sorted(self.failure_points, key=lambda fp: severity_order[fp.severity])[:n]

    def summary_text(self) -> str:
        lines = []
        lines.append(f"=== AUDIT REPORT: Game {self.game_number} ({self.result}) ===")
        lines.append(f"  Ticks: {self.total_ticks}")
        lines.append(f"  Failure points: {len(self.failure_points)}")

        counts = self.severity_counts()
        lines.append(f"  Critical: {counts['critical']}  High: {counts['high']}  Medium: {counts['medium']}  Low: {counts['low']}")
        lines.append(f"  Supply blocked ticks: {self.supply_blocked_ticks}")
        lines.append(f"  Idle production ticks: {self.idle_production_ticks}")
        lines.append(f"  Mineral float ticks: {self.mineral_float_ticks}")
        lines.append(f"  Peak army: {self.peak_army}  Peak minerals: {self.peak_minerals}")
        lines.append(f"  Action diversity: {self.action_diversity:.2f}")

        if self.failure_points:
            lines.append(f"\n  Top failure points:")
            for fp in self.top_failures(5):
                lines.append(f"    [{fp.severity.value.upper()}] tick {fp.tick}: {fp.description}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_number": self.game_number,
            "result": self.result,
            "total_ticks": self.total_ticks,
            "failure_points": [fp.to_dict() for fp in self.failure_points],
            "severity_counts": self.severity_counts(),
            "category_counts": self.category_counts(),
            "supply_blocked_ticks": self.supply_blocked_ticks,
            "idle_production_ticks": self.idle_production_ticks,
            "mineral_float_ticks": self.mineral_float_ticks,
            "total_units_lost": self.total_units_lost,
            "peak_army": self.peak_army,
            "peak_minerals": self.peak_minerals,
            "action_diversity": round(self.action_diversity, 3),
            "top_failures": [fp.to_dict() for fp in self.top_failures(10)],
        }


class ReplayAuditor:
    """Records tick-by-tick state and analyzes for failure points.

    Call record_tick() each game tick, then analyze() at game end.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._snapshots: List[TickSnapshot] = []
        self._failure_points: List[FailurePoint] = []

        # Thresholds (configurable)
        self._supply_block_threshold = 10      # ticks of supply block before flagging
        self._idle_production_threshold = 20   # ticks of no production before flagging
        self._mineral_float_threshold = 15     # ticks of >500 minerals before flagging
        self._army_idle_threshold = 30         # ticks of army doing nothing
        self._late_expansion_tick = 3000       # tick by which we should have expanded
        self._unit_loss_wave_threshold = 5     # units lost in short window
        self._action_degen_threshold = 10      # consecutive same action before flagging
        self._tech_delay_tick = 2000           # tick by which we should have tech

        # Internal tracking
        self._supply_block_streak = 0
        self._idle_prod_streak = 0
        self._mineral_float_streak = 0
        self._mineral_float_peak = 0
        self._army_idle_streak = 0
        self._consecutive_actions: List[str] = []
        self._prev_army_count = 0
        self._prev_worker_count = 0
        self._unit_loss_windows: List[Tuple[int, int]] = []  # (start_tick, count)

    def record_tick(self, tick: int, bot: Any, action_type: str = "") -> None:
        """Record one tick of game state from the bot."""
        # Build snapshot from bot state
        snap = TickSnapshot(
            tick=tick,
            minerals=bot.minerals,
            vespene=bot.vespene,
            supply_used=bot.supply_used,
            supply_cap=bot.supply_cap,
            worker_count=len(bot.workers),
            army_count=0,
            base_count=len(bot.townhalls),
            total_units=len(bot.units),
            action_type=action_type,
        )

        # Classify units
        worker_ids = {u.tag for u in bot.workers}
        townhall_ids = {s.type_id for s in bot.townhalls}
        spawned_names = {"LOCUST", "BROODLING", "INTERCEPTOR", "AUTOTURRET"}
        supply_names = {"OVERLORD", "OVERSEER", "OBSERVER"}

        for u in bot.units:
            uname = u.name.upper()
            if u.is_structure:
                snap.own_structure_names.append(uname)
                continue
            if u.tag in worker_ids or uname in supply_names or uname in spawned_names:
                continue

            snap.own_unit_names.append(uname)
            if u.can_attack:
                snap.army_count += 1
                snap.army_value += u.health + getattr(u, 'shield', 0)
                if u.is_attacking:
                    snap.units_attacking += 1
                elif u.is_moving:
                    snap.units_moving += 1
                else:
                    snap.units_idle += 1

        for eu in bot.known_enemy_units:
            if not eu.is_structure:
                snap.enemy_unit_names.append(eu.name.upper())
                snap.enemy_army_value += eu.health + getattr(eu, 'shield', 0)

        self._snapshots.append(snap)

        # --- Detect failures incrementally ---
        self._detect_supply_block(tick, snap)
        self._detect_idle_production(tick, snap, bot)
        self._detect_mineral_float(tick, snap)
        self._detect_army_idle(tick, snap)
        self._detect_unit_losses(tick, snap)
        self._detect_action_degeneration(tick, action_type)
        self._detect_late_expansion(tick, snap)
        self._detect_defense_gap(tick, snap)
        self._detect_economy_stall(tick, snap)

        self._prev_army_count = snap.army_count
        self._prev_worker_count = snap.worker_count

    def populate_ledger(self, ledger: EpistemicLedger, bot: Any) -> None:
        """Populate the EpistemicLedger with evidence from the latest snapshot.

        Pure recording — no decisions, no recommendations.
        Called from bot's on_step after record_tick.
        """
        if not self._snapshots:
            return
        snap = self._snapshots[-1]
        tick = snap.tick

        # Record capabilities observed
        seen_caps = set()
        for uname in snap.own_unit_names:
            cap_name = uname.lower()
            if cap_name not in seen_caps:
                seen_caps.add(cap_name)
                ledger.observe_capability(
                    name=cap_name, category="unit", tick=tick,
                    evidence_data={"unit_count": snap.own_unit_names.count(uname)},
                )
        for sname in snap.own_structure_names:
            cap_name = sname.lower()
            if cap_name not in seen_caps:
                seen_caps.add(cap_name)
                ledger.observe_capability(
                    name=cap_name, category="structure", tick=tick,
                )

        # Record terrain capability usage
        if hasattr(bot, 'encoder') and hasattr(bot.encoder, 'information'):
            info = bot.encoder.information
            if info.cliff_density > 0.15:
                ledger.observe_capability(
                    name="cliff_map", category="terrain", tick=tick,
                    evidence_data={
                        "cliff_density": info.cliff_density,
                        "terrain_complexity": info.terrain_complexity,
                    },
                )

        # Record enemy capabilities observed
        enemy_caps = set()
        for eu_name in snap.enemy_unit_names:
            if eu_name not in enemy_caps:
                enemy_caps.add(eu_name)
                ledger.observe_capability(
                    name=f"enemy_{eu_name.lower()}", category="unit", tick=tick,
                )

    # ── Failure Detectors ────────────────────────────────────────

    def _detect_supply_block(self, tick: int, snap: TickSnapshot) -> None:
        if snap.supply_blocked and snap.minerals >= 100:
            self._supply_block_streak += 1
        else:
            if self._supply_block_streak >= self._supply_block_threshold:
                self._failure_points.append(FailurePoint(
                    tick=tick - self._supply_block_streak,
                    category=FailureCategory.SUPPLY_BLOCKED,
                    severity=Severity.HIGH if self._supply_block_streak >= 30 else Severity.MEDIUM,
                    description=f"Supply blocked for {self._supply_block_streak} ticks (supply {snap.supply_used}/{snap.supply_cap})",
                    context={"duration": self._supply_block_streak, "supply": f"{snap.supply_used}/{snap.supply_cap}"},
                ))
            self._supply_block_streak = 0

    def _detect_idle_production(self, tick: int, snap: TickSnapshot, bot: Any) -> None:
        if tick < 100:
            return
        prod_buildings = 0
        idle_buildings = 0
        for s in bot.units.structure:
            if s.type_id not in {th.type_id for th in bot.townhalls}:
                prod_buildings += 1
                if hasattr(s, 'is_idle') and s.is_idle:
                    idle_buildings += 1
        if prod_buildings > 0 and idle_buildings == prod_buildings:
            self._idle_prod_streak += 1
        else:
            if self._idle_prod_streak >= self._idle_production_threshold:
                self._failure_points.append(FailurePoint(
                    tick=tick - self._idle_prod_streak,
                    category=FailureCategory.IDLE_PRODUCTION,
                    severity=Severity.HIGH if self._idle_prod_streak >= 50 else Severity.MEDIUM,
                    description=f"All {prod_buildings} production buildings idle for {self._idle_prod_streak} ticks",
                    context={"duration": self._idle_prod_streak, "buildings": prod_buildings},
                ))
            self._idle_prod_streak = 0

    def _detect_mineral_float(self, tick: int, snap: TickSnapshot) -> None:
        if snap.mineral_float:
            self._mineral_float_streak += 1
            if snap.minerals > self._mineral_float_peak:
                self._mineral_float_peak = snap.minerals
        else:
            if self._mineral_float_streak >= self._mineral_float_threshold:
                self._failure_points.append(FailurePoint(
                    tick=tick - self._mineral_float_streak,
                    category=FailureCategory.MINERAL_FLOAT,
                    severity=Severity.MEDIUM if self._mineral_float_streak < 30 else Severity.HIGH,
                    description=f"Minerals floated above 500 for {self._mineral_float_streak} ticks (peak {self._mineral_float_peak})",
                    context={"duration": self._mineral_float_streak, "peak_minerals": self._mineral_float_peak},
                ))
            self._mineral_float_streak = 0
            self._mineral_float_peak = 0

    def _detect_army_idle(self, tick: int, snap: TickSnapshot) -> None:
        if snap.army_count >= 5 and snap.units_idle == snap.army_count and snap.units_attacking == 0:
            self._army_idle_streak += 1
        else:
            if self._army_idle_streak >= self._army_idle_threshold:
                self._failure_points.append(FailurePoint(
                    tick=tick - self._army_idle_streak,
                    category=FailureCategory.ARMY_IDLE,
                    severity=Severity.HIGH if self._army_idle_streak >= 60 else Severity.MEDIUM,
                    description=f"Army ({snap.army_count} units) idle for {self._army_idle_streak} ticks",
                    context={"duration": self._army_idle_streak, "army_size": snap.army_count},
                ))
            self._army_idle_streak = 0

    def _detect_unit_losses(self, tick: int, snap: TickSnapshot) -> None:
        if self._prev_army_count > 0 and snap.army_count < self._prev_army_count:
            lost = self._prev_army_count - snap.army_count
            if lost >= 2:
                self._unit_loss_windows.append((tick, lost))
                # Check if many losses in short window (100 ticks)
                recent = [c for t, c in self._unit_loss_windows if tick - t <= 100]
                total_lost = sum(recent)
                if total_lost >= self._unit_loss_wave_threshold:
                    self._failure_points.append(FailurePoint(
                        tick=tick,
                        category=FailureCategory.UNIT_LOSS_WAVE,
                        severity=Severity.CRITICAL if total_lost >= 10 else Severity.HIGH,
                        description=f"Lost {total_lost} army units in 100 ticks (army: {snap.army_count})",
                        context={"total_lost": total_lost, "army_remaining": snap.army_count},
                    ))
                    self._unit_loss_windows = [(t, c) for t, c in self._unit_loss_windows if tick - t > 100]

    def _detect_action_degeneration(self, tick: int, action_type: str) -> None:
        self._consecutive_actions.append(action_type)
        if len(self._consecutive_actions) > 50:
            self._consecutive_actions = self._consecutive_actions[-50:]

        if len(self._consecutive_actions) >= self._action_degen_threshold:
            last_n = self._consecutive_actions[-self._action_degen_threshold:]
            if len(set(last_n)) == 1 and last_n[0] not in ("hold", ""):
                # Check if we already reported this recently
                recent_degen = any(
                    fp.category == FailureCategory.ACTION_DEGENERATE
                    and tick - fp.tick < 200
                    for fp in self._failure_points
                )
                if not recent_degen:
                    self._failure_points.append(FailurePoint(
                        tick=tick - self._action_degen_threshold,
                        category=FailureCategory.ACTION_DEGENERATE,
                        severity=Severity.MEDIUM,
                        description=f"Action '{action_type}' repeated {self._action_degen_threshold}+ times in a row",
                        context={"action": action_type, "streak": self._action_degen_threshold},
                    ))

    def _detect_late_expansion(self, tick: int, snap: TickSnapshot) -> None:
        if tick == self._late_expansion_tick and snap.base_count <= 1:
            self._failure_points.append(FailurePoint(
                tick=tick,
                category=FailureCategory.LATE_EXPANSION,
                severity=Severity.HIGH,
                description=f"No expansion by tick {tick} (still {snap.base_count} base(s))",
                context={"base_count": snap.base_count},
            ))

    def _detect_defense_gap(self, tick: int, snap: TickSnapshot) -> None:
        enemy_near = len(snap.enemy_unit_names)
        if enemy_near > 0 and snap.army_count == 0 and snap.units_attacking == 0:
            recent_gap = any(
                fp.category == FailureCategory.DEFENSE_GAP and tick - fp.tick < 200
                for fp in self._failure_points
            )
            if not recent_gap:
                self._failure_points.append(FailurePoint(
                    tick=tick,
                    category=FailureCategory.DEFENSE_GAP,
                    severity=Severity.CRITICAL,
                    description=f"Enemy units visible ({enemy_near}) but no army to defend",
                    context={"enemy_count": enemy_near},
                ))

    def _detect_economy_stall(self, tick: int, snap: TickSnapshot) -> None:
        if tick < 200:
            return
        # Worker count dropped significantly
        if self._prev_worker_count > 0 and snap.worker_count < self._prev_worker_count - 2:
            recent_stall = any(
                fp.category == FailureCategory.ECONOMY_STALLED and tick - fp.tick < 300
                for fp in self._failure_points
            )
            if not recent_stall:
                self._failure_points.append(FailurePoint(
                    tick=tick,
                    category=FailureCategory.ECONOMY_STALLED,
                    severity=Severity.HIGH,
                    description=f"Workers dropped from {self._prev_worker_count} to {snap.worker_count}",
                    context={"prev_workers": self._prev_worker_count, "now_workers": snap.worker_count},
                ))

    # ── Analysis ─────────────────────────────────────────────────

    def _flush_streaks(self) -> None:
        """Flush any ongoing streaks as failure points at game end."""
        last_tick = self._snapshots[-1].tick if self._snapshots else 0

        if self._supply_block_streak >= self._supply_block_threshold:
            snap = self._snapshots[-1] if self._snapshots else None
            self._failure_points.append(FailurePoint(
                tick=last_tick - self._supply_block_streak,
                category=FailureCategory.SUPPLY_BLOCKED,
                severity=Severity.HIGH if self._supply_block_streak >= 30 else Severity.MEDIUM,
                description=f"Supply blocked for {self._supply_block_streak} ticks (supply {snap.supply_used}/{snap.supply_cap})" if snap else f"Supply blocked for {self._supply_block_streak} ticks",
                context={"duration": self._supply_block_streak},
            ))

        if self._idle_prod_streak >= self._idle_production_threshold:
            self._failure_points.append(FailurePoint(
                tick=last_tick - self._idle_prod_streak,
                category=FailureCategory.IDLE_PRODUCTION,
                severity=Severity.HIGH if self._idle_prod_streak >= 50 else Severity.MEDIUM,
                description=f"Production buildings idle for {self._idle_prod_streak} ticks",
                context={"duration": self._idle_prod_streak},
            ))

        if self._mineral_float_streak >= self._mineral_float_threshold:
            self._failure_points.append(FailurePoint(
                tick=last_tick - self._mineral_float_streak,
                category=FailureCategory.MINERAL_FLOAT,
                severity=Severity.MEDIUM if self._mineral_float_streak < 30 else Severity.HIGH,
                description=f"Minerals above 500 for {self._mineral_float_streak} ticks (peak {self._mineral_float_peak})",
                context={"duration": self._mineral_float_streak, "peak_minerals": self._mineral_float_peak},
            ))

        if self._army_idle_streak >= self._army_idle_threshold:
            snap = self._snapshots[-1] if self._snapshots else None
            self._failure_points.append(FailurePoint(
                tick=last_tick - self._army_idle_streak,
                category=FailureCategory.ARMY_IDLE,
                severity=Severity.HIGH if self._army_idle_streak >= 60 else Severity.MEDIUM,
                description=f"Army ({snap.army_count if snap else '?'} units) idle for {self._army_idle_streak} ticks",
                context={"duration": self._army_idle_streak},
            ))

    def analyze(self, game_result: str = "", game_number: int = 0) -> AuditReport:
        """Produce full audit report from recorded ticks."""
        # Flush any ongoing streaks as failures
        self._flush_streaks()

        report = AuditReport(game_number=game_number, result=game_result)
        report.timeline = list(self._snapshots)
        report.total_ticks = len(self._snapshots)
        report.failure_points = list(self._failure_points)

        # Aggregate metrics
        if self._snapshots:
            report.supply_blocked_ticks = sum(1 for s in self._snapshots if s.supply_blocked)
            report.idle_production_ticks = self._idle_prod_streak
            report.mineral_float_ticks = sum(1 for s in self._snapshots if s.mineral_float)
            report.peak_army = max(s.army_count for s in self._snapshots)
            report.peak_minerals = max(s.minerals for s in self._snapshots)

            # Unit losses
            for i in range(1, len(self._snapshots)):
                prev = self._snapshots[i - 1]
                curr = self._snapshots[i]
                if prev.army_count > curr.army_count:
                    report.total_units_lost += prev.army_count - curr.army_count

            # Action diversity (Shannon entropy)
            action_counts: Dict[str, int] = {}
            for s in self._snapshots:
                if s.action_type:
                    action_counts[s.action_type] = action_counts.get(s.action_type, 0) + 1
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                import math
                entropy = 0.0
                for count in action_counts.values():
                    p = count / total_actions
                    if p > 0:
                        entropy -= p * math.log2(p)
                max_entropy = math.log2(max(1, len(action_counts)))
                report.action_diversity = entropy / max_entropy if max_entropy > 0 else 0.0

        return report

    def reset(self) -> None:
        """Reset for next game."""
        self._snapshots.clear()
        self._failure_points.clear()
        self._supply_block_streak = 0
        self._idle_prod_streak = 0
        self._mineral_float_streak = 0
        self._mineral_float_peak = 0
        self._army_idle_streak = 0
        self._consecutive_actions.clear()
        self._prev_army_count = 0
        self._prev_worker_count = 0
        self._unit_loss_windows.clear()
