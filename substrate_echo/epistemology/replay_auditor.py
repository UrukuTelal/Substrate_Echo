"""Replay Auditor — Analyzes tick-by-tick data for failure points.

Records every tick's state during a game and produces an audit report
identifying: supply blocks, idle production, economy stalls, army
inaction, composition mismatches, unit losses, and action degeneration.

Architecture:
    TickSnapshot   — one frame of game state (units, resources, supply, actions)
    FailurePoint   — a detected issue with tick, severity, category, description
    AuditReport    — full game analysis with failure points, timeline, summary
    ReplayAuditor  — records ticks, detects failures, produces report

Usage in bot:
    auditor = ReplayAuditor()
    # Each tick:
    auditor.record_tick(tick, bot, action_type)
    # Game end:
    report = auditor.analyze(game_result)
    report.print_summary()
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
        else:
            if self._mineral_float_streak >= self._mineral_float_threshold:
                self._failure_points.append(FailurePoint(
                    tick=tick - self._mineral_float_streak,
                    category=FailureCategory.MINERAL_FLOAT,
                    severity=Severity.MEDIUM if self._mineral_float_streak < 30 else Severity.HIGH,
                    description=f"Minerals floated above 500 for {self._mineral_float_streak} ticks (peak {snap.minerals})",
                    context={"duration": self._mineral_float_streak, "peak_minerals": snap.minerals},
                ))
            self._mineral_float_streak = 0

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
            snap = self._snapshots[-1] if self._snapshots else None
            self._failure_points.append(FailurePoint(
                tick=last_tick - self._mineral_float_streak,
                category=FailureCategory.MINERAL_FLOAT,
                severity=Severity.MEDIUM if self._mineral_float_streak < 30 else Severity.HIGH,
                description=f"Minerals above 500 for {self._mineral_float_streak} ticks",
                context={"duration": self._mineral_float_streak, "peak_minerals": snap.minerals if snap else 0},
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
        self._army_idle_streak = 0
        self._consecutive_actions.clear()
        self._prev_army_count = 0
        self._prev_worker_count = 0
        self._unit_loss_windows.clear()
