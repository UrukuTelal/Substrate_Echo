"""Tests for ReplayAuditor — tick-by-tick failure analysis."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_echo.epistemology.replay_auditor import (
    ReplayAuditor, AuditReport, FailurePoint, TickSnapshot,
    FailureCategory, Severity,
)


class FakeUnit:
    """Minimal unit mock for testing."""
    def __init__(self, tag, name, can_attack=False, is_structure=False,
                 is_idle=True, is_attacking=False, is_moving=False,
                 health=50, shield=0):
        self.tag = tag
        self.name = name
        self.can_attack = can_attack
        self.is_structure = is_structure
        self.is_idle = is_idle
        self.is_attacking = is_attacking
        self.is_moving = is_moving
        self.health = health
        self.shield = shield
        self.type_id = name


class UnitsList(list):
    """List with .structure attribute, like SC2 Units object."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.structure = []


class FakeBot:
    """Minimal bot mock for testing."""
    def __init__(self, minerals=100, vespene=50, supply_used=10, supply_cap=15,
                 n_workers=10, n_army=5, n_bases=1, enemy_count=0):
        self.minerals = minerals
        self.vespene = vespene
        self.supply_used = supply_used
        self.supply_cap = supply_cap
        self.workers = [FakeUnit(i, 'DRONE') for i in range(n_workers)]
        all_units = list(self.workers)
        for i in range(n_army):
            all_units.append(FakeUnit(1000 + i, 'ZERGLING', can_attack=True))
        structures = [FakeUnit(2000 + i, 'HATCHERY', is_structure=True)
                      for i in range(n_bases)]
        self.units = UnitsList(all_units)
        self.units.structure = structures
        self.townhalls = [FakeUnit(2000 + i, 'HATCHERY', is_structure=True)
                          for i in range(n_bases)]
        self.known_enemy_units = [FakeUnit(3000 + i, 'MARINE', can_attack=True)
                                  for i in range(enemy_count)]


# ── Basic Recording ──────────────────────────────────────────────

class TestRecording:

    def test_record_single_tick(self):
        auditor = ReplayAuditor()
        bot = FakeBot()
        auditor.record_tick(0, bot, 'build_army')
        report = auditor.analyze('Defeat', 1)
        assert len(report.timeline) == 1
        assert report.total_ticks == 1

    def test_record_multiple_ticks(self):
        auditor = ReplayAuditor()
        bot = FakeBot()
        for i in range(50):
            auditor.record_tick(i, bot, 'build_army')
        report = auditor.analyze('Defeat', 1)
        assert len(report.timeline) == 50

    def test_snapshot_fields(self):
        auditor = ReplayAuditor()
        bot = FakeBot(minerals=200, vespene=100, supply_used=20, supply_cap=30)
        auditor.record_tick(5, bot, 'attack')
        report = auditor.analyze()
        snap = report.timeline[0]
        assert snap.minerals == 200
        assert snap.vespene == 100
        assert snap.supply_used == 20
        assert snap.supply_cap == 30
        assert snap.worker_count == 10
        assert snap.base_count == 1

    def test_reset_clears_state(self):
        auditor = ReplayAuditor()
        bot = FakeBot()
        auditor.record_tick(0, bot, 'build_army')
        auditor.reset()
        report = auditor.analyze()
        assert len(report.timeline) == 0


# ── Supply Block Detection ───────────────────────────────────────

class TestSupplyBlock:

    def test_supply_blocked_detected(self):
        auditor = ReplayAuditor()
        # 15 ticks of supply blocked + minerals to build
        bot = FakeBot(supply_used=13, supply_cap=15, minerals=200)
        for i in range(20):
            auditor.record_tick(i, bot, 'build_army')
        report = auditor.analyze()
        fp = [f for f in report.failure_points
              if f.category == FailureCategory.SUPPLY_BLOCKED]
        assert len(fp) == 1
        assert fp[0].severity in (Severity.MEDIUM, Severity.HIGH)

    def test_no_supply_block_when_not_blocked(self):
        auditor = ReplayAuditor()
        bot = FakeBot(supply_used=5, supply_cap=20, minerals=200)
        for i in range(20):
            auditor.record_tick(i, bot, 'build_army')
        report = auditor.analyze()
        fp = [f for f in report.failure_points
              if f.category == FailureCategory.SUPPLY_BLOCKED]
        assert len(fp) == 0


# ── Mineral Float Detection ──────────────────────────────────────

class TestMineralFloat:

    def test_mineral_float_detected(self):
        auditor = ReplayAuditor()
        bot = FakeBot(minerals=800)
        for i in range(20):
            auditor.record_tick(i, bot, 'build_army')
        report = auditor.analyze()
        fp = [f for f in report.failure_points
              if f.category == FailureCategory.MINERAL_FLOAT]
        assert len(fp) == 1

    def test_no_float_when_low_minerals(self):
        auditor = ReplayAuditor()
        bot = FakeBot(minerals=50)
        for i in range(20):
            auditor.record_tick(i, bot, 'build_army')
        report = auditor.analyze()
        fp = [f for f in report.failure_points
              if f.category == FailureCategory.MINERAL_FLOAT]
        assert len(fp) == 0


# ── Action Degeneration Detection ────────────────────────────────

class TestActionDegeneration:

    def test_degenerate_action_detected(self):
        auditor = ReplayAuditor()
        bot = FakeBot()
        for i in range(15):
            auditor.record_tick(i, bot, 'build_army')
        report = auditor.analyze()
        fp = [f for f in report.failure_points
              if f.category == FailureCategory.ACTION_DEGENERATE]
        assert len(fp) == 1
        assert 'build_army' in fp[0].description

    def test_diverse_actions_no_degeneration(self):
        auditor = ReplayAuditor()
        bot = FakeBot()
        actions = ['build_army', 'attack', 'defend', 'scout', 'expand']
        for i in range(20):
            auditor.record_tick(i, bot, actions[i % len(actions)])
        report = auditor.analyze()
        fp = [f for f in report.failure_points
              if f.category == FailureCategory.ACTION_DEGENERATE]
        assert len(fp) == 0


# ── Defense Gap Detection ────────────────────────────────────────

class TestDefenseGap:

    def test_defense_gap_detected(self):
        auditor = ReplayAuditor()
        bot = FakeBot(n_army=0, enemy_count=3)
        for i in range(5):
            auditor.record_tick(i, bot, 'defend')
        report = auditor.analyze()
        fp = [f for f in report.failure_points
              if f.category == FailureCategory.DEFENSE_GAP]
        assert len(fp) == 1
        assert fp[0].severity == Severity.CRITICAL


# ── Economy Stall Detection ──────────────────────────────────────

class TestEconomyStall:

    def test_economy_stall_detected(self):
        auditor = ReplayAuditor()
        # Start with 15 workers, then lose some (tick > 200 required)
        bot1 = FakeBot(n_workers=15)
        for i in range(203, 206):
            auditor.record_tick(i, bot1, 'build_army')
        bot2 = FakeBot(n_workers=10)
        for i in range(206, 210):
            auditor.record_tick(i, bot2, 'build_army')
        report = auditor.analyze()
        fp = [f for f in report.failure_points
              if f.category == FailureCategory.ECONOMY_STALLED]
        assert len(fp) == 1


# ── Army Idle Detection ──────────────────────────────────────────

class TestArmyIdle:

    def test_army_idle_detected(self):
        auditor = ReplayAuditor()
        bot = FakeBot(n_army=10)
        for i in range(35):
            auditor.record_tick(i, bot, 'defend')
        report = auditor.analyze()
        fp = [f for f in report.failure_points
              if f.category == FailureCategory.ARMY_IDLE]
        assert len(fp) == 1


# ── Unit Loss Wave Detection ─────────────────────────────────────

class TestUnitLoss:

    def test_unit_loss_wave_detected(self):
        auditor = ReplayAuditor()
        # Army of 10, then army of 3
        bot1 = FakeBot(n_army=10)
        for i in range(5):
            auditor.record_tick(i, bot1, 'attack')
        bot2 = FakeBot(n_army=3)
        for i in range(5, 10):
            auditor.record_tick(i, bot2, 'attack')
        report = auditor.analyze()
        fp = [f for f in report.failure_points
              if f.category == FailureCategory.UNIT_LOSS_WAVE]
        assert len(fp) == 1
        assert fp[0].severity in (Severity.HIGH, Severity.CRITICAL)


# ── Audit Report ─────────────────────────────────────────────────

class TestAuditReport:

    def test_severity_counts(self):
        auditor = ReplayAuditor()
        bot = FakeBot(minerals=800, n_army=0, enemy_count=3)
        for i in range(20):
            auditor.record_tick(i, bot, 'build_army')
        report = auditor.analyze('Defeat', 1)
        counts = report.severity_counts()
        assert sum(counts.values()) > 0

    def test_category_counts(self):
        auditor = ReplayAuditor()
        bot = FakeBot()
        for i in range(20):
            auditor.record_tick(i, bot, 'build_army')
        report = auditor.analyze()
        counts = report.category_counts()
        assert 'action_degenerate' in counts

    def test_top_failures_sorted_by_severity(self):
        auditor = ReplayAuditor()
        bot = FakeBot(minerals=800, n_army=10, enemy_count=3)
        for i in range(50):
            auditor.record_tick(i, bot, 'defend')
        report = auditor.analyze()
        top = report.top_failures(3)
        if len(top) >= 2:
            sev_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
            for j in range(len(top) - 1):
                assert sev_order[top[j].severity] <= sev_order[top[j + 1].severity]

    def test_to_dict(self):
        auditor = ReplayAuditor()
        bot = FakeBot()
        auditor.record_tick(0, bot, 'build_army')
        report = auditor.analyze('Victory', 1)
        d = report.to_dict()
        assert d['game_number'] == 1
        assert d['result'] == 'Victory'
        assert 'failure_points' in d
        assert 'severity_counts' in d

    def test_summary_text(self):
        auditor = ReplayAuditor()
        bot = FakeBot()
        auditor.record_tick(0, bot, 'build_army')
        report = auditor.analyze('Victory', 1)
        text = report.summary_text()
        assert 'AUDIT REPORT' in text
        assert 'Game 1' in text

    def test_peak_metrics(self):
        auditor = ReplayAuditor()
        bot1 = FakeBot(minerals=100, n_army=5)
        auditor.record_tick(0, bot1, 'build_army')
        bot2 = FakeBot(minerals=500, n_army=15)
        auditor.record_tick(1, bot2, 'build_army')
        report = auditor.analyze()
        assert report.peak_minerals == 500
        assert report.peak_army == 15

    def test_action_diversity(self):
        auditor = ReplayAuditor()
        bot = FakeBot()
        # Single action = 0 diversity
        for i in range(10):
            auditor.record_tick(i, bot, 'build_army')
        report1 = auditor.analyze()
        assert report1.action_diversity == 0.0

        # Diverse actions = higher diversity
        auditor2 = ReplayAuditor()
        actions = ['build_army', 'attack', 'defend', 'scout', 'expand']
        for i in range(10):
            auditor2.record_tick(i, bot, actions[i % len(actions)])
        report2 = auditor2.analyze()
        assert report2.action_diversity > 0.0


# ── TickSnapshot Properties ──────────────────────────────────────

class TestTickSnapshot:

    def test_supply_blocked_property(self):
        snap = TickSnapshot(tick=0, supply_used=14, supply_cap=15, minerals=200)
        assert snap.supply_blocked is True

    def test_not_supply_blocked(self):
        snap = TickSnapshot(tick=0, supply_used=5, supply_cap=15)
        assert snap.supply_blocked is False

    def test_supply_ratio(self):
        snap = TickSnapshot(tick=0, supply_used=10, supply_cap=20)
        assert snap.supply_ratio == 0.5

    def test_mineral_float_property(self):
        snap = TickSnapshot(tick=0, minerals=600)
        assert snap.mineral_float is True
        snap2 = TickSnapshot(tick=0, minerals=100)
        assert snap2.mineral_float is False

    def test_economy_ratio(self):
        snap = TickSnapshot(tick=0, worker_count=10, army_count=10)
        assert snap.economy_ratio == 0.5
        snap2 = TickSnapshot(tick=0, worker_count=0, army_count=0)
        assert snap2.economy_ratio == 0.5  # default
