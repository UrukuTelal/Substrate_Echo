"""Tests for Council Layer (S12)."""
from __future__ import annotations
import pytest
import time
from substrate_echo.kernel.council import (
    AuditType, AuditSeverity, AuditTrigger,
    AuditReport, DriftMetrics, CouncilState,
    DriftDetector, ScheduledAuditor, EventAuditor, Council
)


def test_audit_report_to_dict():
    report = AuditReport(
        audit_id=1, tick=100, audit_type=AuditType.SCHEDULED,
        trigger=AuditTrigger.SCHEDULED, timestamp=time.time(),
        observations=["test"], anomalies=["anomaly1"],
        hypotheses=["hypo1"], recommendations=["rec1"],
        confidence=0.7, severity=AuditSeverity.WARNING,
    )
    d = report.to_dict()
    assert d["audit_id"] == 1
    assert d["severity"] == "warning"
    assert "anomaly1" in d["anomalies"]


def test_drift_metrics_to_dict():
    dm = DriftMetrics(tick=10, attractor_drift=0.2, overall_drift=0.15)
    d = dm.to_dict()
    assert d["attractor_drift"] == 0.2
    assert d["overall_drift"] == 0.15


def test_drift_detector_baseline():
    dd = DriftDetector()
    dd.set_baseline({"n_attractors": 10, "coherence": 0.5})
    dm = dd.update({"n_attractors": 12, "coherence": 0.6})
    assert dm.attractor_drift == pytest.approx(2.0)
    assert dm.confidence_drift == pytest.approx(0.1)


def test_drift_detector_no_baseline():
    dd = DriftDetector()
    dm = dd.update({"n_attractors": 10})
    assert dm.overall_drift == 0.0


def test_scheduled_auditor_interval():
    sa = ScheduledAuditor(interval_ticks=100)
    assert not sa.should_audit(50)
    assert sa.should_audit(100)
    assert not sa.should_audit(150)
    assert sa.should_audit(200)


def test_scheduled_auditor_report():
    sa = ScheduledAuditor(interval_ticks=100)
    report = sa.create_report(100, {"n_attractors": 5, "coherence": 0.1,
                                    "volume_entropy": 0.05, "n_goals": 2,
                                    "n_embodiments": 1}, 1)
    assert report.audit_type == AuditType.SCHEDULED
    assert report.severity == AuditSeverity.WARNING
    assert len(report.anomalies) > 0


def test_event_auditor_attractor_collapse():
    ea = EventAuditor()
    prev = {"n_attractors": 10}
    curr = {"n_attractors": 3}
    triggers = ea.check_triggers(curr, prev)
    assert AuditTrigger.ATTRACTOR_COLLAPSE in triggers


def test_event_auditor_no_collapse():
    ea = EventAuditor()
    prev = {"n_attractors": 10}
    curr = {"n_attractors": 8}
    triggers = ea.check_triggers(curr, prev)
    assert AuditTrigger.ATTRACTOR_COLLAPSE not in triggers


def test_event_auditor_memory_explosion():
    ea = EventAuditor()
    triggers = ea.check_triggers({"n_attractors": 150})
    assert AuditTrigger.MEMORY_EXPLOSION in triggers


def test_event_auditor_report():
    ea = EventAuditor()
    report = ea.create_report(100, AuditTrigger.ATTRACTOR_COLLAPSE,
                              {"n_attractors": 3}, 1)
    assert report.audit_type == AuditType.EVENT
    assert report.severity == AuditSeverity.WARNING
    assert len(report.recommendations) > 0


def test_council_tick_no_reports():
    council = Council(audit_interval=1000)
    reports = council.tick({"tick": 10, "n_attractors": 5})
    assert len(reports) == 0


def test_council_tick_scheduled_report():
    council = Council(audit_interval=100)
    reports = council.tick({"tick": 100, "n_attractors": 5, "coherence": 0.5,
                            "volume_entropy": 0.5, "n_goals": 2,
                            "n_embodiments": 1})
    assert len(reports) == 1
    assert reports[0].audit_type == AuditType.SCHEDULED


def test_council_tick_event_report():
    council = Council(audit_interval=1000)
    council.tick({"tick": 10, "n_attractors": 10})
    reports = council.tick({"tick": 20, "n_attractors": 3})
    assert any(r.trigger == AuditTrigger.ATTRACTOR_COLLAPSE for r in reports)


def test_council_manual_audit():
    council = Council()
    report = council.manual_audit(
        {"tick": 50}, observations=["manual check"])
    assert report.audit_type == AuditType.MANUAL
    assert "manual check" in report.observations


def test_council_get_reports():
    council = Council(audit_interval=50)
    for i in range(5):
        council.tick({"tick": i * 50, "n_attractors": 5, "coherence": 0.5,
                      "volume_entropy": 0.5, "n_goals": 2, "n_embodiments": 1})
    reports = council.get_reports(n=3)
    assert len(reports) == 3


def test_council_state():
    council = Council(audit_interval=50)
    council.tick({"tick": 50, "n_attractors": 5, "coherence": 0.5,
                  "volume_entropy": 0.5, "n_goals": 2, "n_embodiments": 1})
    state = council.get_state()
    assert state.n_audits == 1
    assert state.health_score == pytest.approx(1.0)
    assert len(state.pending_reports) == 1


def test_council_health_degrades():
    council = Council(audit_interval=10)
    for i in range(20):
        council.tick({"tick": i * 10, "n_attractors": 0, "coherence": 0.1,
                      "volume_entropy": 0.05, "n_goals": 0, "n_embodiments": 1})
    state = council.get_state()
    assert state.health_score < 1.0


def test_council_set_baseline():
    council = Council()
    council.set_baseline({"n_attractors": 10, "coherence": 0.5})
    assert council.drift._baseline is not None
