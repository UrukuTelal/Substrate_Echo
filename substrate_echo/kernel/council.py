"""Council Layer — Metacognition and periodic health checks.

The Council is not part of the constant cognition loop.
It operates as an audit process — like a periodic health check.

Responsibilities:
  1. Scheduled audits (every N ticks, time-based)
  2. Event-triggered audits (anomaly detection)
  3. Drift detection (architectural, concept, attention)
  4. Experiment suggestions (hypothesis testing)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import time


class AuditType(Enum):
    SCHEDULED = "scheduled"
    EVENT = "event"
    MANUAL = "manual"


class AuditSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AuditTrigger(Enum):
    SCHEDULED = "scheduled"
    ATTRACTOR_COLLAPSE = "attractor_collapse"
    ENTROPY_REDUCTION = "entropy_reduction"
    MEMORY_EXPLOSION = "memory_explosion"
    CONFIDENCE_OSCILLATION = "confidence_oscillation"
    PREDICTION_FAILURE = "prediction_failure"
    ARCHITECTURE_CHANGE = "architecture_change"
    RESOURCE_ANOMALY = "resource_anomaly"
    MANUAL = "manual"


@dataclass
class AuditReport:
    """Output of a council audit."""
    audit_id: int
    tick: int
    audit_type: AuditType
    trigger: AuditTrigger
    timestamp: float
    observations: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.5
    severity: AuditSeverity = AuditSeverity.INFO
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "tick": self.tick,
            "audit_type": self.audit_type.value,
            "trigger": self.trigger.value,
            "timestamp": self.timestamp,
            "observations": self.observations,
            "anomalies": self.anomalies,
            "hypotheses": self.hypotheses,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "metadata": self.metadata,
        }


@dataclass
class DriftMetrics:
    """Tracks drift between current and baseline substrate state."""
    tick: int
    attractor_drift: float = 0.0
    entropy_drift: float = 0.0
    confidence_drift: float = 0.0
    resource_drift: float = 0.0
    overall_drift: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "attractor_drift": self.attractor_drift,
            "entropy_drift": self.entropy_drift,
            "confidence_drift": self.confidence_drift,
            "resource_drift": self.resource_drift,
            "overall_drift": self.overall_drift,
        }


@dataclass
class CouncilState:
    """Current state of the council."""
    last_audit_tick: int = 0
    last_audit_time: float = 0.0
    n_audits: int = 0
    n_reports: int = 0
    n_recommendations: int = 0
    drift_score: float = 0.0
    health_score: float = 1.0
    recent_severity: str = "info"
    pending_reports: List[Dict[str, Any]] = field(default_factory=list)


class DriftDetector:
    """Detects architectural and concept drift in the substrate."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._baseline: Optional[Dict[str, float]] = None
        self._history: List[Dict[str, float]] = []

    def set_baseline(self, metrics: Dict[str, float]):
        self._baseline = metrics.copy()

    def update(self, metrics: Dict[str, float]) -> DriftMetrics:
        self._history.append(metrics)
        if len(self._history) > self.window_size:
            self._history.pop(0)

        if not self._baseline:
            return DriftMetrics(tick=0)

        drift = {}
        for key in self._baseline:
            if key in metrics:
                drift[key] = abs(metrics[key] - self._baseline[key])
            else:
                drift[key] = 0.0

        return DriftMetrics(
            tick=len(self._history),
            attractor_drift=drift.get("n_attractors", 0.0),
            entropy_drift=drift.get("volume_entropy", 0.0),
            confidence_drift=drift.get("coherence", 0.0),
            resource_drift=drift.get("resource_utilization", 0.0),
            overall_drift=sum(drift.values()) / max(1, len(drift)),
        )


class ScheduledAuditor:
    """Runs audits on a schedule."""

    def __init__(self, interval_ticks: int = 500):
        self.interval_ticks = interval_ticks
        self._last_audit_tick = 0

    def should_audit(self, current_tick: int) -> bool:
        if current_tick - self._last_audit_tick >= self.interval_ticks:
            self._last_audit_tick = current_tick
            return True
        return False

    def create_report(self, tick: int, substrate_state: Dict[str, Any],
                      audit_id: int) -> AuditReport:
        observations = []
        anomalies = []
        hypotheses = []
        recommendations = []
        severity = AuditSeverity.INFO

        n_attractors = substrate_state.get("n_attractors", 0)
        coherence = substrate_state.get("coherence", 0.5)
        volume_entropy = substrate_state.get("volume_entropy", 0.5)
        n_goals = substrate_state.get("n_goals", 0)
        n_embodiments = substrate_state.get("n_embodiments", 0)

        observations.append(f"Attractors: {n_attractors}")
        observations.append(f"Coherence: {coherence:.3f}")
        observations.append(f"Volume entropy: {volume_entropy:.3f}")
        observations.append(f"Active goals: {n_goals}")
        observations.append(f"Embodiments: {n_embodiments}")

        if n_attractors == 0:
            anomalies.append("No attractors formed")
            hypotheses.append("Insufficient data or poor dynamics model")
            recommendations.append("Collect more samples or adjust dynamics parameters")
            severity = AuditSeverity.WARNING

        if coherence < 0.2:
            anomalies.append(f"Low coherence: {coherence:.3f}")
            hypotheses.append("Attractors may be too spread or noisy")
            recommendations.append("Consider consolidation or noise reduction")
            severity = AuditSeverity.WARNING

        if volume_entropy < 0.1:
            anomalies.append(f"Low entropy: {volume_entropy:.3f}")
            hypotheses.append("System may be over-specialized")
            recommendations.append("Encourage exploration or add noise")
            severity = AuditSeverity.WARNING

        if volume_entropy > 0.9:
            anomalies.append(f"High entropy: {volume_entropy:.3f}")
            hypotheses.append("System may be under-specialized")
            recommendations.append("Allow more consolidation time")
            severity = AuditSeverity.WARNING

        return AuditReport(
            audit_id=audit_id,
            tick=tick,
            audit_type=AuditType.SCHEDULED,
            trigger=AuditTrigger.SCHEDULED,
            timestamp=time.time(),
            observations=observations,
            anomalies=anomalies,
            hypotheses=hypotheses,
            recommendations=recommendations,
            confidence=0.7 if not anomalies else 0.5,
            severity=severity,
        )


class EventAuditor:
    """Runs audits triggered by specific events."""

    def __init__(self):
        self._triggers: List[AuditTrigger] = []

    def check_triggers(self, substrate_state: Dict[str, Any],
                       prev_state: Optional[Dict[str, Any]] = None
                       ) -> List[AuditTrigger]:
        triggers = []

        if prev_state:
            prev_attractors = prev_state.get("n_attractors", 0)
            curr_attractors = substrate_state.get("n_attractors", 0)
            if prev_attractors > 0 and curr_attractors < prev_attractors * 0.5:
                triggers.append(AuditTrigger.ATTRACTOR_COLLAPSE)

            prev_entropy = prev_state.get("volume_entropy", 0.5)
            curr_entropy = substrate_state.get("volume_entropy", 0.5)
            if prev_entropy > 0.3 and curr_entropy < 0.1:
                triggers.append(AuditTrigger.ENTROPY_REDUCTION)

            prev_coherence = prev_state.get("coherence", 0.5)
            curr_coherence = substrate_state.get("coherence", 0.5)
            if abs(curr_coherence - prev_coherence) > 0.3:
                triggers.append(AuditTrigger.CONFIDENCE_OSCILLATION)

        n_attractors = substrate_state.get("n_attractors", 0)
        if n_attractors > 100:
            triggers.append(AuditTrigger.MEMORY_EXPLOSION)

        return triggers

    def create_report(self, tick: int, trigger: AuditTrigger,
                      substrate_state: Dict[str, Any],
                      audit_id: int) -> AuditReport:
        observations = []
        anomalies = []
        hypotheses = []
        recommendations = []
        severity = AuditSeverity.INFO

        if trigger == AuditTrigger.ATTRACTOR_COLLAPSE:
            anomalies.append("Attractor count dropped significantly")
            hypotheses.append("Possible: dynamics model drift, excessive pruning, or noise")
            recommendations.append("Run deep audit on attractor health")
            recommendations.append("Check dynamics model parameters")
            severity = AuditSeverity.WARNING

        elif trigger == AuditTrigger.ENTROPY_REDUCTION:
            anomalies.append("Entropy dropped below threshold")
            hypotheses.append("System may be over-specializing")
            recommendations.append("Consider exploration boost")
            severity = AuditSeverity.WARNING

        elif trigger == AuditTrigger.MEMORY_EXPLOSION:
            anomalies.append("Attractor count exceeds threshold")
            hypotheses.append("Consolidation may be too slow")
            recommendations.append("Increase consolidation rate or prune weak attractors")
            severity = AuditSeverity.WARNING

        elif trigger == AuditTrigger.CONFIDENCE_OSCILLATION:
            anomalies.append("Coherence fluctuating rapidly")
            hypotheses.append("Predictions may be unstable")
            recommendations.append("Check dynamics model stability")
            severity = AuditSeverity.WARNING

        return AuditReport(
            audit_id=audit_id,
            tick=tick,
            audit_type=AuditType.EVENT,
            trigger=trigger,
            timestamp=time.time(),
            observations=observations,
            anomalies=anomalies,
            hypotheses=hypotheses,
            recommendations=recommendations,
            confidence=0.6,
            severity=severity,
        )


class Council:
    """Orchestrates audits and produces health reports.

    The Council does not modify the substrate directly.
    It produces reports that Executive Function can act on.
    """

    def __init__(self, audit_interval: int = 500):
        self.scheduled = ScheduledAuditor(interval_ticks=audit_interval)
        self.event = EventAuditor()
        self.drift = DriftDetector()

        self._reports: List[AuditReport] = []
        self._next_audit_id = 1
        self._prev_state: Optional[Dict[str, Any]] = None
        self._state_history: List[Dict[str, Any]] = []

    def tick(self, substrate_state: Dict[str, Any]) -> List[AuditReport]:
        """Run audits and return any new reports."""
        new_reports = []
        current_tick = substrate_state.get("tick", 0)

        if self.scheduled.should_audit(current_tick):
            report = self.scheduled.create_report(
                current_tick, substrate_state, self._next_audit_id)
            self._next_audit_id += 1
            self._reports.append(report)
            new_reports.append(report)

        triggers = self.event.check_triggers(substrate_state, self._prev_state)
        for trigger in triggers:
            report = self.event.create_report(
                current_tick, trigger, substrate_state, self._next_audit_id)
            self._next_audit_id += 1
            self._reports.append(report)
            new_reports.append(report)

        drift_metrics = self.drift.update(substrate_state)

        self._prev_state = substrate_state.copy()
        self._state_history.append(substrate_state)
        if len(self._state_history) > 200:
            self._state_history.pop(0)

        return new_reports

    def manual_audit(self, substrate_state: Dict[str, Any],
                     observations: Optional[List[str]] = None) -> AuditReport:
        """Run a manual audit with custom observations."""
        report = AuditReport(
            audit_id=self._next_audit_id,
            tick=substrate_state.get("tick", 0),
            audit_type=AuditType.MANUAL,
            trigger=AuditTrigger.MANUAL,
            timestamp=time.time(),
            observations=observations or [],
            confidence=0.8,
            severity=AuditSeverity.INFO,
        )
        self._next_audit_id += 1
        self._reports.append(report)
        return report

    def get_reports(self, n: int = 10) -> List[AuditReport]:
        """Get recent audit reports."""
        return self._reports[-n:]

    def get_state(self) -> CouncilState:
        """Get current council state."""
        recent = self._reports[-5:] if self._reports else []
        severity_counts = {"info": 0, "warning": 0, "critical": 0}
        for r in recent:
            severity_counts[r.severity.value] = severity_counts.get(r.severity.value, 0) + 1

        worst = "info"
        if severity_counts.get("critical", 0) > 0:
            worst = "critical"
        elif severity_counts.get("warning", 0) > 0:
            worst = "warning"

        n_recommendations = sum(len(r.recommendations) for r in self._reports)

        # Health decays with repeated warnings, but recovers when stable
        health = 1.0
        recent_warnings = sum(1 for r in self._reports[-20:] if r.severity == AuditSeverity.WARNING)
        recent_criticals = sum(1 for r in self._reports[-20:] if r.severity == AuditSeverity.CRITICAL)

        # Warnings reduce health gradually (not linearly)
        if recent_warnings > 0:
            health *= max(0.3, 1.0 - (recent_warnings * 0.05))
        if recent_criticals > 0:
            health *= max(0.1, 1.0 - (recent_criticals * 0.15))

        # Recovery: if no issues in last 10 reports, health improves
        last_10 = self._reports[-10:] if len(self._reports) >= 10 else self._reports
        if all(r.severity == AuditSeverity.INFO for r in last_10):
            health = min(1.0, health + 0.1)

        return CouncilState(
            last_audit_tick=self._reports[-1].tick if self._reports else 0,
            last_audit_time=self._reports[-1].timestamp if self._reports else 0.0,
            n_audits=len(self._reports),
            n_reports=len(self._reports),
            n_recommendations=n_recommendations,
            drift_score=self.drift._baseline is not None and self._reports
                        and self._reports[-1].metadata.get("drift", 0.0),
            health_score=health,
            recent_severity=worst,
            pending_reports=[r.to_dict() for r in recent],
        )

    def set_baseline(self, substrate_state: Dict[str, float]):
        """Set drift detection baseline."""
        self.drift.set_baseline(substrate_state)
