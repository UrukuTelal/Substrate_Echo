"""Governance Gate — Meta-controller that checks actions before execution.

Sits between action selection and execution. Applies rules that
prevent the system from acting on low-confidence beliefs or making
risky moves when uncertainty is high.

Architecture:
    Action Score (from bridge)
           |
           v
    Governance Rules (permission check)
           |
           v
    Decision: ALLOW / MODIFY / DENY
           |
           v
    (if modified) Adjusted Action
           |
           v
    Execution

Rules:
    1. "Do not commit high-cost actions from low-confidence beliefs"
    2. "Prefer information gathering when uncertainty is high"
    3. "Do not attack when army exposure exceeds safety threshold"
    4. "Retreat is always allowed (defensive actions are cheap)"
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum


class GovernanceDecision(Enum):
    """Decision on an action."""
    ALLOW = "allow"
    MODIFY = "modify"
    DENY = "deny"


class RulePriority(Enum):
    """Rule priority levels."""
    CRITICAL = 100   # always enforced
    HIGH = 75        # enforced unless overridden
    MEDIUM = 50      # default
    LOW = 25         # advisory


@dataclass
class GovernanceRule:
    """A rule that checks actions before execution."""
    rule_id: str
    description: str
    priority: RulePriority = RulePriority.MEDIUM

    # The check function: returns (decision, reason, adjusted_action)
    check: Callable = None

    # Whether this rule is enabled
    enabled: bool = True

    # Stats
    times_checked: int = 0
    times_triggered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "priority": self.priority.name,
            "enabled": self.enabled,
            "checked": self.times_checked,
            "triggered": self.times_triggered,
        }


@dataclass
class GovernanceVerdict:
    """Result of governance check."""
    decision: GovernanceDecision
    rule_id: str = ""
    reason: str = ""
    original_action: str = ""
    adjusted_action: Optional[str] = None
    confidence_at_check: float = 0.0
    cost_at_check: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "rule": self.rule_id,
            "reason": self.reason,
            "original": self.original_action,
            "adjusted": self.adjusted_action,
        }


class GovernanceGate:
    """Meta-controller that checks actions before execution.
    
    Usage:
        gate = GovernanceGate()
        
        verdict = gate.check(
            action_type="attack",
            confidence=0.29,
            cost_level=3,
            uncertainty=0.7,
            army_exposure=0.8,
        )
        
        if verdict.decision == GovernanceDecision.DENY:
            print(f"Denied: {verdict.reason}")
        elif verdict.decision == GovernanceDecision.MODIFY:
            print(f"Modified to: {verdict.adjusted_action}")
    """

    def __init__(self):
        self._rules: List[GovernanceRule] = []
        self._verdicts: List[GovernanceVerdict] = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Install default governance rules."""

        # Rule 1: No high-cost actions from low confidence
        def check_confidence_cost(action_type, confidence, cost_level,
                                  uncertainty, army_exposure, **kw):
            if cost_level >= 3 and confidence < 0.3:
                return GovernanceVerdict(
                    decision=GovernanceDecision.DENY,
                    rule_id="confidence_cost",
                    reason=f"High-cost action ({action_type}) from low confidence ({confidence:.2f})",
                    original_action=action_type,
                    confidence_at_check=confidence,
                    cost_at_check=cost_level,
                )
            return None

        self._rules.append(GovernanceRule(
            rule_id="confidence_cost",
            description="Do not commit high-cost actions from low-confidence beliefs",
            priority=RulePriority.HIGH,
            check=check_confidence_cost,
        ))

        # Rule 2: Prefer scouting when uncertainty is high
        def check_uncertainty(action_type, confidence, cost_level,
                              uncertainty, army_exposure, **kw):
            if uncertainty > 0.6 and action_type in ("attack", "expand"):
                return GovernanceVerdict(
                    decision=GovernanceDecision.MODIFY,
                    rule_id="uncertainty_scout",
                    reason=f"High uncertainty ({uncertainty:.2f}) — prefer scouting",
                    original_action=action_type,
                    adjusted_action="scout",
                )
            return None

        self._rules.append(GovernanceRule(
            rule_id="uncertainty_scout",
            description="Prefer information gathering when uncertainty is high",
            priority=RulePriority.MEDIUM,
            check=check_uncertainty,
        ))

        # Rule 3: No attack when army exposure is too high
        def check_army_exposure(action_type, confidence, cost_level,
                                uncertainty, army_exposure, **kw):
            if action_type == "attack" and army_exposure > 0.8:
                return GovernanceVerdict(
                    decision=GovernanceDecision.MODIFY,
                    rule_id="army_exposure",
                    reason=f"Army exposure too high ({army_exposure:.2f}) — build up first",
                    original_action=action_type,
                    adjusted_action="build_army",
                )
            return None

        self._rules.append(GovernanceRule(
            rule_id="army_exposure",
            description="Do not attack when army exposure exceeds safety threshold",
            priority=RulePriority.HIGH,
            check=check_army_exposure,
        ))

        # Rule 4: Always allow defensive actions
        def check_defensive(action_type, confidence, cost_level,
                            uncertainty, army_exposure, **kw):
            if action_type in ("defend", "retreat", "hold"):
                return GovernanceVerdict(
                    decision=GovernanceDecision.ALLOW,
                    rule_id="defensive_always",
                    reason="Defensive actions always permitted",
                    original_action=action_type,
                )
            return None

        self._rules.append(GovernanceRule(
            rule_id="defensive_always",
            description="Retreat is always allowed (defensive actions are cheap)",
            priority=RulePriority.CRITICAL,
            check=check_defensive,
        ))

        # Rule 5: Don't expand when under threat
        def check_expand_threat(action_type, confidence, cost_level,
                                uncertainty, army_exposure,
                                threat_level=0.0, **kw):
            if action_type == "expand" and threat_level > 0.6:
                return GovernanceVerdict(
                    decision=GovernanceDecision.MODIFY,
                    rule_id="expand_threat",
                    reason=f"Threat level high ({threat_level:.2f}) — defend first",
                    original_action=action_type,
                    adjusted_action="defend",
                )
            return None

        self._rules.append(GovernanceRule(
            rule_id="expand_threat",
            description="Do not expand when threat level is high",
            priority=RulePriority.MEDIUM,
            check=check_expand_threat,
        ))

    def add_rule(self, rule: GovernanceRule):
        """Add a custom governance rule."""
        self._rules.append(rule)

    def check(self, action_type: str, confidence: float = 0.5,
              cost_level: float = 0, uncertainty: float = 0.5,
              army_exposure: float = 0.0,
              threat_level: float = 0.0,
              **kwargs) -> GovernanceVerdict:
        """Check an action against all governance rules.
        
        Returns the first verdict from the highest-priority triggered rule.
        If no rules trigger, returns ALLOW.
        """
        # Sort rules by priority (highest first)
        sorted_rules = sorted(self._rules,
                              key=lambda r: r.priority.value, reverse=True)

        for rule in sorted_rules:
            if not rule.enabled or rule.check is None:
                continue

            rule.times_checked += 1

            verdict = rule.check(
                action_type=action_type,
                confidence=confidence,
                cost_level=cost_level,
                uncertainty=uncertainty,
                army_exposure=army_exposure,
                threat_level=threat_level,
                **kwargs,
            )

            if verdict is not None:
                rule.times_triggered += 1
                self._verdicts.append(verdict)
                return verdict

        # No rules triggered — allow
        verdict = GovernanceVerdict(
            decision=GovernanceDecision.ALLOW,
            rule_id="default",
            reason="No rules triggered",
            original_action=action_type,
        )
        self._verdicts.append(verdict)
        return verdict

    def get_verdicts(self) -> List[GovernanceVerdict]:
        return list(self._verdicts)

    def get_rule_stats(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._rules]

    def get_denial_rate(self) -> float:
        """Fraction of actions that were denied."""
        if not self._verdicts:
            return 0.0
        denials = sum(1 for v in self._verdicts
                      if v.decision == GovernanceDecision.DENY)
        return denials / len(self._verdicts)

    def get_modification_rate(self) -> float:
        """Fraction of actions that were modified."""
        if not self._verdicts:
            return 0.0
        mods = sum(1 for v in self._verdicts
                   if v.decision == GovernanceDecision.MODIFY)
        return mods / len(self._verdicts)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_checks": len(self._verdicts),
            "denials": sum(1 for v in self._verdicts
                          if v.decision == GovernanceDecision.DENY),
            "modifications": sum(1 for v in self._verdicts
                                if v.decision == GovernanceDecision.MODIFY),
            "allows": sum(1 for v in self._verdicts
                         if v.decision == GovernanceDecision.ALLOW),
            "denial_rate": round(self.get_denial_rate(), 4),
            "modification_rate": round(self.get_modification_rate(), 4),
            "rules": self.get_rule_stats(),
        }

    def render(self) -> str:
        lines = []
        lines.append("Governance Gate Report")
        lines.append("=" * 60)
        summary = self.get_summary()
        lines.append(f"  Total checks:    {summary['total_checks']}")
        lines.append(f"  Allows:          {summary['allows']}")
        lines.append(f"  Modifications:   {summary['modifications']}")
        lines.append(f"  Denials:         {summary['denials']}")
        lines.append(f"  Denial rate:     {summary['denial_rate']:.1%}")
        lines.append(f"  Modification rate: {summary['modification_rate']:.1%}")
        lines.append("")
        lines.append("  Rule Stats:")
        for rule in self._rules:
            lines.append(
                f"    {rule.rule_id:25s} "
                f"checked={rule.times_checked:4d} "
                f"triggered={rule.times_triggered:4d} "
                f"[{rule.priority.name}]"
            )
        lines.append("")

        # Recent verdicts
        lines.append("  Recent Verdicts (last 10):")
        for v in self._verdicts[-10:]:
            marker = {"allow": "OK", "modify": "MOD", "deny": "DENY"}
            lines.append(
                f"    [{marker[v.decision.value]:4s}] "
                f"{v.original_action:12s}"
                f"{(' -> ' + v.adjusted_action) if v.adjusted_action else ''}"
                f"  ({v.reason})"
            )

        lines.append("=" * 60)
        return "\n".join(lines)
