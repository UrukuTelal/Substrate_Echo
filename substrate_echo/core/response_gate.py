"""Privacy-Aware Response Gating — P6.8

Suppress responses when communicative intent confidence is below threshold.

The problem: an agent perceives behavioral signals (gaze, gesture,
speech) that MIGHT be communicative intent directed at them. But the
signals are noisy and ambiguous. If the agent responds to every
ambiguous signal, it will:
1. Interrupt ongoing activities for false positives
2. Reveal its own state unprompted
3. Over-respond to social noise

The solution: a gating layer that evaluates whether a perceived
signal crosses the response threshold before allowing the agent
to act on it.

The gate considers:
- Intent confidence (from CommunicativeIntentDetector)
- Privacy level (how much state the agent wants to protect)
- Social context (number of observers, relationship strength)
- Response cost (what the agent gives up by responding)

If the gate passes, the response proceeds. If blocked, the agent
observes silently without revealing awareness.

Usage:
    gate = ResponseGate(
        confidence_threshold=0.5,
        privacy_level=0.3,
    )
    
    decision = gate.evaluate(
        intent_confidence=0.7,
        intent_type="REQUEST",
        social_openness=0.6,
        observers=2,
        relationship_strength=0.8,
    )
    
    if decision.allowed:
        # respond to the intent
    else:
        # observe silently
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class GateDecision:
    """Result of response gating evaluation."""
    allowed: bool
    reason: str
    effective_threshold: float   # the actual threshold used
    confidence: float            # the input confidence
    privacy_cost: float          # how much privacy is spent
    suppression_strength: float  # 0.0=fully suppressed, 1.0=fully allowed
    
    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "effective_threshold": self.effective_threshold,
            "confidence": self.confidence,
            "privacy_cost": self.privacy_cost,
            "suppression_strength": self.suppression_strength,
        }


@dataclass
class ResponseGateConfig:
    """Configuration for response gating."""
    confidence_threshold: float = 0.5    # base threshold for intent confidence
    privacy_level: float = 0.3          # 0=no privacy, 1=max privacy
    observer_penalty: float = 0.05     # per additional observer
    relationship_discount: float = 0.2  # reduce threshold for known entities
    cost_threshold: float = 0.3        # max response cost before suppression
    min_dwell_frames: int = 3          # entity must be tracked this long
    cooldown_frames: int = 2           # frames between responses to same entity


class ResponseGate:
    """Controls whether the agent responds to perceived intent.
    
    The gate implements a simple privacy calculus:
    
        effective_threshold = base_threshold + privacy_boost - relationship_discount
        
    Where:
        privacy_boost = privacy_level × (1 + observer_penalty × observers)
        relationship_discount = relationship_discount × relationship_strength
    
    If confidence >= effective_threshold, the response is allowed.
    Otherwise, the agent suppresses its response and observes silently.
    
    Additional heuristics:
    - Entities tracked for fewer than min_dwell_frames are suppressed
    - Responses to the same entity are throttled by cooldown_frames
    - High-cost responses require higher confidence
    
    Usage:
        gate = ResponseGate()
        
        # Each tick, for each detected intent:
        decision = gate.evaluate(
            intent_confidence=0.7,
            intent_type="REQUEST",
            social_openness=0.6,
            observers=1,
            relationship_strength=0.5,
        )
    """
    
    def __init__(self, config: Optional[ResponseGateConfig] = None):
        self.config = config or ResponseGateConfig()
        self._cooldown: dict[str, int] = {}  # entity_id → frames since last response
        self._suppressed_count = 0
        self._allowed_count = 0
    
    @property
    def stats(self) -> dict:
        total = self._allowed_count + self._suppressed_count
        return {
            "allowed": self._allowed_count,
            "suppressed": self._suppressed_count,
            "total": total,
            "suppression_rate": (
                self._suppressed_count / total if total > 0 else 0.0),
        }
    
    def evaluate(self, intent_confidence: float,
                 intent_type: str = "UNKNOWN",
                 social_openness: float = 0.5,
                 observers: int = 0,
                 relationship_strength: float = 0.0,
                 entity_id: Optional[str] = None,
                 dwell_frames: int = 0,
                 response_cost: float = 0.0) -> GateDecision:
        """Evaluate whether to allow a response.
        
        Args:
            intent_confidence: 0-1, how confident we are in the intent
            intent_type: classified intent type
            social_openness: 0-1, agent's current social openness
            observers: number of other entities observing
            relationship_strength: 0-1, strength of relationship
            entity_id: ID of the intent source (for cooldown)
            dwell_frames: how many frames entity has been tracked
            response_cost: 0-1, cost of responding (time/attention)
        
        Returns:
            GateDecision with allowed flag and metadata
        """
        cfg = self.config
        
        # ── Dwell check ──────────────────────────────────────────
        if dwell_frames < cfg.min_dwell_frames:
            self._suppressed_count += 1
            return GateDecision(
                allowed=False,
                reason=f"dwell_frames={dwell_frames} < {cfg.min_dwell_frames}",
                effective_threshold=1.0,
                confidence=intent_confidence,
                privacy_cost=0.0,
                suppression_strength=0.0,
            )
        
        # ── Cooldown check ───────────────────────────────────────
        if entity_id is not None:
            last = self._cooldown.get(entity_id, -cfg.cooldown_frames - 1)
            if last >= 0 and last < cfg.cooldown_frames:
                self._suppressed_count += 1
                return GateDecision(
                    allowed=False,
                    reason=f"cooldown: {last}/{cfg.cooldown_frames}",
                    effective_threshold=1.0,
                    confidence=intent_confidence,
                    privacy_cost=0.0,
                    suppression_strength=0.0,
                )
        
        # ── Compute effective threshold ──────────────────────────
        privacy_boost = cfg.privacy_level * (1 + cfg.observer_penalty * observers)
        rel_discount = cfg.relationship_discount * relationship_strength
        effective_threshold = cfg.confidence_threshold + privacy_boost - rel_discount
        effective_threshold = max(0.0, min(1.0, effective_threshold))
        
        # ── Cost adjustment ──────────────────────────────────────
        if response_cost > cfg.cost_threshold:
            effective_threshold += (response_cost - cfg.cost_threshold) * 0.5
            effective_threshold = min(1.0, effective_threshold)
        
        # ── Social openness modulation ───────────────────────────
        # High openness lowers threshold (agent is more receptive)
        openness_discount = social_openness * 0.15
        effective_threshold -= openness_discount
        effective_threshold = max(0.0, effective_threshold)
        
        # ── Decision ─────────────────────────────────────────────
        allowed = intent_confidence >= effective_threshold
        
        if allowed:
            self._allowed_count += 1
            if entity_id is not None:
                self._cooldown[entity_id] = 0
        else:
            self._suppressed_count += 1
        
        suppression = 1.0 - min(1.0, intent_confidence / max(effective_threshold, 1e-6))
        
        return GateDecision(
            allowed=allowed,
            reason="confidence above threshold" if allowed else "confidence below threshold",
            effective_threshold=effective_threshold,
            confidence=intent_confidence,
            privacy_cost=cfg.privacy_level * (1.0 - suppression),
            suppression_strength=suppression,
        )
    
    def tick(self) -> None:
        """Advance all cooldown timers. Call once per frame."""
        for eid in list(self._cooldown.keys()):
            self._cooldown[eid] += 1
    
    def reset(self) -> None:
        self._cooldown.clear()
        self._suppressed_count = 0
        self._allowed_count = 0
