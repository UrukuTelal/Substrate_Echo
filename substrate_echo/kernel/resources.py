"""Resource Manager — Governs finite cognitive resources.

With multiple embodiments sharing one substrate, resources become finite.
Embodiments compete for:
  - Compute time (GPU/CPU allocation)
  - Memory capacity (attractor count, consolidation)
  - Learning rate (new samples per tick)
  - Attention (events getting full processing)

The Resource Manager arbitrates these competing demands based on:
  - Goal tier priority (safety > maintenance > active > learning > exploration)
  - Embodiment trust level
  - Current resource availability
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import time


class ResourceTier(Enum):
    """Priority tiers for resource allocation."""
    SAFETY = 0      # Critical constraints — highest priority
    MAINTENANCE = 1 # Consolidation, decay
    ACTIVE = 2      # Active goals
    LEARNING = 3    # New attractors, vector field updates
    EXPLORATION = 4 # Novelty-seeking
    IDLE = 5        # Abstraction, meta-attractor formation


class LeaseStatus(Enum):
    GRANTED = "granted"
    ACTIVE = "active"
    DENIED = "denied"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class ResourceBudget:
    """Total available resources in the system."""
    total_compute: float = 1.0
    total_memory: float = 1.0
    total_learning: float = 1.0
    total_attention: float = 10.0

    # Current allocation (sum of active leases)
    used_compute: float = 0.0
    used_memory: float = 0.0
    used_learning: float = 0.0
    used_attention: float = 0.0

    def available_compute(self) -> float:
        return max(0, self.total_compute - self.used_compute)

    def available_memory(self) -> float:
        return max(0, self.total_memory - self.used_memory)

    def available_learning(self) -> float:
        return max(0, self.total_learning - self.used_learning)

    def available_attention(self) -> float:
        return max(0, self.total_attention - self.used_attention)

    def utilization(self) -> Dict[str, float]:
        return {
            "compute": self.used_compute / max(0.001, self.total_compute),
            "memory": self.used_memory / max(0.001, self.total_memory),
            "learning": self.used_learning / max(0.001, self.total_learning),
            "attention": self.used_attention / max(0.001, self.total_attention),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_compute": self.total_compute,
            "total_memory": self.total_memory,
            "total_learning": self.total_learning,
            "total_attention": self.total_attention,
            "used_compute": self.used_compute,
            "used_memory": self.used_memory,
            "used_learning": self.used_learning,
            "used_attention": self.used_attention,
            "available_compute": self.available_compute(),
            "available_memory": self.available_memory(),
            "available_learning": self.available_learning(),
            "available_attention": self.available_attention(),
        }


@dataclass
class ResourceLease:
    """A granted resource allocation to an embodiment."""
    lease_id: int
    embodiment_id: str
    attention: float = 0.1
    compute: float = 0.1
    learning: float = 0.1
    duration: float = 60.0
    tier: ResourceTier = ResourceTier.ACTIVE
    status: LeaseStatus = LeaseStatus.GRANTED
    granted_at: float = 0.0
    expires_at: float = 0.0
    priority_score: float = 0.0

    def is_active(self) -> bool:
        return self.status in (LeaseStatus.GRANTED, LeaseStatus.ACTIVE)

    def is_expired(self) -> bool:
        if self.duration <= 0:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "embodiment_id": self.embodiment_id,
            "attention": self.attention,
            "compute": self.compute,
            "learning": self.learning,
            "duration": self.duration,
            "tier": self.tier.name,
            "status": self.status.value,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "priority_score": self.priority_score,
        }


@dataclass
class ResourceRequest:
    """Request from an embodiment for cognitive resources."""
    embodiment_id: str
    attention: float = 0.1
    compute: float = 0.1
    learning: float = 0.1
    duration: float = 60.0
    tier: ResourceTier = ResourceTier.ACTIVE
    justification: str = ""
    trust_level: float = 0.5


@dataclass
class ResourceAllocation:
    """Decision on a resource request."""
    granted: bool
    lease: Optional[ResourceLease] = None
    reason: str = ""
    modified_attention: float = 0.0
    modified_compute: float = 0.0
    modified_learning: float = 0.0


@dataclass
class ResourceState:
    """Current state of the resource manager."""
    budget: Dict[str, float]
    utilization: Dict[str, float]
    active_leases: int
    total_leases_issued: int
    pending_requests: int
    recent_denials: int
    n_embodiments: int
    tier_allocations: Dict[str, float]


class ResourceManager:
    """Manages finite cognitive resources across embodiments.

    Responsibilities:
      1. Resource accounting (what's available, what's used)
      2. Priority-based allocation (safety > maintenance > active > ...)
      3. Embodiment scheduling (time-sharing, leases)
      4. Lease lifecycle management
    """

    def __init__(self, budget: Optional[ResourceBudget] = None):
        self.budget = budget or ResourceBudget()
        self._leases: Dict[int, ResourceLease] = {}
        self._next_lease_id = 1
        self._requests: List[ResourceRequest] = []
        self._denials: int = 0
        self._total_issued: int = 0

    def request(self, req: ResourceRequest) -> ResourceAllocation:
        """Process a resource request and grant, modify, or deny it."""
        self._cleanup_expired()

        tier_priority = 1.0 - (req.tier.value * 0.15)

        available = {
            "attention": self.budget.available_attention(),
            "compute": self.budget.available_compute(),
            "learning": self.budget.available_learning(),
        }

        if req.attention <= available["attention"] and \
           req.compute <= available["compute"] and \
           req.learning <= available["learning"]:
            lease = self._create_lease(req, tier_priority)
            return ResourceAllocation(
                granted=True, lease=lease,
                reason="Full grant",
                modified_attention=req.attention,
                modified_compute=req.compute,
                modified_learning=req.learning,
            )

        scale_factors = []
        if req.attention > available["attention"]:
            scale_factors.append(available["attention"] / max(0.001, req.attention))
        if req.compute > available["compute"]:
            scale_factors.append(available["compute"] / max(0.001, req.compute))
        if req.learning > available["learning"]:
            scale_factors.append(available["learning"] / max(0.001, req.learning))

        scale = min(scale_factors) if scale_factors else 0.0
        scale *= req.trust_level

        if scale < 0.01:
            self._denials += 1
            return ResourceAllocation(
                granted=False, reason="Insufficient resources",
            )

        modified = ResourceAllocation(
            granted=True,
            reason=f"Partial grant ({scale:.1%} of requested)",
            modified_attention=req.attention * scale,
            modified_compute=req.compute * scale,
            modified_learning=req.learning * scale,
        )

        lease = self._create_lease(
            req, tier_priority,
            attention=modified.modified_attention,
            compute=modified.modified_compute,
            learning=modified.modified_learning,
        )
        modified.lease = lease
        return modified

    def _create_lease(self, req: ResourceRequest, priority: float,
                      attention: Optional[float] = None,
                      compute: Optional[float] = None,
                      learning: Optional[float] = None) -> ResourceLease:
        """Create a lease and allocate resources."""
        lease = ResourceLease(
            lease_id=self._next_lease_id,
            embodiment_id=req.embodiment_id,
            attention=attention or req.attention,
            compute=compute or req.compute,
            learning=learning or req.learning,
            duration=req.duration,
            tier=req.tier,
            status=LeaseStatus.ACTIVE,
            granted_at=time.time(),
            expires_at=time.time() + req.duration,
            priority_score=priority,
        )
        self._next_lease_id += 1
        self._leases[lease.lease_id] = lease
        self._total_issued += 1

        self.budget.used_attention += lease.attention
        self.budget.used_compute += lease.compute
        self.budget.used_learning += lease.learning

        return lease

    def _cleanup_expired(self):
        """Revoke expired leases and free resources."""
        to_revoke = []
        for lid, lease in self._leases.items():
            if lease.is_active() and lease.is_expired():
                to_revoke.append(lid)

        for lid in to_revoke:
            self.revoke(lid)

    def revoke(self, lease_id: int):
        """Revoke a lease and free its resources."""
        if lease_id not in self._leases:
            return
        lease = self._leases[lease_id]
        if lease.is_active():
            self.budget.used_attention = max(0, self.budget.used_attention - lease.attention)
            self.budget.used_compute = max(0, self.budget.used_compute - lease.compute)
            self.budget.used_learning = max(0, self.budget.used_learning - lease.learning)
        lease.status = LeaseStatus.REVOKED

    def release(self, lease_id: int):
        """Gracefully release a lease (embodiment done)."""
        if lease_id not in self._leases:
            return
        lease = self._leases[lease_id]
        if lease.is_active():
            self.budget.used_attention = max(0, self.budget.used_attention - lease.attention)
            self.budget.used_compute = max(0, self.budget.used_compute - lease.compute)
            self.budget.used_learning = max(0, self.budget.used_learning - lease.learning)
        lease.status = LeaseStatus.EXPIRED

    def get_active_leases(self) -> List[ResourceLease]:
        """Get all currently active leases."""
        self._cleanup_expired()
        return [l for l in self._leases.values() if l.is_active()]

    def get_embodiment_leases(self, embodiment_id: str) -> List[ResourceLease]:
        """Get all leases for a specific embodiment."""
        return [l for l in self._leases.values()
                if l.embodiment_id == embodiment_id]

    def get_state(self) -> ResourceState:
        """Get current resource manager state."""
        active = self.get_active_leases()

        tier_alloc = {}
        for tier in ResourceTier:
            tier_alloc[tier.name] = sum(
                l.attention + l.compute + l.learning
                for l in active if l.tier == tier
            )

        embodiments = set(l.embodiment_id for l in active)

        return ResourceState(
            budget=self.budget.to_dict(),
            utilization=self.budget.utilization(),
            active_leases=len(active),
            total_leases_issued=self._total_issued,
            pending_requests=len(self._requests),
            recent_denials=self._denials,
            n_embodiments=len(embodiments),
            tier_allocations=tier_alloc,
        )

    def update_tier_weights(self, tier: ResourceTier,
                            attention: float = 0.1,
                            compute: float = 0.1,
                            learning: float = 0.1):
        """Update default resource allocation for a tier."""
        pass

    def scale_for_safety(self, safety_level: float):
        """Scale all resources based on safety level.

        safety_level: 0.0 = normal, 1.0 = emergency
        """
        if safety_level > 0.5:
            safety_factor = 1.0 - (safety_level - 0.5) * 2
            for lid, lease in self._leases.items():
                if lease.is_active() and lease.tier.value > ResourceTier.SAFETY.value:
                    scale = safety_factor
                    self.budget.used_attention -= lease.attention * (1 - scale)
                    self.budget.used_compute -= lease.compute * (1 - scale)
                    self.budget.used_learning -= lease.learning * (1 - scale)
                    lease.attention *= scale
                    lease.compute *= scale
                    lease.learning *= scale
