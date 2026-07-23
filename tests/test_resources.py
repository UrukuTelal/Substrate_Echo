"""Tests for Resource Manager (S11)."""
from __future__ import annotations
import pytest
import time
from substrate_echo.kernel.resources import (
    ResourceBudget, ResourceLease, ResourceRequest,
    ResourceAllocation, ResourceState, ResourceManager,
    ResourceTier, LeaseStatus
)


def test_budget_defaults():
    b = ResourceBudget()
    assert b.total_compute == 1.0
    assert b.total_memory == 1.0
    assert b.available_compute() == 1.0
    assert b.available_attention() == 10.0


def test_budget_used():
    b = ResourceBudget()
    b.used_compute = 0.3
    b.used_attention = 4.0
    assert b.available_compute() == pytest.approx(0.7)
    assert b.available_attention() == 6.0


def test_budget_utilization():
    b = ResourceBudget()
    b.used_compute = 0.5
    util = b.utilization()
    assert util["compute"] == pytest.approx(0.5)
    assert util["memory"] == pytest.approx(0.0)


def test_budget_to_dict():
    b = ResourceBudget()
    d = b.to_dict()
    assert "total_compute" in d
    assert "available_compute" in d
    assert d["available_compute"] == 1.0


def test_lease_expiry():
    lease = ResourceLease(
        lease_id=1, embodiment_id="test",
        duration=0.01, granted_at=time.time() - 0.1,
        expires_at=time.time() - 0.05,
    )
    assert lease.is_expired()
    assert lease.is_active()  # still active until revoked/expired by manager


def test_lease_no_expiry():
    lease = ResourceLease(
        lease_id=1, embodiment_id="test",
        duration=0, granted_at=time.time(),
        expires_at=time.time() + 100,
    )
    assert not lease.is_expired()


def test_lease_to_dict():
    lease = ResourceLease(
        lease_id=1, embodiment_id="test",
        attention=0.3, compute=0.2, learning=0.1,
    )
    d = lease.to_dict()
    assert d["attention"] == 0.3
    assert d["embodiment_id"] == "test"


def test_manager_full_grant():
    rm = ResourceManager()
    req = ResourceRequest(
        embodiment_id="robot1",
        attention=0.2, compute=0.3, learning=0.1,
        duration=60.0, tier=ResourceTier.ACTIVE,
    )
    result = rm.request(req)
    assert result.granted
    assert result.lease is not None
    assert result.lease.attention == 0.2
    assert rm.budget.used_attention == 0.2


def test_manager_partial_grant():
    # With limited resources, we should get a partial grant
    rm = ResourceManager(budget=ResourceBudget(total_attention=0.1, total_compute=0.1, total_learning=0.1))
    req = ResourceRequest(
        embodiment_id="robot1",
        attention=0.4, compute=0.1, learning=0.1,
        duration=60.0, tier=ResourceTier.ACTIVE,
        trust_level=0.5,
    )
    result = rm.request(req)
    assert result.granted
    assert result.modified_attention < 0.4
    assert result.modified_attention > 0


def test_manager_denial_insufficient():
    rm = ResourceManager(budget=ResourceBudget(total_attention=0.01, total_compute=0.01, total_learning=0.01))
    req = ResourceRequest(
        embodiment_id="robot1",
        attention=0.5, compute=0.5, learning=0.5,
        duration=60.0, tier=ResourceTier.ACTIVE,
        trust_level=0.1,
    )
    result = rm.request(req)
    assert not result.granted
    assert "Insufficient" in result.reason


def test_manager_revocation():
    rm = ResourceManager()
    req = ResourceRequest(
        embodiment_id="robot1",
        attention=0.3, compute=0.2, learning=0.1,
        duration=60.0, tier=ResourceTier.ACTIVE,
    )
    result = rm.request(req)
    lease_id = result.lease.lease_id
    rm.revoke(lease_id)
    assert rm.budget.used_attention == 0.0
    assert rm.budget.used_compute == 0.0
    assert rm.budget.used_learning == 0.0


def test_manager_release():
    rm = ResourceManager()
    req = ResourceRequest(
        embodiment_id="robot1",
        attention=0.3, compute=0.2, learning=0.1,
        duration=60.0, tier=ResourceTier.ACTIVE,
    )
    result = rm.request(req)
    rm.release(result.lease.lease_id)
    assert rm.budget.used_attention == 0.0


def test_manager_active_leases():
    rm = ResourceManager()
    for i in range(3):
        req = ResourceRequest(
            embodiment_id=f"robot{i}",
            attention=0.1, compute=0.1, learning=0.1,
            duration=60.0, tier=ResourceTier.ACTIVE,
        )
        rm.request(req)
    assert len(rm.get_active_leases()) == 3


def test_manager_embodiment_leases():
    rm = ResourceManager()
    for _ in range(2):
        rm.request(ResourceRequest(
            embodiment_id="robot1", attention=0.1, compute=0.1, learning=0.1,
        ))
    rm.request(ResourceRequest(
        embodiment_id="robot2", attention=0.1, compute=0.1, learning=0.1,
    ))
    assert len(rm.get_embodiment_leases("robot1")) == 2
    assert len(rm.get_embodiment_leases("robot2")) == 1


def test_manager_state():
    rm = ResourceManager()
    rm.request(ResourceRequest(
        embodiment_id="robot1", attention=0.2, compute=0.1, learning=0.1,
    ))
    state = rm.get_state()
    assert state.active_leases == 1
    assert state.n_embodiments == 1
    assert "compute" in state.utilization


def test_manager_safety_scale():
    rm = ResourceManager()
    req = ResourceRequest(
        embodiment_id="robot1",
        attention=0.5, compute=0.5, learning=0.5,
        duration=60.0, tier=ResourceTier.LEARNING,
    )
    rm.request(req)
    rm.scale_for_safety(0.8)
    lease = rm.get_active_leases()[0]
    assert lease.attention < 0.5


def test_manager_expired_cleanup():
    rm = ResourceManager()
    req = ResourceRequest(
        embodiment_id="robot1",
        attention=0.3, compute=0.2, learning=0.1,
        duration=0.01, tier=ResourceTier.ACTIVE,
    )
    result = rm.request(req)
    time.sleep(0.02)
    rm._cleanup_expired()
    assert len(rm.get_active_leases()) == 0
    assert rm.budget.used_attention == 0.0
