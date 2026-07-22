"""Tests for Procedural Spatial Memory."""
import sys
sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.spatial_memory import (
    SpatialMemory, SpatialCell, AffordanceSummary,
)


# ── Basic Recording ──────────────────────────────────────────────

def test_record_single():
    m = SpatialMemory()
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    assert m.total_records == 1
    assert m.cell_count == 1


def test_record_multiple():
    m = SpatialMemory()
    for i in range(5):
        m.record(position=(float(i), 0.0, 0.0),
                 entity_type="human", action_type="GATHER")
    assert m.total_records == 5


def test_record_same_cell():
    m = SpatialMemory(cell_size=10.0)
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    m.record(position=(2.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    assert m.cell_count == 1  # same cell


def test_record_different_cells():
    m = SpatialMemory(cell_size=1.0)
    m.record(position=(0.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    m.record(position=(5.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    assert m.cell_count == 2


# ── Querying ─────────────────────────────────────────────────────

def test_query_empty():
    m = SpatialMemory()
    results = m.query(position=(0.0, 0.0, 0.0))
    assert results == []


def test_query_returns_affordances():
    m = SpatialMemory()
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="BUILD")
    
    results = m.query(position=(1.0, 0.0, 0.0))
    assert len(results) == 2
    types = {a.action_type for a in results}
    assert "GATHER" in types
    assert "BUILD" in types


def test_query_sorted_by_count():
    m = SpatialMemory()
    for _ in range(10):
        m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="BUILD")
    
    results = m.query(position=(1.0, 0.0, 0.0))
    assert results[0].action_type == "GATHER"
    assert results[0].count == 10


def test_query_by_entity_type():
    m = SpatialMemory()
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    m.record(position=(1.0, 0.0, 0.0), entity_type="animal", action_type="EAT")
    
    results = m.query(position=(1.0, 0.0, 0.0), entity_type="human")
    assert len(results) == 1
    assert results[0].entity_type == "human"


def test_query_by_action_type():
    m = SpatialMemory()
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="BUILD")
    
    results = m.query(position=(1.0, 0.0, 0.0), action_type="GATHER")
    assert len(results) == 1
    assert results[0].action_type == "GATHER"


# ── Success Rate ─────────────────────────────────────────────────

def test_success_rate():
    m = SpatialMemory()
    for _ in range(8):
        m.record(position=(1.0, 0.0, 0.0), entity_type="human",
                 action_type="GATHER", success=True)
    for _ in range(2):
        m.record(position=(1.0, 0.0, 0.0), entity_type="human",
                 action_type="GATHER", success=False)
    
    results = m.query(position=(1.0, 0.0, 0.0))
    assert abs(results[0].success_rate - 0.8) < 0.01


def test_success_rate_ema():
    m = SpatialMemory()
    for _ in range(10):
        m.record(position=(1.0, 0.0, 0.0), entity_type="human",
                 action_type="GATHER", success=True)
    # Recent failures should pull EMA down
    for _ in range(5):
        m.record(position=(1.0, 0.0, 0.0), entity_type="human",
                 action_type="GATHER", success=False)
    
    results = m.query(position=(1.0, 0.0, 0.0))
    # EMA should be lower than simple rate
    assert results[0].success_rate_ema < results[0].success_rate


# ── Suggest Actions ──────────────────────────────────────────────

def test_suggest_actions():
    m = SpatialMemory()
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="BUILD")
    
    suggestions = m.suggest_actions(position=(1.0, 0.0, 0.0),
                                    entity_type="human", top_k=2)
    assert len(suggestions) == 2
    assert suggestions[0].action_type == "GATHER"


def test_suggest_actions_empty():
    m = SpatialMemory()
    suggestions = m.suggest_actions(position=(1.0, 0.0, 0.0),
                                    entity_type="human")
    assert suggestions == []


# ── Predict Outcome ──────────────────────────────────────────────

def test_predict_outcome():
    m = SpatialMemory()
    for _ in range(7):
        m.record(position=(1.0, 0.0, 0.0), entity_type="human",
                 action_type="GATHER", success=True)
    for _ in range(3):
        m.record(position=(1.0, 0.0, 0.0), entity_type="human",
                 action_type="GATHER", success=False)
    
    prob = m.predict_outcome(position=(1.0, 0.0, 0.0),
                             entity_type="human", action_type="GATHER")
    assert prob is not None
    assert abs(prob - 0.7) < 0.01


def test_predict_outcome_no_data():
    m = SpatialMemory()
    prob = m.predict_outcome(position=(1.0, 0.0, 0.0),
                             entity_type="human", action_type="GATHER")
    assert prob is None


# ── Neighbor Queries ─────────────────────────────────────────────

def test_query_neighbors():
    m = SpatialMemory(cell_size=1.0)
    m.record(position=(0.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    m.record(position=(0.5, 0.0, 0.0), entity_type="human", action_type="BUILD")
    # Far away — should not be included
    m.record(position=(10.0, 0.0, 0.0), entity_type="human", action_type="EAT")
    
    results = m.query(position=(0.0, 0.0, 0.0), include_neighbors=True)
    types = {a.action_type for a in results}
    assert "GATHER" in types
    assert "BUILD" in types
    assert "EAT" not in types


# ── Density ──────────────────────────────────────────────────────

def test_density():
    m = SpatialMemory(cell_size=1.0)
    for _ in range(5):
        m.record(position=(1.0, 0.0, 0.0), entity_type="human",
                 action_type="GATHER")
    
    density = m.get_density(position=(1.0, 0.0, 0.0))
    assert density == 5


def test_density_radius():
    m = SpatialMemory(cell_size=1.0)
    m.record(position=(0.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    m.record(position=(5.0, 0.0, 0.0), entity_type="human", action_type="BUILD")
    
    density_near = m.get_density(position=(0.0, 0.0, 0.0), radius=2.0)
    assert density_near == 1


# ── Decay ────────────────────────────────────────────────────────

def test_decay():
    m = SpatialMemory()
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    
    before = m.query(position=(1.0, 0.0, 0.0))[0].frequency_ema
    m.decay(decay_rate=0.5)
    after = m.query(position=(1.0, 0.0, 0.0))[0].frequency_ema
    assert after < before


# ── Prune ────────────────────────────────────────────────────────

def test_prune_low_count():
    m = SpatialMemory()
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="BUILD")
    # GATHER has 1 count, BUILD has 1 count
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    
    result = m.prune(min_count=2)
    assert result["affordances_pruned"] == 1  # BUILD pruned
    remaining = m.query(position=(1.0, 0.0, 0.0))
    assert len(remaining) == 1
    assert remaining[0].action_type == "GATHER"


def test_prune_empty_cells():
    m = SpatialMemory(cell_size=1.0)
    m.record(position=(0.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    m.prune(min_count=10)
    assert m.cell_count == 0


# ── AffordanceSummary ───────────────────────────────────────────

def test_affordance_summary_to_dict():
    a = AffordanceSummary(entity_type="human", action_type="GATHER")
    a.record(success=True, timestamp=1.0)
    d = a.to_dict()
    assert d["entity_type"] == "human"
    assert d["count"] == 1


def test_affordance_summary_total():
    a = AffordanceSummary(entity_type="human", action_type="GATHER")
    for _ in range(3):
        a.record(success=True)
    for _ in range(2):
        a.record(success=False)
    assert a.total == 5


# ── to_dict ──────────────────────────────────────────────────────

def test_memory_to_dict():
    m = SpatialMemory()
    m.record(position=(1.0, 0.0, 0.0), entity_type="human", action_type="GATHER")
    d = m.to_dict()
    assert d["cell_count"] == 1
    assert d["total_records"] == 1


# ── 2D Position ──────────────────────────────────────────────────

def test_2d_position():
    m = SpatialMemory()
    m.record(position=(1.0, 0.0), entity_type="human", action_type="GATHER")
    results = m.query(position=(1.0, 0.0))
    assert len(results) == 1
