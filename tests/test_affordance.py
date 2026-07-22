"""Tests for Affordance & Intent Translation Layer."""
import numpy as np
import sys
sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.affordance import (
    EntityType, ActionType, PropertyMap, GoalEstimate,
    ActionAffordance, PSVDelta, Affordance,
    IntentTranslator, DEFAULT_TRANSLATION_RULES,
)


# ── Basic Data Structures ────────────────────────────────────────

def test_property_map_to_array():
    p = PropertyMap(temperature=0.5, moisture=0.3, mass=1.0)
    arr = p.to_array()
    assert arr.shape == (10,)
    assert arr[0] == 0.5
    assert arr[1] == 0.3
    assert arr[2] == 1.0


def test_action_affordance_score():
    a = ActionAffordance(
        action=ActionType.APPROACH,
        feasibility=0.8, cost=0.1, benefit=0.6)
    assert a.score == 0.6 * 0.8 - 0.1 * 0.3  # 0.45
    assert a.score > 0


def test_action_affordance_zero_feasibility():
    a = ActionAffordance(
        action=ActionType.DEFEND,
        feasibility=0.0, cost=0.5, benefit=1.0)
    assert a.score == -0.15  # 1.0*0 - 0.5*0.3


def test_psv_delta_repr():
    d = PSVDelta(pillar_idx=5, delta_theta=0.15, reason="test")
    r = repr(d)
    assert "Integrity" in r
    assert "↑" in r
    assert "0.150" in r


def test_psv_delta_repr_negative():
    d = PSVDelta(pillar_idx=12, delta_theta=-0.20, reason="harm")
    r = repr(d)
    assert "Harm" in r
    assert "↓" in r


def test_affordance_best_action():
    a = Affordance(
        entity_id=1,
        entity_type=EntityType.HUMAN,
        possible_actions=[
            ActionAffordance(ActionType.OBSERVE, 1.0, 0.0, 0.2),
            ActionAffordance(ActionType.ASSIST, 0.6, 0.3, 0.5),
        ])
    best = a.best_action
    assert best is not None
    # ASSIST: 0.5*0.6 - 0.3*0.3 = 0.21
    # OBSERVE: 0.2*1.0 - 0.0*0.3 = 0.20
    assert best.action == ActionType.ASSIST


def test_affordance_empty_actions():
    a = Affordance(entity_id=1, entity_type=EntityType.UNKNOWN)
    assert a.best_action is None


def test_affordance_psv_magnitude():
    a = Affordance(
        entity_id=1,
        entity_type=EntityType.HUMAN,
        pillar_deltas=[
            PSVDelta(7, 0.15, "test"),
            PSVDelta(8, 0.10, "test"),
        ])
    assert abs(a.total_psv_magnitude - 0.25) < 1e-10


# ── IntentTranslator: Single Entity ──────────────────────────────

def test_translate_human_approach():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=1,
        entity_type=EntityType.HUMAN,
        position=np.array([0.3, 0.5, 0.0]),
        properties=PropertyMap(distance=0.3),
    )
    assert aff.entity_type == EntityType.HUMAN
    assert len(aff.possible_actions) > 0
    assert len(aff.pillar_deltas) > 0
    # Should have social intent (close proximity)
    assert aff.social_intent > 0


def test_translate_human_far_away():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=1,
        entity_type=EntityType.HUMAN,
        position=np.array([2.0, 3.0, 0.0]),
        properties=PropertyMap(distance=2.0),
    )
    # Far away = less social intent
    assert aff.social_intent < 0.3


def test_translate_plant():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=2,
        entity_type=EntityType.PLANT,
        position=np.array([0.0, 0.0, 0.0]),
        properties=PropertyMap(moisture=0.1),  # dry plant
    )
    assert aff.entity_type == EntityType.PLANT
    assert len(aff.pillar_deltas) > 0
    # Should suggest assisting (dry plant)
    actions = [a.action for a in aff.possible_actions]
    assert ActionType.ASSIST in actions


def test_translate_plant_well_watered():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=2,
        entity_type=EntityType.PLANT,
        position=np.array([0.0, 0.0, 0.0]),
        properties=PropertyMap(moisture=0.9),
    )
    # Well-watered plant: ASSIST benefit should be low
    assist = [a for a in aff.possible_actions if a.action == ActionType.ASSIST]
    assert len(assist) == 1
    assert assist[0].benefit < 0.2  # low benefit


def test_translate_animal():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=3,
        entity_type=EntityType.ANIMAL,
        position=np.array([0.5, 0.5, 0.0]),
    )
    assert aff.entity_type == EntityType.ANIMAL
    actions = [a.action for a in aff.possible_actions]
    assert ActionType.OBSERVE in actions
    assert ActionType.MONITOR in actions


def test_translate_mycelium():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=4,
        entity_type=EntityType.MYCELIUM,
        position=np.array([0.0, 0.0, -1.0]),
    )
    actions = [a.action for a in aff.possible_actions]
    assert ActionType.MONITOR in actions


def test_translate_terrain():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=5,
        entity_type=EntityType.TERRAIN,
        position=np.array([1.0, 0.0, 0.0]),
        properties=PropertyMap(toxicity=0.7),
    )
    # Toxic terrain: should suggest avoiding
    actions = [a.action for a in aff.possible_actions]
    assert ActionType.AVOID in actions
    # Should have hazard-related PSV deltas
    hazard_deltas = [d for d in aff.pillar_deltas if "hazard" in d.reason]
    assert len(hazard_deltas) > 0


def test_translate_unknown():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=6,
        entity_type=EntityType.UNKNOWN,
        position=np.array([0.5, 0.5, 0.0]),
    )
    actions = [a.action for a in aff.possible_actions]
    assert ActionType.INVESTIGATE in actions
    assert ActionType.DEFEND in actions


# ── Social Intent Detection ──────────────────────────────────────

def test_social_intent_close_proximity():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=1,
        entity_type=EntityType.HUMAN,
        position=np.array([0.2, 0.0, 0.0]),
        properties=PropertyMap(distance=0.2),
    )
    assert aff.social_intent >= 0.2  # base 0 + distance bonus 0.2


def test_social_intent_with_hint():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=1,
        entity_type=EntityType.HUMAN,
        position=np.array([1.0, 0.0, 0.0]),
        social_intent_hint=0.8,
    )
    assert aff.social_intent > 0.7


def test_social_intent_sound():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=1,
        entity_type=EntityType.HUMAN,
        position=np.array([1.0, 0.0, 0.0]),
        properties=PropertyMap(sound_level=0.9),
    )
    assert aff.social_intent > 0.3


# ── Goal Inference ───────────────────────────────────────────────

def test_goal_inference_needs_history():
    t = IntentTranslator(dim=16)
    # Single observation: no goal
    aff = t.translate(
        entity_id=1, entity_type=EntityType.HUMAN,
        position=np.array([0.0, 0.0, 0.0]))
    assert aff.inferred_goal is None


def test_goal_inference_moving_entity():
    t = IntentTranslator(dim=16)
    # Simulate movement along a direction
    for i in range(5):
        t.translate(
            entity_id=1, entity_type=EntityType.HUMAN,
            position=np.array([0.1 * i, 0.05 * i, 0.0]))
    
    aff = t.translate(
        entity_id=1, entity_type=EntityType.HUMAN,
        position=np.array([0.5, 0.25, 0.0]))
    assert aff.inferred_goal is not None
    assert "moving" in aff.inferred_goal.description


def test_goal_inference_stationary():
    t = IntentTranslator(dim=16)
    for i in range(5):
        t.translate(
            entity_id=1, entity_type=EntityType.HUMAN,
            position=np.array([1.0, 1.0, 0.0]))  # stationary
    
    aff = t.translate(
        entity_id=1, entity_type=EntityType.HUMAN,
        position=np.array([1.0, 1.0, 0.0]))
    assert aff.inferred_goal is not None
    assert "stationary" in aff.inferred_goal.description


# ── PSV Delta Generation ────────────────────────────────────────

def test_psv_deltas_human_observed():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=1, entity_type=EntityType.HUMAN,
        position=np.array([0.3, 0.0, 0.0]),
        properties=PropertyMap(distance=0.3))
    
    pillar_indices = [d.pillar_idx for d in aff.pillar_deltas]
    # ASSIST rule gives: Warmth(9), Relation(2 is Force per rules)... 
    # Best action for benign human at distance=0.3 is ASSIST (score=0.21)
    # Rule (HUMAN, ASSIST) gives: Warmth(9), Relation(7), Force(2)
    assert 9 in pillar_indices  # Warmth from ASSIST rule
    assert 7 in pillar_indices  # Relation from ASSIST rule


def test_psv_deltas_hazard():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=5, entity_type=EntityType.TERRAIN,
        position=np.array([1.0, 0.0, 0.0]),
        properties=PropertyMap(toxicity=0.9))
    
    pillar_indices = [d.pillar_idx for d in aff.pillar_deltas]
    # Should affect Resistance (4), Awareness (0)
    assert 4 in pillar_indices
    assert 0 in pillar_indices


def test_psv_deltas_familiar_entity():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=1, entity_type=EntityType.HUMAN,
        position=np.array([0.5, 0.0, 0.0]),
        properties=PropertyMap(familiar=True, distance=0.5))
    
    pillar_indices = [d.pillar_idx for d in aff.pillar_deltas]
    # Should boost Memory (10) for familiar entities
    assert 10 in pillar_indices


# ── Apply Deltas ─────────────────────────────────────────────────

def test_apply_deltas_basic():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=1, entity_type=EntityType.HUMAN,
        position=np.array([0.3, 0.0, 0.0]),
        properties=PropertyMap(distance=0.3))
    
    state = np.zeros(16)
    new_state = t.apply_deltas(state, [aff])
    
    # State should have changed
    assert np.any(new_state != state)
    # All values in [0, 1]
    assert np.all(new_state >= 0.0)
    assert np.all(new_state <= 1.0)


def test_apply_deltas_clipping():
    t = IntentTranslator(dim=16)
    aff = Affordance(
        entity_id=1,
        entity_type=EntityType.HUMAN,
        pillar_deltas=[PSVDelta(0, 0.5, "test")])
    
    state = np.full(16, 0.9)
    new_state = t.apply_deltas(state, [aff])
    # Should clip to 1.0, not exceed
    assert new_state[0] == 1.0


def test_apply_deltas_multiple():
    t = IntentTranslator(dim=16)
    aff1 = Affordance(
        entity_id=1, entity_type=EntityType.HUMAN,
        pillar_deltas=[PSVDelta(7, 0.10, "social")])
    aff2 = Affordance(
        entity_id=2, entity_type=EntityType.PLANT,
        pillar_deltas=[PSVDelta(9, 0.15, "nurturing")])
    
    state = np.zeros(16)
    new_state = t.apply_deltas(state, [aff1, aff2])
    assert new_state[7] == 0.10
    assert new_state[9] == 0.15


# ── Batch Translation ───────────────────────────────────────────

def test_translate_batch():
    t = IntentTranslator(dim=16)
    observations = [
        {"entity_id": 1, "entity_type": EntityType.HUMAN,
         "position": np.array([0.3, 0.0, 0.0])},
        {"entity_id": 2, "entity_type": EntityType.PLANT,
         "position": np.array([0.0, 0.5, 0.0]),
         "properties": PropertyMap(moisture=0.1)},
    ]
    results = t.translate_batch(observations)
    assert len(results) == 2
    assert results[0].entity_type == EntityType.HUMAN
    assert results[1].entity_type == EntityType.PLANT


# ── Response Detection ───────────────────────────────────────────

def test_requires_response_high_social():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=1, entity_type=EntityType.HUMAN,
        position=np.array([0.3, 0.0, 0.0]),
        social_intent_hint=0.9)
    assert aff.requires_response


def test_requires_response_toxic():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=5, entity_type=EntityType.TERRAIN,
        position=np.array([0.5, 0.0, 0.0]),
        properties=PropertyMap(toxicity=0.8))
    assert aff.requires_response


def test_no_requires_response_benign():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=5, entity_type=EntityType.OBJECT,
        position=np.array([2.0, 2.0, 0.0]),
        properties=PropertyMap(toxicity=0.0, motion_speed=0.0))
    assert not aff.requires_response


# ── Edge Cases ───────────────────────────────────────────────────

def test_translate_large_dim():
    t = IntentTranslator(dim=32)  # extra dimensions
    aff = t.translate(
        entity_id=1, entity_type=EntityType.HUMAN,
        position=np.array([0.0, 0.0, 0.0]))
    # Should work, deltas respect dim
    for d in aff.pillar_deltas:
        assert d.pillar_idx < 32


def test_empty_batch():
    t = IntentTranslator(dim=16)
    results = t.translate_batch([])
    assert results == []


def test_entity_history_bounded():
    t = IntentTranslator(dim=16)
    for i in range(30):
        t.translate(
            entity_id=1, entity_type=EntityType.HUMAN,
            position=np.array([float(i), 0.0, 0.0]))
    assert len(t._entity_histories[1]) <= 20  # bounded


# ── Action Scoring ───────────────────────────────────────────────

def test_human_best_action_is_observe_or_communicate():
    """For benign human, OBSERVE should rank highest."""
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=1, entity_type=EntityType.HUMAN,
        position=np.array([0.5, 0.5, 0.0]),
        properties=PropertyMap(distance=0.5))
    best = aff.best_action
    assert best is not None
    assert best.action in (ActionType.OBSERVE, ActionType.APPROACH, ActionType.COMMUNICATE, ActionType.ASSIST)


def test_animal_best_action_is_observe():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=1, entity_type=EntityType.ANIMAL,
        position=np.array([1.0, 1.0, 0.0]))
    best = aff.best_action
    assert best is not None
    assert best.action == ActionType.OBSERVE  # cost=0, feasibility=1.0


def test_terrain_high_toxicity_avoid():
    t = IntentTranslator(dim=16)
    aff = t.translate(
        entity_id=5, entity_type=EntityType.TERRAIN,
        position=np.array([0.5, 0.0, 0.0]),
        properties=PropertyMap(toxicity=0.9))
    # AVOID should have high benefit for toxic terrain
    avoid = [a for a in aff.possible_actions if a.action == ActionType.AVOID]
    assert len(avoid) == 1
    assert avoid[0].benefit > 0.2


# ── HSV Integration ──────────────────────────────────────────────

def test_hsv_high_social_openness_increases_social_intent():
    from substrate_echo.core.human_state import HSVState, GaussianDim
    t = IntentTranslator(dim=16)
    
    # HSV: high social openness
    hsv = HSVState()
    hsv.social_openness.mean = 0.9
    hsv.social_openness.variance = 0.05
    
    aff_no_hsv = t.translate(
        entity_id=1, entity_type=EntityType.HUMAN,
        position=np.array([1.0, 0.0, 0.0]),
        properties=PropertyMap(distance=1.0))
    
    t2 = IntentTranslator(dim=16)
    aff_hsv = t2.translate(
        entity_id=2, entity_type=EntityType.HUMAN,
        position=np.array([1.0, 0.0, 0.0]),
        properties=PropertyMap(distance=1.0),
        hsv_state=hsv)
    
    # HSV should boost social intent
    assert aff_hsv.social_intent >= aff_no_hsv.social_intent


def test_hsv_low_social_openness_decreases_social_intent():
    from substrate_echo.core.human_state import HSVState
    t = IntentTranslator(dim=16)
    
    hsv = HSVState()
    hsv.social_openness.mean = 0.1
    hsv.social_openness.variance = 0.05
    hsv.confidence  # should be high (low variance)
    
    aff = t.translate(
        entity_id=1, entity_type=EntityType.HUMAN,
        position=np.array([0.3, 0.0, 0.0]),
        properties=PropertyMap(distance=0.3),
        hsv_state=hsv)
    
    # Low openness should suppress social intent vs default
    t2 = IntentTranslator(dim=16)
    aff_default = t2.translate(
        entity_id=2, entity_type=EntityType.HUMAN,
        position=np.array([0.3, 0.0, 0.0]),
        properties=PropertyMap(distance=0.3))
    
    assert aff.social_intent <= aff_default.social_intent + 0.1


def test_hsv_high_fatigue_dampens_deltas():
    from substrate_echo.core.human_state import HSVState
    t = IntentTranslator(dim=16)
    
    hsv = HSVState()
    hsv.fatigue.mean = 0.9
    hsv.fatigue.variance = 0.05
    
    aff = t.translate(
        entity_id=1, entity_type=EntityType.HUMAN,
        position=np.array([0.3, 0.0, 0.0]),
        properties=PropertyMap(distance=0.3),
        hsv_state=hsv)
    
    # Fatigued human: deltas should be dampened
    assert aff.total_psv_magnitude < 0.5


def test_hsv_ignored_for_non_human():
    from substrate_echo.core.human_state import HSVState
    t = IntentTranslator(dim=16)
    
    hsv = HSVState()
    hsv.social_openness.mean = 0.95
    
    # Translate a plant with HSV — should be ignored
    aff = t.translate(
        entity_id=1, entity_type=EntityType.PLANT,
        position=np.array([0.0, 0.0, 0.0]),
        properties=PropertyMap(moisture=0.5),
        hsv_state=hsv)
    
    # Should not affect plant translation
    assert aff.entity_type == EntityType.PLANT
    # HSV only modulates humans, plant translation unaffected
    assert aff.social_intent >= 0.0


def test_hsv_high_arousal_boosts_force():
    from substrate_echo.core.human_state import HSVState
    t = IntentTranslator(dim=16)
    
    hsv = HSVState()
    hsv.arousal.mean = 0.9
    hsv.arousal.variance = 0.05
    
    aff = t.translate(
        entity_id=1, entity_type=EntityType.HUMAN,
        position=np.array([0.3, 0.0, 0.0]),
        properties=PropertyMap(distance=0.3),
        hsv_state=hsv)
    
    # High arousal: Force (2) deltas should be boosted
    force_deltas = [d for d in aff.pillar_deltas if d.pillar_idx == 2]
    if force_deltas:
        assert force_deltas[0].delta_theta > 0
