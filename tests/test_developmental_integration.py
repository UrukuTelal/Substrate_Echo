"""Integration test: developmental cognition modules with CognitiveLoop."""
import numpy as np
import sys
sys.path.insert(0, r"C:\Projects\Substrate_Echo")

from substrate_echo.core.cognitive_loop import CognitiveLoop, CognitiveLoopConfig
from substrate_echo.core.affordance import IntentTranslator, EntityType, PropertyMap
from substrate_echo.core.goal_tracker import GoalManager, GoalPhase
from substrate_echo.core.communicative_intent import (
    CommunicativeIntentDetector, BehavioralSignals, CommunicativeIntent)


def test_cognitive_loop_with_developmental_modules():
    """Verify the three modules compose in the cognitive loop tick."""
    loop = CognitiveLoop(CognitiveLoopConfig(
        dt=0.1, max_steps_per_tick=1, enforce_conservation=False))
    
    # Wire developmental modules
    loop.set_developmental_modules(
        intent_translator=IntentTranslator(dim=16),
        goal_manager=GoalManager(),
        comm_detector=CommunicativeIntentDetector(ai_position=np.zeros(3)),
    )
    
    # Initialize field and pillars
    dim = 16
    loop.initialize(np.random.randn(dim) * 0.1, np.full(16, 0.5))
    
    # Verify stats
    stats = loop.stats()
    assert stats["has_intent_translator"]
    assert stats["has_goal_manager"]
    assert stats["has_comm_detector"]
    
    # Run tick without external observations (should not crash)
    from substrate_echo.dynamics.field_evolution import FieldEvolver
    gl = FieldEvolver()
    
    result = loop.tick(gl, memory_system=None, agent_ecology=None)
    assert result is not None
    assert "developmental" in result
    assert result["developmental"]["affordances"] == []
    assert result["developmental"]["goal_states"] == {}
    
    print("PASS: integration test")


def test_intent_translator_with_goal_tracker():
    """Verify IntentTranslator → GoalManager pipeline."""
    translator = IntentTranslator(dim=16)
    gm = GoalManager()
    
    # Observe a human approaching over several ticks
    for i in range(5):
        pos = np.array([1.0 - 0.15 * i, 0.0, 0.0])
        
        # Translate
        aff = translator.translate(
            entity_id=1,
            entity_type=EntityType.HUMAN,
            position=pos,
            properties=PropertyMap(distance=float(1.0 - 0.15 * i)),
        )
        
        # Update goal tracker
        gm.update(
            entity_id=1,
            position=pos,
            timestamp=float(i) * 0.5,
            social_intent=aff.social_intent,
        )
    
    goal = gm.get_state(1)
    assert goal is not None
    assert goal.phase in (GoalPhase.EXPLORING, GoalPhase.APPROACHING)
    assert goal.velocity.speed > 0
    
    # Now the human stops nearby
    for i in range(15):
        translator.translate(
            entity_id=1, entity_type=EntityType.HUMAN,
            position=np.array([0.5, 0.0, 0.0]),
            properties=PropertyMap(distance=0.5))
        gm.update(
            entity_id=1, position=np.array([0.5, 0.0, 0.0]),
            timestamp=float(5 + i) * 0.5, social_intent=0.7)
    
    goal = gm.get_state(1)
    assert goal.phase in (GoalPhase.COMMUNICATING, GoalPhase.IDLE)
    
    print("PASS: intent_translator + goal_tracker")


def test_full_perception_pipeline():
    """Full pipeline: perception → affordance → PSV delta → state update."""
    translator = IntentTranslator(dim=16)
    gm = GoalManager()
    detector = CommunicativeIntentDetector(ai_position=np.zeros(3))
    
    # Start from neutral state
    state = np.full(16, 0.5)
    
    # Observe: human approaching, talking, facing me
    aff = translator.translate(
        entity_id=1,
        entity_type=EntityType.HUMAN,
        position=np.array([0.5, 0.0, 0.0]),
        properties=PropertyMap(distance=0.5, sound_level=0.6),
        social_intent_hint=0.5,
    )
    
    # Goal tracking
    gm.update(entity_id=1, position=np.array([0.5, 0.0, 0.0]),
              timestamp=1.0, social_intent=aff.social_intent)
    
    # Communicative intent
    signals = BehavioralSignals(
        speech_level=0.6,
        speech_duration=2.0,
        facing_toward_me=True,
        distance=0.5,
    )
    comm = detector.analyze(signals, entity_position=np.array([0.5, 0.0, 0.0]))
    
    # Apply PSV deltas
    new_state = translator.apply_deltas(state, [aff])
    
    # Verify state changed
    assert not np.allclose(new_state, state)
    assert np.all(new_state >= 0.0) and np.all(new_state <= 1.0)
    
    # Verify affordance has actions and deltas
    assert len(aff.possible_actions) > 0
    assert len(aff.pillar_deltas) > 0
    
    # Verify goal tracking
    goal = gm.get_state(1)
    assert goal is not None
    
    # Verify communicative intent
    assert comm.confidence > 0
    
    print("PASS: full perception pipeline")


if __name__ == "__main__":
    test_cognitive_loop_with_developmental_modules()
    test_intent_translator_with_goal_tracker()
    test_full_perception_pipeline()
    print("\nAll integration tests passed!")
