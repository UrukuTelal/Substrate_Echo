# Cognitive Architecture

## The Cognitive Loop

The `IntegratedAgent.think()` method runs a 16-step cycle each tick:

1. Perception (spatial + affordance)
2. Memory consolidation
3. Emotional contagion
4. Intent generation
5. Self-model update
6. Theory of mind
7. Hierarchical planning
8. Habit formation check
9. Goal tracking
10. Counterfactual reasoning
11. Meta-cognition (confidence calibration)
12. Controller (desired → feasible PSV)
13. World model update
14. Evaluation
15. Prediction verification (wired to DynamicsMemory)
16. Idle exploration (wired to ExperienceScheduler)

## Memory Systems

| System | Purpose | Persistence |
|--------|---------|-------------|
| AttractorMemory | Encode experiences as attractors | Permanent |
| DynamicsMemory | Learn velocity field V(x) = Ax + b | Permanent |
| EpisodicMemory | Narrative chapters with causal links | Permanent |
| SpatialMemory | Location-based recall | Permanent |
| SocialMemory | Interaction episodes with outcomes | Permanent |
| RelationshipMemory | Per-agent trust and collaboration | Permanent |
| HabitFormation | Repeated sequence automation | Permanent |

## Meta-Cognition

- Calibrated confidence: predicted vs actual accuracy (Brier score)
- Per-source trust: auto-adjusts based on predictive usefulness
- Self-correction: detects overconfidence and adjusts
- Disagreement detection: flags when sources conflict

## Planning

- Hierarchical goal decomposition (ROOT → SUBGOAL → ACTION)
- Reusable strategies learned from successful plans
- Counterfactual simulation of alternative decisions
- Risk assessment before action execution
