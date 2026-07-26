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

## Tactical Brain (SC2 Embodiment)

The TacticalBrain (`epistemology/tactical_brain.py`) implements the observe → hypothesize → experiment → learn loop for the SC2 embodiment.

**Key principle:** No hardcoded counter knowledge. Everything is learned from battle outcomes.

### Architecture

```
BattleState (tick snapshot)
    ↓
EnemyComposition (type counts, value, upgrades)
    ↓
CounterHypothesis ("unit X counters Y" — confidence from outcomes)
    ↓
Experiment ("build 6 Zerglings, track K/D")
    ↓
Validation (hypothesis confirmed/rejected)
```

### Components

- **BattleState**: full tick snapshot — per-unit health/shield/attack/speed, base saturation from SC2 API `ideal_harvesters`, enemy composition
- **UnitSnapshot**: individual unit state with effective HP, attack capability, upgrades
- **BaseSaturation**: per-base worker saturation (real SC2 API values, not assumed averages)
- **CounterHypothesis**: "unit X counters composition Y" — confidence from accumulated battle outcomes
- **Experiment**: mid-game test — build N of hypothesized counter unit, track K/D ratio, validate hypothesis
- **TacticalBrain**: orchestrates observe → hypothesize → experiment → evaluate

### Integration with Bot

- State captured every 5 ticks in `on_step()`
- Counter-unit suggestions used in `_exec_train()` when confidence > 0.3
- Battle outcomes tracked in `_exec_attack()` (20-40 tick evaluation window)
- Persistence via JSON across games (`data/tactical_brain.json`)

## Replay Auditor (SC2 Embodiment)

The ReplayAuditor (`epistemology/replay_auditor.py`) records tick-by-tick game state and produces audit reports identifying failure points.

### Failure Detectors

| Detector | Category | Severity | Trigger |
|----------|----------|----------|---------|
| Supply block | `SUPPLY_BLOCKED` | MED/HIGH | supply_used >= cap-2 for 10+ ticks with minerals |
| Idle production | `IDLE_PRODUCTION` | MED/HIGH | All production buildings idle for 20+ ticks |
| Mineral float | `MINERAL_FLOAT` | MED/HIGH | Minerals > 500 for 15+ ticks |
| Army idle | `ARMY_IDLE` | MED/HIGH | 5+ army units doing nothing for 30+ ticks |
| Unit loss wave | `UNIT_LOSS_WAVE` | HIGH/CRIT | 5+ units lost in 100-tick window |
| Action degenerate | `ACTION_DEGENERATE` | MED | Same action 10+ times in a row |
| Late expansion | `LATE_EXPANSION` | HIGH | No expansion by tick 3000 |
| Defense gap | `DEFENSE_GAP` | CRITICAL | Enemy visible, 0 army |
| Economy stall | `ECONOMY_STALLED` | HIGH | Worker count drops significantly |

### Audit Report

- `TickSnapshot`: one frame of game state (units, resources, supply, actions, army value)
- `FailurePoint`: detected issue with tick, severity, category, description, context
- `AuditReport`: full game analysis — severity/category counts, peak metrics, action diversity (Shannon entropy)
- Improvement tracking: failure count + supply block ticks compared across games
