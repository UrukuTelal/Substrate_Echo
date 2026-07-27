# Changelog

All notable changes to Substrate_Echo will be documented in this file.

## [0.15.0] - 2026-07-27

### Fixed — Core Training & Execution Pipeline

**WorldState army_count bug** (`substrate_echo/epistemology/affordance_tracer.py:184`):
- Was counting ALL units (workers, structures, overlords) as "army"
- Fixed to only count combat units (excludes workers, supply, structures, spawned)
- Attack affordance now correctly gates on actual army ≥ 5

**_exec_train execution logic** (`scripts/sc2_iterate_01.py`):
- `build_army` action now correctly forces army training via `force_army=True`
- Removed unconditional general training loop at end of `_exec_train` that always trained drones
- Army training now filters out workers AND supply units (overlords, medivacs, etc.)

**Gas awareness** (`substrate_echo/epistemology/affordance_tracer.py`, `scripts/sc2_iterate_01.py`):
- Drives system now factors actual vespene reserves into GAS satisfaction
- Gas building gate triggers when minerals high but gas low (minerals > 300, gas < 50)
- Build_army affordance scales reward/success with gas availability
- Tech_up affordance penalizes gas-starved tech attempts
- Expand reward boosted when gas-starved (new base = new geysers)

**_exec_tech_up rewritten substrate-agnostic** (`scripts/sc2_iterate_01.py`):
- Discovers BUILD/RESEARCH/MORPH abilities dynamically from capabilities
- No longer hardcoded Zerg tech tree — works for all races
- Phase 1: builds any non-supply/non-gas/non-townhall structure
- Phase 2: researches any idle tech structure
- Phase 3: morphs advanced townhalls

**Gas building fixes** (`scripts/sc2_iterate_01.py`):
- Removed two-step move-then-build pattern; SC2 API handles worker movement
- Removed dead `_drone_to_geyser` tracking dict
- Gas structure filter now race-agnostic (EXTRACTOR/ASSIMILATOR/REFINERY)

**Race lock removed**:
- `--race` CLI arg added (Zerg/Terran/Protoss/Random)
- `Race.Zerg` hardcode replaced with config-driven race selection

**Race-agnostic name constants** (`scripts/sc2_iterate_01.py` class-level):
- `SUPPLY_NAMES`, `SPAWNED_NAMES`, `WORKER_KEYWORDS`, `SUPPLY_KEYWORDS`, `TOWNHALL_KEYWORDS`, `GAS_KEYWORDS`
- Replaced 4× duplicated inline sets

**F1 spam mitigation**:
- TRAIN/MORPH filter excludes BUILD abilities (which need target positions)

### Added
- `scripts/sc2_iterate_01.py`: `--race` argument
- `substrate_echo/epistemology/affordance_tracer.py`: `vespene` field used in drives
- Debug logging for `tech_up` and `build_army` execution

### Changed
- `build_army` action distribution increased from ~1% to ~5-20%
- `tech_up` now successfully builds Spawning Pool, Evolution Chamber
- Zerglings trained and attack when army ≥ 5

### Known Issues
- Zerglings may be classified as scouts by affordance tracer
- Larva TRAIN_DRONE spam errors expected (waiting for minerals/larva limit)
- Gas extractor build errors when drone already on geyser

## [0.14.0] - 2026-07-26

### Added — Representational Layer (shared semantic substrate)

The Representational Layer is the foundation all subsystems reason over.
It answers four questions about the same reality: What did I observe?
What does it mean? What patterns are emerging? What should I do?

**Ontology (representational/ontology.py):**
- `Concept`: atoms of knowledge — what things are (unit, structure, terrain, strategy)
- `TaxonomyNode`: is_a hierarchy enabling transitive reasoning (Marine → Unit → Entity)
- `PropertySchema`: typed, bounded properties entities can have
- `Rule`: causal relationships (condition → consequence) with confidence
- `Constraint`: what cannot happen — shrinks search space dramatically
- `Ontology`: unified static knowledge store with save/load

**Entity Descriptor (representational/entity_descriptor.py):**
- Universal cognitive object — everything reasoned about uses this structure
- 13 sub-structures: Identity, Embodiment, Classification, Composition, Capabilities, Affordances, Relationships, State, Observations, Hypotheses, Evidence, Causality, History
- Key distinction: Capabilities ≠ Affordances. Capabilities = what CAN it do. Affordances = what opportunities it CREATES
- Evidence never overwritten by hypotheses — separate fields
- Uncertainty always explicit — every field carries confidence
- Full serialization roundtrip

**State Graph (representational/state_graph.py):**
- Dynamic entity tracking across ticks
- Typed edges (enemy, allies, counters, adjacent_to) with confidence
- Query methods: by role, taxonomy, capability, affordance, relationship
- Graph traversal: neighbors, connected_by, counters_of
- Persistence with save/load

**Semantic Interpreter (representational/interpreter.py):**
- Translates raw SC2 observations into EntityDescriptors
- SC2 knowledge base: 15 unit types, 4 structure types with taxonomy, capabilities, affordances, counter relationships
- Auto-populates Ontology on first use
- Processes own units, enemy units, own structures, enemy structures
- Updates StateGraph edges based on counter relationships
- Generates terrain EntityDescriptor from computed metrics
- Marks unseen entities with decaying confidence

**Causal Graph (representational/causal_graph.py):**
- Records events: unit created/destroyed/damaged/moved, structure events, resource changes, army engagements
- Auto-generates consequences for high-impact events
- Event indexing by tick and entity for fast queries
- Temporal causality: get_causes() and get_effects()
- Compression: old events summarized by significance

**Frame System (representational/frames.py):**
- 8 frame types: Danger, Opportunity, Uncertainty, Composition, Terrain, Temporal, Economic, ThreatAssessment
- Each frame is a query over the world model answering a strategic question
- 4 built-in Perspectives: EarlyGame, MidGame, UnderAttack, Attacking
- Perspectives weight which frames dominate reasoning
- Dynamic: perspective shifts based on game state

**Narrative Layer (representational/narrative.py):**
- Compresses causal event chains into temporal stories
- Dramatic arc: Setup → Rising Action → Climax → Falling Action → Resolution
- Auto-determines outcome from entity participation
- Narrative types: Engagement, Economic, Strategic, Terrain, Composition, Scouting
- Active narratives auto-close after 500 tick silence or complete arcs

**SC2 Bot Integration (scripts/sc2_iterate_01.py):**
- Imports all representational components
- Constructor initializes Ontology, StateGraph, Interpreter, CausalGraph, FrameSystem, NarrativeLayer
- on_step: every 3 ticks — interpret game state, record causal events, select perspective, update narratives
- `_record_causal_events()`: tracks army/worker/base deltas as causal events
- `_select_perspective()`: early/mid/under_attack/attacking based on threat ratio
- on_end: saves state_graph.json, causal_graph.json, narratives.json; prints all summaries

**Tests:**
- 37 new tests in `tests/test_representational.py`
- Full coverage: Ontology, EntityDescriptor, StateGraph, CausalGraph, FrameSystem, NarrativeLayer, SemanticInterpreter
- 835 tests passing total (798 original + 37 new)

## [0.13.0] - 2026-07-25

### Added — TacticalBrain + ReplayAuditor

**TacticalBrain (epistemology/tactical_brain.py):**
- `BattleState`: full tick snapshot — per-unit health/shield/attack/speed, base saturation, enemy composition
- `UnitSnapshot`: individual unit state with effective HP, attack capability, upgrades
- `BaseSaturation`: per-base worker saturation from SC2 API `ideal_harvesters` (not hardcoded)
- `EnemyComposition`: type counts, total value, upgrade levels, movement profiles
- `CounterHypothesis`: "unit X counters composition Y" — confidence from battle outcomes only
- `Experiment`: mid-game test — build N of hypothesized counter, track K/D, validate
- `TacticalBrain`: orchestrates observe → hypothesize → experiment → evaluate (pure learning, no hardcoded counters)
- Cross-game persistence via JSON (`data/tactical_brain.json`)

**ReplayAuditor (epistemology/replay_auditor.py):**
- `TickSnapshot`: one frame of game state (units, resources, supply, actions, army value)
- `FailurePoint`: detected issue with tick, severity, category, description, context
- `AuditReport`: full game analysis — severity/category counts, peak metrics, action diversity (Shannon entropy)
- 10 failure detectors:
  - `SUPPLY_BLOCKED`: supply_used >= cap-2 for 10+ ticks with minerals
  - `IDLE_PRODUCTION`: all production buildings idle for 20+ ticks
  - `MINERAL_FLOAT`: minerals > 500 for 15+ ticks
  - `ARMY_IDLE`: 5+ army units doing nothing for 30+ ticks
  - `UNIT_LOSS_WAVE`: 5+ units lost in 100-tick window
  - `ACTION_DEGENERATE`: same action 10+ times in a row
  - `LATE_EXPANSION`: no expansion by tick 3000
  - `DEFENSE_GAP`: enemy visible, 0 army (CRITICAL)
  - `ECONOMY_STALLED`: worker count drops significantly
  - `POOR_COMPOSITION`: army comp doesn't counter enemy
- Wired into IterateBot: `record_tick()` every tick, `analyze()` at game end
- Audit report in JSON output, improvement comparison includes failure metrics

**SC2 Bot Wiring (scripts/sc2_iterate_01.py):**
- TacticalBrain state capture every 5 ticks in `on_step()`
- Counter-unit selection in `_exec_train()`: suggests counter units when confidence > 0.3
- Battle outcome tracking in `_exec_attack()`: records pre-battle composition, evaluates 20-40 ticks later
- Real per-base saturation from SC2 API `ideal_harvesters` in `_update_drives()`
- TacticalBrain persistence: saves every 500 ticks + at game end

**External Agent Pipeline:**
- Fixed crowquant dependency (`pip install -e /c/Projects/crowquant`)
- `test_external_agents.py` now runs (65 tests, all passing)

### Tests
- 798 tests passing (26 new auditor tests + 65 external agent tests restored)
- `test_replay_auditor.py`: 26 tests covering all 10 failure detectors, report generation, snapshot properties

## [0.12.0] - 2026-07-23

### Added — SC2 Embodiment (COMPLETE ✓)
SC2 is now a controllable external environment for Substrate_Echo.

**Embodiment Layer:**
- `embodiments/sc2/observation_encoder.py`: SC2 game state → 16D substrate vectors
- `embodiments/sc2/action_decoder.py`: Abstract intent → SC2 commands
- `embodiments/sc2/sc2_bot.py`: Main SC2 embodiment adapter (TESTED - 500 steps)
- `embodiments/sc2/adapter.py`: SC2 ↔ Substrate Kernel bridge

**Social/Cognitive Layer:**
- `embodiments/sc2/trust.py`: Dynamic trust attractor for multi-agent interactions
- `embodiments/sc2/communication.py`: Selective information sharing
- `embodiments/sc2/trickster.py`: Narrative/social intelligence layer
- `embodiments/sc2/council_sc2.py`: Diplomat, Trust Analyst, Negotiator, Adversary Model
- `embodiments/sc2/truce_mode.py`: Alternative optimization landscape for cooperative play

**Connection Verified:**
- 500 steps completed successfully against Easy AI
- 9 Melee maps installed at `C:\Program Files (x86)\StarCraft II\Maps\Melee\`
- sc2 library (v0.11.2) with protobuf 3.20.3 for compatibility

## [0.11.0] - 2026-07-23

### Added — Competing Pressures Stress Test (EXP-SUB-005)
- `scripts/exp_sub_005_competing_pressures.py`: Multi-embodiment stress test
- 3 embodiments: desktop (answer request), robot (avoid obstacle), simulation (explore novelty)
- 4 stress scenarios: resource squeeze, conflicting goals, prediction degradation, resource release
- 6/6 architecture checks passing

### Fixed
- Goal explosion: safety generator now uses per-embodiment cooldown (20-tick window)
- Council health decay: exponential decay with recovery (was linear collapse to 0)
- Trajectory-based observations: cyclical patterns replace pure noise

## [0.10.0] - 2026-07-22

### Added — Integration Tests (S13-S14)
- `tests/test_integration.py`: Full pipeline validation
- Goal → Attention pipeline tests
- Resource → Embodiment sharing tests
- Council → Audit pipeline tests
- Prediction → Confidence tests
- Multi-embodiment learning tests
- End-to-end full cycle tests
- 19 tests passing

## [0.9.0] - 2026-07-22

### Added — Council Layer (S12)
- `kernel/council.py`: Council for metacognition and health checks
- ScheduledAuditor: periodic audits every N ticks
- EventAuditor: trigger-based audits (collapse, entropy, memory explosion)
- DriftDetector: architectural and concept drift detection
- AuditReport: observations, anomalies, hypotheses, recommendations
- CouncilState: health score, drift score, pending reports
- 18 tests passing (test_council.py)

## [0.8.0] - 2026-07-22

### Added — Resource Manager (S11)
- `kernel/resources.py`: ResourceManager for finite cognitive resources
- ResourceBudget: compute, memory, learning, attention tracking
- ResourceLease: time-limited resource grants with tier priorities
- ResourceRequest/Allocation: request/grant protocol
- Safety scaling: emergency resource throttling
- 17 tests passing (test_resources.py)

## [0.7.0] - 2026-07-22

### Added — Executive Function Layer (S10)
- `kernel/executive.py`: Goal lifecycle management (S10.1)
- PriorityScorer: urgency × importance × confidence × expected_value / resource_cost (S10.2)
- AttentionAllocator: finite attention based on prediction errors and novelty (S10.3)
- GoalGenerator: automatic goal creation from safety triggers (S10.4)
- ExecutiveFunction: orchestrates goal management, integrated into SubstrateKernel
- 16 tests passing (test_executive.py)

## [0.6.0] - 2026-07-21

### Added — Substrate Kernel
- `kernel/__init__.py`: SubstrateKernel cognitive backend with two-plane architecture
- `kernel/api.py`: Control Plane (REST) + Cognitive Plane (WebSocket)
- `kernel/client.py`: In-process and streaming client libraries
- State schema: Observation, Goal, Reward, Action, Prediction, EmbodimentState, CognitiveState
- Multiple embodiments share one kernel (cross-embodiment learning)

### Added — Basin Topology
- `dynamics/basin_topology.py`: BasinMetrics, AttractorState with plasticity properties
- Basin depth (energy contrast), volume (isolation), entropy (diversity), balance (dominance)
- Attractor plasticity: stability, plasticity, novelty, confidence, access tracking
- Structural event detection: births, deaths, merges, splits

### Added — Abstraction Engine
- `dynamics/abstraction.py`: AttractorCorrelation, AbstractionEngine, MetaAttractor, CognitiveBudget
- Time-proximity co-activation correlation between attractors
- Meta-attractor creation from correlated clusters (hierarchy building)
- Finite cognitive budget (competition for resources)

### Experiments
- EXP-SUB-002: Closed feedback loop — 2→14 attractors, coherence 0.202→0.924, self-reinforcing
- EXP-SUB-003: Basin topology — 16 attractors, depth 0.457, plasticity分化, 16 births/0 deaths
- EXP-SUB-004: Abstraction hierarchy — 4 meta-attractors from correlated base attractors
- Demo: two embodiments (desktop + robot) sharing one cognitive kernel

## [0.5.0] - 2026-07-21

### Added
- External agent integration pipeline (S8): InteractionEncoder, ForeignEvaluator, IntegrationGate, CandidateQueue
- Foreign ecosystem simulation (S9): 7 synthetic behavioral archetypes, validation harness
- Verification loop (S10): prediction-based verification, confidence decay, provenance tracking
- Domain-conditioned reputation (S11): per-domain trust, Bayesian blending, keyword domain detection
- Social persona ecology (S12): PersonaGenome, PersonaDynamics, 3-layer adaptation, 6 agent genomes
- LatentIntegrationRecord: audit trail from latent vectors back to source
- IntegrationMode: OBSERVATION_ONLY, CANDIDATE_ONLY, FULL safety progression
- WHT relocated to post-acceptance (epistemic firewall fix)
- Security hardening: input truncation, queue caps, rate limiting, error isolation

### Experiments
- EXP-EXT-001: WHT is distance-preserving (ratio=1.0000)
- EXP-EXT-001B: Quantization provides 3.6x invariance
- EXP-EXT-002: Temporal reputation drift (5 drifting agents, FAR=0%)
- EXP-EXT-003: Firewall benchmark (86% contamination rate with heuristic encoder)
- EXP-EXT-004: Prediction-based trust (system values usefulness over presentation)
- EXP-EXT-005: Domain transfer (Physics A>C>B, Social B>C>A)
- EXP-SOC-001: Cognitive ecology stability (6 agents, 2000 interactions, stable divergence)

### Tests
- 702 tests passing across 30+ test suites

## [0.1.0] - 2026-07-19

### Added
- Core models: WorldObject, Experience, MemoryTrace, Action
- Spatial world model with grid indexing
- Ontological field with attractor/repulsor dynamics
- Attractor memory (encode, recall, consolidate)
- 5 cognitive agents with ecology
- Embodiment bridge for AR/robotics
- Integration bridges: PSV, Void, Engine, Council
- Dynamics memory, meta-cognition, episodic memory, hierarchical planner
- Habit formation, counterfactual reasoning, self-model, theory of mind
- Emotional contagion, experience scheduler, goal tracker
- 28 initial tests
