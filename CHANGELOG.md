# Changelog

All notable changes to Substrate_Echo will be documented in this file.

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
