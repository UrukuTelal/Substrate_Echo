# Changelog

All notable changes to Substrate_Echo will be documented in this file.

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
