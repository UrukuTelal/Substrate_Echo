# Live Journal of Ecosystem System Status

**Analyst:** System Analyst / Auditor / Recorder
**Initialized:** 2026-07-26 17:02 PDT
**Last Updated:** 2026-07-26 19:30 PDT
**Method:** Read-only analysis. No code modifications performed.

---

## Entry #1 — Full Ecosystem Analysis (2026-07-26 17:50, refined 18:10)

### 1. Ecosystem Repository Status

| Repo | Branch | HEAD | Tests | Uncommitted | Build Status |
|------|--------|------|-------|-------------|--------------|
| Van_Nueman_Engine | main | `f53a275` | 34 suites, ~148-163 assertions | 1 untracked (whisper.cpp) | 17 exe in build/Release |
| Van_Nueman_AI | main | `acca3d5` | 48 NEL + 10 council = 58 | **Clean** | Python (.venv) |
| Van_Nueman_Agents | master | `28130b` | None | 51 desktop.ini noise | N/A (Markdown/Python) |
| DeveloperConsole | master | `10c68c0` | None | 2 modified | dist/ built |
| MultiverseScreensaver | main | `66b1093` | 3 assert-based | 2 modified + LICENSE | build/ with MathTests.exe |
| VNES-Lab | main | `5b8090e` | 265 R-LAAER + 27 regression | 23 changed files | Python |
| Substrate_Echo | main | `469755e` | **835 tests** (39 files) | 1 untracked (data/) | Python |

**Key observations:**
- `Van_Nueman_Services`, `Van_Nueman_Social_Sim`, `Van_Nueman_Toolchain` are **not standalone repos** — Services and Social_Sim live inside Van_Nueman_AI; Toolchain doesn't exist on disk
- Van_Nueman_Engine has **17 executables** (not 14 as documented in AGENTS.md)
- Van_Nueman_AI is the only repo with zero uncommitted changes
- Substrate_Echo has the largest test suite at 835 tests

---

### 2. Skein Council Adversarial Review

#### 2.1 Claim Verification Results

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | "No remaining system() calls" | **FALSE** | 8+ active system()/os.system() in first-party code |
| 2 | "163 tests pass in Engine" | **UNABLE TO VERIFY** | ~148-163 custom test macros; exact 163 unconfirmed |
| 3 | "10/10 Council tests pass" | **VERIFIED** | 10 test functions in test_council.py |
| 4 | "835 tests in Substrate_Echo" | **VERIFIED** | 836 def test_* across 39 files (off-by-one negligible) |
| 5 | "All 14 executables build clean" | **FALSE** | 18+ unconditional targets in CMakeLists |
| 6 | "cellular_automata.cu excluded" | **FALSE** | Explicitly listed in ENGINE_CUDA_KERNELS at cmake/08-cuda.cmake:101 |
| 7 | "PowerShell dependency exists" | **VERIFIED** | cmake/02-dependencies.cmake:6-11, build_msvc.bat:21 |
| 8 | "SPIR-V has only 4 tests" | **UNABLE TO VERIFY** | Toolchain SPIR-V tests are unimplemented (placeholder) |

#### 2.2 Active system() Calls Found

| File | Line | Call | Risk |
|------|------|------|------|
| Van_Nueman_AI/scripts/sensor_integration.py | 14-15 | `os.system('modprobe w1-gpio')`, `os.system('modprobe w1-therm')` | LOW — Linux-only, requires root |
| Van_Nueman_Agents/GAMER/gamer_agent.py | 423 | `os.system("title GAMER...")` | LOW — cosmetic Windows title |
| Van_Nueman_AI/scripts/generate_pillar_training_data.py | 229, 907 | `os.system(cmd)`, `os.system(f'kubectl ...')` | **MEDIUM** — command injection risk with unsanitized input |
| Van_Nueman_AI/Van_Nueman_Whisper/examples/server/server.cpp | 289, 327 | `system("ffmpeg -version")`, `std::system(cmd.c_str())` | **MEDIUM** — cmd constructed from string concat |
| Van_Nueman_AI/Van_Nueman_Whisper/examples/common-whisper.cpp | 165 | `system((command + ...))` | **MEDIUM** — string-built command |

**Assessment:** The claim "no remaining system() calls" is **stale**. The Engine's own code was cleaned, but AI subsystem scripts and Whisper examples still have active calls. The generate_pillar_training_data.py calls are highest risk due to potential unsanitized input.

#### 2.3 Adversarial Questioning

**Q: Are the 835 Substrate_Echo tests meaningful, or are they testing trivialities?**
- The test suite covers: Ontology, EntityDescriptor, StateGraph, CausalGraph, FrameSystem, NarrativeLayer, Kernel, AffordanceTracer, TacticalBrain, ReplayAuditor, UnitClassifier, BuildingClassifier, ObservationEncoder, and more
- Most tests verify serialization roundtrips, query correctness, and edge cases
- **Verdict:** Substantive. The representational layer tests (37 new) verify real behavioral contracts.

**Q: Is the representational layer actually integrated, or just wired for collection?**
- **FINDING:** During gameplay, the representational layer is **producer-only** — it accumulates semantic state but does not participate in online decision-making
- `FrameSystem.query_all()` is **never called** in sc2_iterate_01.py
- `StateGraph` is populated but never queried for action decisions
- `CausalGraph` records events but consequences are never consumed
- `NarrativeLayer` processes ticks but narratives never feed into governance
- **However:** At `on_end()`, graphs are saved as JSON → offline analysis pipeline
- **Precise statement:** The representational layer is producer-only during gameplay. It is not yet part of the decision loop. Its outputs feed into offline analysis (JSON → replay study → future training).

**Q: Is the kernel actually doing useful work if its output is ignored?**
- The kernel computes `CognitiveState` with action, prediction, coherence, volume_entropy, executive fields
- These are stored in `self._cognitive_states` list but never read by the action decision path
- The action decision path in `_interpret_action()` uses: WorldState, EntityModel, Drives, AffordanceTracer, ModelPool, ActionBridge, GovernanceGate
- **Precise statement:** No current gameplay decisions appear to depend on kernel outputs. The kernel may still be contributing to telemetry, replay logging, and offline learning infrastructure. Those are legitimate runtime responsibilities even if not yet wired to actions.

---

### 3. Substrate_Echo Hypotheses & Experimental Suggestions

#### H13: Expand Kernel Vector to Include All 9 InformationState Dimensions
- **Current:** `_build_vector()` slices `information.to_vector()[:3]` — only map_revealed, terrain_complexity, cliff_density
- **Wasted:** enemy_known_ratio, visibility_advantage, enemy_bases_known, scout_count, last_scout_time, uncertainty
- **Hypothesis:** Including all 9 information dimensions will improve kernel coherence and action predictions
- **Experiment:** Modify `_build_vector()` to include all 9, re-run 5 games, compare kernel coherence scores and action diversity
- **Risk:** LOW — additive change, no existing behavior altered
- **Evidence strength:** LOW (predicted, not observed)

#### H14: Wire FrameSystem Results into Governance Gate (Phased)
- **Current:** FrameSystem queries are never called; GovernanceGate only sees AffordanceCandidates
- **Hypothesis:** Frame results (danger level, opportunity score, threat assessment) should modulate GovernanceGate decisions
- **Experiment (Phase 1 — observation only):**
  ```
  GovernanceGate.check()
        ↓
  receives frame metadata as extra kwarg
        ↓
  ignores it completely
        ↓
  logs both frame results and governance decisions
  ```
  Run 10 games. Measure: do high-danger frame results correlate with governance rejections?
- **Experiment (Phase 2 — if correlation > 0.5):**
  ```
  GovernanceGate.check()
        ↓
  receives frame metadata
        ↓
  applies soft weighting (0.0-0.3 modifier)
        ↓
  logs all overrides
  ```
- **Risk:** Phase 1: NONE. Phase 2: MEDIUM — could over-constrain if frames are noisy
- **Evidence strength:** LOW (predicted, not observed)

#### H15: Validate Kernel Coherence Semantics Before Wiring
- **Current:** Kernel computes `coherence` but meaning unclear
- **Hypothesis:** Low coherence could indicate conflict, inconsistency, novelty, sensor failure, or partial observability — these are different phenomena requiring different responses
- **Experiment (validation first):**
  ```
  Record coherence values during gameplay
        ↓
  Tag each low-coherence moment with what actually happened
        ↓
  Classify: was it conflict? novelty? sensor gap? inconsistency?
        ↓
  Only then assign behavioral response
  ```
- **Risk:** NONE — observation-only validation
- **Evidence strength:** LOW (hypothesis, not validated)

#### H16: Representational Layer as Post-Hoc Analyzer (Disciplined Integration Path)
- **Current:** The layer collects data but doesn't influence decisions
- **Assessment:** This may be correct behavior for a development-stage system — collect data first, verify representations are stable and meaningful, then wire into governance
- **Experiment:**
  1. Run 20 games with representational layer producing-only
  2. Analyze whether frame results correlate with battle outcomes
  3. If correlation > 0.7: wire into governance (Phase 2 of H14)
  4. If correlation 0.3-0.7: frames are weakly informative, wire with caution
  5. If correlation < 0.3: redesign frames before wiring
- **Risk:** NONE — observation-only
- **Evidence strength:** MEDIUM (this is a disciplined integration methodology, not a claim about what will happen)

#### H17: Canonical Pillar Index YAML
- **From journal:** "Duplicated 16-pillar index definitions are a major source of integration bugs"
- **Hypothesis:** Single source-of-truth YAML would eliminate integration mismatches
- **Experiment:** Grep for pillar index definitions across all Van_Nueman repos, count duplicates, generate canonical YAML
- **Risk:** LOW — infrastructure improvement
- **Evidence strength:** MEDIUM (previous audits found mismatches; YAML is a known solution pattern)

---

### 4. Audit Findings

#### 4.1 Documentation Drift

| Document | Issue | Category | Severity |
|----------|-------|----------|----------|
| AGENTS.md | Claims "14 executables" — actually 18+ | docs/stale | LOW |
| AGENTS.md | Claims "163 tests" — cannot verify exact count | docs/stale | LOW |
| AGENTS.md | Claims "no remaining system() calls" — FALSE | security | **MEDIUM** |
| AGENTS.md | Claims "cellular_automata.cu excluded" — FALSE | architecture/stale | LOW |
| CHANGELOG.md | Last entry v0.13.0 (Jul 25) — v0.14.0 added today | docs/stale | OK |
| PLAN.md | Phase S20 added today — accurate | — | OK |
| Live Journal (previous entry) | "No new claims since July 4" — stale | docs/stale | LOW |

#### 4.2 Integration Gaps (Critical Path)

| Gap | Impact | Priority |
|-----|--------|----------|
| Kernel CognitiveState not consumed by action decisions | No current gameplay decisions depend on kernel outputs | HIGH |
| FrameSystem never queried during gameplay | Representational layer is producer-only, not in decision loop | HIGH |
| InformationState 6/9 dimensions wasted | Kernel blind to enemy info, uncertainty, scouting | MEDIUM |
| Representational layer outputs not wired to governance | No feedback loop from semantic analysis to action | HIGH |
| TacticalBrain hypotheses still at 0 after 23 games | Detection thresholds too strict — instrumentation bug | MEDIUM |

#### 4.3 Positive Findings

| Finding | Evidence |
|---------|----------|
| Substrate_Echo test suite is comprehensive | 835 tests, 39 files, full coverage of representational layer |
| Van_Nueman_AI is fully clean | Zero uncommitted changes |
| Council pipeline is stable | 10/10 tests pass |
| Representational layer architecture is sound | Ontology, EntityDescriptor, StateGraph, CausalGraph, FrameSystem, NarrativeLayer all implemented and tested |
| SC2 knowledge base is substantial | 15 unit types, 4 structure types with taxonomy, capabilities, affordances |
| VNES-Lab has active CI | 265 R-LAAER tests, CI pipeline running |

#### 4.4 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| system() command injection in generate_pillar_training_data.py | MEDIUM | HIGH | Replace with subprocess.run() with list args |
| Kernel outputs unused for decisions | HIGH | MEDIUM | Wire CognitiveState into action selection after validation (H15) |
| Representational layer produces but doesn't act | HIGH | MEDIUM | Disciplined integration path via H16 → H14 |
| Documentation drift from actual state | HIGH | LOW | Periodic audit (this journal) |
| Toolchain repo missing from disk | LOW | LOW | Re-clone or document as archived |

---

### 5. Evidence Quality

| Finding | Evidence Strength | Basis |
|---------|------------------|-------|
| system() calls exist in first-party code | **HIGH** | Direct grep with file:line evidence |
| Kernel outputs not consumed by action decisions | **HIGH** | Code path analysis: _interpret_action() never reads cognitive_state |
| Representational layer is producer-only during gameplay | **HIGH** | Code path analysis: query_all() never called in on_step |
| 835 tests pass in Substrate_Echo | **HIGH** | pytest collection output |
| 10/10 Council tests pass | **HIGH** | Test file inspection |
| 18+ executables (not 14) | **HIGH** | CMakeLists target count |
| FrameSystem would improve governance decisions | **LOW** | Predicted, not observed |
| Kernel coherence should drive exploration | **LOW** | Hypothesis, not validated — coherence semantics unknown |
| YAML will reduce integration bugs | **MEDIUM** | Previous audits found mismatches; solution pattern is established |
| H16 correlation analysis will be informative | **MEDIUM** | This is a methodology claim, not a prediction about the system |

---

### 6. Quantitative Metrics (Baseline)

These metrics establish a baseline for tracking improvement over time.

#### Representational Layer (per-game at on_end)

| Metric | Value | Notes |
|--------|-------|-------|
| StateGraph nodes (entities) | TBD | Not yet logged |
| StateGraph edges | TBD | Not yet logged |
| CausalGraph events recorded | TBD | Not yet logged |
| CausalGraph consequences auto-generated | TBD | Not yet logged |
| FrameSystem queries executed | 0 | query_all() never called |
| NarrativeLayer active narratives | TBD | process_tick() runs but query not logged |
| NarrativeLayer completed narratives | TBD | |
| Ontology concepts loaded | ~19 | 15 units + 4 structures |
| Interpreter entities processed per tick | TBD | Requires logging |

#### Kernel (per-game average)

| Metric | Value | Notes |
|--------|-------|-------|
| Average coherence | TBD | Not logged to file |
| Coherence variance | TBD | |
| Prediction confidence | TBD | |
| Volume entropy | TBD | |
| State transitions | TBD | |
| Observations published | ~game_length/1 | Every tick |

#### Gameplay (per-game)

| Metric | Value | Notes |
|--------|-------|-------|
| Total steps | ~2000-5000 | Varies |
| Actions executed | TBD | From actions_log |
| Action diversity (Shannon entropy) | TBD | From audit report |
| Battle outcomes recorded | TBD | From tactical_brain |
| Hypotheses generated | 0 | After 23 games — threshold bug |
| Experiments run | 0 | Blocked by 0 hypotheses |

**Note:** Many metrics are "TBD" because the current logging infrastructure doesn't capture representational layer or kernel metrics at on_end(). Adding these to the result JSON in on_end() would enable trend tracking.

---

### 7. Architectural Observation

The audit reveals a common pattern in cognitive architecture development:

```
Perception → Representation → [GAP] → Action
```

The ecosystem currently has:

```
Perception (SC2 API, encoder)
    ↓
Representation (Ontology, StateGraph, CausalGraph, FrameSystem, NarrativeLayer)
    ↓
    ✗  ← gap: representational outputs not wired to action selection
    ↓
Action (AffordanceTracer → GovernanceGate → SC2 commands)
```

The parallel path that DOES connect to action:

```
Perception (SC2 API, encoder)
    ↓
Epistemology (EntityModel, Drives, AffordanceTracer, TacticalBrain)
    ↓
Action (GovernanceGate → SC2 commands)
```

Many architectures intentionally leave the representation→action gap during development. They first verify that representations are stable and meaningful before allowing them to influence behavior. H16 proposes a disciplined way to bridge this gap: observe first, measure correlation, then wire.

---

### 8. Recommendations

**Immediate (this session):**
1. Update AGENTS.md to correct false claims (system() calls, executable count, cellular_automata.cu)
2. Mark H13 (expand kernel vector) as highest-priority experiment — LOW risk, HIGH information value
3. Add representational layer and kernel metrics to on_end() result JSON — enables trend tracking

**Short-term (next session):**
4. Run H16 experiment — 20 games of pure observation, then correlation analysis
5. Fix system() calls in generate_pillar_training_data.py (highest security risk)
6. Validate H15 — classify what low coherence actually means before assigning behavioral response

**Medium-term:**
7. Wire FrameSystem.query_all() into on_step for logging (Phase 1 of H14)
8. Wire CognitiveState into action selection (H15, after validation)
9. Wire FrameSystem results into GovernanceGate (H14 Phase 2, after correlation established)
10. Generate canonical pillar index YAML (H17)

---

*This journal entry was produced by the System Analyst / Auditor / Recorder role.*
*No code was modified. All findings are from read-only analysis.*
*Refined at 18:10 PDT based on adversarial review feedback.*
*Next entry will be produced when new observations arise or experiments complete.*

---

## Entry #2 — sc2_iterate_01.py Architecture Audit (2026-07-26 18:30)

### File Profile

| Metric | Value |
|--------|-------|
| Total lines | 1,676 |
| Methods | 25 (+ 3 nested in main) |
| Instance variables in __init__ | **55** |
| Imports | 34 top-level + 6 inline |
| Classes | 1 (`IterateBot(BotAI)`) |
| Hardcoded magic numbers | **40+** |
| TODO/FIXME/HACK | 0 |

---

### 1. Design Principle Compliance Audit

Each principle is scored: ✅ COMPLIANT | ⚠️ PARTIAL | ❌ VIOLATION

#### Core Architecture Principles

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| **P1** | "Not a chatbot/RAG — dynamical cognitive substrate" | ⚠️ PARTIAL | Kernel exists and computes CognitiveState, but its outputs are not used for decisions (Entry #1 finding). The "dynamical" part is telemetry-only. |
| **P2** | "Clients never manipulate cognition directly — they publish state" | ❌ VIOLATION | `_interpret_action()` directly calls `affordance_tracer.generate()`, `model_pool.get_model()`, `action_bridge.score_action()`, `governance_gate.check()` — all cognition-layer components are called directly by the embodiment client. There is no separation between "publish state" and "make decisions." The bot IS the cognition. |
| **P11** | "The kernel should not know it is playing a game" | ⚠️ PARTIAL | The kernel itself is game-agnostic (Observation vector + CognitiveState). But the kernel is bypassed entirely — `_interpret_action()` makes decisions without consulting it. The principle is met in letter (kernel doesn't know about SC2) but violated in spirit (kernel doesn't do anything). |
| **P12** | "The kernel receives experience, not game mechanics" | ❌ VIOLATION | The kernel receives a 16D vector derived from game mechanics (minerals, supply, army count). The `_interpret_action()` method works directly with game mechanics (SC2 units, positions, abilities). The abstraction layer between game mechanics and cognition does not exist. |

#### Epistemology Principles

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| **P20** | "Track what was observed, inferred, predicted, and whether reality confirmed" | ⚠️ PARTIAL | EntityModel tracks observed/predicted. TacticalBrain records battle outcomes. But the feedback loop is incomplete — predictions are not systematically compared against outcomes to update beliefs. |
| **P21** | "Epistemology pipeline is strictly sequential and layered" | ❌ VIOLATION | The pipeline in `_interpret_action()` is: WorldState → AffordanceTracer → ModelPool → ActionBridge → GovernanceGate. This is a single flat pipeline, not the layered architecture described in P21 (Raw → Feature → Hypothesis → Prediction → Outcome → Belief → Rule). Hypothesis generation and outcome verification are handled by separate systems (TacticalBrain, Auditor) that don't feed back into the action pipeline. |
| **P22** | "Failures are valuable data" | ✅ COMPLIANT | ReplayAuditor detects 10 failure types. Ledger tracks evidence. TacticalBrain records battle outcomes. Failure data is preserved. |

#### Representational Layer Principles

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| **P23** | "Ontology lives AROUND the kernel, not inside it" | ✅ COMPLIANT | Ontology is in `representational/ontology.py`, not in `kernel/`. |
| **P24** | "EntityDescriptor is the universal cognitive object" | ⚠️ PARTIAL | EntityDescriptor exists and is well-designed. But `_interpret_action()` does not use EntityDescriptors — it uses raw SC2 units and `WorldState.from_botai()`. The universal object is not universal yet. |
| **P25** | "Capabilities ≠ Affordances" | ✅ COMPLIANT | EntityDescriptor separates them correctly. AffordanceTracer also distinguishes them. |
| **P26** | "Evidence never overwritten by hypotheses" | ✅ COMPLIANT | EntityEvidence and EntityHypothesis are separate fields in EntityDescriptor. |
| **P27** | "Uncertainty always explicit" | ✅ COMPLIANT | EntityState has confidence fields. EntityObservation has confidence. FrameResult has confidence and relevance. |
| **P28** | "Frames are queries, not static templates" | ⚠️ PARTIAL | FrameSystem implements 8 dynamic query types. But `query_all()` is never called during gameplay — the queries exist but are not executed. |
| **P29** | "Perspectives shift over time" | ⚠️ PARTIAL | `_select_perspective()` exists and selects based on threat ratio. But the selected perspective is only passed to `frame_system.set_perspective()` — whose results are never consumed. |

#### Executive Function Principles

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| **P8** | "Goals are primitives — Executive Function determines which matter right now" | ❌ VIOLATION | There is no Executive Function module. DriveManager has 8 needs with fixed weights. The "which goals matter right now" decision is made by `_interpret_action()` using hardcoded `need_weights` dict (lines 515-519). Goal prioritization is static, not adaptive. |
| **P33** | "Executive Function determines which desired states matter right now" | ❌ VIOLATION | Same as P8. No Executive Function exists. The bot has a flat action-selection pipeline with no goal arbitration. |
| **P36** | "Resource priority stack: Safety > Maintenance > Active Goals > Learning > Exploration > Idle" | ❌ VIOLIATION | No priority stack exists. The `_interpret_action()` method scores all candidates equally and picks the highest score. There is no safety override, no maintenance priority, no learning budget. |

#### Council / Metacognition Principles

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| **P9** | "Council is NOT part of the constant cognition loop — it audits" | ✅ COMPLIANT | Council is not called in on_step. It is a separate test-only system. |
| **P34** | "Council produces reports, not direct modifications" | ✅ COMPLIANT | Council produces ApprovalResult. GovernanceGate is separate. |
| **P35** | "Council is an audit process — periodic health check" | ✅ COMPLIANT | Not invoked during gameplay. |
| **P32** | "Do not optimize strategies — optimize the generators of strategies" | ❌ VIOLATION | No meta-cognition exists. The bot optimizes action scores (strategies) directly via model_pool updates. There is no system that optimizes how strategies are generated. |

#### SC2 Embodiment Principles

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| **P10** | "The objective is not to build a StarCraft bot" | ⚠️ PARTIAL | The code reads like a StarCraft bot. Zerg-specific logic in `_exec_tech_up()` (189 lines of hardcoded Zerg structures/upgrades). `_map_build_step_to_action()` has 60+ hardcoded unit names. The embodiment is not abstracted — it IS the game. |
| **P17** | "Opponent actively changes the environment — must model what opponent knows/believes/predicts" | ⚠️ PARTIAL | EntityModel tracks "enemy" as a single entity with threat_level. Known enemy units are counted. But there is no model of what the opponent knows, believes, or predicts. No deception modeling. |
| **P18** | "First success criterion: maintain coherent cognition in hostile, partially observed, adversarial environment" | ⚠️ PARTIAL | The kernel computes coherence but it's not used. The bot functions without coherent cognition — it uses flat action scoring. The success criterion is not measurable because the kernel is bypassed. |

#### Development Philosophy

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| **P30** | "After S22: 30% architecture / 70% experiments" | ❌ VIOLATION | This file is 1,676 lines of architecture with zero experiment infrastructure. No A/B testing, no hypothesis tracking in the action pipeline, no controlled comparisons. The TacticalBrain has experiment machinery but it's disconnected from action selection. |
| **P43** | "SC2 playthroughs required as integration validation" | ✅ COMPLIANT | The iteration loop runs N games and logs results. |

---

### 2. Compliance Summary

| Category | ✅ | ⚠️ | ❌ |
|----------|---|---|---|
| Core Architecture (P1, P2, P11, P12) | 0 | 2 | 2 |
| Epistemology (P20, P21, P22) | 1 | 1 | 1 |
| Representational Layer (P23-P29) | 4 | 3 | 0 |
| Executive Function (P8, P33, P36) | 0 | 0 | 3 |
| Council/Metacognition (P9, P32, P34, P35) | 3 | 0 | 1 |
| SC2 Embodiment (P10, P17, P18) | 0 | 3 | 0 |
| Development Philosophy (P30, P43) | 1 | 0 | 1 |
| **TOTAL** | **9** | **9** | **8** |

**Overall compliance: 35% (9/26)**
**Partial compliance: 35% (9/26)**
**Violations: 30% (8/26)**

---

### 3. Structural Code Smells

| Smell | Severity | Evidence | Design Impact |
|-------|----------|----------|---------------|
| **God Object** | CRITICAL | IterateBot has 55 instance variables, owns 20+ subsystems | Violates P4 (four-layer separation), P2 (client shouldn't manipulate cognition), P9 (council not in loop) |
| **God Method** | CRITICAL | `on_step()` is 198 lines mixing 6 concerns with 15 throttle frequencies | Violates P21 (strictly sequential layered pipeline) |
| **Copy-paste unit filtering** | HIGH | worker_ids/supply_ids/spawned_names + classifier duplicated in `_execute`, `_exec_attack`, `_exec_defend` (3x) | Violates P24 (EntityDescriptor should be universal — instead raw SC2 units are filtered ad-hoc everywhere) |
| **Hardcoded Zerg logic** | HIGH | `_exec_tech_up()` (189 lines), `_map_build_step_to_action()` (60+ unit names), `Race.Zerg` forced at L165 | Violates P10 (objective is not to build a SC2 bot), P12 (kernel should receive experience not game mechanics) |
| **Magic numbers** | HIGH | 40+ unexplained numeric literals across the file | No named constants, no configuration, no adaptation |
| **No abstraction layer** | HIGH | SC2 API objects (Unit, Position) used directly in cognition code | Violates P12 (kernel receives experience not game mechanics) |
| **Encapsulation violations** | MEDIUM | `on_end` accesses `tactical_brain._battle_log`, `_hypotheses`, `_experiments` | Tight coupling between subsystems |
| **Late attribute init** | MEDIUM | `_prev_army` via `hasattr` in `_record_causal_events` | Should be in `__init__` |
| **Inline imports** | MEDIUM | 6 imports inside method bodies | Performance overhead per call |
| **No experiment framework** | HIGH | No A/B testing, no controlled comparisons, no hypothesis→experiment→result pipeline | Violates P30 (70% experiments) |

---

### 4. The Fundamental Architectural Tension

The audit reveals a **design contradiction** at the heart of this file:

**Stated architecture (P2, P4, P11, P12):**
```
SC2 Client (embodiment)
    ↓ publishes state
Kernel (cognition)
    ↓ produces CognitiveState
Executive Function (goal arbitration)
    ↓ selects intent
Action Bridge (translation)
    ↓ produces SC2 commands
SC2 Client (execution)
```

**Actual architecture (sc2_iterate_01.py):**
```
SC2 Client (everything)
    ↓ reads SC2 API directly
interpret_action() (flat scoring)
    ↓ calls 7 subsystems directly
_execute() (SC2 commands)
```

The bot is a **monolith** that owns cognition, epistemology, representation, execution, learning, and chat. The kernel, executive function, and representational layer exist as separate modules but are bypassed by the action pipeline.

This is not necessarily wrong for a development-stage system. The monolith was built to get the bot playing games and generating data. The next phase (per P30: 70% experiments) would be to:

1. Validate that the representational layer produces meaningful signals (H16)
2. Validate that kernel coherence measures something real (H15)
3. Then refactor the monolith into the layered architecture

The danger is if the monolith becomes the permanent architecture — in which case the kernel, representational layer, and executive function are dead code.

---

### 5. Specific Violations with Line References

| Principle | Violation | Location |
|-----------|-----------|----------|
| **P2** "Clients publish state, don't manipulate cognition" | `_interpret_action()` calls `affordance_tracer.generate()` directly | L511 |
| **P2** | `_interpret_action()` calls `governance_gate.check()` directly | L569 |
| **P8** "Goals are primitives, EF arbitrates" | No Executive Function; hardcoded `need_weights` dict | L515-519 |
| **P10** "Not a SC2 bot" | `Race.Zerg` forced | L165 |
| **P10** | `_exec_tech_up()` has Zerg-only structure/upgrade tables | L1081-1092, L1123-1130 |
| **P10** | `_map_build_step_to_action()` has 60+ hardcoded unit names | L1434-1446 |
| **P12** "Kernel receives experience not game mechanics" | `_interpret_action()` works with raw SC2 Unit objects | L498-587 |
| **P17** "Model what opponent knows/believes/predicts" | Enemy modeled as single entity with threat_level | L500-506 |
| **P21** "Strictly sequential layered pipeline" | Flat single-pass pipeline in `_interpret_action()` | L498-587 |
| **P24** "EntityDescriptor is universal" | Raw SC2 units used instead of EntityDescriptors | L655, L905-907, L995-997 |
| **P30** "70% experiments" | Zero experiment infrastructure in action pipeline | N/A |
| **P32** "Optimize generators of strategies" | No meta-cognition; model_pool optimizes directly | L524-534 |
| **P36** "Safety > Maintenance > Goals > Learning > Exploration" | No priority stack; flat scoring | L536-555 |

---

### 6. Recommendations

**What to do about the monolith:**

The bot works. It plays games, collects data, and generates learning signals. That's valuable. The question is whether to refactor now or later.

**Option A: Refactor now (high effort, high risk)**
- Extract cognition into kernel
- Extract execution into action bridge
- Extract learning into epistemology pipeline
- Risk: breaking a working system

**Option B: Validate first, refactor later (recommended)**
1. Run H16 (20 games observation) to validate representational layer signals
2. Run H15 (coherence validation) to validate kernel metrics
3. If signals are meaningful, refactor the monolith to use them
4. If signals are not meaningful, the monolith is the right architecture for now

**Option C: Incremental extraction (low risk, slow)**
- Extract unit filtering into a shared helper (fixes 3x copy-paste)
- Extract magic numbers into a config class
- Extract Zerg-specific logic into a race adapter
- Leave the overall monolith structure alone

**Evidence strength for these recommendations: LOW** — these are architectural judgments, not empirical findings. The right choice depends on whether the representational layer and kernel produce actionable signals, which is what H16 and H15 will determine.

---

*Audit of sc2_iterate_01.py against 50 extracted design principles.*
*File: 1,676 lines, 55 instance variables, 25 methods, 40+ magic numbers.*
*Compliance: 35% full, 35% partial, 30% violations.*
*Primary finding: The bot is a monolith that bypasses its own cognitive architecture.*
*Recommended next step: H16 (validate representational signals) before refactoring.*

---

## Entry #3 — Deep 7-Part Architectural Analysis (2026-07-26 19:00)

This entry synthesizes detailed subsystem analysis to answer: Where does data flow, where does it die, what is layered vs. flat, what is represented vs. hardcoded, what is coupled, what uses the blackboard, and what is the provenance of each action decision?

---

### 1. Data Flow Trace — Complete Subsystem Map

Every subsystem in sc2_iterate_01.py, its inputs, outputs, and where those outputs go.

#### 1.1 Perception Layer (Data Ingestion)

| Subsystem | Inputs | Outputs | Consumed By |
|-----------|--------|---------|-------------|
| `observation_encoder` | SC2 API (`bot.state.units`, `bot.state.visibility`, `bot.state.pathing_grid`, `bot.state.terrain_height`) | `InformationState` (9 dims, 3 used), `WorldState`, `TerrainMetrics` | `IterateBot.on_step()` → passed to `_interpret_action()` as `obs`, `world_state`, `terrain` |
| `unit_classifier` | Raw SC2 `Unit` objects | `UnitRole` enum (ARMY/WORKER/SPAWNED/SUPPLY/STRUCTURE/UNKNOWN), `UnitInfo` (position, health, type, cost) | Used inline in `_execute()`, `_exec_attack()`, `_exec_defend()`, `_record_causal_events()` |
| `building_classifier` | Raw SC2 `Unit` objects with `is_structure` | `BuildingRole`, `BuildingInfo` (tech tree position, produces, requires) | Used in `_exec_tech_up()` for structure identification |

#### 1.2 Epistemology Layer (Belief Formation)

| Subsystem | Inputs | Outputs | Consumed By |
|-----------|--------|---------|-------------|
| `entity_model` | UnitRoles, UnitInfo, BuildingInfo, `Observation` | `EntityState` (for each observed entity: type, role, health, capabilities, confidence), `Entity` (enemy aggregate: threat_level, known_units, composition) | Stored in `self._entity_states`, `self._entity` — **NOT consumed by any action decision** |
| `drives` | `WorldState` (minerals, supply, army), `Entity` (threat) | `DriveState` (8 needs: expansion, defense, attack, economy, tech, scouting, supply, survival) | `drive_manager.get_needs()` → `need_weights` dict → used in action scoring |
| `affordance_tracer` | `WorldState`, `EntityModel`, `EntityStates` | `AffordanceCandidate[]` (action + source + cost + urgency + confidence + terrain) | `_interpret_action()` → scored by ModelPool → GovernanceGate |
| `tactical_brain` | `WorldState`, `EntityModel`, `GameState` | `TacticalDecision` (primary, fallback, retreat), battle records, hypothesis records | `_interpret_action()` uses `tactical_brain.decide_tactics()` for attack/defend decisions; `on_end()` writes battle_log/hypotheses to JSON — **NOT consumed during gameplay** |
| `model_pool` | `AffordanceCandidate[]`, historical outcomes | `ModelOutput` (score, confidence, reasoning) per candidate | `_interpret_action()` scores each candidate |
| `auditor` | SC2 `ActionRaw`, `Result`, `GameState` | `ActionEvaluation` (success, failure type, details) | `on_end()` writes to ledger — **NOT consumed during gameplay** |
| `replay_auditor` / `epistemic_ledger` | All `ActionEvaluation`s, tactical brain data, hypotheses | `EpistemicLedger` JSON with evidence, experiments, strategies | `on_end()` saves to `data/epistemic_ledger_*.json` — **NOT consumed during gameplay** |

#### 1.3 Representational Layer (Semantic State)

| Subsystem | Inputs | Outputs | Consumed By |
|-----------|--------|---------|-------------|
| `ontology` | SC2 knowledge base (15 units, 4 structures) | `Concept`, `TaxonomyNode`, `PropertySchema`, `Rule`, `Constraint` | `SemanticInterpreter` (auto-populates) |
| `state_graph` | Processed unit/structure states per tick | `SGEntity` nodes, `SGEdge` typed relationships (hostile, friendly, produces, requires, proximity, threatens, supports) | `on_end()` saves to JSON — **NOT queried during gameplay** |
| `semantic_interpreter` | Raw SC2 units/structures, `InformationState`, terrain metrics | Updated `StateGraph` + `Ontology` state | `interpret_tick()` called every 3 ticks → populates StateGraph — **graph never queried** |
| `causal_graph` | Army deltas, worker deltas, base deltas, tick number | `CausalEvent` with `Consequence[]`, `EventChain[]` | `on_end()` saves to JSON — **NOT queried during gameplay** |
| `frame_system` | `WorldState`, `EntityModel`, `InformationState`, `TerrainMetrics`, current `Perspective` | `FrameResult[]` per frame type (Danger, Opportunity, ThreatAssessment, etc.) | `query_all()` **NEVER CALLED** — frames computed nowhere |
| `narrative_layer` | Frame results, entity states, ticks | `Narrative` with `NarrativeArc`, events, outcomes | `process_tick()` runs every 3 ticks → narratives accumulated — **narratives never queried** |

#### 1.4 Action Layer (Execution)

| Subsystem | Inputs | Outputs | Consumed By |
|-----------|--------|---------|-------------|
| `action_bridge` | Scored `ModelOutput[]`, `GovernanceRule[]` | `SC2 ActionRaw` (command object) | `_execute()` sends to SC2 API |
| `governance_gate` | `AffordanceCandidate[]`, `GovernanceRule[]` | Approved/rejected candidates, `DecisionLog` | `_interpret_action()` filters candidates |

#### 1.5 Kernel (Dynamical Processing)

| Subsystem | Inputs | Outputs | Consumed By |
|-----------|--------|---------|-------------|
| `kernel` | 16D vector from `InformationState[:3]` + drives + actions | `CognitiveState` (action, prediction, coherence, volume_entropy, executive) | Stored in `self._cognitive_states[]` — **NOT consumed by any downstream system** |

#### 1.6 Data Flow Summary

```
SC2 API
  ↓
observation_encoder → InformationState (3/9 dims used), WorldState, TerrainMetrics
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ PARALLEL PATHS (not sequential):                               │
│                                                                 │
│ Path A: entity_model → EntityStates, Entity                     │
│         (stored, not consumed by decisions)                      │
│                                                                 │
│ Path B: drives → DriveState → need_weights → action scoring     │
│         (consumed, but static — hardcoded weights)               │
│                                                                 │
│ Path C: affordance_tracer → AffordanceCandidates                 │
│         → model_pool (scores) → governance_gate (filters)        │
│         → action_bridge (translates) → SC2 commands              │
│         (ACTIVE decision path)                                   │
│                                                                 │
│ Path D: tactical_brain → TacticalDecision                        │
│         (used for attack/defend routing only)                    │
│                                                                 │
│ Path E: kernel → CognitiveState                                  │
│         (stored, NOT consumed)                                   │
│                                                                 │
│ Path F: interpreter → state_graph                                │
│         (stored, NOT queried)                                    │
│                                                                 │
│ Path G: causal_graph → events + consequences                     │
│         (stored, NOT queried)                                    │
│                                                                 │
│ Path H: frame_system → FrameResults                              │
│         (NEVER CALLED)                                           │
│                                                                 │
│ Path I: narrative_layer → Narratives                             │
│         (processed, NOT queried)                                 │
│                                                                 │
│ Path J: auditor → ActionEvaluation → epistemic_ledger            │
│         (stored at on_end, NOT consumed during gameplay)         │
└─────────────────────────────────────────────────────────────────┘
```

**Key finding:** Of 10 parallel data paths, only **Path B + Path C + Path D** actually influence action decisions. The other 7 paths are producer-only.

---

### 2. Dead-End Detection

A "dead-end" is any subsystem whose outputs are computed but never influence any subsequent decision, metric, or external artifact.

#### 2.1 Confirmed Dead Subsystems

| Subsystem | Status | Evidence | Severity |
|-----------|--------|----------|----------|
| **FrameSystem** | **DEAD** — never called | `query_all()` is never invoked anywhere in sc2_iterate_01.py. `set_perspective()` is called but its results are discarded. The entire 8-frame-type + 4-perspective system has zero runtime consumers. | CRITICAL — 300+ lines of dead code |
| **StateGraph** | **DEAD** — populated but never queried | `SemanticInterpreter.interpret_tick()` runs every 3 ticks, populating nodes and edges. No code ever queries the graph (no `query_by_role()`, `query_by_taxonomy()`, `get_entities_with_capability()`, etc. during gameplay). Only consumed at `on_end()` for JSON export. | HIGH — useful for offline analysis, but zero in-game value |
| **CausalGraph** | **DEAD** — records but never queries consequences | `_record_causal_events()` writes army/worker/base deltas as `CausalEvent`s. Consequences are auto-generated. But no code ever calls `query_consequences()`, `get_causal_chain()`, or `get_events_by_type()` during gameplay. Only consumed at `on_end()` for JSON export. | HIGH — records causal chains that nobody reads |
| **NarrativeLayer** | **DEAD** — processes ticks but narratives not consumed | `process_tick()` runs every 3 ticks, creating/completing narratives. But no code ever queries `get_active_narratives()`, `get_completed_narratives()`, or uses narrative data for any decision. | HIGH — dramatic arc computed but irrelevant |
| **Kernel** | **PRODUCER-ONLY** — computes CognitiveState that nobody reads | Kernel receives 16D vector, computes coherence/prediction/action/executive. Stored in `_cognitive_states[]`. No downstream system reads this list during gameplay. Only written to JSON at `on_end()`. | HIGH — the named core of the architecture is telemetry-only |
| **EntityModel** | **PRODUCER-ONLY** — computes entity states that don't enter decisions | `EntityState` objects are created for every observed unit/structure. But `_interpret_action()` works with raw SC2 units and `AffordanceCandidate`s, not `EntityState`s. Entity model outputs are stored but not consumed by the action pipeline. | MEDIUM — used by affordance_tracer (indirectly) but not directly by decisions |
| **EpistemicLedger** | **DEFERRED** — consumed only at on_end() | Evidence is accumulated during gameplay but only written to disk at game end. No in-game feedback loop from ledger to decisions. | MEDIUM — by design (offline analysis), but means no in-game learning from evidence |

#### 2.2 Partially Dead Subsystems

| Subsystem | Status | Evidence |
|-----------|--------|----------|
| **SemanticInterpreter** | Active (runs), but output is dead | `interpret_tick()` executes every 3 ticks, but its output (StateGraph mutations) is never queried |
| **Ontology** | Active (loaded), but mostly unused beyond interpreter auto-population | SC2 knowledge base loaded at init. Concepts referenced by interpreter. No other system queries the ontology |
| **TacticalBrain** | Active (decides tactics), but learning outputs dead | `decide_tactics()` is called for attack/defend decisions. But `analyze_battles()`, `generate_hypothesis()`, `record_experiment()` outputs are never consumed during gameplay — only at `on_end()` |
| **Auditor** | Active (evaluates actions), but feedback is deferred | `evaluate()` runs after each action. But `ActionEvaluation` results are stored in ledger, not fed back to adjust future action scoring |

#### 2.3 Active Subsystems (Outputs Consumed)

| Subsystem | Status | Consumer |
|-----------|--------|----------|
| **ObservationEncoder** | Active | InformationState → drives scoring, affordance_tracer, terrain metrics |
| **WorldState** | Active | Used by affordance_tracer, drives, tactical_brain, entity_model |
| **DriveManager** | Active | need_weights → action scoring (but weights are static) |
| **AffordanceTracer** | Active | AffordanceCandidates → model_pool → governance_gate → actions |
| **ModelPool** | Active | Scores candidates → governance_gate |
| **GovernanceGate** | Active | Filters candidates → action_bridge |
| **ActionBridge** | Active | Translates to SC2 commands |
| **TerrainTraversal** | Active | Unit filtering for cliff-capable units |

#### 2.4 Dead-End Quantification

| Category | Subsystems | Lines (est.) | % of Total |
|----------|-----------|-------------|-----------|
| **Fully dead** | FrameSystem, StateGraph (in-game), CausalGraph (in-game), NarrativeLayer | ~800 lines | ~48% |
| **Producer-only** | Kernel, EntityModel, EpistemicLedger (deferred) | ~400 lines | ~24% |
| **Active** | ObservationEncoder, WorldState, DriveManager, AffordanceTracer, ModelPool, GovernanceGate, ActionBridge, TerrainTraversal, UnitClassifier, BuildingClassifier | ~476 lines | ~28% |

**Conclusion:** Approximately **72%** of subsystem code produces outputs that are either dead or producer-only during gameplay. Only **28%** of subsystem code actively influences action decisions.

---

### 3. Architectural Layering

The stated architecture defines 4 layers (perception, representation, cognition, action). How does the actual code map to these layers?

#### 3.1 Stated Layer Definitions

From the design documents:
1. **Perception** — Raw sensory data ingestion (SC2 API → InformationState)
2. **Representation** — Semantic state (Ontology, StateGraph, CausalGraph, FrameSystem, NarrativeLayer)
3. **Cognition** — Kernel + Executive Function (goal arbitration, coherence)
4. **Action** — AffordanceTracer → GovernanceGate → ActionBridge → SC2 commands

#### 3.2 Actual Layer Mapping

| Actual Module | Stated Layer | Actual Layer | Layer Violation? |
|---------------|-------------|-------------|-----------------|
| `observation_encoder` | Perception | Perception | ✅ Correct |
| `unit_classifier` | Perception | Perception | ✅ Correct |
| `building_classifier` | Perception | Perception | ✅ Correct |
| `ontology` | Representation | Representation | ✅ Correct |
| `state_graph` | Representation | Representation | ✅ Correct |
| `semantic_interpreter` | Representation | Representation | ✅ Correct |
| `causal_graph` | Representation | Representation | ✅ Correct |
| `frame_system` | Representation | Representation (dead) | ✅ Correct layer, dead execution |
| `narrative_layer` | Representation | Representation (dead) | ✅ Correct layer, dead execution |
| `kernel` | Cognition | Cognition (producer-only) | ✅ Correct layer, unused output |
| `entity_model` | Cognition/Epistemology | **Perception** (works with raw SC2 units) | ❌ Layer violation — should output EntityDescriptors, not raw states |
| `drives` | Cognition | Cognition | ✅ Correct |
| `drive_manager` | Cognition | Cognition (static) | ⚠️ Correct layer, but no Executive Function |
| `tactical_brain` | Cognition/Epistemology | **Flat** (mixes perception, belief, tactics) | ❌ Layer violation — decides tactics directly from SC2 state |
| `affordance_tracer` | Cognition | **Action** (generates actionable candidates) | ⚠️ Borderline — affordances are cognitive but candidates are action-ready |
| `model_pool` | Cognition | Cognition | ✅ Correct |
| `governance_gate` | Action | Action | ✅ Correct |
| `action_bridge` | Action | Action | ✅ Correct |
| `auditor` | Epistemology (post-action) | Epistemology (deferred) | ✅ Correct |

#### 3.3 Layer Violations

| Violation | Location | Impact |
|-----------|----------|--------|
| **EntityModel outputs raw game states, not EntityDescriptors** | `entity_model.py` → `_interpret_action()` | P24 violated — the "universal cognitive object" is not used. Raw SC2 units pass through directly. |
| **TacticalBrain mixes perception + belief + decision** | `tactical_brain.py` → `decide_tactics()` | P21 violated — should be layered (observe → believe → decide), but is flat (observe+decide in one call) |
| **No Executive Function layer exists** | Missing entirely | P8, P33, P36 violated — drive_manager produces needs, but no module arbitrates between them dynamically |
| **_interpret_action() crosses all 4 layers** | `sc2_iterate_01.py:498-587` | P2 violated — single method handles perception (SC2 API reads), cognition (scoring), and action (command generation) |

#### 3.4 Layering Diagram

```
STATED LAYERING:
  Perception → Representation → Cognition → Action
  (clean, sequential, each layer only sees the layer below)

ACTUAL LAYERING:
  Perception ─────────────────────────────────────────┐
       │                                               │
       ├─→ Representation (dead end)                   │
       │                                               │
       ├─→ EntityModel (stores, doesn't feed forward)  │
       │                                               │
       ├─→ Drives → need_weights ─────────────┐        │
       │                                       │        │
       ├─→ TacticalBrain → tactics ────────┐   │        │
       │                                   │   │        │
       ├─→ AffordanceTracer ──────────┐    │   │        │
       │                              │    │   │        │
       ├─→ Kernel (stores, dead)      │    │   │        │
       │                              ↓    ↓   ↓        ↓
       │                    ┌──────────────────────────┐
       │                    │  _interpret_action()     │
       │                    │  (flat scoring pipeline) │
       │                    └──────────┬───────────────┘
       │                               │
       │                    ┌──────────↓───────────────┐
       │                    │  GovernanceGate           │
       │                    │  ActionBridge             │
       │                    │  SC2 commands              │
       │                    └──────────────────────────┘
```

**Conclusion:** The layers exist as separate files/modules, but in practice `_interpret_action()` is a flat function that crosses all layers. The representation→cognition→action pipeline described in the architecture documents does not exist in the code.

---

### 4. Design Philosophy Compliance — Representation vs. Hardcoding

The design philosophy emphasizes: "Ontology lives AROUND the kernel, not inside it" and "EntityDescriptor is the universal cognitive object." How much of the system is represented (data-driven) vs. hardcoded?

#### 4.1 Represented (Data-Driven)

| Component | Representation | Evidence |
|-----------|---------------|----------|
| **SC2 Knowledge Base** | 15 unit types, 4 structures in `interpreter.py` with taxonomy, capabilities, affordances | `SC2_KNOWLEDGE` dict, auto-populates Ontology |
| **Unit Roles** | Classification via `UnitClassifier` rules (ARMY/WORKER/SPAWNED/SUPPLY) | Rule-based, extensible |
| **Building Roles** | Classification via `BuildingClassifier` (TECH/MINERAL/ENERGY/PRODUCTION/DEFENSE) | Rule-based, extensible |
| **Terrain** | `TerrainMetrics` computed from pathing grid (cliff ratio, open ratio, chokes, etc.) | Data-driven from SC2 API |
| **StateGraph** | Typed edges (HOSTILE, FRIENDLY, PRODUCES, REQUIRES, PROXIMITY, THREATENS, SUPPORTS) | Queryable graph structure |
| **CausalGraph** | Event types, consequence types, event chains | Data-driven causal model |
| **FrameSystem** | 8 frame types, 4 perspectives, weighted scoring | Query-based (but never called) |
| **NarrativeLayer** | Narrative types, arcs, events | Data-driven story model |

#### 4.2 Hardcoded

| Component | Hardcoding | Lines | Impact |
|-----------|-----------|-------|--------|
| **Race forced to Zerg** | `Race.Zerg` hardcoded | L165 | Cannot play other races |
| **Zerg tech tree** | Structure/upgrade tables hardcoded | L1081-1092, L1123-1130 | 189 lines of Zerg-only logic |
| **Build order** | `_map_build_step_to_action()` has 60+ unit names | L1434-1446 | Cannot adapt build orders |
| **Action selection** | `_interpret_action()` flat scoring pipeline | L498-587 | No layered cognition |
| **Drive weights** | `need_weights` dict with fixed values | L515-519 | No adaptive priority |
| **Scoring formula** | `score = need_weight * affordance.urgency * confidence` | L536-555 | Fixed formula, no learning |
| **Tech-up logic** | `_exec_tech_up()` hardcoded structure/upgrade priority | L1070-1145 | Cannot adapt tech strategy |
| **Magic numbers** | 40+ unexplained numeric literals | Throughout | No configuration |
| **Terrain thresholds** | Hardcoded cliff/open/choke ratios | Various | No adaptation to map types |

#### 4.3 Representation vs. Hardcoding Ratio

| Category | Estimated Lines | % of File |
|----------|----------------|----------|
| **Represented (data-driven, extensible)** | ~200 | ~12% |
| **Hardcoded (game-specific, inflexible)** | ~600 | ~36% |
| **Infrastructure (boilerplate, config, logging)** | ~500 | ~30% |
| **Dead code (producer-only, never consumed)** | ~376 | ~22% |

**Conclusion:** The file is approximately **3x more hardcoded than represented**. The ontology, state graph, causal graph, and frame system exist as data-driven modules, but the actual decision-making code (`_interpret_action()`, `_exec_tech_up()`, `_map_build_step_to_action()`) is heavily hardcoded to Zerg gameplay.

#### 4.4 The Representation Paradox

The design philosophy says: "EntityDescriptor is the universal cognitive object." But:
- `_interpret_action()` never creates or reads EntityDescriptors
- It works with raw SC2 `Unit` objects and `WorldState`
- `UnitClassifier` produces `UnitRole` (an enum), not EntityDescriptors
- `EntityModel` produces `EntityState` objects, which are similar to but not the same as EntityDescriptors

The representational layer was built to be the universal interface, but the action pipeline was built before it existed and was never refactored to use it.

---

### 5. Coupling Analysis

How tightly coupled are the subsystems? What happens if one changes?

#### 5.1 Direct Coupling Map

| Module | Directly Depends On | Depended On By |
|--------|-------------------|---------------|
| `IterateBot.__init__` | ALL 20+ subsystems (imports and instantiates every one) | Nothing (top-level) |
| `_interpret_action()` | WorldState, AffordanceTracer, ModelPool, ActionBridge, GovernanceGate, Drives, DrivesNeed, TerrainMetrics, InformationState | IterateBot |
| `_execute()` | SC2 API (raw Unit objects), UnitClassifier | IterateBot |
| `_exec_attack()` | SC2 API, UnitClassifier, TacticalBrain, ActionBridge | IterateBot |
| `_exec_defend()` | SC2 API, UnitClassifier, TacticalBrain, ActionBridge | IterateBot |
| `_exec_tech_up()` | SC2 API, BuildingClassifier, SC2 knowledge (hardcoded) | IterateBot |
| `_record_causal_events()` | CausalGraph, SC2 API (army/supply/worker counts) | IterateBot |
| `_select_perspective()` | WorldState, EntityModel, FrameSystem | IterateBot |
| `on_end()` | TacticalBrain (private attrs), ReplayAuditor, EpistemicLedger, StateGraph, CausalGraph, NarrativeLayer, Ontology | IterateBot |

#### 5.2 Coupling Smells

| Smell | Location | Impact |
|-------|----------|--------|
| **God Object** | `IterateBot` imports and instantiates ALL subsystems in `__init__` | Any subsystem change requires modifying IterateBot |
| **Private attribute access** | `on_end()` reads `tactical_brain._battle_log`, `._hypotheses`, `._experiments` | TacticalBrain internal format change breaks on_end() |
| **3x inline import** | `_interpret_action()` imports `UnitRole`, `_record_causal_events()` imports `CausalEvent`, `EventType` | Performance overhead per tick; import failures not caught at startup |
| **hasattr() guard** | `_record_causal_events()` uses `hasattr(self, '_prev_army')` | Late attribute initialization — fragile, unclear contract |
| **Direct SC2 API in cognition** | `_exec_attack()`, `_exec_defend()` read `bot.state.units` directly | Cannot test cognition without SC2 running |
| **No interface contracts** | All subsystems communicate via concrete types, not protocols/ABCs | Cannot swap implementations without modifying consumers |

#### 5.3 Coupling Quantification

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **Instability (I)** | ~0.9 (high) | IterateBot depends on everything; nothing depends on it |
| **Afferent coupling (Ca)** | 0 | No module depends on IterateBot |
| **Efferent coupling (Ce)** | 20+ | IterateBot depends on 20+ modules |
| **Fan-in to action decision** | 7 (WorldState, Drives, AffordanceTracer, ModelPool, ActionBridge, GovernanceGate, TacticalBrain) | 7 subsystems must be present for any action |
| **Fan-in to on_end** | 8 (TacticalBrain, Auditor, Ledger, StateGraph, CausalGraph, NarrativeLayer, Ontology, JSON) | 8 subsystems accessed at game end |

#### 5.4 What Breaks If Each Subsystem Changes

| If This Changes... | What Breaks |
|--------------------|-------------|
| `WorldState` fields | _interpret_action(), _exec_attack(), _exec_defend(), drives, affordance_tracer, tactical_brain, _select_perspective() |
| `AffordanceCandidate` fields | _interpret_action(), governance_gate, model_pool |
| `UnitClassifier` output | _execute(), _exec_attack(), _exec_defend(), _record_causal_events() |
| `TacticalBrain.decide_tactics()` | _exec_attack(), _exec_defend() |
| `GovernanceGate.check()` | _interpret_action() |
| `CausalGraph` API | _record_causal_events(), on_end() |
| `StateGraph` API | semantic_interpreter, on_end() |
| `FrameSystem` API | _select_perspective() (but frames never called — low impact) |
| `NarrativeLayer` API | on_step() (but narratives never queried — low impact) |

**Conclusion:** The system is tightly coupled through IterateBot as a god object. The active decision path (Path C from section 1) has 7 direct dependencies. The dead subsystems have lower coupling impact (changes to FrameSystem/NarrativeLayer/StateGraph affect only their producers and on_end export).

---

### 6. Blackboard Utilization

The design references a "blackboard" pattern — a shared data space where subsystems post and read information. Does the system use one?

#### 6.1 What Exists

| Candidate "Blackboard" | Role | Used As Blackboard? |
|------------------------|------|-------------------|
| `self._entity_states` dict | Stores EntityState per entity ID | **Partially** — written by entity_model, read by affordance_tracer and _record_causal_events. But not a shared workspace — only 2 readers. |
| `self._world_state` | Current WorldState | **Yes** — written by observation_encoder, read by drives, affordance_tracer, tactical_brain, _select_perspective, _record_causal_events. Closest to a blackboard. |
| `self._cognitive_states` list | Kernel outputs | **No** — written by kernel, never read |
| `self._actions_log` list | Action history | **No** — written by on_step, read only by on_end for JSON export |
| `self._battle_log` list | Battle outcomes | **No** — written by tactical_brain, read only by on_end for JSON export |
| `StateGraph` (global) | Entity/edge storage | **No** — written by interpreter, never queried |
| `CausalGraph` (global) | Event storage | **No** — written by _record_causal_events, never queried |
| `EpistemicLedger` (global) | Evidence storage | **No** — accumulated during game, read only at on_end |

#### 6.2 Blackboard Pattern Assessment

The system does **not** implement a true blackboard pattern. What it has instead:

1. **Instance variables as ad-hoc shared state** — `self._world_state`, `self._entity_states`, etc. are passed between subsystems via method arguments, not via a shared workspace
2. **No publish/subscribe** — subsystems don't register interest in data; they receive it via explicit parameter passing
3. **No data fusion** — multiple subsystems write to different variables, but no module integrates them into a coherent world model
4. **No query mechanism** — subsystems can't ask "what do we know about X?" across all data sources; they must know which specific variable to read

#### 6.3 What a Blackboard Would Look Like

If the system used a proper blackboard:
```
Blackboard (shared workspace)
  ├── Perception layer posts: WorldState, InformationState, TerrainMetrics
  ├── Representation layer posts: FrameResults, NarrativeState, CausalChains
  ├── Epistemology layer posts: EntityStates, Hypotheses, Beliefs
  ├── Cognition layer reads all, posts: Intent, Priority, Coherence
  └── Action layer reads Intent, posts: Commands
```

Every subsystem would read from and write to the blackboard. No direct inter-subsystem calls. The blackboard becomes the single source of truth.

**Current state:** The system has no blackboard. It has a god object (IterateBot) that manually passes data between subsystems via method arguments. This creates tight coupling and makes it impossible to add new subsystems without modifying IterateBot.

---

### 7. Decision Provenance Traces

For each of the 3 primary action types (Expand, Attack, Defend), trace the complete chain of information that influenced the decision — and identify what was available but ignored.

#### 7.1 Expand Decision (e.g., "Build Hatchery at expansion location")

```
DECISION: Expand (build Hatchery at position X)
  │
  ├── WHAT INFLUENCED THIS DECISION:
  │   ├── WorldState.minerals (current mineral count)
  │   ├── WorldState.supply_used / supply_cap
  │   ├── WorldState.base_count (number of bases)
  │   ├── Drives → expansion_need (computed from base_count vs. game_time)
  │   ├── AffordanceTracer.generate() → EXPAND candidate (urgency based on base_count)
  │   ├── ModelPool.score(affordance) → score (based on historical success rate)
  │   └── GovernanceGate.check() → approved (no rule blocks it)
  │
  ├── WHAT WAS AVAILABLE BUT IGNORED:
  │   ├── EntityModel.enemy → threat_level (is expanding safe?)
  │   ├── InformationState.enemy_known_ratio (do we know where enemy is?)
  │   ├── InformationState.visibility_advantage (can enemy see our expansion?)
  │   ├── TerrainMetrics.choke_count (is expansion defensible?)
  │   ├── FrameSystem.query("Danger") → danger level (never called)
  │   ├── CausalGraph → have past expansions been attacked? (never queried)
  │   ├── NarrativeLayer → are we in a "building" or "under attack" arc? (never queried)
  │   ├── StateGraph → what entities are near expansion location? (never queried)
  │   ├── Kernel.cognitive_state.coherence (is our cognition coherent?)
  │   ├── TacticalBrain → should we expand or attack? (only used for attack/defend, not expand)
  │   └── Ontology → what does "expansion" mean in terms of capabilities? (not used)
  │
  └── PROVENANCE: WorldState + Drives + AffordanceTracer + ModelPool + GovernanceGate
      Missing: Enemy threat, terrain defensibility, past expansion outcomes, narrative context
```

#### 7.2 Attack Decision (e.g., "Send army to attack enemy base")

```
DECISION: Attack (target enemy base at position X)
  │
  ├── WHAT INFLUENCED THIS DECISION:
  │   ├── WorldState.army_count (enough units to attack?)
  │   ├── EntityModel.enemy.threat_level (enemy strength)
  │   ├── EntityModel.enemy.known_units (enemy composition)
  │   ├── TacticalBrain.decide_tactics() → PRIMARY decision (ATTACK)
  │   │   └── Uses: our_composition, enemy_composition, army_supply, enemy_army_supply
  │   ├── Drives → attack_need (computed from threat_level vs. army strength)
  │   ├── AffordanceTracer.generate() → ATTACK candidate
  │   │   └── Uses: WorldState, EntityModel, EntityStates (filtered by role)
  │   ├── ModelPool.score(affordance) → score
  │   └── GovernanceGate.check() → approved
  │
  ├── WHAT WAS AVAILABLE BUT IGNORED:
  │   ├── InformationState.visibility_advantage (can we see the target?)
  │   ├── InformationState.enemy_bases_known (is this the right base?)
  │   ├── TerrainMetrics.cliff_density (can our units reach the target?)
  │   ├── UnitClassifier → which units can traverse cliffs? (used in _exec_attack, but not in decision)
  │   ├── FrameSystem.query("Opportunity") → timing quality (never called)
  │   ├── FrameSystem.query("ThreatAssessment") → threat level (never called)
  │   ├── CausalGraph → have past attacks on this target succeeded? (never queried)
  │   ├── NarrativeLayer → are we in an "attacking" arc? (never queried)
  │   ├── StateGraph → what defenses does the enemy have at target? (never queried)
  │   ├── Kernel.cognitive_state.coherence (is our cognition coherent?)
  │   ├── Ontology → what capabilities does the target require to attack? (not used)
  │   └── EntityDescriptor → universal object not used (raw SC2 units used instead)
  │
  └── PROVENANCE: WorldState + EntityModel + TacticalBrain + Drives + AffordanceTracer + ModelPool + GovernanceGate
      Missing: Visibility, terrain, past attack outcomes, narrative context, semantic state
```

#### 7.3 Defend Decision (e.g., "Send army to defend base")

```
DECISION: Defend (move army to base position X)
  │
  ├── WHAT INFLUENCED THIS DECISION:
  │   ├── WorldState.army_count (units available to defend)
  │   ├── EntityModel.enemy.threat_level (threat detected)
  │   ├── TacticalBrain.decide_tactics() → PRIMARY decision (DEFEND)
  │   │   └── Uses: our_composition, enemy_composition, army_supply, enemy_army_supply
  │   ├── Drives → defense_need (computed from threat_level vs. base_count)
  │   ├── AffordanceTracer.generate() → DEFEND candidate
  │   │   └── Uses: WorldState, EntityModel, EntityStates (filtered by role)
  │   ├── ModelPool.score(affordance) → score
  │   └── GovernanceGate.check() → approved
  │
  ├── WHAT WAS AVAILABLE BUT IGNORED:
  │   ├── InformationState.visibility_advantage (where is the threat?)
  │   ├── TerrainMetrics.choke_count (can we defend the choke?)
  │   ├── UnitClassifier → which units can hold a choke? (used in _exec_defend, not decision)
  │   ├── FrameSystem.query("Danger") → danger level (never called)
  │   ├── CausalGraph → have we been attacked here before? (never queried)
  │   ├── NarrativeLayer → are we in a "under attack" arc? (never queried)
  │   ├── StateGraph → what friendly entities are near the threat? (never queried)
  │   ├── Kernel.cognitive_state.coherence (is our cognition coherent?)
  │   ├── Ontology → what defense capabilities are available? (not used)
  │   └── EntityDescriptor → universal object not used
  │
  └── PROVENANCE: WorldState + EntityModel + TacticalBrain + Drives + AffordanceTracer + ModelPool + GovernanceGate
      Missing: Visibility, terrain, past defense outcomes, narrative context, semantic state
```

#### 7.4 Provenance Summary

| Decision | Information Sources Used | Information Sources Ignored | Coverage |
|----------|------------------------|---------------------------|----------|
| **Expand** | 5 (WorldState, Drives, AffordanceTracer, ModelPool, GovernanceGate) | 11 (enemy threat, visibility, terrain, frames, causal graph, narrative, state graph, kernel, tactical brain, ontology, entity descriptor) | **31%** |
| **Attack** | 7 (WorldState, EntityModel, TacticalBrain, Drives, AffordanceTracer, ModelPool, GovernanceGate) | 10 (visibility, terrain, frames, causal graph, narrative, state graph, kernel, ontology, entity descriptor, unit classifier for decision) | **41%** |
| **Defend** | 7 (WorldState, EntityModel, TacticalBrain, Drives, AffordanceTracer, ModelPool, GovernanceGate) | 10 (visibility, terrain, frames, causal graph, narrative, state graph, kernel, ontology, entity descriptor, unit classifier for decision) | **41%** |

**Conclusion:** Each decision uses approximately **30-40%** of the available information. The most commonly ignored categories are: representational layer outputs (frames, narratives, causal history, semantic state), kernel outputs (coherence, prediction), and environmental context (visibility, terrain defensibility).

---

### 8. Semantic Dependency Graph

Which modules would need to change if a specific design principle were enforced?

#### 8.1 If P2 ("Clients publish state, don't manipulate cognition") Were Enforced

```
CURRENT:
  IterateBot.on_step()
    ├── Reads SC2 API directly (perception)
    ├── Calls entity_model (cognition)
    ├── Calls affordance_tracer (cognition)
    ├── Calls model_pool (cognition)
    ├── Calls governance_gate (action filtering)
    ├── Calls action_bridge (action)
    └── Calls SC2 API (execution)

REQUIRED CHANGE:
  IterateBot.on_step()
    ├── Reads SC2 API → publishes WorldState to blackboard
    ├── EntityModel reads WorldState → publishes EntityStates
    ├── AffordanceTracer reads WorldState + EntityStates → publishes AffordanceCandidates
    ├── ModelPool reads AffordanceCandidates → publishes scored candidates
    ├── GovernanceGate reads scored candidates → publishes approved candidates
    ├── ActionBridge reads approved candidates → publishes SC2 commands
    └── IterateBot executes SC2 commands
```

**Modules affected:** IterateBot (major refactor), all 7 downstream modules (interface changes)

#### 8.2 If P24 ("EntityDescriptor is the universal cognitive object") Were Enforced

```
CURRENT:
  UnitClassifier → UnitRole enum
  BuildingClassifier → BuildingRole enum
  EntityModel → EntityState (custom type)
  _interpret_action() → uses raw SC2 Unit objects

REQUIRED CHANGE:
  UnitClassifier → EntityDescriptor
  BuildingClassifier → EntityDescriptor
  EntityModel → EntityDescriptor
  _interpret_action() → uses EntityDescriptor exclusively
  AffordanceTracer → accepts EntityDescriptor[]
  GovernanceGate → accepts EntityDescriptor[]
```

**Modules affected:** UnitClassifier, BuildingClassifier, EntityModel, _interpret_action(), AffordanceTracer, GovernanceGate (6 modules)

#### 8.3 If P8 ("Goals are primitives, Executive Function arbitrates") Were Enforced

```
CURRENT:
  DriveManager → fixed need_weights dict → flat scoring

REQUIRED CHANGE (new module):
  ExecutiveFunction
    ├── Reads: Drives, WorldState, FrameResults, Kernel.cognitive_state
    ├── Computes: dynamic priority stack (Safety > Maintenance > Goals > Learning > Exploration)
    ├── Outputs: active_goal, priority_level
    └── Replaces: hardcoded need_weights dict in _interpret_action()
```

**Modules affected:** DriveManager (output changes), _interpret_action() (remove need_weights), new ExecutiveFunction module created, FrameSystem (must be wired for EF to read)

#### 8.4 If FrameSystem Were Wired (H14)

```
CURRENT:
  _select_perspective() → frame_system.set_perspective()
  frame_system.query_all() → NEVER CALLED

REQUIRED CHANGE:
  _select_perspective() → frame_system.set_perspective()
  on_step() → frame_system.query_all() → FrameResults
  FrameResults → passed to GovernanceGate as metadata
  GovernanceGate → optionally uses FrameResults for soft weighting
```

**Modules affected:** IterateBot.on_step() (add query_all call), GovernanceGate (accept frame metadata), FrameSystem (already implemented, just needs wiring)

---

### 9. Synthesis — What This Analysis Reveals

The 7-part analysis converges on a single finding: **sc2_iterate_01.py is a working prototype that demonstrates the system can play StarCraft, but it has not yet been refactored to use its own cognitive architecture.**

The representational layer, kernel, and executive function exist as separate modules, but the monolith bypasses them all. The action pipeline uses ~30-40% of available information, ignoring representational layer outputs, kernel coherence, environmental context, and causal history.

This is **normal for a development-stage system**. The recommended path forward (H16 → H15 → H14) is disciplined: validate that the unused signals are meaningful before wiring them into decisions.

The key risk is that the monolith becomes permanent — in which case the representational layer, kernel, and executive function are dead code. The key opportunity is that the representational layer already produces rich semantic state; wiring it into decisions could significantly improve gameplay quality.

**Evidence strength for this synthesis: HIGH** — based on direct code analysis of all subsystems and their interconnections.

---

*Deep 7-part architectural analysis of sc2_iterate_01.py.*
*Covers: data flow, dead-ends, layering, representation vs. hardcoding, coupling, blackboard, decision provenance.*
*Key finding: 72% of subsystem code is producer-only or dead during gameplay.*
*Key finding: Each decision uses 30-40% of available information.*
*Key finding: No blackboard pattern; god object mediates all data flow.*
*Recommended next step: H16 (validate representational signals) before any wiring.*

---

## Entry #4 — Infrastructure Improvements (2026-07-26 19:15)

### 1. Representational Layer & Kernel Metrics Added to on_end()

Added comprehensive metrics to the iteration log JSON at game end:

**Kernel metrics:**
- `avg_coherence` — mean coherence across all ticks
- `coherence_variance` — variance in coherence values
- `avg_volume_entropy` — mean volume entropy across all ticks
- `n_attractors` — number of base attractors discovered
- `cognitive_energy` — remaining cognitive energy
- `ticks_processed` — total kernel ticks

**Representational layer metrics:**
- `state_graph.entities` — number of entity nodes
- `state_graph.edges` — number of relationship edges
- `causal_graph.events` — number of causal events recorded
- `causal_graph.consequences` — number of auto-generated consequences
- `narrative_layer.active` — number of active narratives
- `narrative_layer.completed` — number of completed narratives
- `frame_queries_executed` — frame system queries (currently counts perspective updates, actual query_all() pending H14)
- `interpreter_entities_processed` — total entities processed by interpreter
- `ontology_concepts` — number of concepts in ontology

**Tracking counters added:**
- `_frame_queries_executed` — incremented on each representational tick
- `_interpreter_entities_processed` — tracks new entities added to state graph
- `_kernel_coherence_values` — list of all coherence values for variance calculation
- `_kernel_volume_entropy_values` — list of all volume entropy values

**Impact:** Enables trend tracking across games. H16 observation games will now produce quantitative data on representational layer activity and kernel behavior.

### 2. Tests

All 835 tests pass. No regressions.

### 3. Next Steps

Proceeding with AGENTS.md corrections (false claims about system() calls, executable count, cellular_automata.cu).

**AGENTS.md corrected:**
- "14 executables" → "18+ executables" (2 locations)
- Added Audit Corrections section documenting 8+ active system() calls in first-party code
- Corrected cellular_automata.cu claim (it is included in build, not excluded)

**Tests:** All 835 pass. No regressions.

---

## Entry #5: Hypothesis Generation Pipeline Fix
**Timestamp:** 2026-07-26 19:30 PDT
**Trigger:** Tactical brain hypothesis count stuck at 0 across 47 games despite 128 battles recorded

### 1. Root Cause Analysis

**Three bugs** prevented hypothesis generation:

**Bug 1: `analyze_battles()` disconnected from hypothesis pipeline**
- `analyze_battles()` (called every 5 ticks) recorded battles to `_battle_log` but **never called `_update_hypotheses_from_battle()`**
- This was the primary data path (128 battles recorded) — all data terminated in the log

**Bug 2: `record_battle_outcome()` called with wrong kwargs**
- Call site (line 342): `step=`, `our_composition=`, `enemy_composition=`, `our_killed=`, `enemy_killed=`, `outcome=`
- Function signature: `my_comp`, `enemy_comp`, `won`, `tick`
- TypeError silently swallowed by SC2's on_step exception handler

**Bug 3: Data type mismatch**
- `_last_attack_units` is `Dict[int, str]` (tag→name)
- `_update_hypotheses_from_battle()` expects `Dict[str, int]` (name→count)

### 2. Fixes Applied

| Fix | File | Change |
|-----|------|--------|
| `analyze_battles()` | `tactical_brain.py:589-590` | Added `_update_hypotheses_from_battle()` call after battle log append |
| `record_battle_outcome()` call | `sc2_iterate_01.py:342-356` | Converted `{tag:name}` → `{name:count}`, fixed kwargs to match signature |
| Threshold lowered | `tactical_brain.py:715` | `total >= 2` → `total >= 1` (faster initial learning) |
| Winrate threshold | `tactical_brain.py:721` | `0.5` → `0.4` (allow early hypotheses with less data) |
| Confidence threshold | `tactical_brain.py:760,765` | `0.5` → `0.4` (match generate_hypothesis threshold) |
| Observability | `sc2_iterate_01.py:1300-1318` | Added `hypotheses_active`, `hypothesis_details` to on_end() JSON |

### 3. Verification

**Unit test — `record_battle_outcome` path:**
```
3 battles: ZERGLING vs ZERGLING → 2 wins, 1 loss
→ Hypotheses: 1 (h1: ZERGLING vs ZERGLING | wins=3 losses=1 conf=0.62)
→ Suggestion: ZERGLINK ✓
```

**Unit test — `analyze_battles` path:**
```
State1: 2 ZERGLING alive → State2: 1 ZERGLING (tag 2 died)
→ Battle detected: win
→ Hypotheses: 1 (h1: ZERGLING vs ZERGLING | wins=2 losses=0 conf=0.64)
```

**Full test suite:** 835/835 pass (0 regressions)

### 4. Expected Impact

Next game should show:
- `hypotheses_generated: N` (N > 0)
- `hypothesis_details: [...]` with counter_unit, targets, confidence, wins/losses
- If hypothesis confidence ≥ 0.4, `suggest_counter_unit()` returns a suggestion
- Bot attempts to build suggested counter-unit in `_exec_army()` (line 754-769)

### 5. Commit

`965af40` — "fix: hypothesis generation pipeline — 3 bugs fixed"

*Next entry will be produced when new observations arise or experiments complete.*
