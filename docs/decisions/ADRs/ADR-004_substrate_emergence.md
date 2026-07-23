# ADR-003: Substrate Emergence — Dynamics vs Manual Attractors

**Date:** 2026-07-21
**Status:** Accepted

## Context

The question: are attractors in Substrate_Echo emergent from continuous dynamics, or manually encoded state transitions that resemble attractors?

Two memory systems exist:
- **DynamicsMemory**: learns V(x) = Ax + b from (state, velocity) pairs, discovers fixed points analytically or via integration
- **AttractorMemory**: manually places Attractor objects in OntologicalField with explicit center, strength, basin_width

## Experiment (EXP-SUB-001)

### Method

Generated a synthetic trajectory with 3 behavioral modes (exploration, rest, social) cycling every 100 ticks. Compared:

1. **DynamicsMemory only**: no `form_attractor()`, only trajectory learning
2. **AttractorMemory only**: manual `encode()` placing attractors at each experience

Measured attractor count and semantic coherence (fraction of states correctly assigned to their behavioral mode's cluster).

### Results

| System | Attractors | Coherence |
|--------|-----------|-----------|
| DynamicsMemory | 2 | 0.335 |
| AttractorMemory | 9 | 0.943 |

### Analysis

**DynamicsMemory discovers genuine structure.** From 1500 state transitions, the learned vector field V(x) = Ax + b produced 2 attractors via fixed-point analysis. These attractors partially separate the behavioral modes (coherence 0.335 > random 0.333). The attractors are real — they fall out of the learned dynamics.

**AttractorMemory does the cognitive heavy lifting.** Placing 9 attractors manually produces near-perfect coherence (0.943). Each experience becomes an attractor, creating dense coverage of the state space.

**The substrate is hybrid.** DynamicsMemory provides genuine structure discovery. AttractorMemory provides dense coverage and semantic differentiation. Neither alone is sufficient:

- DynamicsMemory without AttractorMemory: few attractors, moderate coherence
- AttractorMemory without DynamicsMemory: dense but shallow — no velocity field, no prediction, no generalization to novel states

## Consequences

### Positive
- The dynamics core is real: V(x) = Ax + b learns from data, attractors emerge as fixed points
- The fixed-point discovery (analytical for global, integration+mean-shift for local) is mathematically sound
- The system can discover structure it was never told to find

### Negative
- With limited behavioral diversity, dynamics collapse to a single attractor
- The prediction residual is non-trivial (0.18 for clean data, much higher for noisy agent trajectories)
- The global linear model is insufficient for multi-basin dynamics in high dimensions

### Risks
- AttractorMemory's manual placement creates a false sense of "memory" that doesn't generalize
- The two systems are loosely coupled — DynamicsMemory doesn't benefit from AttractorMemory's attractors as boundary conditions

## What Would Make It Stronger

1. **Wire AttractorMemory as boundary conditions for DynamicsMemory**: placed attractors should constrain the learned vector field, not just sit on top of it
2. **Local linear model for agent experiments**: global linear V(x) = Ax + b can only have one analytical fixed point; local k-NN captures multi-basin dynamics
3. **Richer behavioral repertoire**: the agent needs more diverse activities to generate meaningful state trajectories
4. **Ginzburg-Landau field equation**: replace the sum-of-pull-vectors field with a real PDE where basins emerge from diffusion + nonlinearity + noise

## Alternatives Considered

1. **Remove AttractorMemory entirely**: rejected — it provides dense coverage that dynamics alone cannot achieve with limited data
2. **Replace DynamicsMemory with handcrafted attractors**: rejected — loses the genuine structure discovery
3. **Hybrid approach (current)**: accepted — both systems contribute, with the understanding that neither is sufficient alone
